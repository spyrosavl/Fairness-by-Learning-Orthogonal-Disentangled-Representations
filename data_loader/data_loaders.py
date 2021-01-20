import os
import torch
import numpy as np
from torchvision import datasets, transforms
import torchtext
from torch.utils.data import DataLoader, Dataset
from base import BaseDataLoader
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from torch.utils.data.sampler import SubsetRandomSampler


class CIFAR100DataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor()
        ])
        self.data_dir = data_dir
        self.dataset = datasets.CIFAR100(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class CIFAR10DataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.1, num_workers=2, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor()
        ])
        self.data_dir = data_dir
        self.dataset = datasets.CIFAR10(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class MnistDataLoader(BaseDataLoader):
    
    ### MNIST data loading demo using BaseDataLoader
    
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        print(self.dataset)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class YaleDataLoader(BaseDataLoader):    
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = datasets.ImageFolder(self.data_dir)
        print(self.dataset)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class collator(object):
    def __init__(self, device='cpu'):
        self.device = device

    def __call__(self, batch):
        data, sensitive, targets = map(list, zip(*batch))
        data = np.asarray(data)
        outs = []
        for column in data.T:
            if len(np.unique(column)) == 2:
                if (np.unique(column) == [0,1]).all():
                    outs.append(column)
                else:
                    outs.append(column)
            else:
                outs.append(normalize(column.reshape(1, -1))[0]) #TODO check again reshaping
        data = torch.tensor(outs).T
        sensitive = torch.tensor(sensitive)
        targets = torch.tensor(targets)
        return data, sensitive, targets


class GermanDataLoader(BaseDataLoader):
    def __init__(self, data_dir=None, batch_size=16, shuffle=False, validation_split=0.1, num_workers=2):
        trsfm = None #TODO
        txt_file = 'german.data'
        self.dataset = GermanCreditDatasetOneHot(txt_file, data_dir, trsfm)
        self.collator = collator()
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, self.collator)


class GermanCreditDatasetOneHot(Dataset):
    def __init__(self, txt_file, data_dir=None, text_transforms=None):
        self.txt_file = txt_file
        self.data_dir = data_dir
        self.text_transforms = text_transforms
        self.categorical_columns = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]
        self.rows, self.targets, self.sensitive = self.get_data(os.path.join(self.data_dir, self.txt_file))
        self.features = self.get_onehot_attributes(self.rows, self.categorical_columns)

    def get_data(self, _file):
        rows=[];  targets=[]; sensitive=[]
        with open(_file) as txt_file:
          lines = txt_file.readlines()
          for l in lines:
            r = l.split()
            rows.append(r[:-1])
            targets.append(int(r[-1]))
            if r[8] == 'A91' or r[8] == 'A93' or r[8] == 'A94':
                sensitive.append(0)
            else:
                sensitive.append(1)
        cat = np.unique(sensitive)
        cat = list(set(cat))
        cat.sort()
        one_hot = MultiLabelBinarizer(classes=cat).fit([cat])
        sensitive = one_hot.transform(np.asarray(sensitive)[:,None])
        return rows, targets, sensitive

    def get_onehot_attributes(self, rows, columns):
      rows = np.asarray(rows)
      features = None
      for i in range(len(rows[0])):
        if i in columns:
          occ = rows[:,i]
          cat = np.unique(occ)
          cat = list(set(cat))
          cat.sort()
          one_hot = MultiLabelBinarizer(classes=cat).fit([cat])
          transformed = one_hot.transform(occ[:,None])
          if features is not None:
            features = np.column_stack((features, transformed))
          else:
            features = transformed
        else:
          features = np.column_stack((features, rows[:,i,None].astype(int)))
      return features

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        preprocessed_data = self.features[idx]
        if self.text_transforms is not None:
            preprocessed_data = self.text_transforms(preprocessed_data)
        label = 0 if self.targets[idx] == 2 else 1
        sensitive = self.sensitive[idx]
        return preprocessed_data, sensitive, label


class AdultDataLoader(BaseDataLoader):
    def __init__(self, data_dir=None, batch_size=16, shuffle=False, validation_split=0.1, num_workers=2):
        trsfm = None #TODO
        training_file, test_file = 'adult.data', 'adult.test'
        self.dataset = AdultDatasetOneHot(training_file, test_file, data_dir, trsfm)
        self.collator = collator()
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, self.collator, 		
			validation_appended_len=self.dataset.validation_len)


class AdultDatasetOneHot(Dataset):
    def __init__(self, training_file, test_file, data_dir=None, batch_size=16, shuffle=False, validation_split=0.1, num_workers=2):
        self.training_file, self.test_file = training_file, test_file
        self.data_dir = data_dir
        self.text_transforms = None
        self.categorical_columns = [1, 3, 5, 6, 7, 8, 9, 13]
        self.rows, self.targets, self.sensitive = self.get_data(os.path.join(self.data_dir, self.training_file))
        val_rows, val_targets, val_sensitive = self.get_data(os.path.join(self.data_dir, self.test_file))
        self.rows += val_rows; self.targets += val_targets; self.sensitive = np.vstack((self.sensitive, val_sensitive)) #append validation set in the end
        self.training_len, self.validation_len = len(self.rows), len(val_rows)
        self.features = self.get_onehot_attributes(self.rows, self.categorical_columns)

    def get_data(self, _file):
        rows=[];  targets=[]; sensitive=[]
        with open(_file) as txt_file:
          lines = txt_file.readlines()
          if lines[-1] == '\n':
            lines = lines[:-1]
          if lines[0][0] == '|': #for adult test, remove first line
            lines = lines[1:]
          for l in lines:
            r = l.split(",")
            rows.append(r[:-1])
            if len(r) > 1:
              targets.append(r[-1])
              sensitive.append(r[9])
        cat = np.unique(sensitive)
        cat = list(set(cat))
        cat.sort()
        one_hot = MultiLabelBinarizer(classes=cat).fit([cat])
        sensitive = np.asarray(sensitive)
        sensitive = one_hot.transform(sensitive[:,None])
        return rows, targets, sensitive

    def get_onehot_attributes(self, rows, columns):
      rows = np.asarray(rows)
      features = None
      for i in range(len(rows[0])):
        if i in columns:
          occ = rows[:,i]
          cat = np.unique(occ)
          cat = list(set(cat))
          cat.sort()
          one_hot = MultiLabelBinarizer(classes=cat).fit([cat])
          transformed = one_hot.transform(occ[:,None])
          if features is not None:
              features = np.column_stack((features, transformed))
          else:
            features = transformed
        else:
          if features is not None:
            features = np.column_stack((features, normalize(rows[:,i,None].astype(int))))
          else:
            features = normalize(rows[:,i,None].astype(int))       
      return features

    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self, idx):
        preprocessed_data = self.features[idx]
        if self.text_transforms is not None:
            preprocessed_data = self.text_transforms(preprocessed_data)
        #label = np.array(0) if '<=50K' in self.targets[idx] else np.array(1)
        label = 0 if '<=50K' in self.targets[idx] else 1
        sensitive = self.sensitive[idx]
        return preprocessed_data, sensitive, label


if __name__ == '__main__':
    german_dataset = GermanDataLoader('../data/')
    adult_dataset = AdultDatasetOneHot('adult.data', './data/')
    german_dataloader = DataLoader(german_dataset, batch_size=16)
    cifar10 = CIFAR10DataLoader('./', batch_size=64)
    cifar100 = CIFAR100DataLoader('./', batch_size=16)
