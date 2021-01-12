import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from base import BaseDataLoader
from sklearn.preprocessing import MultiLabelBinarizer


class CIFAR100DataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor()
        ])
        self.data_dir = data_dir
        self.dataset = datasets.CIFAR100(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class CIFAR10DataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor()
        ])
        self.data_dir = data_dir
        self.dataset = datasets.CIFAR10(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class GermanCreditDatasetOneHot(Dataset):
    def __init__(self, txt_file, root_dir=None, text_transforms=None):
        self.txt_file = txt_file
        self.root_dir = root_dir
        self.text_transforms = text_transforms
        self.categorical_columns = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]
        self.rows, self.targets, self.sensitive = self.get_data(self.txt_file)
        self.features = self.get_onehot_attributes(self.rows, self.categorical_columns)

    def get_data(self, _file):
        rows=[];  targets=[]; sensitive=[]
        with open(_file) as txt_file:
          lines = txt_file.readlines()
          for l in lines:
            r = l.split(",")[0].split()
            rows.append(r[:-1])
            targets.append(r[-1])
            sensitive.append(r[8])

        cat = np.unique(sensitive)
        cat = list(set(cat))
        cat.sort()
        one_hot = MultiLabelBinarizer(classes=cat).fit(cat)
        sensitive = one_hot.transform(sensitive)
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
          one_hot = MultiLabelBinarizer(classes=cat).fit(cat)
          transformed = one_hot.transform(occ)
          if features is not None:
            features = np.column_stack((features, transformed))
          else:
            features = transformed
        else:
          features = np.column_stack((features, rows[:,i,None]))
      features = np.asarray([[int(i) for i in j] for j in features])        
      return features

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        preprocessed_data = self.features[idx]
        if self.text_transforms is not None:
            preprocessed_data = self.text_transforms(preprocessed_data)
        label = 0 if self.targets[idx] is 2 else 1
        sensitive = self.sensitive[idx]
        return preprocessed_data, sensitive, label


class AdultDatasetOneHot(Dataset):
    def __init__(self, txt_file, root_dir=None, text_transforms=None):
        self.txt_file = txt_file
        self.root_dir = root_dir
        self.text_transforms = text_transforms
        self.categorical_columns = [1, 3, 5, 6, 7, 8, 9, 13]
        self.rows, self.targets, self.sensitive = self.get_data(self.txt_file)
        self.features = self.get_onehot_attributes(self.rows, self.categorical_columns)

    def get_data(self, _file):
        rows=[];  targets=[]; sensitive=[]
        with open(_file) as txt_file:
          lines = txt_file.readlines()
          for l in lines:
            r = l.split(",")
            rows.append(r[:-1])
            if len(r) > 1:
              targets.append(r[-1])
              sensitive.append(r[9])
        cat = np.unique(sensitive)
        cat = list(set(cat))
        cat.sort()
        one_hot = MultiLabelBinarizer(classes=cat).fit(cat)
        sensitive = one_hot.transform(sensitive)
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
          one_hot = MultiLabelBinarizer(classes=cat).fit(cat)
          transformed = one_hot.transform(occ)
          if features is not None:
            features = np.column_stack((features, transformed))
          else:
            features = transformed
        else:
          try:
            if features is not None:
              features = np.column_stack((features, rows[:,i,None]))
            else:
              features = rows[:,i,None]
          except:
            import pdb; pdb.set_trace()
      features = np.asarray([[int(i) for i in j] for j in features])        
      return features

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        preprocessed_data = self.features[idx]
        if self.text_transforms is not None:
            preprocessed_data = self.text_transforms(preprocessed_data)
        sensitive = self.sensitive[idx]
        label = 0 if '<50K' in self.targets[idx] else 1
        return preprocessed_data, sensitive, label


if __name__ == '__main__':
    german_dataset = GermanCreditDataset('./Downloads/GermanCreditDataset/german.data-numeric')
    german_dataloader = DataLoader(german_dataset, batch_size=16)
    cifar10 = CIFAR10DataLoader('./', batch_size=16)
    cifar100 = CIFAR100DataLoader('./', batch_size=16)

