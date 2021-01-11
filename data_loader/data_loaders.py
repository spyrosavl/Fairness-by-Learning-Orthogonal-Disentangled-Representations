from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from base import BaseDataLoader


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


class GermanCreditDataset(Dataset):
    def __init__(self, csv_file, root_dir=None, text_transforms=None):
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.text_transforms = text_transforms
        self.rows, self.targets, self.sensitive = self.get_data(self.csv_file)

    def get_data(self, _file):
        rows=[];  targets=[]; sensitive=[]
        with open(_file) as csv_file:
          lines = csv_file.readlines()
          for l in lines:
            r = l.split(",")[0].split()
            r = [int(i) for i in r]
            rows.append(r[:-1])
            targets.append(r[-1])
            sensitive.append(r[8])
        return rows, targets, sensitive

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        preprocessed_data = self.rows[idx]
        if self.text_transforms is not None:
            preprocessed_data = self.text_transforms(preprocessed_data)

        label = 0 if self.targets[idx] is 2 else 1
        sensitive = self.sensitive[idx]
        return preprocessed_data, sensitive, label


if __name__ == '__main__':
    german_dataset = GermanCreditDataset('./Downloads/GermanCreditDataset/german.data-numeric')
    german_dataloader = DataLoader(german_dataset, batch_size=16)
    cifar10 = CIFAR10DataLoader('./', batch_size=16)
    cifar100 = CIFAR100DataLoader('./', batch_size=16)

