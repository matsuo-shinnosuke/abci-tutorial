import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y, transform):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.transform(self.X[idx])
        y = self.y[idx]
        return  {'X': X, 'y': y}
    
def set_loader(args):
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=args.dataset_dir, train=True, download=True)
        test_dataset = datasets.CIFAR10(root=args.dataset_dir, train=False, download=True)
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=args.dataset_dir, train=True, download=True)
        test_dataset = datasets.CIFAR100(root=args.dataset_dir, train=False, download=True)

    X_train, X_test = train_dataset.data, test_dataset.data
    y_train, y_test = np.array(train_dataset.targets), np.array(test_dataset.targets)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=args.seed)

    # ----
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # ----
    train_dataset = Dataset(X=X_train, y=y_train, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    val_dataset = Dataset(X=X_val, y=y_val, transform=test_transforms)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    test_dataset = Dataset(X=X_test, y=y_test, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    return train_loader, val_loader, test_loader