from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Dataset
import torch
import numpy as np

trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
trans_cifar100_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                                std=[0.267, 0.256, 0.276])])
trans_cifar100_test = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                               std=[0.267, 0.256, 0.276])])
trans_fashionmnist_train = transforms.Compose([transforms.Pad(2),
                                               transforms.RandomCrop(32, padding=4),
                                               transforms.Grayscale(num_output_channels=3),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5,), (0.5,))])
trans_fashionmnist_test = transforms.Compose([transforms.Pad(2),
                                              transforms.Grayscale(num_output_channels=3),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5,), (0.5,))])
trans_emnist_train = transforms.Compose([transforms.Pad(2),
                                         transforms.RandomCrop(32, padding=4),
                                         transforms.Grayscale(num_output_channels=3),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,))])
trans_emnist_test = transforms.Compose([transforms.Pad(2),
                                        transforms.Grayscale(num_output_channels=3),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])


def split_dataset(dataset, split_thresh):
    label_dict = {}
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label not in label_dict.keys():
            label_dict[label] = []
        label_dict[label].append(i)

    # fill a/b_dict as {labelA: [1st sampleA uidx, 2nd sampleA uidx, ...], ...}
    a_dict, b_dict = {}, {}
    for label, idx_list in label_dict.items():
        np.random.shuffle(idx_list)
        b_ct = int(split_thresh * len(idx_list))
        b_dict[label] = idx_list[:b_ct]
        a_dict[label] = idx_list[b_ct:]

    return a_dict, b_dict


def fetch_data(args):
    if args.dataset == 'cifar10':
        dataset_tr = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar10_train)
        dataset_vl = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar10_test)
        dataset_te = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=trans_cifar10_test)
        # dataset_tr and dataset_vl should have the same uidx, different is in the transform
        dict_tr_train, dict_tr_valid = split_dataset(dataset_tr, split_thresh=args.val_split)
        dict_te_test, _ = split_dataset(dataset_te, split_thresh=0)
    elif args.dataset == 'cifar100':
        dataset_tr = datasets.CIFAR100('data/cifar100', train=True, download=True, transform=trans_cifar100_train)
        dataset_vl = datasets.CIFAR100('data/cifar100', train=True, download=True, transform=trans_cifar100_test)
        dataset_te = datasets.CIFAR100('data/cifar100', train=False, download=True, transform=trans_cifar100_test)
        # dataset_tr and dataset_vl should have the same uidx, different is in the transform
        dict_tr_train, dict_tr_valid = split_dataset(dataset_tr, split_thresh=args.val_split)
        dict_te_test, _ = split_dataset(dataset_te, split_thresh=0)
    elif args.dataset == 'fashionmnist':
        dataset_tr = datasets.FashionMNIST('data/fashionmnist', train=True, download=True,
                                           transform=trans_fashionmnist_train)
        dataset_vl = datasets.FashionMNIST('data/fashionmnist', train=True, download=True,
                                           transform=trans_fashionmnist_test)
        dataset_te = datasets.FashionMNIST('data/fashionmnist', train=False, download=True,
                                           transform=trans_fashionmnist_test)
        dict_tr_train, dict_tr_valid = split_dataset(dataset_tr, split_thresh=args.val_split)
        dict_te_test, _ = split_dataset(dataset_te, split_thresh=0)
    elif args.dataset == 'emnist':
        dataset_tr = datasets.EMNIST('data/emnist',
                                     split='byclass',
                                     train=True,
                                     download=True,
                                     transform=trans_emnist_train)
        dataset_vl = datasets.EMNIST('data/emnist',
                                     split='byclass',
                                     train=True,
                                     download=True,
                                     transform=trans_emnist_test)
        dataset_te = datasets.EMNIST('data/emnist',
                                     split='byclass',
                                     train=False,
                                     download=True,
                                     transform=trans_emnist_test)
        dict_tr_train, dict_tr_valid = split_dataset(dataset_tr, split_thresh=args.val_split)
        dict_te_test, _ = split_dataset(dataset_te, split_thresh=0)
    else:
        exit("Dataset {} is not implemented".format(args.dataset))

    # dataset_??: original raw dataset from download
    # dict_??_*: sample_list-to-label mapping (label is key)
    return dataset_tr, dataset_vl, dataset_te, dict_tr_train, dict_tr_valid, dict_te_test


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
