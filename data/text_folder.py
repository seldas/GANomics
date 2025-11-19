"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""
import csv, torch

import torch.utils.data as data

from PIL import Image
import os

TXT_EXTENSIONS = [
    '.txt',
]


def is_text_file(filename):
    return any(filename.endswith(extension) for extension in TXT_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    files = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_text_file(fname):
                path = os.path.join(root, fname)
                files.append(path)
    return files[:min(max_dataset_size, len(files))]


def default_loader(path, normalize=False):
    with open(path, newline='') as file:
        reader = csv.reader(file, delimiter='\t')
        nums = []
        for i in reader.__next__():
            if len(i) > 0:
                nums.append(float(i))

    if normalize:
        # DONE: min-max normalization
        norm_nums = []
        min_val, max_val = min(nums), max(nums)
        for i in nums:
            norm_nums.append((i - min_val) / (max_val - min_val))
        return torch.tensor(norm_nums).unsqueeze(-1)
    else:
        return torch.tensor(nums).unsqueeze(-1)


class TextFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        txts = make_dataset(root)
        if len(txts) == 0:
            raise (RuntimeError("Found 0 text files in: " + root + "\n"
                                                                   "Supported text extensions are: " + ",".join(
                TXT_EXTENSIONS)))

        self.root = root
        self.txts = txts
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.txts[index]
        txt = self.loader(path)
        if self.transform is not None:
            txt = self.transform(txt)
        if self.return_paths:
            return txt, path
        else:
            return txt

    def __len__(self):
        return len(self.txts)
