import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from data.text_folder import make_dataset as make_text_dataset
from data.text_folder import default_loader as text_default_loader
from PIL import Image
import random


class UnalignedPairedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the models with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """


    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # print('I am here')
        if opt.datatype == 'image':

            self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
            self.dir_medianB = os.path.join(opt.dataroot, opt.phase + 'medianB')  # create a path '/path/to/data/trainmedianB'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
            self.dir_medianA = os.path.join(opt.dataroot, opt.phase + 'medianA')  # create a path '/path/to/data/trainmedianA'

            self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'

            self.medianB_paths = sorted(make_dataset(self.dir_fakeB, opt.max_dataset_size))    # load images from '/path/to/data/trainmedianB'
            self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
            self.medianA_paths = sorted(make_dataset(self.dir_fakeA, opt.max_dataset_size))  # load images from '/path/to/data/trainmedianA'

            self.A_size = len(self.A_paths)  # get the size of dataset A
            self.B_size = len(self.B_paths)  # get the size of dataset B

            btoA = self.opt.direction == 'BtoA'
            input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
            output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image

            self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
            self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

        if opt.datatype == 'text':

            # print('text mode')
            self.dir_A = os.path.join(opt.dataroot, opt.phase + 'AG')  # create a path '/path/to/data/trainA'
            self.dir_medianB = os.path.join(opt.dataroot, opt.phase + 'NGS')  # create a path '/path/to/data/trainB'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + 'NGS')  # create a path '/path/to/data/trainB'
            self.dir_medianA = os.path.join(opt.dataroot, opt.phase + 'AG')  # create a path '/path/to/data/trainA'
            # self.dir_A = os.path.join(opt.dataroot, opt.phase + 'HELA')  # create a path '/path/to/data/trainA'
            # self.dir_medianB = os.path.join(opt.dataroot, opt.phase + 'MCF7')  # create a path '/path/to/data/trainB'
            # self.dir_B = os.path.join(opt.dataroot, opt.phase + 'MCF7')  # create a path '/path/to/data/trainB'
            # self.dir_medianA = os.path.join(opt.dataroot, opt.phase + 'HELA')  # create a path '/path/to/data/trainA'

            # self.dir_A = os.path.join(opt.dataroot, opt.phase + 'Affy')  # create a path '/path/to/data/trainA'
            # self.dir_medianB = os.path.join(opt.dataroot, opt.phase + 'AG2')  # create a path '/path/to/data/trainB'
            # self.dir_B = os.path.join(opt.dataroot, opt.phase + 'AG2')  # create a path '/path/to/data/trainB'
            # self.dir_medianA = os.path.join(opt.dataroot, opt.phase + 'Affy')  # create a path '/path/to/data/trainA'
            self.A_paths = sorted(make_text_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
            self.B_paths = sorted(make_text_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
            self.medianB_paths = sorted(make_text_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
            self.medianA_paths = sorted(make_text_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'

            self.A_size = len(self.A_paths)  # get the size of dataset A
            self.B_size = len(self.B_paths)  # get the size of dataset B

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths, B_paths, fakeA, fakeA_paths, fakeB and fakeB_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
            medianA (tensor)   -- an image of input domain corresponding to the median domain of G2
            medianB (tensor)   -- an image of output domain corresponding to the median domain of G1
            medianA_paths (str)    -- image paths
            medianB_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]


        medianA_path = self.medianA_paths[index_B]
        medianB_path = self.medianB_paths[index % self.A_size]

        # print('A_Path',A_path)
        # print('B_path',B_path)
        # print('medianA_path',medianA_path)
        # print('medianB_path',medianB_path)

        if self.opt.datatype == 'image':

            A_img = Image.open(A_path).convert('RGB')
            B_img = Image.open(B_path).convert('RGB')

            medianA_img = Image.open(medianA_path).convert('RGB')
            medianB_img = Image.open(medianB_path).convert('RGB')

            # apply image transformation
            A = self.transform_A(A_img)
            B = self.transform_B(B_img)

            medianA = self.transform_A(medianA_img)
            medianB = self.transform_B(medianB_img)

        if self.opt.datatype == 'text':
            try:
                A = text_default_loader(A_path)
                B = text_default_loader(B_path)
    
                medianA = text_default_loader(medianA_path)
                medianB = text_default_loader(medianB_path)
            except:
                A, B, medianA, medianB = 'na','na','na','na'

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'medianA': medianA, 'medianB': medianB,
               'medianA_paths': medianA_path, 'medianB_paths': medianB_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

