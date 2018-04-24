from __future__ import print_function
from PIL import Image
import os
import os.path
import errno
import torch

import numpy as np
import struct
from torchvision.datasets.mnist import MNIST


class SmallNORB(MNIST):
    """`SmallNORB <https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz",
        "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz",
        "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz",
        "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz"
    ]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def download(self):
        """Download the SmallNORB data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat')),
            read_label_file(os.path.join(self.root, self.raw_folder, 'smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat'))
        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat')),
            read_label_file(os.path.join(self.root, self.raw_folder, 'smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    @staticmethod
    def _parse_small_NORB_header(file_pointer):
        """
        from: https://github.com/ndrplz/small_norb/blob/master/smallnorb/dataset.py
        Parse header of small NORB binary file

        Parameters
        ----------
        file_pointer: BufferedReader
            File pointer just opened in a small NORB binary file
        Returns
        -------
        file_header_data: dict
            Dictionary containing header information
        """
        # Read magic number
        magic = struct.unpack('<BBBB', file_pointer.read(4))  # '<' is little endian)

        # Read dimensions
        dimensions = []
        num_dims, = struct.unpack('<i', file_pointer.read(4))  # '<' is little endian)
        for _ in range(num_dims):
            dimensions.extend(struct.unpack('<i', file_pointer.read(4)))

        file_header_data = {'magic_number': magic,
                            'matrix_type': SmallNORB.matrix_type_from_magic(magic),
                            'dimensions': dimensions}
        return file_header_data

    @staticmethod
    def matrix_type_from_magic(magic_number):
        """
        from: https://github.com/ndrplz/small_norb/blob/master/smallnorb/dataset.py
        Get matrix data type from magic number
        See here: https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/readme for details.
        Parameters
        ----------
        magic_number: tuple
            First 4 bytes read from small NORB files
        Returns
        -------
        element type of the matrix
        """
        convention = {'1E3D4C51': 'single precision matrix',
                      '1E3D4C52': 'packed matrix',
                      '1E3D4C53': 'double precision matrix',
                      '1E3D4C54': 'integer matrix',
                      '1E3D4C55': 'byte matrix',
                      '1E3D4C56': 'short matrix'}
        magic_str = bytearray(reversed(magic_number)).hex().upper()
        return convention[magic_str]


def read_label_file(path):
    with open(path, mode='rb') as f:
        header = SmallNORB._parse_small_NORB_header(f)

        num_examples, = header['dimensions']

        struct.unpack('<BBBB', f.read(4))  # ignore this integer
        struct.unpack('<BBBB', f.read(4))  # ignore this integer

        labels = np.zeros(shape=num_examples, dtype=np.int32)
        for i in range(num_examples):
            category, = struct.unpack('<i', f.read(4))
            labels[i] = category
        labels = labels.repeat(2) # use both left and right image
        return torch.LongTensor(labels)


def read_image_file(path):
    with open(path, mode='rb') as f:

        header = SmallNORB._parse_small_NORB_header(f)

        num_examples, channels, height, width = header['dimensions']

        examples = np.zeros(shape=(num_examples * channels, height, width), dtype=np.uint8)

        for i in range(num_examples * channels):
            # Read raw image data and restore shape as appropriate
            image = struct.unpack('<' + height * width * 'B', f.read(height * width))
            image = np.uint8(np.reshape(image, newshape=(height, width)))

            examples[i] = image

    return torch.ByteTensor(examples)
