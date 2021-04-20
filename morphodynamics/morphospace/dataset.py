import os
import glob
from PIL import Image
import cv2
import torch.utils.data as torch_data


class Dataset(torch_data.Dataset):
    """
    A class for feeding image data into the autoencoder
    """


    def __init__(self, transforms, RGB):
        """
        - transforms: transforms to apply to the images
        - RGB: whether to process in color
        """

        self.transforms = transforms
        self.RGB = RGB

        self.times = ['090min', '105min', '120min', '135min', '150min', '165min', '180min', '195min', '210min']
        self.drugnames = ['DMSO', 'compound_A', 'compound_X', 'compound_B', 'compound_C_0_041', 'compound_C_10']

        self.paths = []
        path_to_here = os.path.dirname(os.path.realpath(__file__))
        path_images = path_to_here+'/../data/images/'
        for time in self.times:
            for drug in self.drugnames:
                ims =  glob.glob(path_images + '{}/{}/*'.format(time, drug))
                self.paths += ims

        self.labels = [0 for path in self.paths]

    def drug_labelled(self):
        """
        Label the images by drug
        """

        self.labels = []
        for path in self.paths:
            for idx, drug_name in enumerate(self.drugnames):
                if path.split('/')[-2] == drug_name:
                    self.labels.append(idx)


    def keep_oneTime_drugLabelled(self, time_0to8):
        """
        Remove images from all but one time, and label by drug
        """
        keep_idxs = []
        for idx, path in enumerate(self.paths):
            if path.split('/')[-3] == self.times[time_0to8]:
                keep_idxs.append(idx)

        self.paths = [self.paths[idx] for idx in keep_idxs]

        self.labels = []
        for path in self.paths:
            for idx, drug_name in enumerate(self.drugnames):
                if path.split('/')[-2] == drug_name:
                    self.labels.append(idx)



    def keep_oneDrug_timeLabelled(self, drug_name):
        """
        Remove images from all but one drug, and label by time
        """

        keep_idxs = []
        for idx, path in enumerate(self.paths):
            if path.split('/')[-2] == drug_name:
                keep_idxs.append(idx)


        self.paths = [self.paths[idx] for idx in keep_idxs]

        self.labels = []
        for path in self.paths:
            time = path.split('/')[-3]
            self.labels.append(self.times.index(time))


    def keep_severalDrugs_timeLabelled(self, drug_names):
        """
        Remove images from all but some drugs, and label by time
        """

        keep_idxs = []
        for idx, path in enumerate(self.paths):
            for drug_name in drug_names:
                if path.split('/')[-2] == drug_name:
                    keep_idxs.append(idx)


        self.paths = [self.paths[idx] for idx in keep_idxs]

        self.labels = []
        for path in self.paths:
            time = path.split('/')[-3]
            self.labels.append(self.times.index(time))



    def __getitem__(self, index):

        path = self.paths[index]
        label = self.labels[index]

        image = cv2.imread(self.paths[index], 0)

        image = Image.fromarray(image)
        if self.RGB == True:
            image = image.convert('RGB')
        image = self.transforms(image)

        return (path, image, label)


    def __len__(self):
        return len(self.labels)
