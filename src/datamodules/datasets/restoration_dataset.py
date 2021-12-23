import os
import cv2
from torch.utils.data import Dataset
from albumentations import HorizontalFlip, RandomCrop, CenterCrop, ShiftScaleRotate, GaussNoise
from albumentations.core.composition import Compose
import random
from src.utils.utils import pad_image
import torchvision.transforms.functional as F

from torchvision.transforms import transforms
import torch
import cv2 as cv
import os
from glob import glob


class RestorationDataset(Dataset):
    """
    Dataset for super-resolution training
    """

    def __init__(self, img_dir, mode: str = "train", train_val_test_split: list = (100, 10, 10), shape=(128, 128),
                 corruption_type="GAUSSIAN_NOISE"):
        """
        :param img_dir: path to dir with simple_images
        :param mode: 'train' or 'test'
        """
        self.mode = mode
        self.img_dir = img_dir
        self.image_shape = shape
        self.corruption_type = corruption_type
        self.train_val_test_split = train_val_test_split

        # transformations
        self.transform_gt = self.get_transform()
        self.transform_corrupt = self.get_corrupt_transform()

        # train/test split
        self.test_set_size = self.train_val_test_split[2]

        images_names = [y for x in os.walk(img_dir) for y in glob(os.path.join(x[0], '*.jpg'))]

        print(f"{len(images_names)=}")

        random.seed(1)
        random.shuffle(images_names)

        if self.mode == 'train':
            self.data = images_names[self.test_set_size:sum(self.train_val_test_split)]
        elif self.mode == 'test':
            self.data = images_names[:self.test_set_size]
        else:
            print(f"Invalid mode {self.mode}")

        # normalization
        self.norm = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                             std=[0.5, 0.5, 0.5])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data[idx])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv.resize(image, dsize=self.image_shape)

        image = self.transform_gt(image=image)['image']

        corrupted = self.transform_corrupt(image=image)['image'] / 255.0
        image = image / 255.0

        image = self.norm(image)
        corrupted = self.norm(corrupted)

        return corrupted.type(torch.FloatTensor), image.type(torch.FloatTensor)

    def get_transform(self):
        transforms_list = [
            HorizontalFlip(),
            ShiftScaleRotate(),
            RandomCrop(height=self.image_shape[0], width=self.image_shape[1]),
        ]

        return Compose(transforms_list)

    def get_corrupt_transform(self):
        transforms_list = [
            GaussNoise(var_limit=(100.0, 150.0), mean=0, per_channel=False, always_apply=False, p=1)
        ]

        return Compose(transforms_list)


if __name__ == "__main__":
    d = RestorationDataset("/home/bohdan/Documents/UCU/3/AI/textures_super-resolution/data/")
    print(len(d))
