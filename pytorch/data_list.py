import numpy as np
from PIL import Image
from torch.utils.data import Dataset

def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split(' ')) > 2:
        images = [(val.split(' ')[0], np.array([int(la) for la in val.split(' ')[1:]])) for val in image_list]
      else:
        images = [(val.split(' ')[0], int(val.split(' ')[1])) for val in image_list]
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None,
                 mode='RGB', val_ratio=0, ds_type='train'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        from sklearn.model_selection import train_test_split
        if val_ratio == 0:
            train_imgs = imgs
            val_imgs = None
        else:
            train_imgs, val_imgs = train_test_split(imgs, test_size = int(len(imgs) * val_ratio), random_state = 42)
        if ds_type == 'train':
            self.imgs = train_imgs
        else:
            self.imgs = val_imgs

        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class MultiTransImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None,
                 mode='RGB', val_ratio=0, ds_type='train', m = 4):
        self.m = m
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        from sklearn.model_selection import train_test_split
        if val_ratio == 0:
            train_imgs = imgs
            val_imgs = None
        else:
            train_imgs, val_imgs = train_test_split(imgs, test_size = int(len(imgs) * val_ratio), random_state = 42)
        if ds_type == 'train':
            self.imgs = train_imgs
        else:
            self.imgs = val_imgs

        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        imgs_aug = []
        if self.transform is not None:
            imgs_aug.append(self.transform[0](img))
            if type(self.transform[1]) == list:
                for i in range(10):
                    imgs_aug.append(self.transform[1][i](img))
            else:
                for i in range(self.m):
                    imgs_aug.append(self.transform[1](img))
        if self.target_transform is not None:
            target = self.target_transform(target)

        return imgs_aug, target

    def __len__(self):
        if self.imgs is None:
            return 0
        return len(self.imgs)
