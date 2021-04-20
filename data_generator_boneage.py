import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io
import PIL
from torchvision import transforms
import pandas as pd
from tqdm import tqdm


class BoneAgeDataset(Dataset):
    """
    Loads the rsna bone age data set
    """

    def __init__(self, data_dir='./rsna-bone-age/',
                 resize_to=(256, 256), augment=False, preload=False, preloaded_data=None):
        """
        Given the root directory of the dataset, this function initializes the
        data set

        :param data_dir: List with paths of raw images
        """

        self._resize_to = resize_to
        self._data_dir = data_dir
        self._augment = augment
        self._preload = preload

        self._df = pd.read_csv(self._data_dir+f'/boneage-training-dataset.csv')

        self._img_file_names = []
        self._labels = []

        for i in range(self._df.shape[0]):
            self._img_file_names.append(self._data_dir+f"/boneage-training-dataset/boneage-training-dataset/"
                                                       f"{self._df['id'][i]}.png")
            self._labels.append(self._df['boneage'][i])

        # normalize labels
        self._labels = np.array(self._labels, dtype=np.float64)
        self._labels = self._labels - self._labels.min()
        self._labels = self._labels / self._labels.max()
        self._labels = torch.tensor(self._labels).float().unsqueeze(-1)

        self._imgs = []

        if self._preload:
            for fname in tqdm(self._img_file_names):
                x = io.imread(fname, as_gray=True)
                x = np.atleast_3d(x)
                max_size = np.max(x.shape)

                trans_always1 = [
                    transforms.ToPILImage(),
                    transforms.CenterCrop(max_size),
                    transforms.Resize(self._resize_to),
                ]

                trans = transforms.Compose(trans_always1)
                x = trans(x)
                self._imgs.append(x)
        else:
            if preloaded_data:
                self._labels = preloaded_data[0]
                self._imgs = preloaded_data[1]
                self._preload = True

    def __len__(self):
        return len(self._img_file_names)

    def __getitem__(self, idx):
        if self._preload:
            x = self._imgs[idx]
            size = x.size
        else:
            x = io.imread(self._img_file_names[idx], as_gray=True)
            x = np.atleast_3d(x)
            max_size = np.max(x.shape)

            trans_always1 = [
                transforms.ToPILImage(),
                transforms.CenterCrop(max_size),
                transforms.Resize(self._resize_to),
            ]

            trans = transforms.Compose(trans_always1)
            x = trans(x)
            w, h = x.size
            size = (h, w)

        y = self._labels[idx]

        trans_augment = []
        if self._augment:
            trans_augment.append(transforms.RandomHorizontalFlip())
            # trans_augment.append(transforms.RandomRotation(10, resample=PIL.Image.BILINEAR))
            trans_augment.append(transforms.CenterCrop(size))
            trans_augment.append(transforms.RandomCrop(size, padding=8))

        mean = [0.14344494]
        std = [0.18635063]

        trans_always2 = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
        trans = transforms.Compose(trans_augment+trans_always2)

        x = trans(x)

        return x, y


def demo():
    from matplotlib import pyplot as plt

    dataset_train = BoneAgeDataset(data_dir='/media/fastdata/laves/rsna-bone-age/', augment=True)
    data_loader_train = DataLoader(dataset_train, batch_size=1, shuffle=False)

    print("Train dataset length:", len(data_loader_train))

    for i_batch, b in enumerate(data_loader_train):
        x, y = b
        print(i_batch, y, x.size(), y.size(),
              x.type(), y.type())
        plt.subplot(1, 1, 1)
        plt.imshow(x.data.cpu().numpy()[0, 0])
        plt.title(str(y.item()))
        # plt.pause(0.5)
        plt.show()
        plt.clf()


def calc_mean_std():
    dataset = BoneAgeDataset(data_dir='/media/fastdata/laves/rsna-bone-age/', augment=False, preload=False)
    data_loader = DataLoader(dataset, batch_size=1)

    accu = []

    for data, _ in tqdm(data_loader):
        accu.append(data.data.cpu().numpy().flatten())

    accu = np.concatenate(accu)

    return accu.mean(), accu.std()


if __name__ == "__main__":
    mean, std = calc_mean_std()
    print("mean =", mean)
    print("std =", std)
    # demo()
