import numpy as np
from torch.utils.data import Dataset, DataLoader
from skimage import io
from torchvision import transforms
import pandas as pd
from tqdm import tqdm


class BreastPathQDataset(Dataset):
    """
    Loads the breastpathq histology data set
    """

    def __init__(self, data_dir='./breastpathq/',
                 resize_to=(512, 512), file_ext='.tif',
                 color=True, augment=False, preload=False):
        """
        Given the root directory of the dataset, this function initializes the
        data set

        :param data_dir: List with paths of raw images
        """

        self._resize_to = resize_to
        self._color = color
        self._data_dir = data_dir
        self._augment = augment
        self._preload = preload

        self._df = pd.read_csv(self._data_dir+f'/train_labels.csv')
        self._img_file_names = [self._data_dir+f"/train/{self._df['slide'][i]}_{self._df['rid'][i]}{file_ext}"
                                for i in range(self._df.shape[0])]

        self._imgs = []

        if self._preload:
            for fname in tqdm(self._img_file_names):
                x = io.imread(fname)
                x = np.atleast_3d(x)
                self._imgs.append(x)

    def __len__(self):
        return len(self._img_file_names)

    def __getitem__(self, idx):
        if self._preload:
            x = self._imgs[idx]
        else:
            x = io.imread(self._img_file_names[idx])
            x = np.atleast_3d(x)

        y = np.array([self._df['y'][int(idx)]], dtype=np.float32)

        trans_always1 = [
            transforms.ToPILImage(),
            transforms.Resize(self._resize_to),
        ]

        trans_augment = []
        if self._augment:
            if np.random.rand() > 0.5:
                x = x.swapaxes(0, 1)
            trans_augment.append(transforms.RandomHorizontalFlip())
            trans_augment.append(transforms.RandomVerticalFlip())
            trans_augment.append(transforms.CenterCrop(self._resize_to))
            # trans_augment.append(transforms.RandomCrop(self._resize_to, padding=8, padding_mode='reflect'))

        mean = [0.78313226, 0.635746, 0.74008995]
        std = [0.1681861, 0.1879578, 0.14389448]

        trans_always2 = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
        trans = transforms.Compose(trans_always1+trans_augment+trans_always2)

        x = trans(x)

        return x, y


def demo():
    from matplotlib import pyplot as plt

    dataset_train = BreastPathQDataset(data_dir='/media/data/breastpathq', augment=True, preload=True)
    data_loader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)

    print("Train dataset length:", len(data_loader_train))

    for i_batch, b in enumerate(data_loader_train):
        x, y = b
        print(i_batch, y, x.size(), y.size(),
              x.min(), x.max())
        plt.subplot(1, 1, 1)
        plt.imshow(x.data.cpu().numpy()[0].transpose(1, 2, 0))
        plt.title(str(y))
        # plt.pause(0.5)
        plt.show()
        plt.clf()


def calc_mean_std():
    dataset = BreastPathQDataset(data_dir='/media/fastdata/laves/breastpathq', augment=False)
    data_loader = DataLoader(dataset, batch_size=1)

    accu = []

    for data, _ in tqdm(data_loader):
        accu.append(data.data.cpu().squeeze().numpy())

    accu = np.array(accu)

    return accu.mean(axis=(0, 2, 3)), accu.std(axis=(0, 2, 3))


if __name__ == "__main__":
    # mean, std = calc_mean_std()
    # print("mean =", mean)
    # print("std =", std)
    demo()
