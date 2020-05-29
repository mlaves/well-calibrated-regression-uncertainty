import numpy as np
from torch.utils.data import Dataset, DataLoader
from skimage import io
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
import os


class EndoVisDataset(Dataset):
    """
    Loads the EndoVis instrument tracking data set
    """

    def __init__(self, data_dir, scale=1.0, augment=False, preload=False):
        """
        Given the root directory of the dataset, this function initializes the
        data set

        :param data_dir: List with paths of raw images
        """

        self._scale = scale
        self._data_dir = data_dir
        self._augment = augment
        self._preload = preload

        self._path_tree = sorted([o for o in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, o))])

        self._img_file_names = []  # list of image file names
        self._imgs = []  # list of PILs or empty
        self._labels = []  # list of normalized x,y pixel coordinates of tool base

        for path in self._path_tree:
            df = pd.read_csv(data_dir+'/'+path+'/Right_Instrument_Pose.txt', header=None, delim_whitespace=True)
            for index, row in df.iterrows():
                if row[0] > -1 and row[1] > -1:
                    self._img_file_names.append(f"{data_dir}/{path}/{(index+1):04}.png")
                    self._labels.append([row[0]/720, row[1]/576])

        if self._preload:
            for f in tqdm(self._img_file_names):
                x = io.imread(f)
                x = np.atleast_3d(x)
                x = self.to_pil_and_resize(x, self._scale)
                self._imgs.append(x)

    @staticmethod
    def to_pil_and_resize(x, scale):
        w, h, _ = x.shape
        new_size = (int(w * scale), int(h * scale))

        trans_always1 = [
            transforms.ToPILImage(),
            transforms.Resize(new_size),
        ]

        trans = transforms.Compose(trans_always1)
        x = trans(x)
        return x

    def __len__(self):
        return len(self._img_file_names)

    def __getitem__(self, idx):
        if self._preload:
            x = self._imgs[idx]
        else:
            x = io.imread(self._img_file_names[idx])
            x = np.atleast_3d(x)
            x = self.to_pil_and_resize(x, self._scale)

        y = np.array(self._labels[idx], dtype=np.float32)

        # horizontal flipping
        if self._augment and np.random.rand() > 0.5:
            x = transforms.functional.hflip(x)
            y[1] = 1 - y[1]

        trans_augment = []
        if self._augment:
            trans_augment.append(transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                                                                saturation=0.2, hue=0.1)], p=0.5))

        trans_always2 = [
            transforms.ToTensor(),
        ]
        trans = transforms.Compose(trans_augment+trans_always2)

        x = trans(x)

        return x, y


def demo():
    from matplotlib import pyplot as plt

    dataset_train = EndoVisDataset(data_dir='/media/data/EndoVis15_instrument_tracking/test',
                                   augment=False, scale=0.5, preload=True)
    data_loader_train = DataLoader(dataset_train, batch_size=1, shuffle=False)

    print("Train dataset length:", len(data_loader_train))

    for i_batch, b in enumerate(data_loader_train):
        x, y = b
        h, w = (x.size(2), x.size(3))
        print(i_batch, y, x.size(), y.size(), x.type(), y.type())
        print(y[0, 0]*w, y[0, 1]*h)
        plt.subplot(1, 1, 1)
        plt.imshow(x.data.cpu().numpy()[0, 0])
        plt.plot(y[0, 0]*w, y[0, 1]*h, 'rx')

        # plt.show()

        plt.pause(0.1)
        ret = plt.waitforbuttonpress(0.1)
        if ret:
            break

        plt.clf()


def perf_test():
    dataset_train = EndoVisDataset(data_dir='/media/data/EndoVis15_instrument_tracking/train',
                                   augment=False, scale=0.5, preload=True)
    data_loader_train = DataLoader(dataset_train, batch_size=1, shuffle=False)

    print("Train dataset length:", len(data_loader_train))

    for b in tqdm(data_loader_train):
        x, y = b


def calc_mean_std():
    dataset = EndoVisDataset(data_dir='/media/fastdata/laves/rsna-bone-age/', augment=False, preload=False)
    data_loader = DataLoader(dataset, batch_size=1)

    accu = []

    for data, _ in tqdm(data_loader):
        accu.append(data.data.cpu().numpy().flatten())

    accu = np.concatenate(accu)

    return accu.mean(), accu.std()


if __name__ == "__main__":
    # mean, std = calc_mean_std()
    # print("mean =", mean)
    # print("std =", std)
    demo()
    # perf_test()
