import torch
import sys
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
sns.set()
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
from data_generator_breast import BreastPathQDataset
from models import BreastPathQModel
import os
from test import test


if __name__ == '__main__':
    device = torch.device("cuda:0")
    matplotlib.rcParams['font.size'] = 8

    out_path = 'test_breastpathq_gaussian'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    batch_size = 16
    resize_to = (384, 384)

    data_set = BreastPathQDataset(data_dir='/media/fastdata/laves/breastpathq/', augment=False, resize_to=resize_to, preload=True)
    assert len(data_set) > 0

    calib_indices = torch.load('./breastpathq_valid_indices.pth')
    test_indices = torch.load('./breastpathq_test_indices.pth')

    print(calib_indices.shape)
    print(test_indices.shape)

    calib_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size,
                                               sampler=SubsetRandomSampler(calib_indices))
    test_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size,
                                              sampler=SubsetRandomSampler(test_indices))

    # redirect stdout to file
    sys.stdout = open(f'./{out_path}/results.log', 'w')
    print('calib_indices.shape', calib_indices.shape)
    print('test_indices.shape', test_indices.shape)
    print('')

    for base_model in tqdm([
        # 'resnet50',
        'resnet101',
        # 'densenet121',
        'densenet201',
        # 'efficientnetb0',
        'efficientnetb4'
    ]):
        print(base_model)

        from glob import glob

        model = BreastPathQModel(base_model, out_channels=1).to(device)

        checkpoint_path = glob(f"/media/fastdata/laves/regression_snapshots/{base_model}_gaussian_breastpathq_*.pth.tar")[0]

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print("Loading previous weights at epoch " + str(checkpoint['epoch']) + " from\n" + checkpoint_path)
        model.eval()
        test(out_path, model, base_model, calib_loader, test_loader)
