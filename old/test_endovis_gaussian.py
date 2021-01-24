import torch
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
sns.set()
import os
from data_generator_endovis import EndoVisDataset
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from models import BreastPathQModel
import sys
from test import test


if __name__ == '__main__':
    device = torch.device("cuda:0")
    matplotlib.rcParams['font.size'] = 8

    out_path = 'test_endovis_gaussian'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    batch_size = 16

    data_dir = '/media/fastdata/laves/EndoVis15_instrument_tracking'
    data_set_valid = EndoVisDataset(data_dir=data_dir + '/valid', augment=False, scale=0.5, preload=True)
    data_set_test = EndoVisDataset(data_dir=data_dir + '/test', augment=False, scale=0.5, preload=True)

    assert len(data_set_valid) > 0
    assert len(data_set_test) > 0

    calib_loader = torch.utils.data.DataLoader(data_set_valid, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(data_set_test, batch_size=batch_size, shuffle=False)

    # redirect stdout to file
    sys.stdout = open(f'./{out_path}/results.log', 'w')
    print(len(data_set_valid))
    print(len(data_set_test))
    print('')

    for base_model in tqdm([
        #'resnet50',
        #'resnet101',
        #'densenet121',
        'densenet201',
        #'efficientnetb0',
        #'efficientnetb4'
    ]):
        print(base_model)

        from glob import glob

        model = BreastPathQModel(base_model, out_channels=2).to(device)

        checkpoint_path = glob(f"/media/fastdata/laves/regression_snapshots/{base_model}_gaussian_endovis.pth.tar")[0]

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print("Loading previous weights at epoch " + str(checkpoint['epoch']) + " from\n" + checkpoint_path)

        model.eval()
        test(out_path, model, base_model, calib_loader, test_loader)
