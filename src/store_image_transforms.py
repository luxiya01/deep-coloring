import torch
import torchvision
import train
import os
import cv2

datadir = '../tmp_images/test_10_images/'
targetdir = '../tmp_images/test_10_images_npz/'
#datadir = '../../data/stl10/test/'
#targetdir = '../../data/stl10/test_npz/'

bins_dict = train.read_prior_bins_dict('./all_bins.npz')
transform = train.get_transforms(bins_dict['ab_bins'])

if not os.path.exists(targetdir):
    os.mkdir(targetdir)

for folder in os.listdir(datadir):
    folder_abs_path = os.path.join(datadir, folder)
    target_folder_abs_path = os.path.join(targetdir, folder)

    if not os.path.exists(target_folder_abs_path):
        os.mkdir(target_folder_abs_path)

    for img_name in os.listdir(folder_abs_path):
        img = cv2.cvtColor(
            cv2.imread(os.path.join(folder_abs_path, img_name)),
            cv2.COLOR_BGR2RGB)
        img_transformed = transform(img)

        path = os.path.join(target_folder_abs_path,
                            img_name.split('.')[0] + '.pt')
        torch.save({
            'lightness': img_transformed['lightness'],
            'z_truth': img_transformed['z_truth'],
            'original_lab_image': img_transformed['original_lab_image']
        }, path)
