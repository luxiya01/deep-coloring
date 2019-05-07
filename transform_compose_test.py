import lab_distribution
from custom_transforms import RGB2LAB, ToTensor
from network import Net
import torch
import torchvision
import torchvision.transforms as transforms

ab_bins = lab_distribution.get_ab_bins_from_data(
    '../data/stl10/data/stl10_binary/train_X.bin')

transform = transforms.Compose([RGB2LAB(ab_bins), ToTensor()])

trainset = torchvision.datasets.ImageFolder(
    root='../data/stl10/train', transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=False, num_workers=1)

cnn = Net()

print('------ Starting... ------')
for i, data in enumerate(trainloader):
    if i > 5:
        break
    print('i = ', i)
    # inputs = {'lightness': L-channel, 'z-truth': true ab distribution}
    inputs, labels = data
    print(inputs['lightness'].shape, inputs['lightness'].type())
    print(inputs['z_truth'].shape, inputs['z_truth'].type())
    outputs = cnn(inputs['lightness'])
