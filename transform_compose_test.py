import lab_distribution
from custom_transforms import RGB2LAB, ToTensor
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

dataiter = iter(trainloader)
data, labels = dataiter.next()

for key, val in data.items():
    print(key, ': ', type(key), ', ', val.shape)

print(labels)
