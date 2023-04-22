import torch
import torchvision
from torch.utils.data import DataLoader
from utils import *
from config import batch_size
import pickle

test_sets = torchvision.datasets.EMNIST(root="./alpha", split="letters", download=False,
                                        train=False, transform=transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                             (0.1307,), (0.3081,))
                                            ]))

test_dataloader = DataLoader(dataset=test_sets, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
acc_dict = {'model': [], 'acc': []}
model_name = ['SpinalVGG']
total = len(test_sets)
for i in range(1):
    acc = 0
    model_ = torch.load('./param/' + model_name[i] + '.pth')
    model_ = model_.cuda()
    for data in test_dataloader:
        img, target = data
        img = img.cuda()
        target = target.cuda()
        output = model_(img)
        acc += (output.argmax(1) == target).sum()
    acc_dict['model'].append(model_name[i])
    acc_dict['acc'].append(acc / total)
    print(acc_dict)
with open('./dat.pkl', 'wb') as f:
    pickle.dump(acc_dict, f)
