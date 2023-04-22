import torch
from model import *
from model_resnet import *
from model_new import *
from torchvision import transforms
import torchvision

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_lst = ['lenet', 'SpinalVGG', 'WaveMix', 'model_improve', 'resnet18', 'resnet34', 'resnet50', 'resnet101']
# 选择模型
model_choice = 1
if model_choice == 0:
    model = LeNet()
elif model_choice == 1:
    model = SpinalVGG(27)
elif model_choice == 2:
    model = WaveMix(
        num_classes=27,
        depth=7,
        mult=2,
        ff_channel=256,
        final_dim=256,
        dropout=0.5
    )
elif model_choice == 3:
    model = model_improved()
elif model_choice == 4:
    model = resnet18(num_classes=52)
elif model_choice == 5:
    model = resnet34(num_classes=52)
elif model_choice == 6:
    model = resnet50(num_classes=52)
elif model_choice == 7:
    model = resnet101(num_classes=52)
data_transform = transforms.Compose([
                                   torchvision.transforms.RandomPerspective(),
                                   torchvision.transforms.RandomRotation(10, fill=(0,)),
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                    ])
if model_choice > 3 and model_choice != 8:
    data_transform = transforms.Compose([transforms.Resize((28, 28)),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize(
                                             mean=(0.485, 0.456, 0.206),
                                             std=(0.229, 0.224, 0.225))
                                         ])
epoch = 50
learning_rate = 2e-3
batch_size = 192
