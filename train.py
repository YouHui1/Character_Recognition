import time
import torch.utils.data as Data
import torch.optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from config import *
from utils import *
import pickle
import sys
import getopt

opts, args = getopt.getopt(sys.argv[1:], 'k:', ['kind='])
for opt_name, opt_value in opts:
    if opt_name in ('-k', 'kind'):
        model_choice = int(opt_value)
        # print(model_lst[model_choice])

create_log(model_choice)

# 创建模型
model = model
model_name = model_lst[model_choice]
print(model_name)
model = model.to(device)

# train_set = ImageFolder(root="./dataset/train", transform=data_transform)
# train_dataloader = DataLoader(dataset=train_set, batch_size=192, shuffle=True, num_workers=0, drop_last=False)
# val_set = ImageFolder(root="./dataset/test", transform=data_transform)
# val_dataloader = DataLoader(dataset=val_set, batch_size=192, shuffle=True, num_workers=0, drop_last=False)
_set = torchvision.datasets.EMNIST(root="./alpha", split="letters", download=True,
                                   train=True, transform=data_transform)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()
# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
state = {'name': model_name, 'model': model, 'optimizer': optimizer, 'val_loss': 0, 'train_loss': 0, 'best_acc': 0}
acc_loss_lst = {'epoch': 0, 'train_loss': [], 'val_loss': [], 'val_acc': []}


def main():
    cnt = 0

    logger.info('======================================================')
    logger.info('Initialize')
    logger.info('model:{}, device:{}'.format(model_name, device))
    logger.info('learning_rate:{}, epoch:{}'.format(learning_rate, epoch))
    logger.info('======================================================')
    logger.info('======================================================')
    logger.info('Training')

    for i in range(epoch):
        train_set, val_set = Data.random_split(_set,
                                               [int(0.8 * len(_set)), len(_set) - int(0.8 * len(_set))])
        val_dataloader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True, num_workers=0,
                                    drop_last=False)
        train_dataloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0,
                                      drop_last=False)

        logger.info("...epoch {} ...".format(i + 1))
        acc_loss_lst['epoch'] = i + 1
        # 训练
        model.train()
        timer1 = time.perf_counter()
        total_train_step = 0
        total_train_loss = 0

        for data in train_dataloader:
            img, target = data
            if torch.cuda.is_available():
                img = img.cuda()
                target = target.cuda()
            output = model(img)
            loss = loss_fn(output, target)
            total_train_loss += loss.item()
            # 优化器优化模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step = total_train_step + 1
            process = total_train_step / len(train_dataloader)
            a = '*' * int(process * 50)
            b = '.' * int((1 - process) * 50)
            print("\rloss:{:^3.0f}%[{}->{}]{:.5f}".format(int(process * 100), a, b, loss.item()), end="")
        print()

        logger.info("epoch:{}, train_loss:{:.5f}".format(i + 1, total_train_loss))

        acc_loss_lst['train_loss'].append(total_train_loss)

        total_val_loss = 0
        total_acc = 0
        model.eval()
        with torch.no_grad():
            for data in val_dataloader:
                img, target = data
                if torch.cuda.is_available():
                    img = img.cuda()
                    target = target.cuda()
                output = model(img)
                loss = loss_fn(output, target)
                total_val_loss = total_val_loss + loss.item()
                acc = (output.argmax(1) == target).sum()
                total_acc = total_acc + acc

        logger.info("val_loss:{}".format(total_val_loss))
        acc_loss_lst['val_loss'].append(total_val_loss)
        logger.info("val_acc:{}".format(total_acc / len(val_set)))
        rate = total_acc / len(val_set)
        acc_loss_lst['val_acc'].append(rate)

        if abs(rate - state['best_acc']) < 0.008 and rate <= state['best_acc']:
            cnt += 1
        elif abs(rate - state['best_acc']) < 0.004:
            cnt += 1
        if rate > state['best_acc']:
            state['best_acc'] = rate
            state['val_loss'] = total_val_loss
            state['train_loss'] = total_train_loss
            state['model'] = model.state_dict()
            state['optimizer'] = optimizer
            logger.info('======================================================')
            logger.info('======================================================')
            logger.info('Save')
            with open('./param/' + model_lst[model_choice] + '_acc_loss_lst.pkl', 'wb') as f:
                pickle.dump(acc_loss_lst, f)
            with open('./param/' + model_lst[model_choice] + '_param.pkl', 'wb') as f:
                pickle.dump(state, f)
            torch.save(model, './param/' + model_lst[model_choice] + '_dict.pth')

        logger.info('epoch:{},time:{}'.format(i + 1, time.perf_counter() - timer1))

    logger.info('======================================================')
    logger.info('======================================================')
    logger.info('End')
    logger.info('======================================================')


if __name__ == '__main__':
    main()
