import math
import os

import resnet
import torch
import torch.utils.data as data
from dataset import TrainDataset
from opt import parse_opts
from torch import nn
from torch import optim
from transform import transform_images


def main(opt):
    # モデル定義
    model = resnet.resnet152(pretrained=True)
    if not os.path.exists(opt.output):
        os.makedirs(opt.output)

    if torch.cuda.is_available():  # GPUが利用可能か確認
        device = 'cuda'
    else:
        device = 'cpu'
    print('device is {0}'.format(device))
    model.to(device)
    model.fc = nn.Linear(2048, opt.ftclass)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    transform = transform_images()
    training_data = TrainDataset(opt.input, transform)
    train_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True)
    for epoch in range(50):
        train(epoch + 1, model, criterion, optimizer, train_loader, device)
        if (epoch + 1) % 2 == 0:
            savefile = os.path.join(opt.output, 'save_{:0>3}.pth'.format(epoch + 1))
            torch.save(model.state_dict(), savefile)


def train(epoch, model, criterion, optimizer, train_loader, device):
    total_loss = 0
    total_size = 0
    model.train()
    digit = int(math.log10(len(train_loader.dataset)) + 1)
    for batch_idx, (img, target) in enumerate(train_loader):
        img, target = img.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, target)
        total_loss += loss.item()
        total_size += img.size(0)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{:{dig}}/{} ({:3.0f}%)] Average loss: {:.6f}'.format(
                epoch, batch_idx * len(img), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), total_loss / total_size, dig=digit))


if __name__ == '__main__':
    opt = parse_opts()
    main(opt)
