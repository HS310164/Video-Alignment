# coding: UTF-8
import glob
import os
import subprocess

import HandDetection as hd
import cv2
import numpy as np
import resnet
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from dataset import TransDataset
from opt import parse_opts
from torch import nn
from tqdm import tqdm


def main(opt):
    # モデル定義
    model = resnet.resnet101(pretrained=True)
    if torch.cuda.is_available():  # GPUが利用可能か確認
        device = 'cuda'
    else:
        device = 'cpu'
    print('device is {0}'.format(device))
    # print(model.weight.type)
    model.to(device)
    if opt.trained:
        print('load pretrained model')
        model.fc = nn.Linear(2048, opt.ftclass)
        model_data = torch.load(opt.model)
        model.load_state_dict(model_data)
    model.eval()
    # 絶対パスに変換
    outpath = os.path.abspath(opt.output)
    apath = os.path.abspath(opt.input)
    video_names = sorted(glob.glob(os.path.join(apath, '*')))

    if os.path.exists('tmp'):
        subprocess.call('rm -rf tmp', shell=True)

    for vpath in video_names:
        vname = os.path.splitext(os.path.basename(vpath))[0]

        subprocess.call('mkdir tmp', shell=True)
        subprocess.call('ffmpeg -loglevel warning -i {} -r 5 tmp/image_%05d.jpg'.format(vpath), shell=True)
        images = sorted(glob.glob('tmp/*.jpg'))
        if opt.only_hand:
            print('convert to masked images')
            for im in tqdm(images):
                frame = cv2.imread(im)
                maskedframe, _ = hd.detect(frame)
                cv2.imwrite(im, maskedframe)
            print('complete convert images')

        print('extract {}\'s DeepFeatrue'.format(vname))

        outputs = input_image(images, model)

        # ファイルに保存
        if not os.path.exists(outpath):
            subprocess.call('mkdir {}'.format(outpath), shell=True)

        savename = os.path.join(outpath, vname + '.npy')
        # savename1 = os.path.join(outpath, 'layer1', vname + '.npy')
        # savename2 = os.path.join(outpath, 'layer2', vname + '.npy')
        # savename3 = os.path.join(outpath, 'layer3', vname + '.npy')
        # savename4 = os.path.join(outpath, 'layer4', vname + '.npy')
        np.save(savename, outputs)
        # np.save(savename1, ops1)
        # np.save(savename2, ops2)
        # np.save(savename3, ops3)
        # np.save(savename4, ops4)
        subprocess.call('rm -rf tmp', shell=True)


def input_image(im_paths, model):
    # 画像のリサイズ，テンソル化
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                    ])
    dataset = TransDataset(im_paths, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    outputs = []
    ops1 = []
    ops2 = []
    ops3 = []
    ops4 = []
    for idx, input in enumerate(data_loader):
        if torch.cuda.is_available():
            input = input.to('cuda')
        # output, op1, op2, op3, op4 = model(input)
        output = model(input)
        outputs.append(output[0].to('cpu').data.numpy())
    #     ops1.append(op1[0].to('cpu').data.numpy())
    #     ops2.append(op2[0].to('cpu').data.numpy())
    #     ops3.append(op3[0].to('cpu').data.numpy())
    #     ops4.append(op4[0].to('cpu').data.numpy())
    # outputs = np.array(outputs)
    # ops1 = np.array(ops1)
    # ops2 = np.array(ops2)
    # ops3 = np.array(ops3)
    # ops4 = np.array(ops4)
    # print(outputs.shape)
    # print(ops1.shape)
    # print(ops2.shape)
    # print(ops3.shape)
    # print(ops4.shape)
    # return outputs, ops1, ops2, ops3, ops4
    return outputs


class TransDataset(data.Dataset):
    def __init__(self, im_paths, transform=None):
        self.im_paths = im_paths
        self.transform = transform

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, index):
        im = pil_loader(self.im_paths[index])
        if self.transform:
            im = self.transform(im)
        return im


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


if __name__ == '__main__':
    opt = parse_opts()
    main(opt)

'''
ImageNetでpretrainしたResNetからのディープ特徴抽出
python main.py --input inputs_dir/ --output outputs_dir/
input_dirには動画が格納されたフォルダを指定してください
'''
