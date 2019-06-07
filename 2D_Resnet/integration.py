import numpy as np
import argparse
import glob
import os


def main(arg):
    featlist = sorted(glob.glob(os.path.join(arg.feat, '*.npy')))
    for f in featlist:
        feat = np.load(f)
        fname = os.path.basename(f)
        layer1 = np.load(os.path.join(arg.layer1, fname))
        layer2 = np.load(os.path.join(arg.layer2, fname))
        layer3 = np.load(os.path.join(arg.layer3, fname))
        layer4 = np.load(os.path.join(arg.layer4, fname))
        print('Integrating {}\'s Deep Features ...'.format(os.path.splitext(fname)[0]))
        nf = integrate_df(feat, layer1, layer2, layer3, layer4)
        # print(nf.shape)
        np.save(os.path.join(arg.output, fname), nf)
        print('Save {}\'s Deep Features'.format(os.path.splitext(fname)[0]))


def integrate_df(feat, layer1, layer2, layer3, layer4):
    new_feat = feat.copy()
    # print(new_feat.shape)
    l1ratio = feat.shape[1] // layer1.shape[1]
    l2ratio = feat.shape[1] // layer2.shape[1]
    l3ratio = feat.shape[1] // layer3.shape[1]
    l4ratio = feat.shape[1] // layer4.shape[1]
    print('layer1 shape {}'.format(layer1.shape))
    print('layer2 shape {}'.format(layer2.shape))
    print('layer3 shape {}'.format(layer3.shape))
    print('layer4 shape {}'.format(layer4.shape))
    print('Integrate Deep Features')

    for idx1, l1f in enumerate(layer1):
        nfidx = 0
        for idx, d in enumerate(l1f):
            # print(d.shape)
            ave = np.average(d)
            # print(l1f)
            # print(ave)
            for i in range(l1ratio):
                index = nfidx + i
                # print(index)
                # print(new_feat[idx1,index])
                new_feat[idx1, index] = new_feat[idx1, index] + ave
                # print(new_feat[idx1,index])
            nfidx = nfidx + l1ratio
    print('Finish Integrating Deep Feature from layer1')

    for idx1, l2f in enumerate(layer2):
        nfidx = 0
        # print(l2f.shape)
        for idx, d in enumerate(l2f):
            # print(d.shape)
            ave = np.average(d)
            # print(l1f)
            # print(ave)
            for i in range(l2ratio):
                index = nfidx + i
                # print(index)
                # print(new_feat[idx1,index])
                new_feat[idx1, index] = new_feat[idx1, index] + ave
                # print(new_feat[idx1,index])
            nfidx = nfidx + l2ratio
    print('Finish Integrating Deep Feature from layer2')

    for idx1, l3f in enumerate(layer3):
        nfidx = 0
        for idx, d in enumerate(l3f):
            # print(d.shape)
            ave = np.average(d)
            # print(l1f)
            # print(ave)
            for i in range(l3ratio):
                index = nfidx + i
                # print(index)
                # print(new_feat[idx1,index])
                new_feat[idx1, index] = new_feat[idx1, index] + ave
                # print(new_feat[idx1,index])
            nfidx = nfidx + l3ratio
    print('Finish Integrating Deep Feature from layer3')

    for idx1, l4f in enumerate(layer4):
        nfidx = 0
        for idx, d in enumerate(l4f):
            # print(d.shape)
            ave = np.average(d)
            # print(l1f)
            # print(ave)
            for i in range(l4ratio):
                index = nfidx + i
                # print(index)
                # print(new_feat[idx1,index])
                new_feat[idx1, index] = new_feat[idx1, index] + ave
                # print(new_feat[idx1,index])
            nfidx = nfidx + l4ratio
    print('Finish Integrating Deep Feature from layer4')

    return new_feat


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat', '-f', default='', type=str, help='DF')
    parser.add_argument('--layer1', '-l1', default='input', type=str, help='layer1\'s output')
    parser.add_argument('--layer2', '-l2', default='input', type=str, help='layer2\'s output')
    parser.add_argument('--layer3', '-l3', default='input', type=str, help='layer3\'s output')
    parser.add_argument('--layer4', '-l4', default='outputs/', type=str, help='layer4\'s output')
    parser.add_argument('--output', '-o', default='new_outputs/', type=str, help='output dir')
    parser.add_argument('--only_hand', '-oh', action='store_true',
                        help='If this option true, extract feature from region near hand')
    args = parser.parse_args()

    main(args)
