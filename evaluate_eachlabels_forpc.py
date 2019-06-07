# -*- coding: utf-8 -*-

import numpy as np
import pickle
from math import *
import glob
import sys
import os
# import matplotlib.pyplot as plt
import cv2
import argparse
import csv
# from tqdm import tqdm

from module import function as func


def argparser():
    parser = argparse.ArgumentParser(description='This script evaluate any alignment method.')

    parser.add_argument('--linear',
                        dest='linear',
                        action='store_true',
                        help='linear matching result (default: False)')

    parser.add_argument('--num_videos',
                        dest='vnum',
                        action='store',
                        nargs='?',
                        type=int,
                        default=6,
                        help='number of training and test videos (default: %(default)s)')

    parser.add_argument('--num_pattern',
                        dest='pattern',
                        action='store',
                        nargs='?',
                        type=int,
                        default=30,
                        help='number of validation pattern (default: %(default)s)')

    parser.add_argument('--weight',
                        dest='weight',
                        action='store',
                        nargs='?',
                        type=float,
                        default=0.2,
                        help='fixed weight (default: %(default)s)')

    parser.add_argument('video',
                        action='store',
                        type=str,
                        help='path to dir include video data')

    parser.add_argument('result',
                        action='store',
                        type=str,
                        help='path to dir include alignment result')

    parser.add_argument('groundtruth',
                        action='store',
                        type=str,
                        help='path to dir include ground truth data')

    return parser.parse_args()


def main():
    args = argparser()

    # obtain available video data name list
    f = open("/mnt/serverhome03/hsato/aligmentsato/dataset/pc/train.csv", "r")
    reader = csv.reader(f)
    videos = []
    for i in range(args.pattern):
        all_videos = os.listdir(args.video)
        for j in range(args.vnum + 1):
            if j == 0:
                next(reader)[0]
            else:
                line = next(reader)[0]
                all_videos.remove(os.path.join(line))
        videos.append(all_videos)
    f.close()
    videos = np.array(videos)

    # main process
    each_result = None
    for p in range(args.pattern):
        result = None
        for i, q_vfile in enumerate(videos[p]):

            # obtain query video name
            q_name = os.path.splitext(q_vfile)[0]

            # number of frame about query video
            cap = cv2.VideoCapture(args.video + q_vfile)
            q_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            q_seq = [q for q in range(1, q_len + 1)]

            # load ground truth about query video
            filename = args.groundtruth + q_name + ".csv"
            q_labels = func.loadcsvgt(filename)

            for j, r_vfile in enumerate(videos[p]):

                if q_vfile == r_vfile:
                    continue

                # obtain reference video name
                r_name = os.path.splitext(r_vfile)[0]

                # number of frame about reference video
                cap = cv2.VideoCapture(args.video + r_vfile)
                r_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                r_seq = [q for q in range(1, r_len + 1)]

                # load ground truth about reference video
                filename = args.groundtruth + r_name + ".csv"
                r_labels = func.loadcsvgt(filename)

                if not args.linear:

                    filename = args.result + q_name + r_name + ".dp"
                    dp_result = func.loadfile(filename)

                    r_dp = []
                    r = -1
                    for ref in dp_result[0]:
                        if r == -1 and ref == 0:
                            r_dp.append(0)
                        else:
                            if not ref == 0:
                                r += 1
                            r_dp.append(r_seq[r])

                    q_dp = []
                    q = -1
                    for que in dp_result[1]:
                        if q == -1 and que == 0:
                            q_dp.append(0)
                        else:
                            if not que == 0:
                                q += 1
                            q_dp.append(q_seq[q])

                    f_all = np.zeros(q_labels.max() + 1)
                    f_correct = np.zeros(q_labels.max() + 1)
                    for q in q_seq:
                        f_all[q_labels[q - 1]] += 1
                        idx = q_dp.index(q)
                        r = r_dp[idx]
                        if q_labels[q - 1] == r_labels[r - 1]:
                            f_correct[q_labels[q - 1]] += 1


                else:
                    rate = float(len(r_seq)) / len(q_seq)
                    filename = args.groundtruth + q_name + ".csv"
                    ql_labels = func.loadcsvgt_linear(filename, rate)
                    filename = args.groundtruth + r_name + ".csv"
                    rl_labels = func.loadcsvgt(filename)

                    f_all = np.zeros(q_labels.max() + 1)
                    f_correct = np.zeros(q_labels.max() + 1)
                    for q, r in zip(ql_labels, rl_labels):
                        f_all[q] += 1
                        if q == r:
                            f_correct[q] += 1

                result = func.vstack(f_correct.astype(np.float32) / f_all, result)

        each_result = func.vstack(result.sum(axis=0) / len(result), each_result)

    for i, j in enumerate(each_result.sum(axis=0) / len(each_result)):
        print(i, j)


if __name__ == "__main__":
    main()
