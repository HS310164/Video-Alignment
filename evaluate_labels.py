# -*- coding: utf-8 -*-

import numpy as np
import pickle
from math import *
import glob
import sys
import os
import cv2
import argparse
import csv

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
                        default=0.4,
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
    f = open("dataset/train.csv", "r")
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
    all_result = []
    for p in range(args.pattern):
        result = []
        for i, q_vfile in enumerate(videos[p]):

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

                    filename = os.path.join(args.result, q_name + r_name + ".dp")
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

                    f_all = 0
                    f_non = 0
                    f_correct = 0
                    for q in q_seq:
                        f_all += 1
                        idx = q_dp.index(q)
                        r = r_dp[idx]
                        if q_labels[q - 1] == r_labels[r - 1]:
                            f_correct += 1


                else:
                    rate = float(len(r_seq)) / len(q_seq)
                    filename = args.groundtruth + q_name + ".csv"
                    ql_labels = func.loadcsvgt_linear(filename, rate)
                    filename = args.groundtruth + r_name + ".csv"
                    rl_labels = func.loadcsvgt(filename)

                    f_all = 0
                    f_non = 0
                    f_correct = 0
                    for q, r in zip(ql_labels, rl_labels):
                        f_all += 1
                        if q == r:
                            f_correct += 1

                result.append(float(f_correct) / (f_all - f_non))

        all_result.append(sum(result) / len(result))

    print(sum(all_result) / len(all_result))


if __name__ == "__main__":
    main()
