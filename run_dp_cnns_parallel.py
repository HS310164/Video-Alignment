# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import glob
import argparse
import os
import sys
import cv2
import itertools
import csv
from module import DPmatching as dp
from module import function as func
from module import myplot as plot
from joblib import Parallel, delayed


def argparser():
    parser = argparse.ArgumentParser(description='This script run DP-matching and visualize matching result.')

    parser.add_argument('-r',
                        dest='rate',
                        action='store',
                        nargs='?',
                        type=float,
                        default=25,
                        help='neighbor rate (default: %(default)s)')

    parser.add_argument('-th',
                        dest='th',
                        action='store',
                        nargs='?',
                        type=float,
                        default=0.1,
                        help='threshold (default: %(default)s)')

    parser.add_argument('-f',
                        '--flag',
                        dest='flag',
                        action='store_true',
                        help='compute cost matrix from scratch or not (default: False)')

    parser.add_argument('videodata',
                        action='store',
                        type=str,
                        help='path to dir where video data is stored')

    parser.add_argument('resnet2dcnn',
                        action='store',
                        type=str,
                        help='path to dir where BoF histograms is stored')

    parser.add_argument('resnext3dcnn',
                        action='store',
                        type=str,
                        help='path to dir where motion histograms is stored')

    parser.add_argument('result',
                        action='store',
                        type=str,
                        help='path to dir to save result data')

    parser.add_argument('matrix',
                        action='store',
                        type=str,
                        help='path to dir to save substitution score matrix')

    parser.add_argument('plotfig',
                        action='store',
                        type=str,
                        help='path to dir to save figure that include cost matrix and minimum cost path')

    parser.add_argument('truth',
                        action='store',
                        type=str,
                        help='path to dir where ground truth is stored')

    return parser.parse_args()


def dpm(refer, query, args):
    ##########################　preprocessing　#############################

    # exception handling
    if refer == query:
        sys.exit()

    # obtain video name
    r_name = os.path.splitext(refer)[0]
    q_name = os.path.splitext(query)[0]


    r_start = []
    r_end = []
    with open(args.truth + r_name + ".csv", "r") as f:
        reader = csv.reader(f)
        for r in reader:
            tmp = list(map(lambda x: int(x), r))
            if not tmp[2] == 0:
                r_start.append(tmp[0])
                r_end.append(tmp[1])

    q_start = []
    q_end = []
    with open(args.truth + q_name + ".csv", "r") as f:
        reader = csv.reader(f)
        for r in reader:
            tmp = list(map(lambda x: int(x), r))
            if not tmp[2] == 0:
                q_start.append(tmp[0])
                q_end.append(tmp[1])

    # obtain reference video information
    cap = cv2.VideoCapture(args.videodata + refer)
    if not cap.isOpened():
        print("reference video is not display")
        sys.exit()
    r_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    seq1 = []
    for i in range(1, r_len + 1):
        seq1.append(i)

    # obtain query video information
    cap = cv2.VideoCapture(args.videodata + query)
    if not cap.isOpened():
        print("query video is not display")
        sys.exit()
    q_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    seq2 = []
    for i in range(1, q_len + 1):
        seq2.append(i)

    func.makedir(args.result)
    func.makedir(args.matrix)
    func.makedir(args.plotfig)

    re_filename = args.result
    mat_filename = args.matrix
    pf_filename = args.plotfig

    ############################## run DP matching ###############################
    if args.flag:
        print("DP matching {} and {}".format(r_name, q_name))

        sub_matrix, weights = dp.SubstitutionScoreMatrixMultiFeatAdaptive(
            r_name,
            q_name,
            args.rate / 100,
            [args.resnet2dcnn + q_name + ".npy", args.resnet2dcnn + r_name + ".npy"],
            [args.resnext3dcnn + q_name + ".npy", args.resnext3dcnn + r_name + ".npy"],
        )


        filename = mat_filename + r_name + q_name + ".matrix"
        func.savefile(sub_matrix, filename)
    else:
        filename = mat_filename + r_name + q_name + "_motion.matrix"
        sub_matrix = func.loadfile(filename)

    # run DP
    trback = dp.Matching(sub_matrix)

    # traceback
    dp_matching, _ = dp.TraceBack(sub_matrix, trback)

    # save result data
    filename = re_filename + r_name + q_name + ".dp"
    func.savefile(dp_matching, filename)

    ########################### create cost matrix figure ##############################
    best_path = []
    best_path.append((len(seq2), len(seq1)))
    len1 = len(seq1)
    len2 = len(seq2)
    path = trback[(len2, len1)]
    best_path.append(path)

    while not path[0] == 0:
        path = trback[path]
        best_path.append(path)

    best_path.reverse()

    xbar = []
    ybar = []
    for y, x in best_path:
        xbar.append(x)
        ybar.append(y)
    xbar = np.array(xbar)
    ybar = np.array(ybar)

    df = pd.DataFrame(sub_matrix)

    filename = pf_filename + r_name + q_name + ".png"
    plot.CostMatPlot4(r_name, q_name, xbar, ybar, df, r_start, r_end, q_start, q_end, filename, gt=True)
    # print filename
    n = 3
    cnn = pd.DataFrame(weights[0][q_start[n]:q_end[n], r_start[n]:r_end[n]])
    thdcnn = pd.DataFrame(weights[1][q_start[n]:q_end[n], r_start[n]:r_end[n]])
    mini = weights.min()
    maxi = weights.max()
    plot.VisualizeWeight_cnns(cnn, thdcnn, mini, maxi, pf_filename)


def main(args):
    vlist = sorted(glob.glob(os.path.join(args.videodata, '*')))

    Parallel(n_jobs=-1)(
        [delayed(dpm)(os.path.basename(i), os.path.basename(j), args) for i, j in itertools.permutations(vlist, 2)])


if __name__ == "__main__":
    args = argparser()
    main(args)
