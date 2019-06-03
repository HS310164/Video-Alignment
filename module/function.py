#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import os
import sys
import pickle
import shutil
import csv
from math import *


def savefile(obj, filename):
    # save file using pickle module
    with open(filename, "wb") as f:
        pickle.dump(obj, f, protocol=0)


def loadfile(filename):
    # load file using pickle module
    with open(filename, "rb") as f:
        obj = pickle.load(f, encoding='bytes')
    return obj


def makedir(dirname):
    # if output directory dese not exist, make one
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def delAndmakedir(dirname):
    # if output directory exist, deleate it and make new one
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname)


def vstack(array, var):
    if var is None or not len(var):
        return array

    if len(array):
        array = np.vstack((array, var))
    else:
        array = var

    return array


def drange(begin, end, step):
    n = begin
    while n + step < end:
        yield n
        n += step


def loadgt(filename):
    with open(filename, "rb") as f:
        gt = []
        for line in f:
            line = line.rstrip()
            data = line.split('\t')
            gt.append(data)

    return gt


def loadcsvgt(filename):
    labels = []
    with open(filename, "r",) as f:
        reader = csv.reader(f)
        for r in reader:
            tmp = list(map(lambda x: int(x), r))
            labels = labels + [tmp[2] for i in range(tmp[0], tmp[1] + 1)]

    return np.array(labels)


def loadcsvgt_linear(filename, rate):
    labels = []
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for r in reader:
            tmp = list(map(lambda x: int(x), r))
            length = tmp[1] - tmp[0] + 1
            length = int(floor(length * rate))
            labels = labels + [tmp[2] for i in range(length)]

    return np.array(labels)
