#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import math
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import csv
import pandas as pd
from time import time
import sys

from module import function as func
from module import BagofFeatures as bof


# this function make substitution score matrix using only one feature
def SubstitutionScoreMatrix(b_feat, s_feat):
    # dirname = os.path.join(project_dir,"distmap",fname)
    # filename = os.path.join(dirname,r_name+q_name+".npy")
    # if os.path.exists(filename):
    #         print "submat already exists"
    #         return np.load(filename)

    # load histograms for reference video and compute number of neighbor
    sup_feat = np.load(s_feat)

    # load histograms for query video and compute number of neighbor
    beg_feat = np.load(b_feat)

    sup_len = len(sup_feat)
    beg_len = len(beg_feat)

    # initialize substitution score matrix
    sub_matrix = np.zeros((beg_len, sup_len), dtype=np.float32)
    sub_matrix[:, :] = float("inf")

    sup_len = len(sup_feat)
    beg_len = len(beg_feat)

    # compute substitution score matrix
    for b in range(beg_len):
        # compute distance between BoF histograms
        sub_matrix[b] = np.linalg.norm(beg_feat[b] - sup_feat, axis=1)

    # 正規化してみてる
    scaler = MinMaxScaler()
    smatrix = sub_matrix.reshape((sup_len * beg_len, 1))
    sscaled = scaler.fit_transform(smatrix)
    sub_matrix = sscaled.reshape((beg_len, sup_len))

    return sub_matrix


# this function make substitution score matrix using two features integrated adaptively
def SubstitutionScoreMatrixFixed(b_bof, s_bof, b_motion, s_motion, w=0.5):
    # load histograms for reference video and compute number of neighbor
    sup_shist = np.load(s_bof)
    sup_idt = np.load(s_motion)

    # load histograms for query video and compute number of neighbor
    beg_shist = np.load(b_bof)
    beg_idt = np.load(b_motion)

    sup_len = len(sup_shist)
    beg_len = len(beg_shist)

    # initialize substitution score matrix
    sub_matrix = np.zeros((beg_len, sup_len), dtype=np.float32)
    sub_matrix[:, :] = float("inf")

    smatrix = np.zeros((beg_len, sup_len), dtype=np.float32)  # for object features
    mmatrix = np.zeros((beg_len, sup_len), dtype=np.float32)  # for motion features

    # compute substitution score matrix for boject and motion features separately
    for b in range(beg_len):
        # compute distance between BoF histograms
        sscore = np.linalg.norm(beg_shist[b] - sup_shist, axis=1)
        smatrix[b] = sscore

        # compute distance between motion histograms
        mscore = np.linalg.norm(beg_idt[b] - sup_idt, axis=1)
        mmatrix[b] = mscore

    # each substitution score matrix is scaled by mini-max
    smatrix = smatrix.reshape((sup_len * beg_len, 1))
    mmatrix = mmatrix.reshape((sup_len * beg_len, 1))

    scaler = MinMaxScaler()
    sscaled = scaler.fit_transform(smatrix)
    mscaled = scaler.fit_transform(mmatrix)

    sscaled = sscaled.reshape((beg_len, sup_len))
    mscaled = mscaled.reshape((beg_len, sup_len))

    # integrating two features adaptively
    for b in range(beg_len):
        # compute distance between two frames after integration
        score = w * sscaled[b] + (1.0 - w) * mscaled[b]
        sub_matrix[b] = score

    return sub_matrix


# this function make substitution score matrix using two features integrated adaptively
def SubstitutionPCAScoreMatrixFixed(b_bof, s_bof, b_motion, s_motion, w=0.5):
    # load histograms for reference video and compute number of neighbor
    sup_shist = np.load(s_bof)
    sup_idt = np.load(s_motion)

    # load histograms for query video and compute number of neighbor
    beg_shist = np.load(b_bof)
    beg_idt = np.load(b_motion)

    sup_len = len(sup_shist)
    beg_len = len(beg_shist)

    pca = PCA(n_components=512)

    # print('SIFT feature shape:{},{}'.format(sup_shist.shape, beg_shist.shape))

    sift = np.concatenate([sup_shist, beg_shist])
    idt = np.concatenate([sup_idt, beg_idt])

    # print('Concatinated feature: {}'.format(sift.shape))

    sift_fit = pca.fit_transform(sift)
    idt_fit = pca.fit_transform(idt)
    # print('fit concatinated feature: {}'.format(sift_fit.shape))

    sup_shist, beg_shist = np.split(sift_fit, [sup_len])
    sup_idt, beg_idt = np.split(idt_fit, [sup_len])

    # print('SIFT feature shape:{},{}'.format(sup_shist.shape, beg_shist.shape))

    # initialize substitution score matrix
    sub_matrix = np.zeros((beg_len, sup_len), dtype=np.float32)
    sub_matrix[:, :] = float("inf")

    smatrix = np.zeros((beg_len, sup_len), dtype=np.float32)  # for object features
    mmatrix = np.zeros((beg_len, sup_len), dtype=np.float32)  # for motion features

    # compute substitution score matrix for boject and motion features separately
    for b in range(beg_len):
        # compute distance between BoF histograms
        sscore = np.linalg.norm(beg_shist[b] - sup_shist, axis=1)
        smatrix[b] = sscore

        # compute distance between motion histograms
        mscore = np.linalg.norm(beg_idt[b] - sup_idt, axis=1)
        mmatrix[b] = mscore

    # each substitution score matrix is scaled by mini-max
    smatrix = smatrix.reshape((sup_len * beg_len, 1))
    mmatrix = mmatrix.reshape((sup_len * beg_len, 1))

    scaler = MinMaxScaler()
    sscaled = scaler.fit_transform(smatrix)
    mscaled = scaler.fit_transform(mmatrix)

    sscaled = sscaled.reshape((beg_len, sup_len))
    mscaled = mscaled.reshape((beg_len, sup_len))

    # integrating two features adaptively
    for b in range(beg_len):
        # compute distance between two frames after integration
        score = w * sscaled[b] + (1.0 - w) * mscaled[b]
        sub_matrix[b] = score

    return sub_matrix


def SubstitutionConcPCAScoreMatrixFixed(b_bof, s_bof, b_motion, s_motion):
    # load histograms for reference video and compute number of neighbor
    sup_shist = np.load(s_bof)
    sup_idt = np.load(s_motion)

    # load histograms for query video and compute number of neighbor
    beg_shist = np.load(b_bof)
    beg_idt = np.load(b_motion)

    sup_len = len(sup_shist)
    beg_len = len(beg_shist)

    pca = PCA(n_components=512)

    # print('SIFT feature shape:{},{}'.format(sup_shist.shape, beg_shist.shape))

    sift = np.concatenate([sup_shist, beg_shist])
    idt = np.concatenate([sup_idt, beg_idt])

    beg_feat = np.concatenate([sup_shist, sup_idt], axis=1)
    sup_feat = np.concatenate([beg_shist, beg_idt], axis=1)

    conc_feat = np.concatenate([sup_feat, beg_feat])

    # print('Concatinated feature:{}, {}'.format(sup_feat.shape, beg_feat.shape))

    conc_fit = pca.fit_transform(conc_feat)
    # print('fit concatinated feature: {}'.format(conc_fit.shape))

    sup_feat, beg_feat = np.split(conc_fit, [sup_len])

    # print('Video feature shape:{},{}'.format(sup_feat.shape, beg_feat.shape))

    # initialize substitution score matrix
    sub_matrix = np.zeros((beg_len, sup_len), dtype=np.float32)
    sub_matrix[:, :] = float("inf")

    # compute substitution score matrix for object and motion features separately
    for b in range(beg_len):
        # compute distance between BoF histograms
        sub_matrix[b] = np.linalg.norm(beg_feat[b] - sup_feat, axis=1)

    # each substitution score matrix is scaled by mini-max
    scaler = MinMaxScaler()
    smatrix = sub_matrix.reshape((sup_len * beg_len, 1))
    sscaled = scaler.fit_transform(smatrix)
    sub_matrix = sscaled.reshape((beg_len, sup_len))

    return sub_matrix


# this function make substitution score matrix using two features integrated adaptively
def SubstitutionScoreMatrixAdaptive(b_bof, s_bof, b_motion, s_motion, rate, exp=1, conv=1):
    # load histograms
    sup_shist = bof.load(s_bof)
    beg_shist = bof.load(b_bof)

    # compute number of neighbor
    sup_shist_neighbor = CalucNeighbor(sup_shist, rate)
    # sup_shist_neighbor = CalucNeighborExp(sup_shist,rate,exp,conv)
    # sup_shist_neighbor = CalucNeighborOpposite(sup_shist,beg_shist)
    sup_shist_neighbor = sup_shist_neighbor.astype("float")

    beg_shist_neighbor = CalucNeighbor(beg_shist, rate)
    # beg_shist_neighbor = CalucNeighborExp(beg_shist,rate,exp,conv)
    # beg_shist_neighbor = CalucNeighborOpposite(beg_shist,sup_shist)
    beg_shist_neighbor = beg_shist_neighbor.astype("float")

    sup_idt = np.load(s_motion)
    beg_idt = np.load(b_motion)

    sup_idt_neighbor = CalucNeighbor(sup_idt, rate)
    # sup_idt_neighbor = CalucNeighborExp(sup_idt,rate,exp,conv)
    # sup_idt_neighbor = CalucNeighborOpposite(sup_idt,beg_idt)
    sup_idt_neighbor = sup_idt_neighbor.astype("float")

    beg_idt_neighbor = CalucNeighbor(beg_idt, rate)
    # beg_idt_neighbor = CalucNeighborExp(beg_idt,rate,exp,conv)
    # beg_idt_neighbor = CalucNeighborOpposite(beg_idt,sup_idt)
    beg_idt_neighbor = beg_idt_neighbor.astype("float")

    sup_len = len(sup_shist)
    beg_len = len(beg_shist)

    # initialize substitution score matrix
    sub_matrix = np.zeros((beg_len, sup_len), dtype=np.float32)
    select_matrix = np.zeros((beg_len, sup_len), dtype=np.float32)
    sub_matrix[:, :] = float("inf")

    smatrix = np.zeros((beg_len, sup_len), dtype=np.float32)  # for object features
    mmatrix = np.zeros((beg_len, sup_len), dtype=np.float32)  # for motion features

    # #compute substitution score matrix for boject and motion features separately
    for b in range(beg_len):
        # compute distance between BoF histograms
        # euclid distance
        sscore = np.linalg.norm(beg_shist[b] - sup_shist, axis=1)
        smatrix[b] = sscore

        # compute distance between motion histograms
        mscore = np.linalg.norm(beg_idt[b] - sup_idt, axis=1)
        mmatrix[b] = mscore

    # smatrix = bhattacharyya(sup_shist,beg_shist)
    # mmatrix = bhattacharyya(sup_idt,beg_idt)

    # each substitution score matrix is scaled by mini-max
    smatrix = smatrix.reshape((sup_len * beg_len, 1))
    mmatrix = mmatrix.reshape((sup_len * beg_len, 1))

    scaler = MinMaxScaler()
    sscaled = scaler.fit_transform(smatrix)
    mscaled = scaler.fit_transform(mmatrix)

    sscaled = sscaled.reshape((beg_len, sup_len))
    mscaled = mscaled.reshape((beg_len, sup_len))

    # integrating two features adaptively
    for b in range(beg_len):
        for s in range(sup_len):

            s_neighbor = sup_shist_neighbor[s] + beg_shist_neighbor[b]
            m_neighbor = sup_idt_neighbor[s] + beg_idt_neighbor[b]

            # compute importance
            if s_neighbor + m_neighbor == 0:
                s_weight = 0.5
                m_weight = 0.5
            else:
                s_weight = m_neighbor / (s_neighbor + m_neighbor)
                m_weight = 1 - s_weight

            # compute distance between two frames after integration
            score = s_weight * sscaled[b][s] + m_weight * mscaled[b][s]
            sub_matrix[b][s] = score
            select_matrix[b][s] = s_weight

    return sub_matrix, select_matrix


# this function make substitution score matrix using multiple features
def SubstitutionScoreMatrixMultiFeatFixed(project_dir, r_name, q_name, *feat):
    ref_feat = []
    que_feat = []
    weight = []
    for i, f in enumerate(feat):
        # load histograms for reference video
        ref_feat.append(np.load(f[1]))

        # load histograms for query video
        que_feat.append(np.load(f[0]))

        weight.append(f[2])

    ref_len = len(ref_feat[0])
    que_len = len(que_feat[0])

    # initialize substitution score matrix
    sub_matrix = np.zeros((que_len, ref_len), dtype=np.float32)
    sub_matrix[:, :] = float("inf")

    num_feat = len(feat)
    tmp_mat = np.zeros((num_feat, que_len, ref_len), dtype=np.float32)

    fname = ["bof", "idt", "color"]
    # compute substitution score matrix for each feature separately
    for f in range(num_feat):

        for q in range(que_len):
            # compute distance between fature histograms
            tmp_mat[f][q] = np.linalg.norm(que_feat[f][q] - ref_feat[f], axis=1)

        # dirname = os.path.join(project_dir,"distmap",fname[f])
        # if not os.path.exists(dirname):
        #         func.makedir(dirname)

        # filename = os.path.join(dirname,r_name+q_name+".npy")
        # if os.path.exists(filename):
        #         tmp_mat[f] = np.load(filename)
        # else:
        #         for q in range(que_len):
        #                 #compute distance between fature histograms
        #                 tmp_mat[f][q] = np.linalg.norm(que_feat[f][q]-ref_feat[f],axis=1)
        #         np.save(filename,tmp_mat[f])

    scaler = MinMaxScaler()
    for f in range(num_feat):
        # each substitution score matrix is scaled by mini-max
        tmp2_mat = tmp_mat[f].reshape((ref_len * que_len, 1))
        scaled_mat = scaler.fit_transform(tmp2_mat)
        tmp_mat[f] = scaled_mat.reshape((que_len, ref_len))

    # integrating two features adaptively
    for q in range(que_len):

        # compute distance between two frames after integration
        score = np.zeros(ref_len)
        for f in range(num_feat):
            score += weight[f] * tmp_mat[f][q]
        sub_matrix[q] = score

    return sub_matrix


# this function make substitution score matrix using multiple features integrated adaptively
def SubstitutionScoreMatrixMultiFeatAdaptive(r_name, q_name, rate, *feat):
    ref_feat = []
    ref_neighbor = []
    que_feat = []
    que_neighbor = []
    for i, f in enumerate(feat):
        # load histograms
        ref_feat.append(np.load(f[1]))
        que_feat.append(np.load(f[0]))

        # compute number of neighbor
        tmp = CalucNeighbor(ref_feat[i], rate)
        # tmp = CalucNeighborExp(ref_feat[i],rate,exp,conv)
        # tmp = CalucNeighborOpposite(ref_feat[i],que_feat[i])
        ref_neighbor.append(tmp.astype("float"))

        tmp = CalucNeighbor(que_feat[i], rate)
        # tmp = CalucNeighborExp(que_feat[i],rate,exp,conv)
        # tmp = CalucNeighborOpposite(que_feat[i],ref_feat[i])
        que_neighbor.append(tmp.astype("float"))

    ref_neighbor = np.array(ref_neighbor)
    que_neighbor = np.array(que_neighbor)

    ref_len = len(ref_feat[0])
    que_len = len(que_feat[0])

    # initialize substitution score matrix
    sub_matrix = np.zeros((que_len, ref_len), dtype=np.float32)
    sub_matrix[:, :] = float("inf")

    num_feat = len(feat)
    tmp_mat = np.zeros((num_feat, que_len, ref_len), dtype=np.float32)

    fname = ["bof", "idt", "color"]
    # compute substitution score matrix for boject and motion features separately
    for f in range(num_feat):

        for q in range(que_len):
            # compute distance between fature histograms
            tmp_mat[f][q] = np.linalg.norm(que_feat[f][q] - ref_feat[f], axis=1)

        # tmp_mat[f] = bhattacharyya(ref_feat[f],que_feat[f])

        # dirname = os.path.join(project_dir,"distmap",fname[f])
        # if not os.path.exists(dirname):
        #         func.makedir(dirname)

        # filename = os.path.join(dirname,r_name+q_name+".npy")
        # if os.path.exists(filename):
        #         tmp_mat[f] = np.load(filename)
        # else:
        #         for q in range(que_len):
        #                 #compute distance between fature histograms
        #                 tmp_mat[f][q] = np.linalg.norm(que_feat[f][q]-ref_feat[f],axis=1)
        #         np.save(filename,tmp_mat[f])

    # scaler = MinMaxScaler()
    # for f in range(num_feat):
    #         #each substitution score matrix is scaled by mini-max
    #         tmp2_mat = tmp_mat[f].reshape((ref_len*que_len,1))
    #         scaled_mat = scaler.fit_transform(tmp2_mat)
    #         tmp_mat[f] = scaled_mat.reshape((que_len,ref_len))

    # #integrating two features adaptively
    # for q in range(que_len):
    #         neighbor = np.zeros(num_feat)
    #         weight = np.zeros(num_feat)
    #         for r in range(ref_len):

    #                 for f in range(num_feat):

    #                         neighbor[f] = ref_neighbor[f][r] + que_neighbor[f][q]

    #                 #compute importance
    #                 if neighbor.sum() == 0:
    #                         weight[:] = 1./num_feat
    #                 else:
    #                         for f in range(num_feat):
    #                                 weight[f] = sum(np.delete(neighbor,f))/((num_feat-1)*neighbor.sum())

    #                 #compute distance between two frames after integration
    #                 score = 0
    #                 for f in range(num_feat):
    #                         score += weight[f]*tmp_mat[f][q][r]
    #                 sub_matrix[q][r] = score

    scaler = MinMaxScaler()
    neighbors = np.zeros((num_feat, que_len, ref_len))
    for f in range(num_feat):
        # each substitution score matrix is scaled by mini-max
        tmp2_mat = tmp_mat[f].reshape((ref_len * que_len, 1))
        scaled_mat = scaler.fit_transform(tmp2_mat)
        tmp_mat[f] = scaled_mat.reshape((que_len, ref_len))
        for q in range(que_len):
            neighbors[f][q] = que_neighbor[f][q] + ref_neighbor[f]

    weights = np.zeros((num_feat, que_len, ref_len))
    for q in range(que_len):
        tmp = neighbors[:, q, :].sum(axis=0)
        for f in range(num_feat):
            weights[f, q, :] = np.delete(neighbors, f, axis=0)[:, q, :].sum(axis=0) / ((num_feat - 1) * tmp)
        weights[:, q, :][:, tmp == 0] = 1. / num_feat
    # weights[weights!=weights] = 1./num_feat

    sub_matrix = np.sum(tmp_mat * weights, axis=0)

    return sub_matrix, weights


def SubstitutionScoreMatrixMultiFeatConected(*feat):
    ref_feat = []
    que_feat = []
    for i, f in enumerate(feat):
        # load histograms
        ref_feat.append(np.load(f[1]))
        que_feat.append(np.load(f[0]))

    ref_feat = np.concatenate(ref_feat, axis=1)
    que_feat = np.concatenate(que_feat, axis=1)

    ref_len = len(ref_feat)
    que_len = len(que_feat)

    # initialize substitution score matrix
    sub_matrix = np.zeros((que_len, ref_len), dtype=np.float32)

    # compute substitution score matrix for boject and motion features separately
    for q in range(que_len):
        # compute distance between fature histograms
        sub_matrix[q] = np.linalg.norm(que_feat[q] - ref_feat, axis=1)

    return sub_matrix


# this function make substitution score matrix using two features integrated adaptively based on one side neighbor
def SubstitutionScoreMatrixAdaptiveOneSide(b_bof, s_bof, b_motion, s_motion, rate, start=None, end=None):
    # load histograms for reference video and compute number of neighbor
    sup_shist = bof.load(s_bof)
    sup_shist_neighbor = CalucNeighbor(sup_shist, rate)
    sup_shist_neighbor = sup_shist_neighbor.astype("float")

    sup_idt = np.load(s_motion)
    sup_idt_neighbor = CalucNeighbor(sup_idt, rate)
    sup_idt_neighbor = sup_idt_neighbor.astype("float")

    # load histograms for query video
    beg_shist = bof.load(b_bof)
    beg_idt = np.load(b_motion)

    # cut only the range to be used
    if not start is None or not end is None:
        beg_shist = beg_shist[start:end]
        beg_idt = beg_idt[start:end]

    sup_len = len(sup_shist)
    beg_len = len(beg_shist)

    # initialize substitution score matrix
    sub_matrix = np.zeros((beg_len, sup_len), dtype=np.float32)
    select_matrix = np.zeros((beg_len, sup_len), dtype=np.float32)
    sub_matrix[:, :] = float("inf")

    smatrix = np.zeros((beg_len, sup_len), dtype=np.float32)  # for object features
    mmatrix = np.zeros((beg_len, sup_len), dtype=np.float32)  # for motion features

    # compute substitution score matrix for boject and motion features separately
    for b in range(beg_len):
        # compute distance between BoF histograms
        sscore = np.linalg.norm(beg_shist[b] - sup_shist, axis=1)
        smatrix[b] = sscore

        # compute distance between motion histograms
        mscore = np.linalg.norm(beg_idt[b] - sup_idt, axis=1)
        mmatrix[b] = mscore

    # each substitution score matrix is scaled by mini-max
    smatrix = smatrix.reshape((sup_len * beg_len, 1))
    mmatrix = mmatrix.reshape((sup_len * beg_len, 1))

    scaler = MinMaxScaler()
    sscaled = scaler.fit_transform(smatrix)
    mscaled = scaler.fit_transform(mmatrix)

    sscaled = sscaled.reshape((beg_len, sup_len))
    mscaled = mscaled.reshape((beg_len, sup_len))

    # integrating two features adaptively
    for b in range(beg_len):
        for s in range(sup_len):

            s_neighbor = sup_shist_neighbor[s]
            m_neighbor = sup_idt_neighbor[s]

            # compute importance
            if s_neighbor + m_neighbor == 0:
                s_weight = 0.5
                m_weight = 0.5
            else:
                s_weight = m_neighbor / (s_neighbor + m_neighbor)
                m_weight = 1 - s_weight

            # compute distance between two frames after integration
            score = s_weight * sscaled[b][s] + m_weight * mscaled[b][s]
            sub_matrix[b][s] = score
            select_matrix[b][s] = s_weight

    return sub_matrix, select_matrix


# this function run DP matching
def Matching(sub_matrix):
    beg_len, sup_len = sub_matrix.shape

    # initialize cost matrix
    matrix = np.zeros((beg_len + 1, sup_len + 1))
    matrix[:, :] = float("inf")
    matrix[0][0] = 0

    # initialize dictionary for traceback
    traceback = {}
    for i in range(sup_len):
        traceback[(i + 1, 0)] = (i, 0)

    for j in range(beg_len):
        traceback[(0, j + 1)] = (0, j)

    # DP matching
    for i in range(1, beg_len + 1):
        for j in range(1, sup_len + 1):
            score = []
            f = {}

            score.append(matrix[i][j - 1] + sub_matrix[i - 1][j - 1])
            f[matrix[i][j - 1] + sub_matrix[i - 1][j - 1]] = (i, j - 1)

            score.append(matrix[i - 1][j] + sub_matrix[i - 1][j - 1])
            f[matrix[i - 1][j] + sub_matrix[i - 1][j - 1]] = (i - 1, j)

            score.append(matrix[i - 1][j - 1] + (1) * sub_matrix[i - 1][j - 1])
            f[matrix[i - 1][j - 1] + (1) * sub_matrix[i - 1][j - 1]] = (i - 1, j - 1)

            matrix[i][j] = min(score)
            traceback[(i, j)] = f[matrix[i][j]]

    return traceback


# this function run continuous DP matching
def ContinuousMatching(sub_matrix):
    beg_len, sup_len = sub_matrix.shape

    # initialize cost matrix
    matrix = np.zeros((beg_len + 1, sup_len + 1))
    C = np.zeros((beg_len + 1, sup_len + 1))
    matrix[1:, :] = float("inf")
    matrix[0, :] = 0

    # path weight
    diag = 1
    hori_or_verti = 1

    # DP matching
    traceback = {}
    for i in range(1, beg_len + 1):
        if i == 1:
            for j in range(1, sup_len + 1):
                matrix[i][j] = matrix[i - 1][j - 1] + diag * sub_matrix[i - 1][j - 1]
                traceback[(i, j)] = (i - 1, j - 1)

                C[i][j] = C[i - 1][j - 1] + diag

        else:
            for j in range(1, sup_len + 1):
                score = []
                f = {}
                c = {}

                # val = matrix[i-2][j-1]+diag*sub_matrix[i-2][j-1]+sub_matrix[i-1][j-1]
                # score.append(val)
                # f[val] = (i-1,j)
                # c[val] = C[i-2][j-1]+diag+hori_or_verti

                # if j != 1:
                #         val = matrix[i-1][j-2]+diag*sub_matrix[i-1][j-2]+sub_matrix[i-1][j-1]
                #         score.append(val)
                #         f[val] = (i,j-1)
                #         c[val] = C[i-1][j-2]+diag+hori_or_verti

                # val = matrix[i-1][j-1]+diag*sub_matrix[i-1][j-1]
                # score.append(val)
                # f[val] = (i-1,j-1)
                # c[val] = C[i-1][j-1]+diag

                val = matrix[i - 1][j] + sub_matrix[i - 1][j - 1]
                score.append(val)
                f[val] = (i - 1, j)
                c[val] = C[i - 1][j] + hori_or_verti

                val = matrix[i][j - 1] + sub_matrix[i - 1][j - 1]
                score.append(val)
                f[val] = (i, j - 1)
                c[val] = C[i][j - 1] + hori_or_verti

                val = matrix[i - 1][j - 1] + diag * sub_matrix[i - 1][j - 1]
                score.append(val)
                f[val] = (i - 1, j - 1)
                c[val] = C[i - 1][j - 1] + diag

                matrix[i][j] = min(score)
                traceback[(i, j)] = f[matrix[i][j]]
                C[i][j] = c[matrix[i][j]]

    G = matrix[-1, 1:] / C[-1, 1:]
    idx = np.argmin(G) + 1

    return traceback, (beg_len, idx)


# this function run traceback to find minimuim cost path
def TraceBack(sub_matrix, trback, spos=None):
    beg_len, sup_len = sub_matrix.shape

    seq1 = []
    for i in range(1, beg_len + 1):
        seq1.append(i)

    seq2 = []
    for i in range(1, sup_len + 1):
        seq2.append(i)

    if spos is None:
        path = (beg_len, sup_len)
    else:
        path = spos

    gap = 0
    dp_seq1 = []
    dp_seq2 = []
    best_path = []
    best_path.append(path)
    path = trback[path]
    best_path.append(path)

    while not path[0] == 0:
        path = trback[path]
        best_path.append(path)

    best_path.reverse()

    pre1 = best_path[0][0]
    pre2 = best_path[0][1]

    for i in range(1, len(best_path)):
        if best_path[i][0] == pre1:
            dp_seq1.append(gap)
        elif best_path[i][0] == pre1 + 1:
            # dp_seq1.append(seq1.pop(0))
            dp_seq1.append(best_path[i][0])
        pre1 = best_path[i][0]

    for j in range(1, len(best_path)):
        if best_path[j][1] == pre2:
            dp_seq2.append(gap)
        elif best_path[j][1] == pre2 + 1:
            # dp_seq2.append(seq2.pop(0))
            dp_seq2.append(best_path[j][1])
        pre2 = best_path[j][1]

    return (dp_seq1, dp_seq2), best_path


# this function compute number of neighbors for each frame
def CalucNeighbor(feat, rate):
    target_movie_len = len(feat)
    # neighbor = target_movie_len/8.
    neighbor = target_movie_len * rate

    self_distance = np.zeros((len(feat), len(feat)))
    # euclid distance
    for i in range(len(feat)):
        self_distance[i] = np.linalg.norm(feat - feat[i], axis=1)

    # battatyalia distance
    # feat_sum = feat.sum(axis=1)
    # feat_sum[feat_sum==0] = 1.0
    # for i in range(len(feat)):
    #         tmp1 = np.sqrt(feat*feat[i])
    #         tmp1 = tmp1.sum(axis=1)
    #         tmp2 = np.sqrt(feat_sum*feat_sum[i])
    #         self_distance[i] = np.sqrt(1-tmp1.astype(np.float32)/tmp2)
    # self_distance[self_distance!=self_distance] = 0.0

    threshold = otsu(self_distance.flatten())

    neighbor_num = np.zeros(target_movie_len)
    idx = [True for i in range(target_movie_len)]
    for i in range(target_movie_len):
        s = i - int(neighbor / 2.) if i - int(neighbor / 2.) >= 0 else 0
        e = i + int(neighbor / 2.) if i + int(neighbor / 2.) < target_movie_len else target_movie_len - 1
        # tmp1 = len(self_distance[i,0:s+1][self_distance[i,0:s+1]<threshold])
        # tmp2 = len(self_distance[i,e:target_movie_len+1][self_distance[i,e:target_movie_len+1]<threshold])
        # neighbor_num[i] = tmp1+tmp2
        tmp = np.copy(idx)
        tmp[s:e + 1] = False
        neighbor_num[i] = sum(self_distance[i][tmp] < threshold)

    # num = nf
    num = 3
    kernel = np.ones(num) / num
    neighbor_num = np.convolve(neighbor_num, kernel, mode='same')

    neighbor_num = neighbor_num / float(target_movie_len)

    return neighbor_num


def CalucNeighborExp(feat, rate, exp, conv):
    target_movie_len = len(feat)
    neighbor = target_movie_len * rate

    self_distance = np.zeros((len(feat), len(feat)))
    # euclid distance
    for i in range(len(feat)):
        self_distance[i] = np.linalg.norm(feat - feat[i], axis=1)

    # battatyalia distance
    # feat_sum = feat.sum(axis=1)
    # feat_sum[feat_sum==0] = 1.0
    # for i in range(len(feat)):
    #         tmp1 = np.sqrt(feat*feat[i])
    #         tmp1 = tmp1.sum(axis=1)
    #         tmp2 = np.sqrt(feat_sum*feat_sum[i])
    #         self_distance[i] = np.sqrt(1-tmp1.astype(np.float32)/tmp2)
    # self_distance[self_distance!=self_distance] = 0.0

    threshold = otsu(self_distance.flatten())

    alpha = exp
    beta = alpha / float(target_movie_len)
    f = lambda x: (math.e) ** (beta * x)
    neighbor_num = np.zeros(target_movie_len)
    idx = [True for i in range(target_movie_len)]
    for i in range(target_movie_len):
        s = i - int(neighbor / 2.) if i - int(neighbor / 2.) >= 0 else 0
        e = i + int(neighbor / 2.) if i + int(neighbor / 2.) < target_movie_len else target_movie_len - 1

        l = np.arange(len(feat))
        # first_len = len(l[:s])
        # latter_len = len(l[e+1:])
        inter_len = len(l[s:e + 1])
        # first = np.arange(first_len).astype(np.float32)
        # if first_len != 0 and first_len != 1: first /= first.max()
        # latter = np.arange(latter_len).astype(np.float32)
        # if latter_len != 0 and latter_len != 1: latter /= latter.max()

        first = np.array(l[:s])
        latter = np.array(l[e + 1:])
        first = np.array(list(map(f, first - s + 1)))
        latter = np.array(list(map(f, -(latter - e - 1))))
        inter = np.array([np.nan for _ in range(inter_len)])
        ll = np.concatenate((first, inter, latter))
        ll[ll != ll] = 0

        # r = np.array(list(map(f,ll)))
        # r[r!=r] = 0

        tmp = np.copy(idx)
        tmp[s:e + 1] = False
        # neighbor_num[i] = sum(self_distance[i][tmp]<threshold)
        neighbor_num[i] = sum(ll[self_distance[i] < threshold])

    num = conv
    kernel = np.ones(num) / num
    neighbor_num = np.convolve(neighbor_num, kernel, mode='same')

    # neighbor_num = neighbor_num/float(target_movie_len)

    return neighbor_num


def CalucNeighborGaussian(feat, rate):
    target_movie_len = len(feat)
    neighbor = target_movie_len * rate

    self_distance = np.zeros((len(feat), len(feat)))
    for i in range(len(feat)):
        self_distance[i] = np.linalg.norm(feat - feat[i], axis=1)

    con_dist = np.concatenate(self_distance)

    # compute threshold using otsu method
    thresholds = []
    ths = []
    interval = con_dist.max() / 1000.
    for th in func.drange(con_dist.min(), con_dist.max() + interval, interval):
        w1 = len(con_dist[con_dist >= th])
        m1 = con_dist[con_dist >= th].mean()
        w2 = len(con_dist[con_dist < th])
        m2 = con_dist[con_dist < th].mean()
        thresholds.append(w1 * w2 * ((m1 - m2) ** 2))
        ths.append(th)
    threshold = ths[np.nanargmax(thresholds)]

    neighbor_num = np.zeros(target_movie_len)
    idx_list = np.arange(1, target_movie_len + 1)
    idx_list = idx_list.astype(np.float32)
    idx = [True for i in range(target_movie_len)]
    one_list = np.ones(target_movie_len, dtype=np.float32)
    f = lambda x: (math.exp(-x ** 2 / (2. * ((30) ** 2)))) / (math.sqrt(2 * math.pi) * (30))
    for i in range(target_movie_len):
        s = i - int(neighbor / 2.) if i - int(neighbor / 2.) >= 0 else 0
        e = i + int(neighbor / 2.) if i + int(neighbor / 2.) < target_movie_len else target_movie_len - 1
        tmp = np.copy(idx)
        tmp[s:e + 1] = False
        idxl_cp = np.copy(idx_list)
        idxl_cp[~tmp] = np.nan
        idxl_cp[self_distance[i] < threshold] = np.nan
        idxl_cp[:s] = idxl_cp[:s] - int(len(idxl_cp[:s]) / 2.)
        idxl_cp[:s] = list(map(f, idxl_cp[:s]))
        idxl_cp[e + 1:] = idxl_cp[e + 1:] - e - 1 - int(len(idxl_cp[e + 1:]) / 2.)
        idxl_cp[e + 1:] = list(map(f, idxl_cp[e + 1:]))
        idxl_cp[idxl_cp != idxl_cp] = 0
        n = one_list * idxl_cp
        neighbor_num[i] = sum(n)

    neighbor_num = neighbor_num / float(sum(neighbor_num))

    return neighbor_num


def CalucNeighborFixed(feat, rate, th):
    target_movie_len = len(feat)
    neighbor = target_movie_len * rate

    self_distance = np.zeros((len(feat), len(feat)))
    for i in range(len(feat)):
        self_distance[i] = np.linalg.norm(feat - feat[i], axis=1)

    threshold = otsu(self_distance.flatten())
    threshold = threshold * th

    neighbor_num = np.zeros(target_movie_len)
    idx = [True for i in range(target_movie_len)]
    for i in range(target_movie_len):
        s = i - int(neighbor / 2.) if i - int(neighbor / 2.) >= 0 else 0
        e = i + int(neighbor / 2.) if i + int(neighbor / 2.) < target_movie_len else target_movie_len - 1
        tmp = np.copy(idx)
        tmp[s:e + 1] = False
        neighbor_num[i] = sum(self_distance[i][tmp] < threshold)

    # num = nf
    # kernel = np.ones(num)/num
    # neighbor_num = np.convolve(neighbor_num,kernel,mode='same')

    neighbor_num = neighbor_num / float(target_movie_len)

    return neighbor_num


def CalucNeighborAve(feat, rate):
    target_movie_len = len(feat)
    neighbor = target_movie_len * rate

    self_distance = np.zeros((len(feat), len(feat)))
    neighbor_num = np.zeros(len(feat))
    r = 50
    nsum = 0
    nnum = 0
    for i in range(len(feat)):
        self_distance[i] = np.linalg.norm(feat - feat[i], axis=1)
        s = i - r if i - r >= 0 else 0
        e = i + r if i + r <= len(feat) else len(feat)
        nsum += self_distance[i][s:e].sum()
        nnum += len(self_distance[i][s:e])

    # selfdist_cp = np.copy(self_distance)
    # #self_distance = (selfdist_cp - selfdist_cp.mean()) / selfdist_cp.std()
    # self_distance = (selfdist_cp - selfdist_cp.min()) / (selfdist_cp.max() - selfdist_cp.min())

    # for i in tqdm(range(len(feat))):
    #         s = i-r if i-r >= 0 else 0
    #         e = i+r if i+r <= len(feat) else len(feat)
    #         nsum += self_distance[i][s:e].sum()
    #         nnum += len(self_distance[i][s:e])

    th = nsum / nnum

    idx = [True for i in range(target_movie_len)]
    for i in range(len(feat)):
        s = i - int(neighbor / 2.) if i - int(neighbor / 2.) >= 0 else 0
        e = i + int(neighbor / 2.) if i + int(neighbor / 2.) < target_movie_len else target_movie_len - 1
        tmp = np.copy(idx)
        tmp[s:e + 1] = False
        neighbor_num[i] = sum(self_distance[i][tmp] < th)
    neighbor_num = neighbor_num / float(target_movie_len)

    return neighbor_num


def CalucNeighborLog(feat, rate):
    target_movie_len = len(feat)
    neighbor = target_movie_len * rate

    self_distance = np.zeros((len(feat), len(feat)))

    for i in range(len(feat)):
        self_distance[i] = np.linalg.norm(feat - feat[i], axis=1)

    selfdist_cp1 = np.log(self_distance)

    selfdist_cp2 = np.copy(selfdist_cp1)
    selfdist_cp3 = selfdist_cp2[selfdist_cp2 != -np.inf]
    selfdist_cp1 = (selfdist_cp2 - selfdist_cp3.mean()) / selfdist_cp3.std()
    # selfdist_cp1 = (selfdist_cp2 - selfdist_cp2.min()) / (selfdist_cp2.max() - selfdist_cp2.min())

    th = selfdist_cp1.max() * 0.2

    neighbor_num = np.zeros(len(feat))
    idx = [True for i in range(target_movie_len)]
    for i in range(len(feat)):
        s = i - int(neighbor / 2.) if i - int(neighbor / 2.) >= 0 else 0
        e = i + int(neighbor / 2.) if i + int(neighbor / 2.) < target_movie_len else target_movie_len - 1
        tmp = np.copy(idx)
        tmp[s:e + 1] = False
        neighbor_num[i] = sum(selfdist_cp1[i][tmp] < th)
    neighbor_num = neighbor_num / float(target_movie_len)

    return neighbor_num


def CalucNeighborCluster(feat, rate, nf=1):
    target_movie_len = len(feat)
    neighbor = target_movie_len * rate

    kmodel = KMeans(n_clusters=nf,
                    max_iter=300,
                    n_init=100
                    ).fit(feat)
    labels = kmodel.labels_

    neighbor_num = np.zeros(target_movie_len)
    idx = [True for i in range(target_movie_len)]
    for i in range(target_movie_len):
        s = i - int(neighbor / 2.) if i - int(neighbor / 2.) >= 0 else 0
        e = i + int(neighbor / 2.) if i + int(neighbor / 2.) < target_movie_len else target_movie_len - 1
        tmp = np.copy(idx)
        tmp[s:e + 1] = False
        neighbor_num[i] = sum(labels[tmp] == labels[i])

    # num = nf
    # kernel = np.ones(num)/num
    # neighbor_num = np.convolve(neighbor_num,kernel,mode='same')

    neighbor_num = neighbor_num / float(target_movie_len)

    return neighbor_num


def CalucNeighborOpposite(main_feat, sub_feat, nf=1):
    main_len = len(main_feat)
    sub_len = len(sub_feat)

    dist_map = np.zeros((main_len, sub_len))

    for i in range(main_len):
        dist_map[i] = np.linalg.norm(sub_feat - main_feat[i], axis=1)

    th = otsu(dist_map.flatten())

    # neighbor_num = np.zeros(main_len)
    # idx = [True for i in range(main_len)]
    # for i in range(main_len):
    #         neighbor_num[i] = sum(dist_map[i]<th)
    neighbor_num = np.sum(dist_map < th, axis=1)

    # num = nf
    # kernel = np.ones(num)/num
    # neighbor_num = np.convolve(neighbor_num,kernel,mode='same')

    neighbor_num = neighbor_num / float(main_len)

    return neighbor_num


def VisualizeWightMultiFeatAdaptive(r_name, q_name, rate, *feat):
    ref_feat = []
    ref_neighbor = []
    que_feat = []
    que_neighbor = []
    for i, f in enumerate(feat):
        # load histograms for reference video and compute number of neighbor
        ref_feat.append(np.load(f[1]))
        tmp = CalucNeighborFixed(ref_feat[i], rate)
        ref_neighbor.append(tmp.astype("float"))

        # load histograms for query video and compute number of neighbor
        que_feat.append(np.load(f[0]))
        tmp = CalucNeighborFixed(que_feat[i], rate)
        que_neighbor.append(tmp.astype("float"))

    ref_len = len(ref_feat[0])
    que_len = len(que_feat[0])

    num_feat = len(feat)

    # integrating two features adaptively
    weight = np.zeros((num_feat, que_len, ref_len))
    for q in range(que_len):
        neighbor = np.zeros(num_feat)
        for r in range(ref_len):

            for f in range(num_feat):
                neighbor[f] = ref_neighbor[f][r] + que_neighbor[f][q]

            # compute importance
            if neighbor.sum() == 0:
                weight[:] = 1. / num_feat
            else:
                for f in range(num_feat):
                    weight[f][q][r] = sum(np.delete(neighbor, f)) / ((num_feat - 1) * neighbor.sum())

    return weight


def otsu(dist):
    sys.setrecursionlimit(10000)

    hist, edge = np.histogram(dist, bins=1000)
    norm_hist = hist / float(hist.sum())
    mid_edge = np.array([(edge[i] + edge[i + 1]) / 2. for i in range(len(edge) - 1)])

    u_t = 0
    for i, me in enumerate(mid_edge):
        u_t += me * norm_hist[i]

    def calc_threshold(sigma, norm_hist, mid_edge, w, u, u_t, i):
        sigma[i] = ((u_t * w - u) ** 2) / (w * (1 - w))
        i += 1
        if i == len(sigma) - 1:
            return 0
        w = w + norm_hist[i]
        u = u + mid_edge[i] * norm_hist[i]
        calc_threshold(sigma, norm_hist, mid_edge, w, u, u_t, i)

    sigma = np.zeros(len(mid_edge))
    w = norm_hist[0]
    u = mid_edge[0] * norm_hist[0]
    i = 0
    calc_threshold(sigma, norm_hist, mid_edge, w, u, u_t, i)
    idx = np.argmax(sigma)

    return mid_edge[idx]


def bhattacharyya(f1, f2):
    sdist = np.zeros((len(f2), len(f1)), dtype=np.float32)
    f1_sum = f1.sum(axis=1)
    f1_sum[f1_sum == 0] = 1.0
    f2_sum = f2.sum(axis=1)
    f2_sum[f2_sum == 0] = 1.0
    for i in range(len(f2)):
        tmp1 = np.sqrt(f1 * f2[i])
        tmp1 = tmp1.sum(axis=1)
        tmp2 = np.sqrt(f1_sum * f2_sum[i])
        sdist[i] = np.sqrt(1 - tmp1.astype(np.float32) / tmp2)
    sdist[sdist != sdist] = 0.0

    return sdist
