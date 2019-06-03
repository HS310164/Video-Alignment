# -*- coding:utf-8 -*-

# import numpy as np
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
# import matplotlib.patches as patches


def CostMatPlot(r_name, q_name, xbar, ybar, df, r_correct, q_correct, filename, gt=True):
    if gt: plt.scatter(r_correct, q_correct, color='lime', label="Ground Truth")
    plt.plot(xbar, ybar, color='cyan', label="Minimum Cost Path")
    im = plt.imshow(df, aspect='auto', interpolation='None', cmap=plt.cm.hot)
    plt.colorbar(label="Inter-Frame Distance")
    # cb = plt.colorbar(im,label="Inter-Frame Distance")
    # font = mpl.font_manager.FontProperties(size=20)
    # cb.ax.yaxis.label.set_font_properties(font)
    # l = plt.legend(fontsize=20)
    # f = l.get_frame()
    # f.set_facecolor("lightgray")
    plt.ylabel("Query (" + q_name + ")")
    plt.xlabel("Reference (" + r_name + ")")
    # plt.xlabel("Video B Frames",fontsize=20)
    # plt.ylabel("Video A Frames",fontsize=20)
    plt.legend()

    plt.savefig(filename)
    plt.clf()


def CostMatPlot2(r_name, q_name, xbar, ybar, df, r_correct, q_correct, r_label, r_task, q_label, q_task, filename,
                 gt=True):
    if gt: plt.scatter(r_correct, q_correct, color='lime', label="Ground Truth")
    plt.plot(xbar, ybar, color='cyan', label="Minimum Cost Path")
    im = plt.imshow(df, aspect='auto', interpolation='None', cmap=plt.cm.hot)
    plt.colorbar(label="Inter-Frame Distance")
    # cb = plt.colorbar(im,label="Inter-Frame Distance")
    # font = mpl.font_manager.FontProperties(size=20)
    # cb.ax.yaxis.label.set_font_properties(font)
    # l = plt.legend(fontsize=20)
    # f = l.get_frame()
    # f.set_facecolor("lightgray")
    plt.ylabel("Query (" + q_name + ")")
    plt.xlabel("Reference (" + r_name + ")")
    # plt.xlabel("Video B Frames",fontsize=20)
    # plt.ylabel("Video A Frames",fontsize=20)
    plt.yticks(q_label, q_task, rotation=90, fontsize=5)
    plt.xticks(r_label, r_task, fontsize=5)
    plt.legend()
    plt.tight_layout()

    plt.savefig(filename)
    plt.clf()


def CostMatPlot3(r_name, q_name, xbar, ybar, df, r_start, r_end, q_start, q_end, filename, gt=True):
    if gt:
        plt.scatter(r_start, q_start, color='lime', label="Ground Truth", s=45)
    #                plt.scatter(r_start,q_start,color='lime',label = "Ground Truth")
    # plt.scatter(r_end,q_end,color='yellow',label = "Ground Truth (end)")
    plt.plot(xbar, ybar, color='cyan', label="Minimum Cost Path", linewidth=3)
    #        plt.plot(xbar,ybar,color='cyan',label = "Minimum Cost Path")
    im = plt.imshow(df, aspect='auto', interpolation='None', cmap=plt.cm.hot)
    # plt.colorbar(label="Inter-Frame Distance")
    cb = plt.colorbar(im, label="Inter-Frame Distance")
    font = mpl.font_manager.FontProperties(size=20)
    cb.ax.yaxis.label.set_font_properties(font)
    l = plt.legend(fontsize=20)
    f = l.get_frame()
    f.set_facecolor("lightgray")
    # plt.ylabel("Query ("+q_name+")")
    # plt.xlabel("Reference ("+r_name+")")
    plt.xlabel("Video A Frames", fontsize=20)
    plt.ylabel("Video B Frames", fontsize=20)
    plt.legend()

    plt.savefig(filename)
    plt.clf()


def CostMatPlot4(r_name, q_name, xbar, ybar, df, r_start, r_end, q_start, q_end, filename, gt=True):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(xbar, ybar, color='cyan', label="Minimum Cost Path", linewidth=3)
    if gt:
        for i in range(len(r_start)):
            w = r_end[i] - r_start[i]
            h = q_end[i] - q_start[i]
            r = plt.Rectangle(xy=(r_start[i], q_start[i]), width=w, height=h, linewidth='3.0', ec='g', fill=False)
            ax.add_patch(r)
        # plt.scatter(r_start,q_start,color='lime',label = "Ground Truth",s=45)
    im = plt.imshow(df, aspect='auto', interpolation='None', cmap=plt.cm.hot)
    # plt.colorbar(label="Inter-Frame Distance")
    cb = plt.colorbar(im, label="Inter-Frame Distance")
    font = mpl.font_manager.FontProperties(size=20)
    cb.ax.yaxis.label.set_font_properties(font)
    # l = plt.legend(fontsize=20)
    # f = l.get_frame()
    # f.set_facecolor("lightgray")
    plt.xlabel("Video A Frames", fontsize=20)
    plt.ylabel("Video B Frames", fontsize=20)
    # plt.legend()

    plt.savefig(filename)
    plt.clf()


def CostMatPlotCrop(r_name, q_name, xbar, ybar, df, r_start, r_end, q_start, q_end, xlim, ylim, filename, gt=True):
    if gt:
        plt.scatter(r_start, q_start, color='lime', label="Ground Truth", s=250)
        # plt.scatter(r_start,q_start,color='lime',label = "Ground Truth")
        # plt.scatter(r_end,q_end,color='yellow',label = "Ground Truth (end)")
    plt.plot(xbar, ybar, color='cyan', label="Minimum Cost Path", linewidth=7)
    # plt.plot(xbar,ybar,color='cyan',label = "Minimum Cost Path")
    im = plt.imshow(df, aspect='auto', interpolation='None', cmap=plt.cm.hot)
    # plt.colorbar(label="Inter-Frame Distance")
    # cb = plt.colorbar(im,label="Inter-Frame Distance")
    # font = mpl.font_manager.FontProperties(size=20)
    # cb.ax.yaxis.label.set_font_properties(font)
    # l = plt.legend(fontsize=20)
    # f = l.get_frame()
    # f.set_facecolor("lightgray")
    # plt.ylabel("Query ("+q_name+")")
    # plt.xlabel("Reference ("+r_name+")")
    # plt.xlabel("Video A Frames",fontsize=20)
    # plt.ylabel("Video B Frames",fontsize=20)
    # plt.legend()
    plt.ylim(ylim[::-1])
    plt.xlim(xlim)
    plt.axis("off")

    plt.savefig(filename)
    plt.clf()


def CostMatPlotOnly(df, filename):
    im = plt.imshow(df, aspect='auto', interpolation='None', cmap=plt.cm.hot)
    plt.colorbar(label="Inter-Frame Distance")
    # plt.ylabel("Query ("+q_name+")")
    # plt.xlabel("Reference ("+r_name+")")
    # plt.xlabel("Video A Frames",fontsize=20)
    # plt.ylabel("Video B Frames",fontsize=20)
    plt.legend()

    plt.savefig(filename)
    plt.clf()


def SequencePosition(r_len, q_len, r_range, q_range, filename):
    r_range.reverse()
    q_range.reverse()

    xh = [0, 1]
    height = 0.2

    plt.barh(xh, [q_len, r_len], height=height, align="center", color="w")

    color = "rw"

    for i, (r, q) in enumerate(zip(r_range, q_range)):
        plt.barh(xh, [q, r], height=height, align="center", color=color[i])

    plt.yticks(xh, ["Query", "Reference"])

    plt.savefig(filename)


def VisualizeWeight(bof, idt, color, mini, maxi, pf_filename):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(bof, aspect='auto', interpolation='None', cmap=plt.cm.seismic)
    im.set_clim(mini, maxi)
    cb = fig.colorbar(im, label="Weight", ax=ax)
    font = mpl.font_manager.FontProperties(size=20)
    cb.ax.yaxis.label.set_font_properties(font)
    ax.set_xlabel("Video A Frames", fontsize=20)
    ax.set_ylabel("Video B Frames", fontsize=20)
    plt.savefig(pf_filename + "bof.png")
    plt.clf()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(idt, aspect='auto', interpolation='None', cmap=plt.cm.seismic)
    im.set_clim(mini, maxi)
    cb = fig.colorbar(im, label="Weight", ax=ax)
    font = mpl.font_manager.FontProperties(size=20)
    cb.ax.yaxis.label.set_font_properties(font)
    ax.set_xlabel("Video A Frames", fontsize=20)
    ax.set_ylabel("Video B Frames", fontsize=20)
    plt.savefig(pf_filename + "idt.png")
    plt.clf()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(color, aspect='auto', interpolation='None', cmap=plt.cm.seismic)
    im.set_clim(mini, maxi)
    cb = fig.colorbar(im, label="Weight", ax=ax)
    font = mpl.font_manager.FontProperties(size=20)
    cb.ax.yaxis.label.set_font_properties(font)
    ax.set_xlabel("Video A Frames", fontsize=20)
    ax.set_ylabel("Video B Frames", fontsize=20)
    plt.savefig(pf_filename + "color.png")
    plt.clf()


def VisualizeWeight_3dadd(bof, idt, color, thdcnn, mini, maxi, pf_filename):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(bof, aspect='auto', interpolation='None', cmap=plt.cm.seismic)
    im.set_clim(mini, maxi)
    cb = fig.colorbar(im, label="Weight", ax=ax)
    font = mpl.font_manager.FontProperties(size=20)
    cb.ax.yaxis.label.set_font_properties(font)
    ax.set_xlabel("Video A Frames", fontsize=20)
    ax.set_ylabel("Video B Frames", fontsize=20)
    plt.savefig(pf_filename + "bof.png")
    plt.clf()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(idt, aspect='auto', interpolation='None', cmap=plt.cm.seismic)
    im.set_clim(mini, maxi)
    cb = fig.colorbar(im, label="Weight", ax=ax)
    font = mpl.font_manager.FontProperties(size=20)
    cb.ax.yaxis.label.set_font_properties(font)
    ax.set_xlabel("Video A Frames", fontsize=20)
    ax.set_ylabel("Video B Frames", fontsize=20)
    plt.savefig(pf_filename + "idt.png")
    plt.clf()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(color, aspect='auto', interpolation='None', cmap=plt.cm.seismic)
    im.set_clim(mini, maxi)
    cb = fig.colorbar(im, label="Weight", ax=ax)
    font = mpl.font_manager.FontProperties(size=20)
    cb.ax.yaxis.label.set_font_properties(font)
    ax.set_xlabel("Video A Frames", fontsize=20)
    ax.set_ylabel("Video B Frames", fontsize=20)
    plt.savefig(pf_filename + "color.png")
    plt.clf()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(thdcnn, aspect='auto', interpolation='None', cmap=plt.cm.seismic)
    im.set_clim(mini, maxi)
    cb = fig.colorbar(im, label="Weight", ax=ax)
    font = mpl.font_manager.FontProperties(size=20)
    cb.ax.yaxis.label.set_font_properties(font)
    ax.set_xlabel("Video A Frames", fontsize=20)
    ax.set_ylabel("Video B Frames", fontsize=20)
    plt.savefig(pf_filename + "3dcnn.png")
    plt.clf()


def VisualizeWeight_cnns(cnn2d, cnn3d, mini, maxi, pf_filename):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(cnn2d, aspect='auto', interpolation='None', cmap=plt.cm.seismic)
    im.set_clim(mini, maxi)
    cb = fig.colorbar(im, label="Weight", ax=ax)
    font = mpl.font_manager.FontProperties(size=20)
    cb.ax.yaxis.label.set_font_properties(font)
    ax.set_xlabel("Video A Frames", fontsize=20)
    ax.set_ylabel("Video B Frames", fontsize=20)
    plt.savefig(pf_filename + "cnn2d.png")
    plt.clf()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(cnn3d, aspect='auto', interpolation='None', cmap=plt.cm.seismic)
    im.set_clim(mini, maxi)
    cb = fig.colorbar(im, label="Weight", ax=ax)
    font = mpl.font_manager.FontProperties(size=20)
    cb.ax.yaxis.label.set_font_properties(font)
    ax.set_xlabel("Video A Frames", fontsize=20)
    ax.set_ylabel("Video B Frames", fontsize=20)
    plt.savefig(pf_filename + "cnn3d.png")
    plt.clf()
