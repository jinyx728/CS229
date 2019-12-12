######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
import argparse
import json

def plot_overfit():
    out_file='figs/0227/overfit.png'
    fig=plt.gcf()
    ax=plt.gca()
    ax.set_title('overfit')

    in_file='../../rundir/midres/overfit_1/file_logs/train_total.txt'
    data=np.loadtxt(in_file)
    X=data[:,0]/data[-1,0]
    Y=medfilt(np.log10(data[:,1]),11)
    ax.plot(X.copy(),Y.copy(),label='overfit_1')

    in_file='../../rundir/midres/overfit_16/file_logs/train_total.txt'
    data=np.loadtxt(in_file)
    X=data[:,0]/data[-1,0]
    Y=medfilt(np.log10(data[:,1]),11)
    ax.plot(X.copy(),Y.copy(),label='overfit_16')

    in_file='../../rundir/midres/overfit_256/file_logs/train_total.txt'
    data=np.loadtxt(in_file)
    X=data[:,0]/data[-1,0]
    Y=medfilt(np.log10(data[:,1]),11)
    ax.plot(X.copy(),Y.copy(),label='overfit_256')

    in_file='../../rundir/midres/overfit_1024/file_logs/train_total.txt'
    data=np.loadtxt(in_file)
    X=data[:,0]/data[-1,0]
    Y=medfilt(np.log10(data[:,1]),11)
    ax.plot(X.copy(),Y.copy(),label='overfit_1024')

    in_file='../../rundir/midres/lr_0.001/file_logs/train_total.txt'
    data=np.loadtxt(in_file)
    X=data[:,0]/data[-1,0]
    Y=medfilt(np.log10(data[:,1]),11)
    ax.plot(X.copy(),Y.copy(),label='dataset_9k')

    ax.legend()
    print('save to ',out_file)
    fig.savefig(out_file)

def plot_lr():
    out_file='figs/0227/lr.png'
    fig=plt.gcf()
    ax=plt.gca()
    ax.set_title('lr')

    in_file='../../rundir/midres/lr_0.00025/file_logs/train_total.txt'
    data=np.loadtxt(in_file)
    X=data[:,0]/data[-1,0]
    Y=np.log10(data[:,1])
    ax.plot(X.copy(),Y.copy(),label='lr_0.00025')

    in_file='../../rundir/midres/lr_0.0005/file_logs/train_total.txt'
    data=np.loadtxt(in_file)
    X=data[:,0]/data[-1,0]
    Y=np.log10(data[:,1])
    ax.plot(X.copy(),Y.copy(),label='lr_0.0005')

    in_file='../../rundir/midres/lr_0.001/file_logs/train_total.txt'
    data=np.loadtxt(in_file)
    X=data[:,0]/data[-1,0]
    Y=np.log10(data[:,1])
    ax.plot(X.copy(),Y.copy(),label='lr_0.001')

    in_file='../../rundir/midres/lr_0.1/file_logs/train_total.txt'
    data=np.loadtxt(in_file)
    X=data[:,0]/data[-1,0]
    Y=np.log10(data[:,1])
    ax.plot(X.copy(),Y.copy(),label='lr_0.1')

    ax.legend()
    print('save to ',out_file)
    fig.savefig(out_file)

def plot_bar():
    out_file='figs/0314/err_bar.png'
    fig=plt.gcf()
    ax=plt.gca()
    ax.bar([0,1],[1.6620021034099674e-05,1.3487488421426924e-05])
    ax.set_xticks([0,1])
    ax.set_xticklabels(['before','after'])
    fig.savefig(out_file)

def plot_err():
    in_file='../../rundir/midres_diff/diff/file_logs/train_total.json'
    out_file='figs/0314/err2.png'
    fig=plt.gcf()
    ax=plt.gca()
    data=np.array(json.load(open(in_file)))
    X,Y=data[:,1][:],data[:,2][:]
    ax.plot(X,Y)
    ax.plot([0,1000],[1.6620021034099674e-05,1.6620021034099674e-05])
    ax.set_ylim([1e-5,5e-5])
    fig.savefig(out_file)

def plot_pd():
    in_file='opt_test/primal_dual_log.txt'
    iters,etas,r_duals,cg_iters=[],[],[],[]
    with open(in_file) as f:
        while True:
            line=f.readline()
            if line=='':
                break
            parts=line.rstrip().split(',')
            for part in parts:
                k,v=part.split(':')
                if k=="iter":
                    iters.append(int(v))
                elif k=="eta":
                    etas.append(float(v))
                elif k=="r_dual":
                    r_duals.append(float(v))
                elif k=="cg_iters":
                    cg_iters.append(int(v))

    out_file='figs/0317/eta.png'
    fig=plt.figure()
    ax=plt.gca()
    X=np.array(iters)
    Y=np.log10(np.array(etas))
    ax.plot(X,Y)
    ax.set_title("eta")
    fig.savefig(out_file)

    out_file='figs/0317/r_dual.png'
    fig=plt.figure()
    ax=plt.gca()
    X=np.array(iters)
    Y=np.log10(np.array(r_duals))
    ax.plot(X,Y)
    ax.set_title("r_dual")
    fig.savefig(out_file)

    out_file='figs/0317/cg_iters.png'
    fig=plt.figure()
    ax=plt.gca()
    X=np.array(iters)
    Y=np.array(cg_iters)
    ax.bar(X,Y)
    ax.set_title("cg_iters")
    fig.savefig(out_file)

def plot_cvx():
    in_file='opt_test/cvx_log.txt'
    iters,gaps,r_duals=[],[],[]
    with open(in_file) as f:
        while True:
            line=f.readline()
            if line=='':
                break
            parts=line.rstrip().split()
            iters.append(int(parts[0]))
            gaps.append(float(parts[3]))
            r_duals.append(float(parts[5]))

    out_file='figs/0317/cvx/gap.png'
    fig=plt.figure()
    ax=plt.gca()
    X=np.array(iters)
    Y=np.log10(np.array(gaps))
    ax.plot(X,Y)
    ax.set_title("gap")
    fig.savefig(out_file)

    out_file='figs/0317/cvx/r_dual.png'
    fig=plt.figure()
    ax=plt.gca()
    X=np.array(iters)
    Y=np.log10(np.array(r_duals))
    ax.plot(X,Y)
    ax.set_title("r_dual")
    fig.savefig(out_file)

# 8.795517E-02 9.335280E-02 9.043594E-02
# 1.359658E-01 1.070004E-01 1.014696E-01 1.031643E-01 
def plot_0430():
    in_file1='figs/0430/run_logs-tag-train_post_proj_loss.json'
    in_file2='figs/0430/run_logs-tag-train_total.json'
    out_file='figs/0430/train_loss.png'
    out_file2='figs/0430/test_loss.png'
    N=90
    data1=json.load(open(in_file1))
    data1=np.array(data1)
    Y1=data1[:N,2]
    data2=json.load(open(in_file2))
    data2=np.array(data2)
    Y2=data2[:N,2]

    fig=plt.figure()
    ax=plt.gca()
    X=np.arange(1200,1200+N)
    ax.plot(X,Y1,label='w cvx')
    ax.plot(X,Y2,label='w/o cvx')
    ax.legend()
    ax.set_title("train_loss")
    fig.savefig(out_file)

    l1=np.array([8.795517E-02, 9.335280E-02, 9.043594E-02])
    l2=np.array([1.359658E-01, 1.070004E-01, 1.014696E-01, 1.031643E-01])
    fig=plt.figure()
    ax=plt.gca()
    ax.bar(['w cvx','w/o cvx'],[np.mean(l1),np.mean(l2)])
    ax.set_title("test_loss")
    fig.savefig(out_file2)

def plot_0508():
    in_file1='figs/0508/run-logs-tag-train_post_proj_loss.json'
    in_file2='figs/0508/run_logs-tag-train_total.json'
    out_file='figs/0508/train_loss.png'
    N=1000
    data_start=619
    plot_start=640
    data1=json.load(open(in_file1))
    data1=np.array(data1)
    Y1=data1[plot_start-data_start:N,2]
    data2=json.load(open(in_file2))
    data2=np.array(data2)
    Y2=data2[plot_start-data_start:N,2]

    fig=plt.figure()
    ax=plt.gca()
    ax.plot(np.arange(plot_start,plot_start+len(Y1)),Y1,label='w cvx')
    ax.plot(np.arange(plot_start,plot_start+len(Y2)),Y2,label='w/o cvx')
    ax.legend()
    ax.set_title("train_loss")
    fig.savefig(out_file)

def plot_0514():
    in_file1='figs/0514/run_t0_waist_front_hip_front_hip_right_waist_right_logs-tag-train_post_proj_loss.json'
    in_file2='figs/0514/run_t0_waist_front_hip_front_hip_right_waist_right_logs-tag-train_total.json'
    in_file3='figs/0514/run_avg_waist_front_hip_front_hip_right_waist_right_logs-tag-train_post_proj_loss.json'
    out_file='figs/0514/train_loss.png'
    N=10000
    data_start=631
    plot_start=631
    data1=json.load(open(in_file1))
    data1=np.array(data1)
    X1=data1[:,1]
    Y1=data1[:,2]
    I1=X1>=data_start
    X1=X1[I1]
    Y1=Y1[I1]
    data2=json.load(open(in_file2))
    data2=np.array(data2)
    X2=data2[:,1]
    Y2=data2[:,2]
    I2=X2>=data_start
    X2=X2[I2]
    Y2=Y2[I2]
    data3=json.load(open(in_file3))
    data3=np.array(data3)
    X3=data3[:,1]
    Y3=data3[:,2]
    I3=X3>=data_start
    X3=X3[I3]
    Y3=Y3[I3]

    fig=plt.figure()
    ax=plt.gca()
    ax.plot(X1,Y1,label='w cvx')
    ax.plot(X2,Y2,label='w/o cvx')
    ax.plot(X3,Y3,label='avg')
    ax.legend()
    ax.set_title("train_loss")
    ax.set_ylim((0,2e-3))
    fig.savefig(out_file)

def plot_0528():
    out_file='figs/0514/test_loss.png'
    avg_loss=2.650227E-02
    cvx_loss=2.553819E-02
    vt_loss=2.773340E-02
    fig=plt.figure()
    ax=plt.gca()
    ax.bar(['w/o cvx','w cvx','avg'],[vt_loss,cvx_loss,avg_loss])
    ax.set_title("test_loss")
    ax.set_ylim((0,3e-2))
    fig.savefig(out_file)



if __name__=='__main__':
    # plot_overfit()
    # plot_lr()
    # plot_bar()
    # plot_err()
    # plot_pd()
    # plot_cvx()
    # plot_0430()
    # plot_0508()
    # plot_0514()
    plot_0528()