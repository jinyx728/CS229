######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import os
from os.path import join,isfile,isdir
from obj_io import Obj,read_obj
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from ply_io import write_ply

def avg_vt_loss(pd,gt):
    return np.mean(norm(gt-pd,axis=1))

def compute_loss_stats(samples_dir,gt_pattern,pd_pattern,loss_fn):
    sample_dirs=os.listdir(samples_dir)
    loss_list=[]
    for sample_dir in sample_dirs:
        sample_id=int(sample_dir)
        gt_path=join(samples_dir,sample_dir,gt_pattern.format(sample_id))
        gt_v=read_obj(gt_path).v
        pd_path=join(samples_dir,sample_dir,pd_pattern.format(sample_id))
        # print('pd_path',pd_path)
        pd_v=read_obj(pd_path).v
        loss_list.append(loss_fn(pd_v,gt_v))
    loss_list=np.array(loss_list)
    print('mean:',np.mean(loss_list),'max',np.max(loss_list),'min',np.min(loss_list),'med',np.median(loss_list),'#',len(loss_list))

def write_diff_ply(samples_dir,gt_pattern,pd_pattern,out_pattern,max_diff=0.04):
    sample_dirs=os.listdir(samples_dir)
    for sample_dir in sample_dirs:
        sample_id=int(sample_dir)
        gt_path=join(samples_dir,sample_dir,gt_pattern.format(sample_id))
        gt_obj=read_obj(gt_path)
        gt_v=gt_obj.v
        pd_path=join(samples_dir,sample_dir,pd_pattern.format(sample_id))
        pd_v=read_obj(pd_path).v
        d=norm(pd_v-gt_v,axis=1)
        t=np.clip(d/max_diff,0,1).reshape((-1,1))
        color0=np.array([[1,1,1]])
        color1=np.array([[1,0,0]])
        colors=color0*(1-t)+color1*t
        colors=np.uint8(colors*255)
        out_path=join(samples_dir,sample_dir,out_pattern.format(sample_id))
        print('write to',out_path)
        write_ply(out_path,pd_v,gt_obj.f,colors)
        # break

def plot_loss():
    plot_dir='figs/0812'
    # fig_path=join(plot_dir,'test_loss.png')
    fig_path=join(plot_dir,'train_loss.png')
    fig=plt.gcf()
    ax=plt.gca()
    bars=["Jenny's","Jenny's+vt","Jenny's+vt+physics",'pre_project','post_project']
    # ax.bar(bars,[0.012593081972769258,0.004671841745178377,0.008857540784726748,0.004955258924425789,0.004505562495241727])
    ax.bar(bars,[0.012738917874209147,0.003539662819566783,0.008377284988852757,0.00392188471588926,0.003307121408179632])
    plt.xticks(range(len(bars)), bars, rotation=15)
    ax.set_ylim([0,0.013])
    # ax.set_title('Test Loss')
    ax.set_title('Train Loss (1000)')
    plt.savefig(fig_path)

def plot_loss2():
    plot_dir='figs/0828'
    fig_path=join(plot_dir,'avg_max_loss.png')
    fig=plt.gcf()
    ax=plt.gca()
    bars=["[30]","untrained\npostprocess",'trained\npostprocess']
    y1=[0.0047675504487428655,0.00444710089300181,0.004510685677361758]
    y2=[0.012339322888638992,0.012284045374463304,0.010929053726769413]
    index=np.arange(len(bars))
    bar_width=0.35
    ax.bar(index+0.15,y2,bar_width,color='r',label='max')
    ax.bar(index+bar_width+0.15,y1,bar_width,color='b',label='average')
    plt.xticks(index+bar_width, bars)
    ax.set_ylim([0,0.013])
    ax.set_title('Loss')
    ax.legend()
    fig.tight_layout()
    plt.savefig(fig_path)
    
def compare_loss(sample_ids,gt_pattern,pd_pattern1,pd_pattern2,loss_fn=avg_vt_loss):
    loss_list=[]
    for sample_id in sample_ids:
        gt_v=read_obj(gt_pattern.format(sample_id)).v
        pd1_v=read_obj(pd_pattern1.format(sample_id)).v
        loss_1=loss_fn(pd1_v,gt_v)
        pd2_v=read_obj(pd_pattern2.format(sample_id)).v
        loss_2=loss_fn(pd2_v,gt_v)
        loss_list.append((loss_2/loss_1,loss_2,loss_1,sample_id))
    loss_list.sort(key=lambda t:t[0])
    for t in loss_list:
        print(t)


if __name__=='__main__':
    # mean: 0.004889731838952239 max 0.023030382565417774 min 0.0012008278636293176 med 0.004289960923671694 # 996
    # mean: 0.004796929691159972 max 0.022564514720320138 min 0.0009257828147118647 med 0.004178054215691429 # 996
    # compute_loss_stats('../../rundir/lowres_ecos_patch/uvn_1e-2/eval_test_cp1000/waist_front_hip_front_hip_right_waist_right','gt_{:08d}.obj','cr_{:08d}.obj',avg_vt_loss)
    # compute_loss_stats('../../rundir/lowres_ecos_patch/uvn_1e-2_noavg/eval_test_cp1000/waist_front_hip_front_hip_right_waist_right','gt_{:08d}.obj','cr_{:08d}.obj',avg_vt_loss)
    # mean: 0.004852270513510341 max 0.02378903721275101 min 0.0011532454330402354 med 0.004242863886207595 # 996
    # mean: 0.004799745673971046 max 0.022542217448052446 min 0.0010713731570782565 med 0.004180446273225437 # 996
    # compute_loss_stats('../../rundir/lowres_ecos_patch/uvn_1e-2/eval_test_train_best/waist_front_hip_front_hip_right_waist_right','gt_{:08d}.obj','cr_{:08d}.obj',avg_vt_loss)
    # compute_loss_stats('../../rundir/lowres_ecos_patch/uvn_1e-2_noavg/eval_test_train_best/waist_front_hip_front_hip_right_waist_right','gt_{:08d}.obj','cr_{:08d}.obj',avg_vt_loss)

    # mean: 0.004817502038841633 max 0.022958856958761528 min 0.0012304685007010108 med 0.004269609130035205 # 996
    # mean: 0.004789351554872722 max 0.021127078898185984 min 0.0010391442808711036 med 0.004184190344671026 # 996
    # compute_loss_stats('../../rundir/lowres_ecos_patch/uvn_1e-2/eval_test/waist_front_hip_front_hip_right_waist_right','gt_{:08d}.obj','cr_{:08d}.obj',avg_vt_loss)
    # compute_loss_stats('../../rundir/lowres_ecos_patch/uvn_1e-2_noavg/eval_test/waist_front_hip_front_hip_right_waist_right','gt_{:08d}.obj','cr_{:08d}.obj',avg_vt_loss)
    # mean: 0.004908780425397083 max 0.024085367372955017 min 0.0013295314949064618 med 0.004389629327482721 # 996
    # mean: 0.004849724535411348 max 0.022683868119522676 min 0.0009430241143857745 med 0.004275311986428977 # 996
    # compute_loss_stats('../../rundir/lowres_ecos_patch/uvn_1e-2/eval_test_cp500/waist_front_hip_front_hip_right_waist_right','gt_{:08d}.obj','cr_{:08d}.obj',avg_vt_loss)
    # compute_loss_stats('../../rundir/lowres_ecos_patch/uvn_1e-2_noavg/eval_test_cp500/waist_front_hip_front_hip_right_waist_right','gt_{:08d}.obj','cr_{:08d}.obj',avg_vt_loss)

    # plot_loss2()
    # mean: 0.0009720427281668103 max 0.0018763627838264184 min 0.0005672437227401924 med 0.0009359505547443116 # 1000
    # mean: 0.0009117980365106131 max 0.0017888505569503416 min 0.0004618540751990755 med 0.0008869196817961203 # 1000
    # compute_loss_stats('../../rundir/lowres_vt_patch/uvn_1e-2/eval_train/waist_front_hip_front_hip_right_waist_right','gt_{:08d}.obj','pd_{:08d}.obj',avg_vt_loss)
    # compute_loss_stats('../../rundir/lowres_vt_patch/uvn_1e-2/eval_train/waist_front_hip_front_hip_right_waist_right','gt_{:08d}.obj','cr_ineq_{:08d}.obj',avg_vt_loss)

    # test_ids=np.loadtxt('/data/zhenglin/poses_v3/sample_lists/lowres_test_samples.txt').astype(int)
    # compare_loss(test_ids,'../../rundir/lowres_ecos/uvn_1e-2/eval_test/{0:08d}/gt_cloth_{0:08d}.obj','../../rundir/lowres_ecos/uvn_1e-2/eval_test/{0:08d}/cr_cloth_{0:08d}.obj','../../rundir/lowres_vt/uvn_1e-2/eval_test/{0:08d}/cr_ineq_cloth_{0:08d}.obj')
    # write_diff_ply('../../rundir/lowres_vt/uvn_1e-2/eval_test','gt_cloth_{:08d}.obj','pd_cloth_{:08d}.obj','diff_pd_{:08d}.ply')
    # write_diff_ply('../../rundir/lowres_vt/uvn_1e-2/eval_test','gt_cloth_{:08d}.obj','cr_ineq_cloth_{:08d}.obj','diff_cr_ineq_{:08d}.ply')
    # write_diff_ply('../../rundir/lowres_ecos/uvn_1e-2/eval_test','gt_cloth_{:08d}.obj','cr_cloth_{:08d}.obj','diff_cr_{:08d}.ply')

    # compute_loss_stats('../../rundir/lowres_ecos/uvn_1e-2/eval_test','gt_cloth_{:08d}.obj','avg_cloth_{:08d}.obj',avg_vt_loss)
    # mean: 0.0049607489085779655 max 0.011179582955311894 min 0.0026632602974575375 med 0.004784484131825566 # 996
    # mean: 0.004510685677361758 max 0.010929053726769413 min 0.002251292198741549 med 0.0043566290729017305 # 996
    # mean: 0.0047675504487428655 max 0.012339322888638992 min 0.002258904649895548 med 0.004586880489024075 # 996
    # mean: 0.00444710089300181 max 0.012284045374463304 min 0.001978559399353082 med 0.0042800224937731635 # 996
    # compute_loss_stats('../../rundir/lowres_ecos/uvn_1e-2/eval_test','gt_cloth_{:08d}.obj','pd_cloth_{:08d}.obj',avg_vt_loss)
    # compute_loss_stats('../../rundir/lowres_ecos/uvn_1e-2/eval_test','gt_cloth_{:08d}.obj','cr_cloth_{:08d}.obj',avg_vt_loss)
    # compute_loss_stats('../../rundir/lowres_vt/uvn_1e-2/eval_test','gt_cloth_{:08d}.obj','pd_cloth_{:08d}.obj',avg_vt_loss)
    # compute_loss_stats('../../rundir/lowres_vt/uvn_1e-2/eval_test','gt_cloth_{:08d}.obj','cr_ineq_cloth_{:08d}.obj',avg_vt_loss)

    # compute_loss_stats('../../rundir/midres_vt_patch/t0/eval_train/waist_front_hip_front_hip_right_waist_right','gt_{:08d}.obj','cr_ineq_{:08d}.obj',avg_vt_loss)
    # compute_loss_stats('../../rundir/midres_vt_patch/t0/eval_train/waist_front_hip_front_hip_right_waist_right','gt_{:08d}.obj','pd_{:08d}.obj',avg_vt_loss)
    # compute_loss_stats('../../rundir/midres_ecos_patch/t0/eval_train/waist_front_hip_front_hip_right_waist_right','gt_{:08d}.obj','cr_{:08d}.obj',avg_vt_loss)
    # compute_loss_stats('../../rundir/midres_ecos_patch/t0/eval_train/waist_front_hip_front_hip_right_waist_right','gt_{:08d}.obj','pd_{:08d}.obj',avg_vt_loss)

    # mean: 0.005052757843821019 max 0.011054679385397885 min 0.0024520525707736997 med 0.00486488502916778 # 996
    # mean: 0.004926643201374547 max 0.010921328636414753 min 0.002575983179540266 med 0.004721641969560057 # 996
    # compute_loss_stats('../../rundir/lowres_ecos/uvn/eval_test','gt_cloth_{:08d}.obj','pd_cloth_{:08d}.obj',avg_vt_loss)
    # compute_loss_stats('../../rundir/lowres_ecos/uvn/eval_test','gt_cloth_{:08d}.obj','cr_cloth_{:08d}.obj',avg_vt_loss)
    # mean: 0.004518036151209187 max 0.011472803064493137 min 0.002159028028890106 med 0.0043353605894128275 # 996
    # compute_loss_stats('../../rundir/lowres_vt/uvn/eval_test','gt_cloth_{:08d}.obj','cr_ineq_cloth_{:08d}.obj',avg_vt_loss)


    # mean: 0.012593081972769258 max 0.036256628519002645 min 0.0066463475719026185 med 0.01194800280269745 # 996
    # compute_loss_stats('../../rundir/lowres/uvn/eval_test','gt_cloth_{:08d}.obj','pd_cloth_{:08d}.obj',avg_vt_loss)
    # mean: 0.004955258924425789 max 0.01126658577176017 min 0.0027297560200611973 med 0.004758586334414015 # 996
    # compute_loss_stats('../../rundir/lowres_cudaqs/uvn/eval_test','gt_cloth_{:08d}.obj','pd_cloth_{:08d}.obj',avg_vt_loss)
    # mean: 0.004505562495241727 max 0.011180528375393881 min 0.002340096334531933 med 0.004328286895601959 # 996
    # compute_loss_stats('../../rundir/lowres_cudaqs/uvn/eval_test','gt_cloth_{:08d}.obj','cr_cloth_{:08d}.obj',avg_vt_loss)
    # plot_loss()

    # mean: 0.012738917874209147 max 0.03952320122217042 min 0.006759048609807598 med 0.012062080560084537 # 1000
    # mean: 0.00392188471588926 max 0.008321928721681466 min 0.002700981004594125 med 0.00386624729453582 # 1000
    # mean: 0.003307121408179632 max 0.0078381370995324 min 0.002261658137247294 med 0.0032515564684173683 # 1000
    # compute_loss_stats('../../rundir/lowres/uvn/eval_train','gt_cloth_{:08d}.obj','pd_cloth_{:08d}.obj',avg_vt_loss)
    # compute_loss_stats('../../rundir/lowres_cudaqs/uvn/eval_train','gt_cloth_{:08d}.obj','pd_cloth_{:08d}.obj',avg_vt_loss)
    # compute_loss_stats('../../rundir/lowres_cudaqs/uvn/eval_train','gt_cloth_{:08d}.obj','cr_cloth_{:08d}.obj',avg_vt_loss)
    # plot_loss()
    # write_diff_ply('../../rundir/lowres/uvn/eval_test','gt_cloth_{:08d}.obj','pd_cloth_{:08d}.obj','diff_{:08d}.ply')
    # write_diff_ply('../../rundir/lowres_cudaqs/uvn/eval_test','gt_cloth_{:08d}.obj','pd_cloth_{:08d}.obj','diff_pd_{:08d}.ply')
    # write_diff_ply('../../rundir/lowres_cudaqs/uvn/eval_test','gt_cloth_{:08d}.obj','cr_cloth_{:08d}.obj','diff_cr_{:08d}.ply')

    # pd
    # mean: 0.004671841745178377 max 0.011489779255354118 min 0.0022785963041176255 med 0.004497664740176618 # 996
    # mean: 0.003539662819566783 max 0.006877333024388877 min 0.002364348241751012 med 0.0035122080310037848 # 1000
    # cr
    # mean: 0.008857540784726748 max 0.018886309309745775 min 0.004504520795747197 med 0.00838045449673857 # 996
    # mean: 0.008377284988852757 max 0.017792459068028857 min 0.0043180574922201415 med 0.007909824435561121 # 1000
    # compute_loss_stats('../../rundir/lowres_vt/uvn/eval_test','gt_cloth_{:08d}.obj','cr_cloth_{:08d}.obj',avg_vt_loss)
    # compute_loss_stats('../../rundir/lowres_vt/uvn/eval_train','gt_cloth_{:08d}.obj','cr_cloth_{:08d}.obj',avg_vt_loss)
    # plot_loss()

    # write_diff_ply('../../rundir/lowres_vt/uvn/eval_test','gt_cloth_{:08d}.obj','pd_cloth_{:08d}.obj','diff_pd_{:08d}.ply')
    # write_diff_ply('../../rundir/lowres_vt/uvn/eval_test','gt_cloth_{:08d}.obj','cr_cloth_{:08d}.obj','diff_cr_{:08d}.ply')

    # mean: 0.004547267131454219 max 0.011032674458990492 min 0.0020775603254484867 med 0.004351729140948805 # 996
    # mean: 0.006935451627509349 max 0.012405780587740918 min 0.004672662528648324 med 0.006752061232398425 # 996
    # compute_loss_stats('../../rundir/lowres_cudaqs/uvn_noavg/eval_test','gt_cloth_{:08d}.obj','cr_cloth_{:08d}.obj',avg_vt_loss)
    # compute_loss_stats('../../rundir/lowres_cudaqs/uvn_noavg/eval_test','gt_cloth_{:08d}.obj','pd_cloth_{:08d}.obj',avg_vt_loss)
    # compute_loss_stats('../../rundir/lowres_cudaqs/uvn_noavg/eval_train','gt_cloth_{:08d}.obj','cr_cloth_{:08d}.obj',avg_vt_loss)
    # plot_loss()
