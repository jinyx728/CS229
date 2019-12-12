import os
from os.path import join,isfile,isdir

def generate_network_disp(trial_name,sample_list_file,out_dir,mode='val'):
    os.chdir('/data/zhenglin/PhysBAM/Private_Projects/cloth_texture/pixel_network/Scripts/Train_Conv_Mix')
    cmd='python generate_test_args.py -test_case highres_tex -data_root_dir /data/yxjin/poses_v3 -cp ../../rundir/highres_tex/{0}/saved_models/{3}_model_best.pth.tar -trial {0}  -eval_out_dir {1} -pd_only -sample_list_file {2}'.format(trial_name,out_dir,sample_list_file,mode)
    print(cmd)
    os.system(cmd)

if __name__=='__main__':
    # generate_network_disp('l2pix_final_1','/data/zhenglin/poses_v3/sample_lists/lowres_test_samples.txt','/data/zhenglin/PhysBAM/Private_Projects/cloth_texture/pixel_network/rundir/highres_tex/l2pix_final_1/eval_test')
    # generate_network_disp('l2pix_final_2','/data/zhenglin/poses_v3/sample_lists/lowres_test_samples.txt','/data/zhenglin/PhysBAM/Private_Projects/cloth_texture/pixel_network/rundir/highres_tex/l2pix_final_2/eval_test')
    # generate_network_disp('l2pix_final_3','/data/zhenglin/poses_v3/sample_lists/lowres_test_samples.txt','/data/zhenglin/PhysBAM/Private_Projects/cloth_texture/pixel_network/rundir/highres_tex/l2pix_final_3/eval_test')
    # generate_network_disp('l2pix_final_4','/data/zhenglin/poses_v3/sample_lists/lowres_test_samples.txt','/data/zhenglin/PhysBAM/Private_Projects/cloth_texture/pixel_network/rundir/highres_tex/l2pix_final_4/eval_test')

    generate_network_disp('l2pix_final_1','/data/zhenglin/PhysBAM/Private_Projects/cloth_texture/data_generation/script/reconstruct_test/train_samples.txt','/data/zhenglin/PhysBAM/Private_Projects/cloth_texture/pixel_network/rundir/highres_tex/l2pix_final_1/eval_train',mode='train')
    generate_network_disp('l2pix_final_2','/data/zhenglin/PhysBAM/Private_Projects/cloth_texture/data_generation/script/reconstruct_test/train_samples.txt','/data/zhenglin/PhysBAM/Private_Projects/cloth_texture/pixel_network/rundir/highres_tex/l2pix_final_2/eval_train',mode='train')
    generate_network_disp('l2pix_final_3','/data/zhenglin/PhysBAM/Private_Projects/cloth_texture/data_generation/script/reconstruct_test/train_samples.txt','/data/zhenglin/PhysBAM/Private_Projects/cloth_texture/pixel_network/rundir/highres_tex/l2pix_final_3/eval_train',mode='train')
    generate_network_disp('l2pix_final_4','/data/zhenglin/PhysBAM/Private_Projects/cloth_texture/data_generation/script/reconstruct_test/train_samples.txt','/data/zhenglin/PhysBAM/Private_Projects/cloth_texture/pixel_network/rundir/highres_tex/l2pix_final_4/eval_train',mode='train')