import os
from os.path import isfile,isdir,join
from obj_io import read_obj,write_obj,Obj
from io_utils import read_vt
import numpy as np

def replace_textures(in_dir,disp_dir,out_dir):
    if not isdir(out_dir):
        os.makedirs(out_dir)
    files=os.listdir(in_dir)
    rest_vt=read_vt('../vt_groundtruth_div.txt')
    for sample_id in range(15000,30000):
        in_obj_path=join(in_dir,'pd_div_{:08d}_tex.obj'.format(sample_id))
        if not isfile(in_obj_path):
            continue
        disp_path=join(disp_dir,'displace_{:08d}.txt'.format(sample_id))
        if not isfile(disp_path):
            continue
        obj=read_obj(in_obj_path)
        disp=np.loadtxt(disp_path)
        out_obj_path=join(out_dir,'pd_{:08d}.obj'.format(sample_id))
        print('write to',out_obj_path)
        write_obj(Obj(v=obj.v,f=obj.f,vt=rest_vt+disp,mat='tshirt.mtl'),out_obj_path)
        # break

if __name__=='__main__':
    data_root_dir='/data/zhenglin/dataset_subdivision'
    replace_textures(join(data_root_dir,'dataset_pd_new_subdiv'),join(data_root_dir,'displacement_div_filled'),join(data_root_dir,'pd'))


