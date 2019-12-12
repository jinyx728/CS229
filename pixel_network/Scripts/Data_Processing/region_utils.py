######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import os
import numpy as np
from render_offsets import get_vts_normalize_m,normalize_vts
from obj_io import Obj,read_obj,write_obj
from offset_img_utils import get_pix_pos
from PIL import Image
class RegionManager:
    def __init__(self,shared_data_dir='../../shared_data'):
        # get vts
        # shared_data_dir='../../shared_data'
        obj_path=os.path.join(shared_data_dir,'flat_tshirt.obj')
        obj=read_obj(obj_path)
        vts=obj.v
        m=get_vts_normalize_m(vts)
        vts=normalize_vts(vts[:,:2],m)
        self.vts=vts
        self.fcs=obj.f

        # get_region_names
        self.region_path=os.path.join(shared_data_dir,'regions')
        region_names_path=os.path.join(self.region_path,'region_names.txt')
        self.region_names=self.load_region_names(region_names_path)
        self.n_regions=len(self.region_names)
        print('num_regions',len(self.region_names))
        self.img_size=512
        self.crop_size=512*5//16

        region_crop_dir=os.path.join(self.region_path,'region_crops')
        if not os.path.isdir(region_crop_dir):
            os.mkdir(region_crop_dir)
        self.region_crop_dir=region_crop_dir

        self.mask_margin_size=16
        mask_path=os.path.join(shared_data_dir,'offset_img_mask.npy')
        mask=np.load(mask_path)
        self.front_mask=mask[:,:,0]
        self.back_mask=mask[:,:,1]


    def load_region_names(self,path):
        region_names=[]
        with open(path) as f:
            while True:
                line=f.readline()
                if line=='':
                    break
                region_names.append(line.rstrip())
        return region_names

    def get_region_crop(self,region_id):
        print('region_name',self.region_names[region_id])
        # load region_vts_id
        region_name=self.region_names[region_id]
        region_bdry_vt_ids=np.loadtxt(os.path.join(self.region_path,'{}_boundary_vertices.txt'.format(region_name))).astype(np.int32)-1
        region_intr_vt_ids=np.loadtxt(os.path.join(self.region_path,'{}_interior_vertices.txt'.format(region_name))).astype(np.int32)-1
        region_vt_ids=np.hstack([region_bdry_vt_ids,region_intr_vt_ids])
        region_vts=self.vts[region_vt_ids]
        max_xy=np.max(region_vts,axis=0)
        min_xy=np.min(region_vts,axis=0)
        size=max_xy-min_xy
        # fit=size[0]<5/8 and size[1]<5/8
        # print('fit:',fit,'x:',size[0],'y:',size[1])
        # has_front=region_name.find('front')!=-1
        # has_back=region_name.find('back')!=-1
        # unique_side=(has_front and not has_back) or (has_back and not has_front)
        # get min max
        # check if larger than 0.625
        # get front and back
        side='front' if region_name.find('front')!=-1 else 'back'
        xcenter=(max_xy[0]+min_xy[0])/2
        ycenter=(max_xy[1]+min_xy[1])/2
        pix_pos=get_pix_pos(np.array([xcenter,ycenter]),W=self.img_size,H=self.img_size)
        x=int(pix_pos[0]-self.crop_size//2)
        y=int(pix_pos[1]-self.crop_size//2)

        return (x,y,self.crop_size,self.crop_size,side)

    def save_region_crop(self,region_id,crop):
        x,y,w,h,side=crop
        region_name=self.region_names[region_id]
        crop_path=os.path.join(self.region_crop_dir,'{}_crop.txt'.format(region_name))
        with open(crop_path,'w') as f:
            f.write('{} {} {} {} {}'.format(x,y,w,h,side))

    def load_region_crop(self,region_path):
        with open(region_path) as f:
            line=f.readline()
        parts=line.split()
        x,y,w,h,side=int(parts[0]),int(parts[1]),int(parts[2]),int(parts[3]),parts[4]
        return (x,y,w,h,side)

    def load_region_vt_ids(self,region_id):
        region_name=self.region_names[region_id]
        region_bdry_vt_ids=np.loadtxt(os.path.join(self.region_path,'{}_boundary_vertices.txt'.format(region_name))).astype(np.int32)-1
        region_intr_vt_ids=np.loadtxt(os.path.join(self.region_path,'{}_interior_vertices.txt'.format(region_name))).astype(np.int32)-1
        region_vt_ids=np.hstack([region_bdry_vt_ids,region_intr_vt_ids])
        return region_vt_ids

    def get_region_mask(self,region_id,crop):
        x,y,w,h,side=crop
        region_vt_ids=self.load_region_vt_ids(region_id)
        region_vts=self.vts[region_vt_ids]
        max_xy=np.max(region_vts,axis=0)
        min_xy=np.min(region_vts,axis=0)

        max_pix_pos=get_pix_pos(max_xy,W=self.img_size,H=self.img_size).astype(int)
        min_pix_pos=get_pix_pos(min_xy,W=self.img_size,H=self.img_size).astype(int)
        max_pix_pos+=self.mask_margin_size
        min_pix_pos-=self.mask_margin_size

        min_pix_pos[min_pix_pos<0]=0
        max_pix_pos[max_pix_pos>self.img_size]=self.img_size
        
        if side=='front':
            mask=self.front_mask
        else:
            mask=self.back_mask

        local_mask=np.zeros(mask.shape)
        local_mask[min_pix_pos[1]:max_pix_pos[1],min_pix_pos[0]:max_pix_pos[0]]=1
        mask=mask*local_mask

        return mask[y:y+h,x:x+w]

    def save_region_mask(self,region_id,mask,save_img=False):
        region_name=self.region_names[region_id]
        mask_path=os.path.join(self.region_crop_dir,'{}_mask.npy'.format(region_name))
        np.save(mask_path,mask)
        if save_img:
            mask_img_path=os.path.join(self.region_crop_dir,'{}_mask.png'.format(region_name))
            Image.fromarray(np.uint8(mask*255)).save(mask_img_path)

    def get_region_crop_path(self,test_id):
        region_name=self.region_names[test_id]
        return os.path.join(self.region_crop_dir,'{}_crop.txt'.format(region_name))

    def get_region_mask_path(self,test_id):
        region_name=self.region_names[test_id]
        return os.path.join(self.region_crop_dir,'{}_mask.npy'.format(region_name))

    # offset managing:
    def init_for_offset_manage(self):
        self.all_region_vt_ids=self.load_all_region_vt_ids()
        self.all_region_intr_fcs=self.load_all_region_intr_fcs()
        self.skin_dir='/data/zhenglin/poses_v3/lowres_skin_npys'

    def load_all_region_vt_ids(self):
        all_region_vt_ids=[]
        for region_id in range(self.n_regions):
            all_region_vt_ids.append(self.load_region_vt_ids(region_id))
        return all_region_vt_ids

    def get_intr_fcs(self,vt_ids,all_fcs):
        fcs=[]
        vt_id_set=set(vt_ids)
        for fc in all_fcs:
            if fc[0] in vt_id_set and fc[1] in vt_id_set and fc[2] in vt_id_set:
                fcs.append(fc)
        return np.array(fcs,dtype=int)

    def load_all_region_intr_fcs(self):
        all_region_intr_fcs=[]
        for region_id in range(self.n_regions):
            region_vt_ids=self.all_region_vt_ids[region_id]
            all_region_intr_fcs.append(self.get_intr_fcs(region_vt_ids,self.fcs))
        return all_region_intr_fcs

    def get_region_local_fcs(self,region_id):
        vt_id_map={}
        region_vt_ids=self.all_region_vt_ids[region_id]
        n_vts=len(region_vt_ids)
        for i in range(n_vts):
            vt_id=region_vt_ids[i]
            vt_id_map[vt_id]=i
        region_intr_fcs=self.all_region_intr_fcs[region_id]
        region_local_fcs=[]
        for fc in region_intr_fcs:
            region_local_fcs.append([vt_id_map[fc[0]],vt_id_map[fc[1]],vt_id_map[fc[2]]])
        return np.array(region_local_fcs,dtype=np.int32)

    def fcs_to_edges(self,fcs):
        n_vts=np.max(fcs)+1
        nbs=[set() for i in range(n_vts)]
        for fc in fcs:
            nbs[fc[0]].add(fc[1])
            nbs[fc[0]].add(fc[2])
            nbs[fc[1]].add(fc[0])
            nbs[fc[1]].add(fc[2])
            nbs[fc[2]].add(fc[0])
            nbs[fc[2]].add(fc[1])

        edges=[]
        for i in range(n_vts):
            nb=nbs[i]
            for n in nb:
                if n<i:
                    continue
                edges.append([i,n])
        return np.array(edges).astype(np.int32)

    def get_region_local_edges(self,region_id):
        local_fcs=self.get_region_local_fcs(region_id)
        return self.fcs_to_edges(local_fcs)

    def save_region_pieces_obj(self,samples_dir,sample_id):
        skin_path=os.path.join(self.skin_dir,'skin_{:08d}.npy'.format(sample_id))
        skin=np.load(skin_path)
        n_vts=len(skin)
        sample_dir=os.path.join(samples_dir,'{:08d}'.format(sample_id))
        vts=[]
        fcs=[]
        for region_id in range(self.n_regions):
            region_name=self.region_names[region_id]
            region_offset_path=os.path.join(sample_dir,'{}_offset.npy'.format(region_name))
            region_offset=np.load(region_offset_path).reshape((-1,3))

            region_vt_ids=self.all_region_vt_ids[region_id]
            region_fcs=self.all_region_intr_fcs[region_id]
            region_vts=skin.copy()
            region_vts[region_vt_ids]+=region_offset
            vts.append(region_vts)
            fcs.append(region_fcs+region_id*n_vts)
            # break

        vts=np.concatenate(vts,axis=0)
        fcs=np.concatenate(fcs,axis=0)

        obj_path=os.path.join(sample_dir,'region_pieces.obj')
        obj=Obj(v=vts,f=fcs)
        write_obj(obj,obj_path)

    def save_stitched_regions_obj(self,samples_dir,sample_id):
        skin_path=os.path.join(self.skin_dir,'skin_{:08d}.npy'.format(sample_id))
        skin=np.load(skin_path)
        n_vts=len(skin)
        sample_dir=os.path.join(samples_dir,'{:08d}'.format(sample_id))
        vt_offsets=[[] for i in range(n_vts)]
        for region_id in range(self.n_regions):
            region_name=self.region_names[region_id]
            region_offset_path=os.path.join(sample_dir,'{}_offset.npy'.format(region_name))
            region_offset=np.load(region_offset_path).reshape((-1,3))

            region_vt_ids=self.all_region_vt_ids[region_id]
            for i in range(len(region_offset)):
                vt_id=region_vt_ids[i]
                vt_offsets[vt_id].append(region_offset[i])

        for i in range(n_vts):
            vt_offsets[i]=np.mean(np.array(vt_offsets[i]),axis=0)

        vt_offsets=np.array(vt_offsets)
        obj_path=os.path.join(sample_dir,'stitched_regions.obj')
        obj=Obj(v=skin+vt_offsets,f=self.fcs)
        write_obj(obj,obj_path)

        gt_vt_offsets=np.load('/data/zhenglin/poses_v3/lowres_offset_npys/offset_{:08d}.npy'.format(sample_id))
        print('error',self.get_error(vt_offsets-gt_vt_offsets))

    def get_error(self,diff):
        return np.sqrt(np.sum(diff**2)/len(diff))

    def save_region_piece_obj(self,samples_dir,sample_id,region_id):
        skin_path=os.path.join(self.skin_dir,'skin_{:08d}.npy'.format(sample_id))
        skin=np.load(skin_path)
        n_vts=len(skin)
        region_name=self.region_names[region_id]
        sample_dir=os.path.join(samples_dir,'{:08d}'.format(sample_id))
        region_offset_path=os.path.join(sample_dir,'{}_offset.npy'.format(region_name))
        region_offset=np.load(region_offset_path).reshape((-1,3))
        region_vt_ids=self.all_region_vt_ids[region_id]
        region_vts=skin[region_vt_ids]+region_offset
        region_intr_fcs=self.get_region_local_fcs(region_id)
        region_obj_path=os.path.join(sample_dir,'{}.obj'.format(region_name))
        region_obj=Obj(v=region_vts,f=region_intr_fcs)
        write_obj(region_obj,region_obj_path)

    def save_region_from_whole_obj(self,whole_obj_path,out_dir,region_id,prefix=''):
        whole_obj=read_obj(whole_obj_path)
        region_obj=Obj(v=whole_obj.v,f=self.all_region_intr_fcs[region_id])
        region_name=self.region_names[region_id]
        region_obj_path=os.path.join(out_dir,'{}{}.obj'.format(prefix,region_name))
        write_obj(region_obj,region_obj_path)


class PCManager:
    def __init__(self):
        shared_data_dir='../../shared_data'
        data_root_dir='/data/zhenglin/poses_v3'
        pc_dir=os.path.join(data_root_dir,'lowres_pcs')
        self.skin_dir=os.path.join(data_root_dir,'lowres_skin_npys')
        self.mean=np.load(os.path.join(pc_dir,'mean.npy'))
        self.pcs=np.load(os.path.join(pc_dir,'pcs.npy'))

    def project_pc(self,offsets,n_pcs=2048):
        offsets=offsets.reshape(-1)
        offsets-=self.mean
        pcs_mat=self.pcs[:,:n_pcs]
        coefs=offsets.reshape((1,-1)).dot(self.pcs[:,:n_pcs]).reshape(-1)
        proj_offsets=pcs_mat.dot(coefs)
        proj_offsets+=self.mean
        return proj_offsets.reshape(-1,3)

    def save_projected_obj(self,in_path,out_dir,test_id,n_pcs=2048):
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        in_obj=read_obj(in_path)
        skin=np.load(os.path.join(self.skin_dir,'skin_{:08d}.npy'.format(test_id)))
        offsets=in_obj.v-skin
        proj_offsets=self.project_pc(offsets,n_pcs=n_pcs)
        out_obj=Obj(v=skin+proj_offsets,f=in_obj.f)
        out_path=os.path.join(out_dir,'proj_{}.obj'.format(n_pcs))
        write_obj(out_obj,out_path)


if __name__=='__main__':
    # region_manager=RegionManager()
    # for i in range(region_manager.n_regions):
    #     crop=region_manager.get_region_crop(i)
    #     print(crop)
    #     region_manager.save_region_crop(i,crop)
    #     mask=region_manager.get_region_mask(i,crop)
    #     region_manager.save_region_mask(i,mask,save_img=True)

    region_manager=RegionManager()
    region_manager.init_for_offset_manage()
    pc_manager=PCManager()
    samples_dir='../../rundir/lowres_normal_regions/eval_test_full_conv_lap'
    dirs=os.listdir(samples_dir)
    for d in dirs:
        test_id=int(d)
        print('test_id',test_id)
        region_manager.save_region_pieces_obj(samples_dir,test_id)
        region_manager.save_stitched_regions_obj(samples_dir,test_id)
        sample_dir=os.path.join(samples_dir,'{:08d}'.format(test_id))
        pc_manager.save_projected_obj(os.path.join(sample_dir,'stitched_regions.obj'),sample_dir,test_id=test_id,n_pcs=2048)

    # sample_id=414
    # region_manager.save_region_piece_obj(samples_dir,sample_id,0)
    # out_dir='/data/zhenglin/PhysBAM/Private_Projects/cloth_on_virtual_body/Learning/rundir/conv_l1/trial/eval_test/00000414'
    # whole_obj_path='/data/zhenglin/PhysBAM/Private_Projects/cloth_on_virtual_body/Learning/rundir/conv_l1/trial/eval_test/00000414/cloth_00000414.obj'
    # region_manager.save_region_from_whole_obj(whole_obj_path,out_dir,0)

    # region_manager.save_region_pieces_obj('../../rundir/lowres_regions/eval_train',658)
    # region_manager.save_stitched_regions_obj('../../rundir/lowres_regions/eval_train',658)

    # region_manager.save_region_piece_obj('../../rundir/lowres_normal_regions/eval_test_l2',5205,12)
    # region_manager.save_region_from_whole_obj('../../rundir/lowres_normal_regions/eval_test_l2/00005205/proj_2048.obj','../../rundir/lowres_normal_regions/eval_test_l2/00005205/',12)
    # region_manager.save_region_from_whole_obj('../../rundir/lowres_normal/1024/eval_test/00005205/cloth_00005205.obj','../../rundir/lowres_normal/1024/eval_test/00005205/',12)
    # region_manager.save_region_from_whole_obj('../../rundir/lowres_normal/1024/eval_test/00005205/gt_cloth_00005205.obj','../../rundir/lowres_normal/1024/eval_test/00005205/',12,prefix='gt_')

    # pc_manager=PCManager()
    # for n_pcs in [128,256,512,1024,2048,4096,8192]:
    #     print('n_pcs',n_pcs)
    #     pc_manager.save_projected_obj('/data/zhenglin/PhysBAM/Private_Projects/cloth_on_virtual_body/Learning/rundir/lowres_normal_regions/eval_test_full_conv_init_10/00005205/stitched_regions.obj','/data/zhenglin/PhysBAM/Private_Projects/cloth_on_virtual_body/Learning/rundir/eval_test_full_conv_init_10/00005205/proj/',test_id=5205,n_pcs=n_pcs)
