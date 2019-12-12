######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import os
import numpy as np
from obj_io import Obj, read_obj, write_obj
import torch
from torch.nn.functional import grid_sample

def get_pix_pos(vt,W,H):
    pix_pos=(vt+1)/2*np.array([W,H])-0.5
    return pix_pos

class OffsetManager:
    def __init__(self,use_torch=False,ctx=None):
       
        cloth_type=kwargs.get('cloth_type','cloth')
        assert(cloth_type in ['cloth','tie'])
        data_root_dir=kwargs['data_root_dir']
        skin_dir=kwargs['skin_dir']
        if 'only_pred' not in kwargs:
            offset_img_dir=kwargs['offset_img_dir']
            vt_offset_dir=kwargs['vt_offset_dir']
            uvn_dir=kwargs.get('uvn_dir',None)
            uvn_offset_img_dir=kwargs.get('uvn_offset_img_dir',None)
        skin_img_dir=kwargs.get('skin_img_dir',os.path.join(data_root_dir,'lowres_skin_imgs'))

        if cloth_type == 'cloth':
            img_size=kwargs.get('img_size', -1)
            assert(img_size>0)
            shared_data_dir=kwargs['shared_data_dir']
            front_obj_path=os.path.join(shared_data_dir,'flat_tshirt_front.obj')
            front_obj=read_obj(front_obj_path)
            front_vts,front_fcs=front_obj.v,front_obj.f
            front_vts=self.normalize_vts(front_vts)
            front_vts=front_vts.astype(np.float32)
            front_vt_ids=np.loadtxt(os.path.join(shared_data_dir,'front_vertices.txt')).astype(np.int32)
            n_vts=len(front_vts)
            all_vts=front_vts[:,:2].copy()
            front_vts=front_vts[front_vt_ids,:2]

            back_obj_path=os.path.join(shared_data_dir,'flat_tshirt_back.obj')
            back_obj=read_obj(back_obj_path)
            back_vts,back_fcs=back_obj.v,back_obj.f
            back_vts=self.normalize_vts(back_vts)
            back_vts=back_vts.astype(np.float32)
            back_vt_ids=np.loadtxt(os.path.join(shared_data_dir,'back_vertices.txt')).astype(np.int32)
            back_vts=back_vts[back_vt_ids,:2]

            bdry_vt_ids_path=os.path.join(shared_data_dir,'bdry_vertices.txt')
            self.front_vt_ids=front_vt_ids
            self.back_vt_ids=back_vt_ids
            self.shared_data_dir=shared_data_dir
            if 'only_pred' not in kwargs:
                self.vt_offset_dir=vt_offset_dir
            if not os.path.isfile(bdry_vt_ids_path):
                self.save_bdry_ids()
            bdry_vt_ids=np.loadtxt(bdry_vt_ids_path).astype(np.int32)
                
            self.img_size=img_size
            mask_path=os.path.join(shared_data_dir,'offset_img_mask_{}.npy'.format(img_size))
            mask=np.load(mask_path)
            
        elif cloth_type == 'tie':
            shared_data_dir='../../shared_data_necktie'
            tie_obj_path=os.path.join(shared_data_dir,'necktie_uv_flat.obj')
            tie_obj=read_obj(tie_obj_path)
            tie_vts,tie_fcs=tie_obj.v,tie_obj.f
            tie_vts[:,0]=(tie_vts[:,0]-1/2)*4+1/2
            tie_vts=self.normalize_vts(tie_vts).astype(np.float32)
            tie_vts=tie_vts[:,:2]
            n_vts=len(tie_vts)
            mask_path=kwargs['mask_path']
            mask=np.load(mask_path)
        # data_root_dir='offset_test'
        # skin_dir=os.path.join(data_root_dir,'lowres_skin_npys')
        # skin_dir=os.path.join(data_root_dir,skin_dir)
        # offset_img_dir=os.path.join(data_root_dir,offset_img_dir)

        if use_torch:
            if cloth_type == 'cloth':
                all_vts=torch.from_numpy(all_vts)
                front_vts=torch.from_numpy(front_vts)
                front_vt_ids=torch.from_numpy(front_vt_ids).long()
                back_vts=torch.from_numpy(back_vts)
                back_vt_ids=torch.from_numpy(back_vt_ids).long()
                bdry_vt_ids=torch.from_numpy(bdry_vt_ids).long()
            elif cloth_type == 'tie':
                tie_vts=torch.from_numpy(tie_vts)      
        else:
            if cloth_type == 'tie':
                tie_vts=torch.from_numpy(tie_vts)            
        self.device=device    
            
        print('use_torch',use_torch)

        self.shared_data_dir=shared_data_dir
        if cloth_type == 'cloth':
            self.all_vts=all_vts
            self.front_vts=front_vts
            self.front_vt_ids=front_vt_ids
            self.back_vts=back_vts
            self.back_vt_ids=back_vt_ids
            self.bdry_vt_ids=bdry_vt_ids
            if 'only_pred' not in kwargs:
                self.uvn_dir=uvn_dir
                self.uvn_offset_img_dir=kwargs.get('uvn_offset_img_dir',None)
            self.fcs=np.concatenate([front_fcs,back_fcs],axis=0)
            self.n_vts=n_vts
            self.front_mask=mask[:,:,0]
            self.back_mask=mask[:,:,1]
        elif cloth_type == 'tie':
            self.tie_vts=tie_vts
            self.fcs=tie_fcs
            self.n_vts=n_vts
            self.tie_mask=mask            

        self.data_root_dir=data_root_dir
        self.cloth_type = cloth_type
        self.skin_dir=skin_dir
        if 'only_pred' not in kwargs:
            self.offset_img_dir=offset_img_dir
        self.skin_img_dir=skin_img_dir
        self.out_dir='offset_test'

    def normalize_vts(self,vts):
        xyzmin,xyzmax=np.min(vts,axis=0),np.max(vts,axis=0)
        ymin,ymax=xyzmin[1],xyzmax[1]
        ymin-=0.1
        ymax+=0.1
        xcenter=(xyzmin[0]+xyzmax[0])/2
        normalized_vts=vts.copy()
        normalized_vts[:,0]=(vts[:,0]-xcenter)/(ymax-ymin)*2
        normalized_vts[:,1]=(vts[:,1]-ymin)/(ymax-ymin)*2-1
        return normalized_vts

    def get_offsets_torch(self,offset_imgs):
        if self.cloth_type=='cloth':
            assert(self.img_size==offset_imgs.size()[2])
        device=offset_imgs.device
        offset_imgs=offset_imgs.float()
        N=len(offset_imgs)
        H=offset_imgs.size(2)
        offsets=torch.zeros(N,self.n_vts,3).to(device)

        if self.cloth_type == 'cloth':
            # front_vts=self.front_vts.to(device).view(1,1,self.front_vts.size(0),2).repeat(N,1,1,1)
            front_offset_imgs=offset_imgs[:,:3,:,:]
            # front_offsets=grid_sample(front_offset_imgs,front_vts)
            # front_offsets=front_offsets.view(N,3,-1).permute(0,2,1)

            front_offsets=self.get_offsets_from_offset_imgs_torch(front_offset_imgs,self.front_vts)

            # back_vts=self.back_vts.to(device).view(1,1,self.back_vts.size(0),2).repeat(N,1,1,1)
            back_offset_imgs=offset_imgs[:,3:6,:,:]
            # back_offsets=grid_sample(back_offset_imgs,back_vts)
            # back_offsets=back_offsets.view(N,3,-1).permute(0,2,1)

            back_offsets=self.get_offsets_from_offset_imgs_torch(back_offset_imgs,self.back_vts)


            front_vt_ids=self.front_vt_ids.to(device)
            back_vt_ids=self.back_vt_ids.to(device)
            bdry_vt_ids=self.bdry_vt_ids.to(device)
            offsets[:,front_vt_ids,:]+=front_offsets
            offsets[:,back_vt_ids,:]+=back_offsets
            offsets[:,bdry_vt_ids,:]/=2
        elif self.cloth_type == 'tie':
            tie_vts=self.tie_vts.to(device).view(1,1,self.tie_vts.size(0),2).repeat(N,1,1,1)
            tie_offset_imgs=offset_imgs[:,:3,:,:]
            tie_offsets=grid_sample(tie_offset_imgs,tie_vts)
            tie_offsets=tie_offsets.view(N,3,-1).permute(0,2,1)
            offsets[:,:,:]+=tie_offsets
                
        return offsets

    def write_cloth(self,test_id,offset,out_dir=None,prefix='cloth'):
        skin_path=os.path.join(self.skin_dir,'skin_{:08d}.npy'.format(test_id))
        skin=np.load(skin_path)
        cloth=skin+offset
        obj=Obj(v=cloth,f=self.fcs)
        if out_dir is None:
            out_dir=self.out_dir
        out_path=os.path.join(out_dir,'{}_{:08d}.obj'.format(prefix,test_id))
        write_obj(obj,out_path)

    def write_cloth_from_id_torch(self,test_id):
        offset_img_path=os.path.join(self.offset_img_dir,'offset_img_{:08d}.npy'.format(test_id))
        offset_img=np.load(offset_img_path)
        offset_img=torch.from_numpy(offset_img).to(self.device)
        offset_imgs=offset_img.permute((2,0,1)).unsqueeze(0)
        offsets=self.get_offsets_torch(offset_imgs)
        offset=offsets[0].detach().cpu().numpy()

        gt_offset=np.load(os.path.join(self.vt_offset_dir,'offset_{:08d}.npy'.format(test_id)))
        front_vt_ids=self.front_vt_ids.detach().cpu().numpy()
        back_vt_ids=self.back_vt_ids.detach().cpu().numpy()
        front_gt_offset=gt_offset[front_vt_ids]
        front_offset=offset[front_vt_ids]
        back_gt_offset=gt_offset[back_vt_ids]
        back_offset=offset[back_vt_ids]
        print('error',self.get_error(gt_offset-offset),'front',self.get_error(front_gt_offset-front_offset),'back',self.get_error(back_gt_offset-back_offset))

        self.write_cloth(test_id,offset)

    def write_tie(self,test_id,offset,out_dir=None,prefix='tie'):
        skin_path=os.path.join(self.skin_dir,'skin_{:08d}.npy'.format(test_id))
        skin=np.load(skin_path)
        tie=skin+offset
        obj=Obj(v=tie,f=self.fcs)
        if out_dir is None:
            out_dir=self.out_dir
        out_path=os.path.join(out_dir,'{}_{:08d}.obj'.format(prefix,test_id))
        write_obj(obj,out_path)

    def write_tie_from_id_torch(self,test_id):
        offset_img_path=os.path.join(self.offset_img_dir,'offset_img_{:08d}.npy'.format(test_id))
        offset_img=np.load(offset_img_path)
        offset_img=torch.from_numpy(offset_img).to(self.device)
        offset_imgs=offset_img.permute((2,0,1)).unsqueeze(0)
        offsets=self.get_offsets_torch(offset_imgs)
        offset=offsets[0].detach().cpu().numpy()

        tie_gt_offset=np.load('offset_test/offset_{:08d}.npy'.format(test_id))
        offset_error = self.get_error(tie_gt_offset-offset)
        print('error',offset_error)

        self.write_tie(test_id,offset)
        
#         1024: 0.0004042721082389447
#         512: 0.0003930717036202588
#         256: 0.0004917499285979957
#         128: 0.0011131667874338578
        
    def load_uvn_hats(self,ids):
        front_uvnhats=[]
        back_uvnhats=[]
        for id in ids:
            front_uhat=np.load(os.path.join(self.uvn_dir,'front_uhats_{:08d}.npy'.format(id)))
            front_uhat=np.expand_dims(front_uhat,2)
            front_vhat=np.load(os.path.join(self.uvn_dir,'front_vhats_{:08d}.npy'.format(id)))
            front_vhat=np.expand_dims(front_vhat,2)
            front_nhat=np.load(os.path.join(self.uvn_dir,'front_nhats_{:08d}.npy'.format(id)))
            front_nhat=np.expand_dims(front_nhat,2)
            front_uvnhat=np.concatenate([front_uhat,front_vhat,front_nhat],axis=2)
            front_uvnhats.append(front_uvnhat)
            back_uhat=np.load(os.path.join(self.uvn_dir,'back_uhats_{:08d}.npy'.format(id)))
            back_uhat=np.expand_dims(back_uhat,2)
            back_vhat=np.load(os.path.join(self.uvn_dir,'back_vhats_{:08d}.npy'.format(id)))
            back_vhat=np.expand_dims(back_vhat,2)
            back_nhat=np.load(os.path.join(self.uvn_dir,'back_nhats_{:08d}.npy'.format(id)))
            back_nhat=np.expand_dims(back_nhat,2)
            back_uvnhat=np.concatenate([back_uhat,back_vhat,back_nhat],axis=2)
            back_uvnhats.append(back_uvnhat)
        return np.array(front_uvnhats),np.array(back_uvnhats)

    def get_offsets_from_offset_imgs_torch(self,offset_imgs,vts):
        N=len(offset_imgs)
        device=offset_imgs.device
        size=offset_imgs.size(2)
        # vts=((vts+1)/2*(size-1)+0.5)/size*2-1
        vts=((vts+1)/2*size-0.5)/(size-1)*2-1

        vts=vts.type(offset_imgs.type()).to(device).view(1,1,vts.size(0),2).repeat(N,1,1,1)
        offsets=grid_sample(offset_imgs,vts,mode='bilinear')
        offsets=offsets.view(N,3,-1).permute(0,2,1)
        return offsets

    # def get_offsets_from_uvn_offsets_torch(self,front_uvn_offsets,back_uvn_offsets,ids):
    #     device=front_uvn_offsets.device
    #     N=len(ids)
    #     n_vts=front_uvn_offsets.size(1)
    #     front_uvnhats,back_uvnhats=self.load_uvn_hats(ids)
    #     front_uvnhats=torch.from_numpy(front_uvnhats).to(device)
    #     back_uvnhats=torch.from_numpy(back_uvnhats).to(device)
    #     front_uvn_offsets=front_uvn_offsets.unsqueeze(3)
    #     back_uvn_offsets=back_uvn_offsets.unsqueeze(3)
    #     print('front_uvnhats',front_uvnhats.size(),'front_uvn_offsets',front_uvn_offsets.size())
    #     front_offsets=torch.matmul(front_uvnhats,front_uvn_offsets)
    #     back_offsets=torch.matmul(back_uvnhats,back_uvn_offsets)
    #     return front_offsets,back_offsets

    def get_offsets_from_uvn_offsets_torch(self,uvn_offsets,uvn_hats):
        uvn_offsets=uvn_offsets.unsqueeze(3)
        return torch.matmul(uvn_hats,uvn_offsets).squeeze(3)

    def merge_front_back_offsets_torch(self,front_offsets,back_offsets):
        N=len(front_offsets)
        device=front_offsets.device
        offsets=torch.zeros(N,self.n_vts,3).type(front_offsets.type()).to(device)
        front_vt_ids=self.front_vt_ids.to(device)
        back_vt_ids=self.back_vt_ids.to(device)
        bdry_vt_ids=self.bdry_vt_ids.to(device)

        offsets[:,front_vt_ids,:]+=front_offsets
        offsets[:,back_vt_ids,:]+=back_offsets
        offsets[:,bdry_vt_ids,:]/=2
        return offsets

    def get_offsets_from_uvn_offset_imgs_torch(self,uvn_offset_imgs,front_uvn_hats,back_uvn_hats):
        device=uvn_offset_imgs.device
        front_uvn_offset_imgs=uvn_offset_imgs[:,:3,:,:]
        back_uvn_offset_imgs=uvn_offset_imgs[:,3:6,:,:]
        front_uvn_offsets=self.get_offsets_from_offset_imgs_torch(front_uvn_offset_imgs,self.front_vts)
        back_uvn_offsets=self.get_offsets_from_offset_imgs_torch(back_uvn_offset_imgs,self.back_vts)

        front_uvn_hats=front_uvn_hats[:,self.front_vt_ids,:].type(uvn_offset_imgs.type())
        back_uvn_hats=back_uvn_hats[:,self.back_vt_ids,:].type(uvn_offset_imgs.type())

        front_offsets=self.get_offsets_from_uvn_offsets_torch(front_uvn_offsets,front_uvn_hats)
        back_offsets=self.get_offsets_from_uvn_offsets_torch(back_uvn_offsets,back_uvn_hats)

        offsets=self.merge_front_back_offsets_torch(front_offsets,back_offsets)

        return offsets        


    def get_offsets_from_uvn_offset_imgs_torch_obsolete(self,uvn_offset_imgs,ids):
        device=self.device
        front_uvn_offset_imgs=uvn_offset_imgs[:,:3,:,:]
        back_uvn_offset_imgs=uvn_offset_imgs[:,3:6,:,:]
        front_uvn_offsets=self.get_offsets_from_offset_imgs_torch(front_uvn_offset_imgs,self.front_vts)
        back_uvn_offsets=self.get_offsets_from_offset_imgs_torch(back_uvn_offset_imgs,self.back_vts)

        front_uvn_hats,back_uvn_hats=self.load_uvn_hats(ids)
        front_uvn_hats=torch.from_numpy(front_uvn_hats)[:,self.front_vt_ids,:].type(uvn_offset_imgs.type()).to(device)
        back_uvn_hats=torch.from_numpy(back_uvn_hats)[:,self.back_vt_ids,:].type(uvn_offset_imgs.type()).to(device)

        # print('front_uvn_offsets',front_uvn_offsets.size(),'front_uvn_hats',front_uvn_hats.size())
        front_offsets=self.get_offsets_from_uvn_offsets_torch(front_uvn_offsets,front_uvn_hats)
        back_offsets=self.get_offsets_from_uvn_offsets_torch(back_uvn_offsets,back_uvn_hats)

        offsets=self.merge_front_back_offsets_torch(front_offsets,back_offsets)

        return offsets

    def get_offsets_from_uvn_offset_patch_torch(self,uvn_offset_imgs,vts,uvn_hats):
        uvn_offsets=self.get_offsets_from_offset_imgs_torch(uvn_offset_imgs,vts)
        offsets=self.get_offsets_from_uvn_offsets_torch(uvn_offsets,uvn_hats)
        return offsets

    def write_cloth_uvn_from_id_torch(self,test_id):
        device=self.device
        uvn_offset_img=np.load(os.path.join(self.uvn_offset_img_dir,'offset_img_{:08d}.npy'.format(test_id)))
        uvn_offset_imgs=torch.from_numpy(uvn_offset_img).to(device).unsqueeze(0).permute(0,3,1,2)
        ids=[test_id]
        front_uvn_hats,back_uvn_hats=self.load_uvn_hats(ids)
        front_uvn_hats=torch.from_numpy(front_uvn_hats)
        back_uvn_hats=torch.from_numpy(back_uvn_hats)
        offsets=self.get_offsets_from_uvn_offset_imgs_torch(uvn_offset_imgs,front_uvn_hats,back_uvn_hats)
        offset=offsets[0].detach().cpu().numpy()
        self.write_cloth(test_id,offset)

        gt_offset=np.load(os.path.join(self.vt_offset_dir,'offset_{:08d}.npy'.format(test_id)))
        front_vt_ids=self.front_vt_ids.detach().cpu().numpy()
        back_vt_ids=self.back_vt_ids.detach().cpu().numpy()
        front_gt_offset=gt_offset[front_vt_ids]
        front_offset=offset[front_vt_ids]
        back_gt_offset=gt_offset[back_vt_ids]
        back_offset=offset[back_vt_ids]
        print('error',self.get_error(gt_offset-offset),'front',self.get_error(front_gt_offset-front_offset),'back',self.get_error(back_gt_offset-back_offset))


    def get_offset_np(self,offset_img):
        H,W,D=offset_img.shape
        print('H',H,'W',W)
        offset=np.zeros((self.n_vts,3))
        
        def get_pix_pos(vt):
            # pix_pos=(vt+1)/2*np.array([W-1,H-1])
            pix_pos=(vt+1)/2*np.array([W,H])-0.5
            return pix_pos

        def is_masked(pix_pos,mask):
            x,y=pix_pos[0],pix_pos[1]
            if mask[y,x]>0.5:
                return True
            else:
                return False

        def get_bilinear_weights(pix_pos):
            cell_pos=pix_pos%1
            x,y=cell_pos[0],cell_pos[1]
            return [(1-x)*(1-y),(1-x)*y,x*y,x*(1-y)]

        def get_masked_offset(vt,offset_img,mask,print_info=False):
            pix_pos=get_pix_pos(vt)
            bilinear_weights=get_bilinear_weights(pix_pos)
            left=int(pix_pos[0])
            top=int(pix_pos[1])
            interp_pos=[]
            weights=[]

            p=(left,top)
            if is_masked(p,mask):
                interp_pos.append(p)
                weights.append(bilinear_weights[0])
            p=(left,top+1)
            if is_masked(p,mask):
                interp_pos.append(p)
                weights.append(bilinear_weights[1])
            p=(left+1,top+1)
            if is_masked(p,mask):
                interp_pos.append(p)
                weights.append(bilinear_weights[2])
            p=(left+1,top)
            if is_masked(p,mask):
                interp_pos.append(p)
                weights.append(bilinear_weights[3])

            weights=np.array(weights)
            if len(weights)!=4:
                sum_weights=np.sum(weights)
                weights/=sum_weights

            r=np.zeros(3)

            for i in range(len(weights)):
                w=weights[i]
                p=interp_pos[i]
                x,y=p[0],p[1]
                r+=offset_img[y,x,:]*w

            if print_info:
                print('pix_pos',pix_pos,'weights',weights,'offset',r)
            return r

        front_offset_img=offset_img[:,:,:3]
        back_offset_img=offset_img[:,:,3:6]

        for i in range(len(self.front_vt_ids)):
            vt_id=self.front_vt_ids[i]
            vt=self.front_vts[i]
            offset[vt_id]+=get_masked_offset(vt,front_offset_img,self.front_mask)

        for i in range(len(self.back_vt_ids)):
            vt_id=self.back_vt_ids[i]
            vt=self.back_vts[i]
            offset[vt_id]+=get_masked_offset(vt,back_offset_img,self.back_mask)

        offset[self.bdry_vt_ids]/=2

        # error 0.000232548565691 front 0.00023752278384 back 0.00023145499004
        # error 0.000232548565691 front 0.00023752278384 back 0.00023145499004
        return offset

    def get_error(self,diff):
        return np.sqrt(np.sum(diff**2)/len(diff))

    def write_cloth_from_id_np(self,test_id):
        offset_img_path=os.path.join(self.offset_img_dir,'offset_img_{:08d}.npy'.format(test_id))
        offset_img=np.load(offset_img_path)
        offset=self.get_offset_np(offset_img)

        gt_offset=np.load(os.path.join(self.vt_offset_dir,'offset_{:08d}.npy'.format(test_id)))
        
        front_gt_offset=gt_offset[self.front_vt_ids]
        front_offset=offset[self.front_vt_ids]
        back_gt_offset=gt_offset[self.back_vt_ids]
        back_offset=offset[self.back_vt_ids]
        print('error',self.get_error(gt_offset-offset),'front',self.get_error(front_gt_offset-front_offset),'back',self.get_error(back_gt_offset-back_offset))

        self.write_cloth(test_id,offset)


    def save_bdry_ids(self):
        front_vt_ids_set=set(self.front_vt_ids)
        bdry_vt_ids=[]
        for vt_id in self.back_vt_ids:
            if vt_id in front_vt_ids_set:
                bdry_vt_ids.append(vt_id)
        bdry_vt_ids=np.array(bdry_vt_ids)
        print('bdry_vt_ids',bdry_vt_ids.shape)
        bdry_vt_ids_path=os.path.join(self.shared_data_dir,'bdry_vertices.txt')
        np.savetxt(bdry_vt_ids_path,bdry_vt_ids)

    def get_pix_pos(self,vt,W,H):
        pix_pos=(vt+1)/2*np.array([W,H])-0.5
        return pix_pos

    def get_vts_in_crop(self,crop,original_size):
        x,y,w,h,side=crop
        original_W,original_H=original_size
        vt_ids_in_crop=[]
        vts_in_crop=[]
        if side=="front":
            side_vt_ids=self.front_vt_ids
            vts=self.front_vts
        elif side=="back":
            side_vt_ids=self.back_vt_ids
            vts=self.back_vts
        else:
            print("unrecognized side",side)
            assert(False)

        def in_crop(pix_pos):
            pix_x,pix_y=pix_pos[0],pix_pos[1]
            if pix_x>=x and pix_x<x+w and pix_y>=y and pix_y<y+h:
                return True
            else:
                return False

        vts=vts.numpy()
        for i in range(len(vts)):
            vt_id=side_vt_ids[i]
            vt=vts[i]
            pix_pos=self.get_pix_pos(vt,original_W,original_H)
            if in_crop(pix_pos):
                vt_ids_in_crop.append(vt_id)
                vts_in_crop.append(vt)

        return vt_ids_in_crop,np.array(vts_in_crop)

    def convert_vts_to_crop_coord(self,vts,crop,original_size):
        x,y,w,h,_=crop
        original_W,original_H=original_size
        pix_pos=self.get_pix_pos(vts,W=original_W,H=original_H)
        new_vts=(pix_pos-np.array([x,y])+0.5)/np.array([w,h])*2-1
        return new_vts

    def crop_offset_img(self,offset_img,crop):
        x,y,w,h,side=crop
        if side=='front':
            side_slice=slice(0,3)
        elif side=='back':
            side_slice=slice(3,6)
        return offset_img[y:y+h,x:x+w,side_slice]

    def write_crop_cloth_from_id(self,test_id,crop):
        device=self.device
        x,y,w,h,side=crop
        offset_img=np.load(os.path.join(self.offset_img_dir,'offset_img_{:08d}.npy'.format(test_id)))
        size=offset_img.shape[0]
        # offset_imgs=torch.from_numpy(offset_img[:,:,:3]).to(device).permute(2,0,1).unsqueeze(0)

        crop_offset_img=self.crop_offset_img(offset_img,crop)
        crop_offset_imgs=torch.from_numpy(crop_offset_img).to(device).permute(2,0,1).unsqueeze(0)
        vt_ids_in_crop,original_vts_in_crop=self.get_vts_in_crop(crop,(size,size))
        vts_in_crop=self.convert_vts_to_crop_coord(original_vts_in_crop,crop,(size,size))

        vts_in_crop=torch.from_numpy(vts_in_crop).to(device)
        crop_offsets=self.get_offsets_from_offset_imgs_torch(crop_offset_imgs,vts_in_crop)
        # crop_offsets=self.get_offsets_from_offset_imgs_torch(offset_imgs,original_vts_in_crop)
        crop_offset=crop_offsets[0].detach().cpu().numpy()
        offset=np.zeros((self.n_vts,3))
        offset[vt_ids_in_crop]=crop_offset
        self.write_cloth(test_id,offset,prefix='crop_cloth')

        gt_offset=np.load(os.path.join(self.vt_offset_dir,'offset_{:08d}.npy'.format(test_id)))
        gt_crop_offset=gt_offset[vt_ids_in_crop]
        print('error',self.get_error(gt_crop_offset-crop_offset))

    def write_skin_img_cloth_from_id(self,test_id,prefix='cloth',out_dir=None):
        device=self.device
        offset_img=np.load(os.path.join(self.offset_img_dir,'offset_img_{:08d}.npy'.format(test_id)))
        skin_img=np.load(os.path.join(self.skin_img_dir,'skin_img_{:08d}.npy'.format(test_id)))
        cloth_img=offset_img+skin_img
        cloth_imgs=torch.from_numpy(cloth_img).permute(2,0,1).unsqueeze(0)
        cloth_vt=self.get_offsets_torch(cloth_imgs)[0].detach().cpu().numpy()

        gt_cloth_path=os.path.join(self.data_root_dir,'lowres_tshirts','tshirt_{:08d}.obj'.format(test_id))
        gt_cloth_vt=read_obj(gt_cloth_path).v
        print('error',self.get_error(gt_cloth_vt-cloth_vt))
        obj=Obj(v=cloth_vt,f=self.fcs)
        if out_dir is None:
            out_dir=self.out_dir
        out_path=os.path.join(self.out_dir,'{}_{:08d}.obj'.format(prefix,test_id))
        write_obj(obj,out_path)


if __name__=='__main__':
    # offset_manager=OffsetManager(use_torch=False)
    # offset_manager.write_cloth_from_id_np(10)
    data_root_dir='/data/zhenglin/poses_v3'
    # offset_manager=OffsetManager(img_size=512,shared_data_dir='../../shared_data_midres',data_root_dir=data_root_dir,vt_offset_dir=os.path.join(data_root_dir,'midres_offset_npys'),skin_dir=os.path.join(data_root_dir,'midres_skin_npys'),offset_img_dir=os.path.join(data_root_dir,'midres_offset_imgs'),use_torch=True)
    offset_manager=OffsetManager(img_size=512,shared_data_dir='../../shared_data',data_root_dir=data_root_dir,vt_offset_dir=os.path.join(data_root_dir,'lowres_offset_npys'),skin_dir=os.path.join(data_root_dir,'lowres_skin_npys'),offset_img_dir=os.path.join(data_root_dir,'lowres_offset_imgs'),uvn_dir=os.path.join(data_root_dir,'lowres_skin_tshirt_nuvs'),uvn_offset_img_dir=os.path.join(data_root_dir,'lowres_uvn_offset_imgs'),use_torch=True)
    # offset_manager.write_cloth_from_id_torch(10)

    # offset_manager.save_bdry_ids()
    
    # offset_manager.write_cloth_uvn_from_id_torch(10)
    # offset_manager.write_crop_cloth_from_id(10,(20*4,81*4,32*4,32*4,'front'))
    offset_manager.write_skin_img_cloth_from_id(5205)
