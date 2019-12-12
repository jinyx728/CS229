import os
from os.path import isfile,isdir,join,abspath
import shutil
import numpy as np
from numpy.linalg import norm,inv
from camera_test import CameraTest
from obj_io import read_obj,write_obj,Obj
from io_utils import read_vt
from pathos.multiprocessing import ProcessPool
from ply_io import write_ply
import pickle as pkl
import matplotlib.pyplot as plt
from obj_utils import get_edge_from_fcs
import trimesh
from cudaqs_utils import CudaqsUtils
from draw_obj_utils import DrawObjUtils

def normalized(v):
    return v/norm(v)

class ReconstructUtils(CameraTest):
    def __init__(self,sample_id=16107,reconstruct_type='_square',net_mode='test'):
        self.sample_id=sample_id
        self.type=reconstruct_type
        # self.type=''
        self.sample_dir='reconstruct_test/{:08d}{}'.format(self.sample_id,self.type)
        self.frames_dir=join(self.sample_dir,'frames')
        if not isdir(self.frames_dir):
            os.makedirs(self.frames_dir)
        self.keys_dir=self.frames_dir
        self.use_ff_prune=False
        self.use_visible_prune=False
        # self.use_gt_camera=self.type.find('pd')!=-1
        self.use_gt_camera=False
        self.use_avg_reconst=False
        self.use_screen_bc=False
        self.use_obj_mat=False
        print('use_ff_prune',self.use_ff_prune,'use_visible_prune',self.use_visible_prune,'use_gt_camera',self.use_gt_camera)
        if self.use_gt_camera:
            self.gt_reconstruct_info_dir=self.sample_dir[:-3]
        
        self.obj2tri_path='/data/yxjin/PhysBAM/Tools/obj2tri/obj2tri-relwithdebinfo'
        self.prepare_objs()

        if self.type=='mocap':
            self.gt_v=pd_obj.v
            self.gt_vt=read_vt('../vt_groundtruth_div.txt')
        else:
            self.gt_path=join(self.sample_dir,'gt_{}_div.obj'.format(self.sample_id))
            gt_obj=read_obj(self.gt_path)
            self.gt_v=gt_obj.v
            self.gt_vt=gt_obj.vt

        shared_data_dir='../../pixel_network/shared_data_highres/'
        self.shared_data_dir=shared_data_dir
        self.front_vt_ids=np.loadtxt(join(shared_data_dir,'front_vertices.txt')).astype(np.int)
        front_obj=read_obj(join(shared_data_dir,'flat_tshirt_front.obj'))
        self.front_fcs=front_obj.f
        self.vt_gt=read_vt('../vt_groundtruth_div.txt')
        # self.net_out_dir_pattern='../../pixel_network/rundir/highres_tex/l2pix_final_{}/eval_'+net_mode
        if self.type=='mocap':
            self.net_out_dir_pattern='/phoenix/yxjin/mocap_result/mocap_result_{}'
        else:
            self.net_out_dir_pattern='/data/zhenglin/dataset_subdivision/tsnn_output/test_result_{}'
        self.cudaqs_utils=None

    def load_cam(self):
        cam_path=join(self.sample_dir,'camera.txt')
        cams=np.loadtxt(cam_path)
        return cams

    def generate_circle_cams(self,r,thetas):
        self.load_center()
        cx,cy,cz=self.center
        camera_list=[]
        for i,deg in enumerate(thetas):
            theta=deg*(np.pi)/180
            x=r*np.sin(theta)+cx
            z=r*np.cos(theta)+cz
            camera_pos=np.array([x,cy,z])
            camera_list.append(camera_pos)
            view_pos=np.array([cx,cy,cz])
            meshlab_R=self.get_meshlab_R(camera_pos,view_pos)
            self.write_meshlab_camera(join(self.keys_dir,'meshlab_camera_{}.txt'.format(i)),camera_pos,meshlab_R)
            blender_camera_transform=self.get_blender_camera_transform(camera_pos,view_pos)
            print('transform',blender_camera_transform)
            self.write_camera_transform(join(self.keys_dir,'cam_{}.txt'.format(i)),blender_camera_transform)
        with open(join(self.keys_dir,'last_frame.txt'),'w') as f:
            f.write('{0} {0}\n'.format(len(thetas)))

        cam_path=join(self.sample_dir,'camera.txt')
        np.savetxt(cam_path,np.array(camera_list))

    def generate_square_cams(self):
        src_path='reconstruct_test/square_camera_array.txt'
        camera_path=join(self.sample_dir,'camera.txt')
        shutil.copy(src_path,camera_path)
        cams=np.loadtxt(camera_path)
        self.load_center()
        view_pos=self.center
        for i,camera_pos in enumerate(cams):
            meshlab_R=self.get_meshlab_R(camera_pos,view_pos)
            self.write_meshlab_camera(join(self.keys_dir,'meshlab_camera_{}.txt'.format(i)),camera_pos,meshlab_R)
            blender_camera_transform=self.get_blender_camera_transform(camera_pos,view_pos)
            print('transform',blender_camera_transform)
            self.write_camera_transform(join(self.keys_dir,'cam_{}.txt'.format(i)),blender_camera_transform)
        n_cams=len(cams)
        with open(join(self.keys_dir,'last_frame.txt'),'w') as f:
            f.write('{0} {0}\n'.format(n_cams))

    def run_vt(self):
        cams=self.load_cam()
        def f(cam_i):
            cam_pos=cams[cam_i]
            self.get_fill_obj(cam_pos,'{}'.format(cam_i),cam_i,check=False)
        n_cams=len(cams)
        n_threads=n_cams
        pool=ProcessPool(nodes=n_threads)
        pool.map(f,range(n_cams))

    def copy_net_output(self):
        cams=self.load_cam()
        n_cams=len(cams)
        for cam_i in range(n_cams):
            src_path=join(self.net_out_dir_pattern.format(cam_i+1),'displace_{:08d}.txt'.format(self.sample_id))
            tgt_path=join(self.frames_dir,'fill_disp_{}.txt'.format(cam_i))
            shutil.copyfile(src_path,tgt_path)
            cmds=self.get_obj_cmds('{}'.format(cam_i))
            self.run_cmds(cmds)

    def project_vs(self,vs,camera_pos):
        vs=vs-camera_pos.reshape((1,-1))
        vs[:,0]/=-vs[:,2]
        vs[:,1]/=-vs[:,2]
        vs[:,0]+=camera_pos[0]
        vs[:,1]+=camera_pos[1]
        vs[:,2]=camera_pos[2]-1
        return vs

    def get_bbox_bin_id(self,x,size):
        return min(int(x*size),size-1)

    def get_bbox_bins(self,bboxs,n_bins):
        nW,nH=n_bins
        bbox_bins=[[[] for w in range(nW)] for h in range(nH)]
        for f_i,bbox in enumerate(bboxs):
            (l,r),(b,t)=bbox
            li=self.get_bbox_bin_id(l,nW)
            ri=self.get_bbox_bin_id(r,nW)
            bi=self.get_bbox_bin_id(b,nH)
            ti=self.get_bbox_bin_id(t,nH)
            for c in range(li,ri+1):
                for r in range(bi,ti+1):
                    bbox_bins[r][c].append(f_i)
        return bbox_bins

    def get_n_frames(self):
        with open(join(self.frames_dir,'last_frame.txt')) as f:
            line=f.readline()
            parts=line.split()
            n_frames=int(parts[1])
        return n_frames

    def draw_disp_mesh(self,vt,out_path):
        plane_vt=np.hstack([vt,np.zeros((len(vt),1))])
        print('write to',out_path)
        write_obj(Obj(v=plane_vt,f=self.front_fcs),out_path)

    def draw_disp_meshes(self):
        n_frames=self.get_n_frames()
        self.draw_disp_mesh(self.vt_gt,join(self.sample_dir,'gt_vt.obj'))
        for frame_i in range(n_frames):
            vt_path=join(self.keys_dir,'fill_vt_{}.txt'.format(frame_i))
            vt=read_vt(vt_path)
            self.draw_disp_mesh(vt,join(self.keys_dir,'fill_vt_{}.obj'.format(frame_i)))

    def draw_visible(self,cam_i):
        gt_obj=read_obj(self.gt_path)
        n_vts=len(gt_obj.v)
        colors=np.full((n_vts,3),255,dtype=np.uint8)
        gt_visible=pkl.load(open(join(self.sample_dir,'cam_visible.pkl'),'rb'))
        gt_visible=np.array(gt_visible)
        visible_color=np.array([255,0,0],dtype=np.uint8)
        colors[gt_visible[:,cam_i]]=visible_color
        # for v_i,visible in enumerate(gt_visible[:,cam_i]):
        #     if visible:
        #         colors[v_i]=visible_color
        out_path=join(self.sample_dir,'visible_{}.ply'.format(cam_i))
        print('write to',out_path)
        write_ply(out_path,gt_obj.v,gt_obj.f,colors)

    def get_gt_camera(self):
        gt_reconstruct_info_path=join(self.gt_reconstruct_info_dir,'reconstruct_front_info.pkl')
        if not isfile(gt_reconstruct_info_path):
            print(gt_reconstruct_info_path,'not exist')
            assert(False)
        reconstruct_v_ids,reconstruct_cams=pkl.load(open(gt_reconstruct_info_path,'rb'))
        gt_camera={}
        for v_id,cams in zip(reconstruct_v_ids,reconstruct_cams):
            gt_camera[v_id]=cams
        return gt_camera

    def reconstruct(self,out_path,n_threads=1,max_n_v=-1):
        self.cams=self.load_cam()
        self.load_center()
        # v_ids=np.loadtxt(join(self.sample_dir,'vt_ids.txt')).astype(np.int)
        if max_n_v>0:
            v_ids=self.front_vt_ids[:max_n_v]
        else:
            v_ids=self.front_vt_ids
        n_cams=len(self.cams)
        vt_displaced=[read_vt(join(self.keys_dir,'fill_vt_{}.txt'.format(cam_i))) for cam_i in range(n_cams)]
        v_3d=read_obj(join(self.sample_dir,'pd_{}_div.obj'.format(self.sample_id))).v
        if self.use_ff_prune:
            is_fc_ff=[np.loadtxt(join(self.keys_dir,'is_fc_ff_{}.txt'.format(cam_i))) for cam_i in range(n_cams)]
        if self.use_gt_camera:
            gt_camera=self.get_gt_camera()

        fcs=self.front_fcs # otherwise is_fc_ff won't work
        vt_displaced_bboxs=[self.get_bboxs(vt,fcs) for vt in vt_displaced]
        bbox_bin_size=(25,25)
        bbox_bins=[self.get_bbox_bins(bboxs,bbox_bin_size) for bboxs in vt_displaced_bboxs]
        if self.use_visible_prune:
            gt_visible=pkl.load(open(join(self.sample_dir,'cam_visible.pkl'),'rb'))
        def f(v_id):
            vt_gt_i=self.vt_gt[v_id]
            reconstruct_info=[]
            for cam_i in range(n_cams):
                intersects=self.get_intersects(vt_gt_i,vt_displaced[cam_i],v_3d,fcs,self.cams[cam_i],aux_data={'vt_displaced_bboxs':vt_displaced_bboxs[cam_i],'bbox_bins':bbox_bins[cam_i],'bbox_bin_size':bbox_bin_size})
                if intersects is None:
                    continue
                reconstruct_info.append((intersects,cam_i))
            aux_data={'v_id':v_id}
            if self.use_visible_prune:
                aux_data['gt_visible']=gt_visible
            if self.use_ff_prune:
                aux_data['is_fc_ff']=is_fc_ff[cam_i]
            if self.use_gt_camera:
                aux_data['gt_camera']=gt_camera
            reconstruct_result_i=self.reconstruct_v(reconstruct_info,aux_data=aux_data)
            print('v_id',v_id,'reconstruct_result_i',reconstruct_result_i)
            if reconstruct_result_i is None:
                return None
            else:
                return reconstruct_result_i,reconstruct_info

        reconstruct_results=[]
        if n_threads==1:
            for v_id in v_ids:
                reconstruct_results.append(f(v_id))
        else:
            # not working...
            pool=ProcessPool(nodes=n_threads)
            reconstruct_results=pool.map(f,v_ids)

        reconstructed_vs=[]
        reconstructed_v_ids=[]
        reconstruct_cams=[]
        reconstruct_full_info=[]
        for v_id,r in zip(v_ids,reconstruct_results):
            if r is not None:
                (reconstruct_v_i,reconstruct_cams_i,reconstruct_tris_i),reconstruct_ray_info_i=r
                reconstructed_v_ids.append(v_id)
                reconstructed_vs.append(reconstruct_v_i)
                reconstruct_cams.append(reconstruct_cams_i)
                reconstruct_full_info.append((v_id,reconstruct_v_i,reconstruct_cams_i,reconstruct_tris_i,reconstruct_ray_info_i))
        reconstructed_vs=np.array(reconstructed_vs)
        gt_vs=self.gt_v[reconstructed_v_ids]
        print('3d sqrtMSE',self.get_avg_pt_dist(reconstructed_vs,gt_vs))
        # print('screen sqrtMSE',self.get_avg_pt_dist(self.project_vs(reconstructed_vs),self.project_vs(gt_vs)))
        v_3d[reconstructed_v_ids]=reconstructed_vs
        print('# reconstructed vs',len(reconstructed_v_ids))
        if self.use_obj_mat:
            write_obj(Obj(v=v_3d,f=self.get_fc_subset(reconstructed_v_ids,fcs),vt=self.gt_vt,mat='tshirt.mtl'),out_path)
        else:
            write_obj(Obj(v=v_3d,f=self.get_fc_subset(reconstructed_v_ids,fcs)),out_path)

        info_path='{}_info.pkl'.format(out_path[:out_path.rfind('.')])
        print('write to',info_path)
        pkl.dump((reconstructed_v_ids,reconstruct_cams),open(info_path,'wb'))
        full_info_path='{}_full_info.pkl'.format(out_path[:out_path.rfind('.')])
        print('write to',full_info_path)
        pkl.dump(reconstruct_full_info,open(full_info_path,'wb'))

    def get_avg_pt_dist(self,v0,v1):
        n_vts=len(v0)
        return np.sqrt(np.sum((v0-v1)**2)/n_vts)

    def write_2d_array(self,out_path,arr):
        print('write to',out_path)
        with open(out_path,'w') as f:
            for a in arr:
                f.write('{}\n'.format(' '.join([str(v) for v in a])))

    def read_2d_array(self,in_path):
        arr=[]
        with open(in_path) as f:
            while True:
                line=f.readline()
                if line=='':
                    break
                a=[float(v) for v in line.split()]
                arr.append(a)
        return arr

    # def filter_reconstruct_info(self,reconstruct_info,aux_data,method='distance'):
    #     if len(reconstruct_info)<=2:
    #         return reconstruct_info
    #     max_num_cams=2
    #     v_id=aux_data['v_id']
    #     gt_visible=aux_data['gt_visible']
    #     sorted_info=[]
    #     for cam_info_i in reconstruct_info:
    #         vi,cam_i=cam_info_i
    #         if not gt_visible[v_id][cam_i]:
    #             continue
    #         camera_pos=self.cams[cam_i]
    #         view_pos=self.center
    #         gt_vi=self.gt_v[v_id]
    #         if method=='normal':
    #             cam_normal=normalized(view_pos-camera_pos)
    #             v_normal=normalized(gt_vi-camera_pos)
    #             score=-np.inner(cam_normal,v_normal)
    #         elif method=='distance':
    #             score=norm(camera_pos-gt_vi)
    #         else:
    #             assert(False)
    #         sorted_info.append((cam_info_i,score))
    #     sorted_info.sort(key=lambda x:x[1])
    #     return [x[0] for x in sorted_info[:max_num_cams]]

    def triangulate(self,reconstruct_info):
        A=0
        b=0
        for vi,ci in reconstruct_info:
            ni=vi-ci
            ni/=norm(ni)
            D=len(ci)
            M=np.eye(D)-np.outer(ni,ni)
            A+=M
            b+=M.dot(ci)
        return inv(A).dot(b)

    def point_to_line_distance(self,p,line):
        x1,x2=line
        n=(x2-x1)
        n/=norm(n)
        v=p-x1
        proj_v=np.inner(v,n)*n
        return norm(v-proj_v)

    def filter_cam_candidates(self,intersects,cam_pos,gt_v_i):
        if len(intersects)==1:
            return intersects[0]
        min_d=float('Inf')
        min_intersect=None
        for intersect in intersects:
            v_3d_i,_=intersect
            d=self.point_to_line_distance(gt_v_i,(cam_pos,v_3d_i))
            if d<min_d:
                min_d=d
                min_intersect=intersect
        return min_intersect

    def robust_average(self,arr):
        pass

    def reconstruct_v(self,reconstruct_info,aux_data):
        # reconstruct_info=self.filter_reconstruct_info(reconstruct_info,aux_data)
        if len(reconstruct_info)<=1:
            return None
        v_id=aux_data['v_id']
        if self.use_visible_prune:
            gt_visible=aux_data['gt_visible']
        gt_v_i=self.gt_v[v_id]
        if self.use_ff_prune:
            is_fc_ff=aux_data['is_fc_ff']
        if self.use_gt_camera:
            gt_camera=aux_data['gt_camera']
            if not v_id in gt_camera:
                return None
        if self.use_avg_reconst:
            intersects=[]
            cams=[]
            for intersects_i,cam_i in reconstruct_info:
                ci=self.cams[cam_i]
                v_3d_i,tri_i=self.filter_cam_candidates(intersects_i,ci,gt_v_i)
                intersects.append((v_3d_i,ci))
                cams.append(cam_i)
            # return self.triangulate(intersects),cams
            triangulate_results=[]
            for i in range(len(intersects)):
                for j in range(len(intersects)):
                    if i==j:
                        continue
                    triangulate_results.append(self.triangulate([intersects[i],intersects[j]]))

            return np.mean(np.array(triangulate_results),axis=0),cams
        else:
            candidates=[]
            for intersects_i,cam_i in reconstruct_info:
                if self.use_visible_prune and not gt_visible[v_id][cam_i]:
                    continue
                if self.use_gt_camera and not cam_i in gt_camera[v_id]:
                    continue
                for intersects_j,cam_j in reconstruct_info:
                    if cam_i==cam_j:
                        continue
                    if self.use_visible_prune and not gt_visible[v_id][cam_j]:
                        continue
                    if self.use_gt_camera and not cam_j in gt_camera[v_id]:
                        continue
                    ci=self.cams[cam_i]
                    cj=self.cams[cam_j]
                    for v_3d_i,tri_i in intersects_i:
                        for v_3d_j,tri_j in intersects_j:
                            if self.use_ff_prune and (is_fc_ff[tri_i] or is_fc_ff[tri_j]): 
                                continue
                            v=self.triangulate([(v_3d_i,ci),(v_3d_j,cj)])
                            candidates.append(((v,(cam_i,cam_j),(tri_i,tri_j)),norm(v-gt_v_i)))
            if len(candidates)==0:
                return None
            else:
                return min(candidates,key=lambda x:x[1])[0]

    def color_obj(self,out_ply_path,in_obj_path,aux_info_path):
        obj=read_obj(in_obj_path)
        aux_info=pkl.load(open(aux_info_path,'rb'))
        v_ids,cams=aux_info
        cams=np.array(cams)
        n_cams=np.max(cams)+1
        def f(attr):
            i0,i1=attr
            if i1>i0:
                i0,i1=i1,i0
            return (i0*n_cams+i1)/(n_cams*(n_cams-1))*0.75
        cmap=plt.get_cmap('hsv')
        colors=np.full((len(obj.v),3),255,dtype=np.uint8)
        for i,v_id in enumerate(v_ids):
            color=np.array(cmap(f(cams[i]))[:3])
            colors[v_id]=color*255
        print('write to',out_ply_path)
        write_ply(out_ply_path,obj.v,obj.f,colors)

    def get_vt_ff(self,vt_disp):
        return norm(vt_disp,axis=1)==0

    def write_vt_ff(self):
        n_frames=self.get_n_frames()
        for frame_i in range(n_frames):
            vt_disp_path=join(self.keys_dir,'vt_disp_{}.txt'.format(frame_i))
            vt_disp=np.loadtxt(vt_disp_path)
            is_vt_ff=self.get_vt_ff(vt_disp)
            out_path=join(self.keys_dir,'is_vt_ff_{}.txt'.format(frame_i))
            print('write to',out_path)
            np.savetxt(out_path,is_vt_ff)

    def get_fc_ff(self,fcs,is_vt_ff):
        vt_ff_set=set()
        for vt_id,is_ff in enumerate(is_vt_ff):
            if is_ff!=0:
                vt_ff_set.add(vt_id)
        n_fcs=len(fcs)
        is_fc_ff=np.zeros(n_fcs)
        for fc_i,(i0,i1,i2) in enumerate(fcs):
            if i0 in vt_ff_set or i1 in vt_ff_set or i2 in vt_ff_set:
                is_fc_ff[fc_i]=1
        return is_fc_ff

    def write_fc_ff(self):
        n_frames=self.get_n_frames()
        for frame_i in range(n_frames):
            vt_ff_path=join(self.keys_dir,'is_vt_ff_{}.txt'.format(frame_i))
            is_vt_ff=np.loadtxt(vt_ff_path)
            is_fc_ff=self.get_fc_ff(self.front_fcs,is_vt_ff)
            out_path=join(self.keys_dir,'is_fc_ff_{}.txt'.format(frame_i))
            print('write to',out_path)
            np.savetxt(out_path,is_fc_ff)

    def trim_obj(self,in_path,out_pattern,n_iters=1):
        def trim_vts(ids,trim_id_set):
            left_ids=[]
            for i in ids:
                if not i in trim_id_set:
                    left_ids.append(i)
            return left_ids

        def add_to_set(trim_id_set,next_trim_list):
            for v in next_trim_list:
                trim_id_set.add(v)

        def next_trim(trim_id_set,edges):
            next_trim_id_set=set()
            for i0,i1 in edges:
                i0_in=i0 in trim_id_set
                i1_in=i1 in trim_id_set
                if i0_in and not i1_in:
                    next_trim_id_set.add(i1)
                if not i0_in and i1_in:
                    next_trim_id_set.add(i0)
            return list(next_trim_id_set)

        def trim_fcs(fcs,ids):
            id_set=set(ids)
            result_fcs=[]
            for fc in fcs:
                i0,i1,i2=fc
                if i0 in id_set and i1 in id_set and i2 in id_set:
                    result_fcs.append(fc)
            return np.array(result_fcs).astype(np.int)

        bdry_ids=np.loadtxt(join(self.shared_data_dir,'bdry_vertices.txt')).astype(np.int)
        trim_id_set=set(bdry_ids)

        left_ids=trim_vts(self.front_vt_ids,trim_id_set)
        obj=read_obj(in_path)
        edges=get_edge_from_fcs(self.front_fcs)
        fcs=obj.f

        iters=0
        while True:
            fcs=trim_fcs(fcs,left_ids)
            out_path=out_pattern.format(iters)
            print('write to',out_path)
            write_obj(Obj(obj.v,f=fcs),out_path)
            iters+=1
            if iters>=n_iters:
                break
            next_trim_list=next_trim(trim_id_set,edges)
            add_to_set(trim_id_set,next_trim_list)
            left_ids=trim_vts(left_ids,trim_id_set)

    def is_visible(self,mesh,v,camera_pos):
        locations,_,_=mesh.ray.intersects_location(ray_origins=np.array([camera_pos]), ray_directions=np.array([v-camera_pos]))
        dist=norm(v-camera_pos)-1e-10
        for x in locations:
            if norm(x-camera_pos)<dist:
                return False
        return True

    def compute_visible(self,out_path):
        cams=self.load_cam()
        gt_mesh=trimesh.load(self.gt_path,process=False)
        vs_cams_visible=[]
        n_vts=len(gt_mesh.vertices)
        for v_i,v in enumerate(gt_mesh.vertices):
            v_cams_visible=[self.is_visible(gt_mesh,v,cam) for cam in cams]
            vs_cams_visible.append(v_cams_visible)
            print('finished {}/{}'.format(v_i,n_vts))
        print('write to',out_path)
        pkl.dump(vs_cams_visible,open(out_path,'wb'))

    def get_fc_subset(self,v_ids,fcs):
        v_id_set=set(v_ids)
        subset_fc_ids=[]
        for fc_i,fc in enumerate(fcs):
            if fc[0] in v_id_set and fc[1] in v_id_set and fc[2] in v_id_set:
                subset_fc_ids.append(fc_i)
        return fcs[subset_fc_ids]

    def get_intersects(self,vt_i,vt_2d,v_3d,fcs,cam_pos,aux_data):
        tri_list=self.get_tris(vt_i,vt_2d,fcs,aux_data)
        if len(tri_list)==0:
            return None
        candidates=[]
        for tri_id,bc in tri_list:
            vs=self.get_vs(v_3d,fcs[tri_id])
            if self.use_screen_bc:
                vs=self.project_vs(vs,cam_pos)
            v_3d_i=self.get_v_from_bc(vs,bc)
            candidates.append((v_3d_i,tri_id))
        if len(candidates)==0:
            return None
        else:
            return candidates

    def get_vs(self,vs,ids):
        return vs[ids]

    def get_v_from_bc(self,vs,bc):
        return bc.reshape((1,-1)).dot(vs).reshape(-1)

    def get_bboxs(self,vt,fcs):
        return [self.get_bbox(self.get_vs(vt,fc)) for fc in fcs]

    def get_bbox(self,vt):
        vmin=np.min(vt,axis=0)
        vmax=np.max(vt,axis=0)
        return [(x,y) for x,y in zip(vmin,vmax)]

    def get_tris(self,vt_i,vt_2d,fcs,aux_data):
        tri_bboxs=aux_data['vt_displaced_bboxs']
        n_fcs=len(fcs)
        tris=[]
        # for fc_i in range(n_fcs):
        #     bbox=tri_bboxs[fc_i]
        #     if self.in_bbox(vt_i,bbox):
        #         vs=self.get_vs(vt_2d,fcs[fc_i])
        #         bc=self.get_bc(vt_i,vs)
        #         if bc is None:
        #             continue
        #         if bc[0]>=0 and bc[1]>=0 and bc[2]>=0:
        #             tris.append((fc_i,bc))

        bbox_bins=aux_data['bbox_bins']
        nW,nH=aux_data['bbox_bin_size']
        ci=self.get_bbox_bin_id(vt_i[0],nW)
        ri=self.get_bbox_bin_id(vt_i[1],nH)
        for fc_i in bbox_bins[ri][ci]:
            bbox=tri_bboxs[fc_i]
            if self.in_bbox(vt_i,bbox):
                vs=self.get_vs(vt_2d,fcs[fc_i])
                bc=self.get_bc(vt_i,vs)
                if bc is None:
                    continue
                if bc[0]>=0 and bc[1]>=0 and bc[2]>=0:
                    tris.append((fc_i,bc))
        return tris

    def in_bbox(self,v,bbox):
        D=len(v)
        for i in range(D):
            l,h=bbox[i]
            vi=v[i]
            if vi<l or vi>h:
                return False
        return True

    def get_bc(self,x,vs):
        eps=1e-16
        x1,x2,x3=vs
        u=x2-x1
        v=x3-x1
        w=x-x1
        u_dot_u=np.inner(u,u)
        v_dot_v=np.inner(v,v)
        u_dot_v=np.inner(u,v)
        u_dot_w=np.inner(u,w)
        v_dot_w=np.inner(v,w)
        denominator=u_dot_u*v_dot_v-u_dot_v*u_dot_v
        if np.abs(denominator)<eps:
            return None
        one_over_denominator=1/denominator
        a=(v_dot_v*u_dot_w-u_dot_v*v_dot_w)*one_over_denominator
        b=(u_dot_u*v_dot_w-u_dot_v*u_dot_w)*one_over_denominator
        return np.array([1-a-b,a,b])

    def apply_tex(self,in_path,out_path):
        obj=read_obj(in_path)
        obj.vt=self.gt_vt
        print('write to',out_path)
        write_obj(obj,out_path)

    def draw_error_img(self,pd_obj_path):
        cwd=os.getcwd()
        default_camera_pos=np.array([0.0437965,0.705956,0.787657])
        default_camera_str=' '.join([str(v) for v in default_camera_pos])
        cmds=['cd ../../figure_scripts']
        gt_obj_path=join(abspath(self.sample_dir),'gt_{}_div.obj'.format(self.sample_id))
        gt_img_path=join(abspath(self.sample_dir),'gt_img.npy')
        pd_obj_path=abspath(pd_obj_path)
        pd_img_path=pd_obj_path[:-4]+".npy"
        error_img_path=pd_obj_path[:-4]+'_error.png'
        cmds.append('python draw_uv_img.py -i {} -o{} -c {}'.format(gt_obj_path,gt_img_path,default_camera_str))
        cmds.append('python draw_uv_img.py -i {} -o {} -c {}'.format(pd_obj_path,pd_img_path,default_camera_str))
        cmds.append('python draw_error_img.py -img1 {} -img2 {} -out_path {}'.format(gt_img_path,pd_img_path,error_img_path))
        cmds.append('cd {}'.format(cwd))
        self.run_cmds(cmds)

    def draw_area_ratio_ply(self,in_path,out_path):
        cwd=os.getcwd()
        cmds=['cd ../../figure_scripts']
        cmds.append('python compute_area_stats.py -in_path {} -out_path {}'.format(abspath(in_path),abspath(out_path)))
        cmds.append('cd {}'.format(cwd))
        self.run_cmds(cmds)

    def get_vt_ids_from_fcs(self,fcs):
        vt_set=set()
        for i0,i1,i2 in fcs:
            vt_set.add(i0)
            vt_set.add(i1)
            vt_set.add(i2)
        return list(vt_set)

    def write_cleaned_vt_ids(self,in_path,out_path):
        obj=read_obj(in_path)
        vt_ids=self.get_vt_ids_from_fcs(obj.f)
        print('write to',out_path)
        np.savetxt(out_path,np.array(vt_ids))

    def run_ff(self,reconstruct_obj_path,vt_id_path):
        ff_dir=join(self.sample_dir,'ff')
        if not isdir(ff_dir):
            os.makedirs(ff_dir)
        frame=0
        pd_obj_path=join(self.sample_dir,'pd_{}_div.obj'.format(self.sample_id))
        pd_tri_path=pd_obj_path[:-3]+"tri"
        reconstruct_tri_path=reconstruct_obj_path[:-3]+"tri"
        cmds=['{} {} {}'.format(self.obj2tri_path,pd_obj_path,pd_tri_path),
              '{} {} {}'.format(self.obj2tri_path,reconstruct_obj_path,reconstruct_tri_path)]
        cmds.append('../../tex3d {} {} {} {} {}'.format(pd_tri_path,reconstruct_tri_path,vt_id_path,ff_dir,frame))
        cmds.append('cp {} {}'.format(join(ff_dir,'reconstruct_filled.obj'),join(self.sample_dir,'reconstruct_front_filled.obj')))
        self.run_cmds(cmds)

    def run_pp(self,in_path,out_path):
        if self.cudaqs_utils is None:
            self.cudaqs_utils=CudaqsUtils('../../pixel_network/shared_data_hr_front')
        obj=read_obj(in_path)
        obj.v=self.cudaqs_utils.forward(obj.v)
        print('write to',out_path)
        write_obj(obj,out_path)

    def get_clean_mesh(self,in_path,out_path,area_threshold=6,ratio_threshold=4):
        in_obj=read_obj(in_path)
        fcs=self.prune_fcs(in_obj.v,self.gt_v,in_obj.f,area_threshold=area_threshold,ratio_threshold=ratio_threshold)
        print('write to',out_path)
        write_obj(Obj(v=in_obj.v,f=fcs),out_path)


    def prune_fcs(self,pd_vts,gt_vts,fcs,area_threshold,ratio_threshold):
        prune_fcs=[]
        def get_area(vts):
            return norm(np.cross(vts[2]-vts[0],vts[1]-vts[0]))

        def get_max_division(a,b):
            return a/b if a>b else b/a

        def get_ratio(vts,area):
            e0=norm(vts[1]-vts[2])
            e1=norm(vts[0]-vts[2])
            e2=norm(vts[0]-vts[1])
            r0=get_max_division(e0,area/e0)
            r1=get_max_division(e1,area/e1)
            r2=get_max_division(e2,area/e2)
            return max(r0,r1,r2)

        for fc in fcs:
            pd_tri=pd_vts[fc]
            pd_area=get_area(pd_tri)
            pd_ratio=get_ratio(pd_tri,pd_area)
            gt_tri=gt_vts[fc]
            gt_area=get_area(gt_tri)
            gt_ratio=get_ratio(gt_tri,gt_area)
            if pd_area/gt_area>area_threshold or pd_ratio/gt_ratio>ratio_threshold:
                continue
            prune_fcs.append(fc)
        return np.array(prune_fcs).astype(int)

    def filter_front_fcs(self,in_path,out_path):
        in_obj=read_obj(in_path)
        print('write to',out_path)
        write_obj(Obj(v=in_obj.v,f=self.front_fcs,vt=in_obj.vt),out_path)

    def filter_front_fcs_in_sample_dir(self):
        self.filter_front_fcs(join(self.sample_dir,'gt_{}_div.obj'.format(self.sample_id)),join(self.sample_dir,'gt_{}_div_front.obj'.format(self.sample_id)))
        self.filter_front_fcs(join(self.sample_dir,'pd_{}_div.obj'.format(self.sample_id)),join(self.sample_dir,'pd_{}_div_front.obj'.format(self.sample_id)))

    def get_circle_camera(self,out_dir,n_frames,rz,ry):
        self.load_center()
        if not isdir(out_dir):
            os.makedirs(out_dir)
        center=self.center+np.array([0,0,rz])
        for frame_i in range(n_frames):
            theta=frame_i/n_frames*(2*np.pi)
            dx=ry*np.sin(theta)
            dy=ry*np.cos(theta)
            camera_pos=center+np.array([dx,dy,0])
            camera_transform=self.get_blender_camera_transform(camera_pos,self.center)
            out_path=join(out_dir,'cam_{}.txt'.format(frame_i))
            self.write_camera_transform(out_path,camera_transform)

    def analyze_overlap_error(self):
        print('analyze_overlap_error')
        out_dir=join(self.sample_dir,'error_analysis')
        if not isdir(out_dir):
            os.makedirs(out_dir)
        reconstruct_full_info=pkl.load(open(join(self.sample_dir,'reconstruct_front_full_info.pkl'),'rb'))
        overlap_info=[[] for i in range(4)]
        for info_i in reconstruct_full_info:
            v_id,reconstruct_v_i,reconstruct_cams_i,reconstruct_tris_i,reconstruct_ray_info_i=info_i
            for cam_i_p in reconstruct_cams_i:
                for intersects_i_q,cam_i_q in reconstruct_ray_info_i:
                    if cam_i_p==cam_i_q:
                        max_dis=0
                        locations=[]
                        if len(intersects_i_q)>1:
                            for a,(v_3d_a,tri_id_a) in enumerate(intersects_i_q):
                                for b,(v_3d_b,tri_id_b) in enumerate(intersects_i_q):
                                    if a==b: continue
                                    max_dis=max(norm(v_3d_a-v_3d_b),max_dis)
                                locations.append(v_3d_a)
                        overlap_info[cam_i_p].append((v_id,max_dis,locations))
        for i in range(4):
            overlap_info[i].sort(key=lambda x:x[1],reverse=True)
        out_overlap_info=[info_c[0] for info_c in overlap_info]
        out_path=join(out_dir,'overlap_error.pkl')
        print('write to',out_path)
        pkl.dump(out_overlap_info,open(out_path,'wb'))
        out_path=join(out_dir,'overlap_error.txt')
        print('write to',out_path)
        with open(out_path,'w') as f:
            for i in range(4):
                f.write('{}\n'.format(out_overlap_info[i]))

    def draw_overlap_error(self):
        draw_obj_utils=DrawObjUtils()
        error_dir=join(self.sample_dir,'error_analysis')
        info_path=join(error_dir,'overlap_error.pkl')
        info=pkl.load(open(info_path,'rb'))
        for cam_i,(v_id,max_dis,locations) in enumerate(info):
            draw_obj_utils.reset()
            for x in locations:
                draw_obj_utils.add_sphere(x,0.005)
            out_path=join(error_dir,'overlap_error_{}.obj'.format(cam_i))
            print('write to',out_path)
            draw_obj_utils.write_obj(out_path)

    def get_skew_dis(self,ray1,ray2):
        c1,v1=ray1
        n1=v1-c1
        n1/=norm(n1)
        c2,v2=ray2
        n2=v2-c2
        n2/=norm(n2)
        p=np.cross(n1,n2)
        p/=norm(p)
        A=np.zeros((3,3))
        A[:,0]=n1
        A[:,1]=n2
        A[:,2]=p
        b=c2-c1
        sol=np.linalg.inv(A).dot(b)
        p1=c1+n1*sol[0]
        p2=c2-n2*sol[1]
        return np.abs(sol[2]),p1,p2

    def analyze_skew_error(self):
        print('analyze_skew_error')
        out_dir=join(self.sample_dir,'error_analysis')
        if not isdir(out_dir):
            os.makedirs(out_dir)
        reconstruct_full_info=pkl.load(open(join(self.sample_dir,'reconstruct_front_full_info.pkl'),'rb'))
        cams=self.load_cam()
        skew_info=[]
        for info_i in reconstruct_full_info:
            v_id,reconstruct_v_i,reconstruct_cams_i,reconstruct_tris_i,reconstruct_ray_info_i=info_i
            rays=[]
            for cam_i_p,tri_id_p in zip(reconstruct_cams_i,reconstruct_tris_i):
                for intersects_i_q,cam_i_q in reconstruct_ray_info_i:
                    if cam_i_p==cam_i_q:
                        for v_3d_r,tri_id_r in intersects_i_q:
                            if tri_id_p==tri_id_r:
                                rays.append((cams[cam_i_p],v_3d_r))
            assert(len(rays)==2)
            skew_dis,p1,p2=self.get_skew_dis(rays[0],rays[1])
            skew_info.append((v_id,skew_dis,rays,(p1,p2)))
        skew_info.sort(key=lambda x:x[1],reverse=True)
        out_path=join(out_dir,'skew_error.pkl')
        print('write to',out_path)
        pkl.dump(skew_info[0],open(out_path,'wb'))
        out_path=join(out_dir,'skew_error.txt')
        print('write to',out_path)
        with open(out_path,'w') as f:
            f.write('{}\n'.format(skew_info[0]))

    def draw_skew_error(self):
        draw_obj_utils=DrawObjUtils()
        error_dir=join(self.sample_dir,'error_analysis')
        info_path=join(error_dir,'skew_error.pkl')
        info=pkl.load(open(info_path,'rb'))
        v_id,skew_dis,((c1,p1),(c2,p2)),(q1,q2)=info
        draw_obj_utils.add_sphere(c1,0.005)
        draw_obj_utils.add_sphere(p1,0.005)
        draw_obj_utils.add_cylinder(c1,p1,0.002)
        draw_obj_utils.add_sphere(c2,0.005)
        draw_obj_utils.add_sphere(p2,0.005)
        draw_obj_utils.add_cylinder(c2,p2,0.002)
        draw_obj_utils.add_sphere(q1,0.005)
        draw_obj_utils.add_sphere(q2,0.005)
        draw_obj_utils.add_cylinder(q1,q2,0.002)
        out_path=join(error_dir,'skew_error.obj')
        print('write to',out_path)
        draw_obj_utils.write_obj(out_path)


    def get_max_v_error(self,v,id_set):
        max_dis=0
        max_v_id=-1
        for v_id in self.front_vt_ids:
            if v_id in id_set:
                gt_v_i=self.gt_v[v_id]
                pd_v_i=v[v_id]
                dis=norm(gt_v_i-pd_v_i)
                if dis>max_dis:
                    max_dis=dis
                    max_v_id=v_id
        return max_v_id,max_dis

    def analyze_reconstruct_error(self):
        print('analyze_reconstruct_error')
        out_dir=join(self.sample_dir,'error_analysis')
        if not isdir(out_dir):
            os.makedirs(out_dir)
        reconstruct_v_ids=np.loadtxt(join(self.sample_dir,'cleaned_vt_ids.txt')).astype(int)
        reconstruct_v_id_set=set(reconstruct_v_ids.tolist())
        reconstruct_v=read_obj(join(self.sample_dir,'reconstruct_front.obj')).v
        max_v_id,max_dis=self.get_max_v_error(reconstruct_v,reconstruct_v_id_set)
        out_path=join(out_dir,'reconstruct_error.txt')
        print('write to',out_path)
        with open(out_path,'w') as f:
            f.write('{} {}\n'.format(max_v_id,max_dis))

    def analyze_ff_error(self):
        print('analyze_reconstruct_error')
        out_dir=join(self.sample_dir,'error_analysis')
        if not isdir(out_dir):
            os.makedirs(out_dir)
        reconstruct_v_ids=np.loadtxt(join(self.sample_dir,'cleaned_vt_ids.txt')).astype(int)
        reconstruct_v_id_set=set(reconstruct_v_ids.tolist())
        ff_v_id_set=set()
        for v_id in self.front_vt_ids:
            if not v_id in reconstruct_v_id_set:
                ff_v_id_set.add(v_id)
        reconstruct_v=read_obj(join(self.sample_dir,'reconstruct_front_filled.obj')).v
        max_v_id,max_dis=self.get_max_v_error(reconstruct_v,ff_v_id_set)
        out_path=join(out_dir,'ff_error.txt')
        print('write to',out_path)
        with open(out_path,'w') as f:
            f.write('{} {}\n'.format(max_v_id,max_dis))

    def analyze_pp_error(self):
        print('analyze_reconstruct_error')
        out_dir=join(self.sample_dir,'error_analysis')
        if not isdir(out_dir):
            os.makedirs(out_dir)
        v_id_set=set(self.front_vt_ids.tolist())
        reconstruct_v=read_obj(join(self.sample_dir,'reconstruct_front_pp.obj')).v
        max_v_id,max_dis=self.get_max_v_error(reconstruct_v,v_id_set)
        out_path=join(out_dir,'pp_error.txt')
        print('write to',out_path)
        with open(out_path,'w') as f:
            f.write('{} {}\n'.format(max_v_id,max_dis))

    def draw_vt_error(self,error_path,cloth_obj_path,out_path):
        draw_obj_utils=DrawObjUtils()
        with open(error_path) as f:
            line=f.readline()
            parts=line.split()
            max_v_id=int(parts[0])
        v=read_obj(cloth_obj_path).v
        p1,p2=v[max_v_id],self.gt_v[max_v_id]
        draw_obj_utils.add_sphere(p1,0.005)
        draw_obj_utils.add_sphere(p2,0.005)
        draw_obj_utils.add_cylinder(p1,p2,0.002)
        print('write to',out_path)
        draw_obj_utils.write_obj(out_path)

    def draw_reconstruct_error(self):
        error_dir=join(self.sample_dir,'error_analysis')
        self.draw_vt_error(join(error_dir,'reconstruct_error.txt'),join(self.sample_dir,'reconstruct_front.obj'),join(error_dir,'reconstruct_error.obj'))

    def draw_ff_error(self):
        error_dir=join(self.sample_dir,'error_analysis')
        self.draw_vt_error(join(error_dir,'ff_error.txt'),join(self.sample_dir,'reconstruct_front_filled.obj'),join(error_dir,'ff_error.obj'))

    def draw_pp_error(self):
        error_dir=join(self.sample_dir,'error_analysis')
        self.draw_vt_error(join(error_dir,'pp_error.txt'),join(self.sample_dir,'reconstruct_front_pp.obj'),join(error_dir,'pp_error.obj'))

    def analyze_errors(self):
        self.analyze_overlap_error()
        self.analyze_skew_error()
        self.analyze_reconstruct_error()
        self.analyze_ff_error()
        self.analyze_pp_error()

if __name__=='__main__':
<<<<<<< .mine
    test=ReconstructUtils(sample_id=0,reconstruct_type='_square_pd',net_mode='mocap')
    test.generate_square_cams()
||||||| .r79176
    test=ReconstructUtils(sample_id=16107,reconstruct_type='_square',net_mode='test')
    # test.generate_square_cams()
=======
    test=ReconstructUtils(sample_id=16107,reconstruct_type='_square_pd',net_mode='test')
    # test.generate_square_cams()
>>>>>>> .r79188
    # test.run_vt()
<<<<<<< .mine
    # test.filter_front_fcs_in_sample_dir()
    test.copy_net_output()
    test.reconstruct(join(test.sample_dir,'reconstruct_front.obj'),max_n_v=-1)
    test.get_clean_mesh(join(test.sample_dir,'reconstruct_front.obj'),join(test.sample_dir,'reconstruct_front_clean.obj'),area_threshold=6,ratio_threshold=4)
    test.write_cleaned_vt_ids(join(test.sample_dir,'reconstruct_front_clean.obj'),join(test.sample_dir,'cleaned_vt_ids.txt'))
    test.run_ff(join(test.sample_dir,'reconstruct_front_clean.obj'),join(test.sample_dir,'cleaned_vt_ids.txt'))
    test.run_pp(join(test.sample_dir,'reconstruct_front_filled.obj'),join(test.sample_dir,'reconstruct_front_pp.obj'))
||||||| .r79176
    test.filter_front_fcs_in_sample_dir()
    # test.copy_net_output()
    # test.reconstruct(join(test.sample_dir,'reconstruct_front.obj'),max_n_v=-1)
    # test.get_clean_mesh(join(test.sample_dir,'reconstruct_front.obj'),join(test.sample_dir,'reconstruct_front_clean.obj'),area_threshold=6,ratio_threshold=4)
    # test.write_cleaned_vt_ids(join(test.sample_dir,'reconstruct_front_clean.obj'),join(test.sample_dir,'cleaned_vt_ids.txt'))
    # test.run_ff(join(test.sample_dir,'reconstruct_front_clean.obj'),join(test.sample_dir,'cleaned_vt_ids.txt'))
=======
    # test.filter_front_fcs_in_sample_dir()
    # test.copy_net_output()
    # test.reconstruct(join(test.sample_dir,'reconstruct_front.obj'),max_n_v=-1)
    # test.get_clean_mesh(join(test.sample_dir,'reconstruct_front.obj'),join(test.sample_dir,'reconstruct_front_clean.obj'),area_threshold=6,ratio_threshold=4)
    # test.write_cleaned_vt_ids(join(test.sample_dir,'reconstruct_front_clean.obj'),join(test.sample_dir,'cleaned_vt_ids.txt'))
    # test.run_ff(join(test.sample_dir,'reconstruct_front_clean.obj'),join(test.sample_dir,'cleaned_vt_ids.txt'))
>>>>>>> .r79188
    # test.run_pp(join(test.sample_dir,'reconstruct_front_filled.obj'),join(test.sample_dir,'reconstruct_front_pp.obj'))
    # test.draw_error_img(join(test.sample_dir,'reconstruct_front_pp.obj'))
    # test.draw_area_ratio_ply(join(test.sample_dir,'reconstruct_front_pp.obj'),join(test.sample_dir,'reconstruct_front_pp.ply'))
    # test.run_pp(join(test.sample_dir,'pd_{}_div_front.obj'.format(test.sample_id)),join(test.sample_dir,'pd_{}_pp.obj'.format(test.sample_id)))
    # test.draw_error_img(join(test.sample_dir,'pd_{}_pp.obj'.format(test.sample_id)))
    # test.draw_area_ratio_ply(join(test.sample_dir,'pd_{}_pp.obj'.format(test.sample_id)),join(test.sample_dir,'pd_{}_pp.ply'.format(test.sample_id)))
    # test.get_circle_camera(join(test.sample_dir,'blender_cam'),120,0.8,0.1)

    # test.analyze_overlap_error()
    # test.analyze_skew_error()
    # test.analyze_reconstruct_error()
    # test.analyze_ff_error()
    # test.analyze_pp_error()
    # test.draw_overlap_error()
    # test.draw_skew_error()
    # test.draw_reconstruct_error()
    # test.draw_ff_error()
    # test.draw_pp_error()

    # test.color_obj(join(test.sample_dir,'reconstruct_front.ply'),join(test.sample_dir,'reconstruct_front.obj'),join(test.sample_dir,'reconstruct_front_info.pkl'))
    # test.draw_error_img()
    # test.draw_area_ratio_ply()
    # test.trim_obj('reconstruct_test/00015129/reconstruct_front.obj','reconstruct_test/00015129/reconstruct_front_{}.obj',n_iters=10)
    # for i in range(8):
    #     test.draw_visible(i)
    # test.draw_disp_meshes()

    # test.generate_circle_cams(0.75,np.arange(-45,45,10))
    # test.generate_circle_cams(0.75,[-60,-45,-30,-15,0,15,30,45])
    # test.apply_tex(join(test.sample_dir,'reconstruct_front_filled.obj'),join(test.sample_dir,'reconstruct_front_filled.obj'))
    # test.draw_error_img()
    # test.draw_area_ratio_ply(join(test.sample_dir,'reconstruct_front_filled.obj'),join(test.sample_dir,'reconstruct_front_filled.ply'))



