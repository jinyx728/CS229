import os
from os.path import isfile,isdir,join,abspath
import numpy as np
from numpy.linalg import norm
import trimesh
# from multiprocessing import Pool
from pathos.multiprocessing import ProcessPool
from io_utils import read_vt,write_vt,write_uv
from obj_io import read_obj,write_obj,Obj
from pyquaternion import Quaternion
import shutil

def compute_pixel_error(img1,img2):
    error_map=np.sum((img1[:,:,1:3]-img2[:,:,1:3])**2,axis=2)
    alpha_map=img1[:,:,3]*img2[:,:,3]
    return np.sqrt(np.sum(error_map*alpha_map)/np.sum(alpha_map))

class CameraTest:
    def __init__(self,sample_id=16338,camera_type='_grid'):
        self.sample_id=sample_id
        self.sample_dir='camera_test/{}'.format(self.sample_id)
        self.type=camera_type
        self.grid_size=(10,10)
        self.frames_dir=join(self.sample_dir,'frames{}'.format(self.type))
        self.keys_dir=join(self.sample_dir,'keys{}'.format(self.type))
        if not isdir(self.frames_dir):
            os.makedirs(self.frames_dir)
        if not isdir(self.keys_dir):
            os.makedirs(self.keys_dir)
        self.obj2tri_path='/data/zhenglin/PhysBAM/Tools/obj2tri/obj2tri'
        self.prepare_objs()
        self.load_center()
        obj=read_obj(join(self.sample_dir,'pd_{}_div.obj'.format(self.sample_id)))
        self.pd_v=obj.v
        self.fcs=obj.f

    def prepare_objs(self):
        data_root_dir='/phoenix/yxjin/test_infer'
        pd_path=join(self.sample_dir,'pd_{}_div.obj'.format(self.sample_id))
        if not isfile(pd_path):
            src_path=join(data_root_dir,'mocap_13_30_subdiv/pd_div_{:08d}_tex.obj'.format(self.sample_id))
            if isfile(src_path):
                shutil.copy(src_path,pd_path)
            else:
                src_path=join(data_root_dir,'test_objs_hr/{:08d}.obj'.format(self.sample_id))
                shutil.copy(src_path,pd_path)
        if self.type != 'mocap':
            gt_path=join(self.sample_dir,'gt_{}_div.obj'.format(self.sample_id))
            if not isfile(gt_path):
                shutil.copy(join(data_root_dir,'dataset_gt_subdiv/tshirt_div_{:08d}_tex.obj'.format(self.sample_id)),gt_path)
        cmds=[]
        cmds+=self.get_tri_cmds()
        cmds+=self.get_mat_cmds()
        self.run_cmds(cmds)

    def get_tri_cmds(self):
        tri_file='{0}/pd_{1}_div.tri.gz'.format(self.sample_dir,self.sample_id)
        if not isfile(tri_file):
            return ['{0} {1}/pd_{2}_div.obj {1}/pd_{2}_div.tri'.format(self.obj2tri_path,self.sample_dir,self.sample_id)]
        else:
            return []

    def get_mat_cmds(self):
        cmds=[]
        if not isfile(join(self.frames_dir,'tshirt.mtl')):
            cmds+=['cp camera_test/tshirt.mtl {}'.format(self.frames_dir),
                   'cp camera_test/light_checker_cross.png {}'.format(self.frames_dir)]
        if not isfile(join(self.keys_dir,'tshirt.mtl')):
            cmds+=['cp camera_test/tshirt.mtl {}'.format(self.keys_dir),
                   'cp camera_test/light_checker_cross.png {}'.format(self.keys_dir)]
        if not isfile(join(self.sample_dir,'tshirt.mtl')):
            cmds+=['cp camera_test/tshirt.mtl {}'.format(self.sample_dir),
                   'cp camera_test/light_checker_cross.png {}'.format(self.sample_dir)]
        return cmds

    def get_disp_cmds(self,camera_pos,postfix):
        return ['python dataset_lowres_new.py -g {0}/gt_{1}_div.obj -b {7} -p {0}/pd_{1}_div.obj -c {2} {3} {4} -t ../vt_groundtruth_div.txt > {5}/vt_disp_{6}.txt'.format(self.sample_dir,self.sample_id,camera_pos[0],camera_pos[1],camera_pos[2],self.keys_dir,postfix,'../../pixel_network/shared_data_highres/back_vertices.txt')]

    def get_fill_cmds(self,postfix,frame):
        cmds=['../../tex2d {0}/pd_{1}_div.tri ../vt_groundtruth_div.txt {2}/vt_disp_{3}.txt {2} {4}'.format(self.sample_dir,self.sample_id,self.keys_dir,postfix,frame),
        'mv {0}/displace_{1}.txt {0}/fill_disp_{2}.txt'.format(self.keys_dir,frame,postfix)]
        return cmds

    def get_obj_cmds(self,postfix):
        return ['python recover_texcoord.py -d {0}/fill_disp_{1}.txt -g ../vt_groundtruth_div.txt > {0}/fill_vt_{1}.txt'.format(self.keys_dir,postfix),
            'python replace_texture.py -i {0} -o {1}/pd_fill_{2}.obj -t {1}/fill_vt_{2}.txt'.format(join(self.sample_dir,'pd_{}_div.obj'.format(self.sample_id)),self.keys_dir,postfix)]

    def get_mean_camera(self):
        cam_path=join(self.sample_dir,'camera.txt')
        cams=np.loadtxt(cam_path)
        print(np.mean(cams,axis=0))

    def get_fill_obj(self,camera_pos,postfix,frame=0,check=True):
        cmds=[]
        cmds+=self.get_disp_cmds(camera_pos,postfix)
        cmds+=self.get_fill_cmds(postfix,frame)
        cmds+=self.get_obj_cmds(postfix)
        run=True
        if check:
            run=self.check_cmds(cmds)
        if run:
            self.run_cmds(cmds)

    def load_center(self):
        center_path=join(self.sample_dir,'center.txt')
        if isfile(center_path):
            self.center=np.loadtxt(center_path)
        else:
            print('compute center...')
            gt_mesh=trimesh.load(join(self.sample_dir,'gt_{}_div.obj'.format(self.sample_id)))
            gt_vertices=np.array(gt_mesh.vertices)
            self.center=np.mean(gt_vertices,axis=0)
            np.savetxt(center_path,self.center)
            print('center',self.center)

    def check_cmds(self,cmds):
        for cmd in cmds:
            print(cmd)
        ans=input('continue to run cmds? (y/n):')
        if ans[0].lower()=='y':
            return True
        else:
            return False

    def run_cmds(self,cmds):
        for cmd in cmds:
            print(cmd)
            if cmd.startswith('cd'):
                path=cmd.split()[1]
                os.chdir(path)
            else:
                ret=os.system(cmd)
                if ret!=0:
                    return

    def write_meshlab_camera(self,out_path,camera_pos,R):
        print('write to',out_path)
        T=np.zeros((4,4))
        T[:3,:3]=R
        T[3,3]=1
        T_str=' '.join([str(v) for v in T.T.reshape(-1)])
        with open(out_path,'w') as f:
            f.write('''<!DOCTYPE ViewState>
<project>
<VCGCamera TranslationVector="{} {} {} 1" LensDistortion="0 0" ViewportPx="1920 1069" PixelSizeMm="0.0369161 0.0369161" CenterPx="960 534" FocalMm="34.1762" RotationMatrix="{}"/>
<ViewSettings NearPlane="1.03109" TrackScale="4.81453" FarPlane="13.0311"/>
<Render Lighting="0" DoubleSideLighting="0" SelectedVert="0" ColorMode="3" SelectedFace="0" BackFaceCull="0" FancyLighting="0" DrawMode="5" TextureMode="3"/>
</project>
        '''.format(-camera_pos[0],-camera_pos[1],-camera_pos[2],T_str))

    def get_circle_fill_objs(self,start_camera_pos,n_cameras):
        max_n_threads=40
        n_threads=min(n_cameras,max_n_threads)
        cx,cy=start_camera_pos[0],start_camera_pos[1]
        cz=self.center[2]
        r=np.abs(start_camera_pos[2]-cz)
        def f(camera_i):
            theta=camera_i/n_cameras*(2*np.pi)
            x=r*np.sin(theta)+cx
            z=r*np.cos(theta)+cz
            camera_pos=np.array([x,cy,z])
            deg=theta/np.pi*180
            deg_str='{}'.format(int(deg))
            meshlab_R=self.get_meshlab_R(camera_pos,np.array([cx,cy,cz]))
            self.write_meshlab_camera(join(self.frames_dir,'meshlab_camera_{}.txt'.format(deg_str)),camera_pos,meshlab_R)
            self.get_fill_obj(camera_pos,postfix=deg_str,frame=camera_i,check=False)
        pool=ProcessPool(nodes=n_threads)
        pool.map(f,range(n_cameras))

    def get_square_fill_objs(self):
        n_threads=4
        cam_path=join(self.sample_dir,'camera.txt')
        cams=np.loadtxt(cam_path)
        def f(camera_i):
            camera_pos=cams[camera_i]
            self.write_meshlab_camera(join(self.keys_dir,'meshlab_camera_{}.txt'.format(camera_i)),camera_pos,np.eye(3))
            self.get_fill_obj(camera_pos,postfix=str(camera_i),frame=camera_i,check=False)
        pool=ProcessPool(nodes=n_threads)
        pool.map(f,range(n_threads))


    def get_meshlab_R(self,camera_pos,view_pos):
        z=-(view_pos-camera_pos)
        z=z/norm(z)
        y=np.array([0,1,0])
        x=np.cross(y,z)
        return np.hstack([x.reshape((-1,1)),y.reshape((-1,1)),z.reshape((-1,1))])

    def blender_pos(self,x):
        return np.array([x[0],-x[2],x[1]])

    def blender_rot(self,q):
        # print('q',q)
        # return Quaternion(axis=[1,0,0],angle=np.pi/2)*q
        return q

    def write_camera_transform(self,out_path,transform):
        x,q=transform
        print('write to',out_path)
        with open(out_path,'w') as f:
            f.write('{} {} {}\n'.format(x[0],x[1],x[2]))
            f.write('{} {} {} {}\n'.format(q[0],q[1],q[2],q[3]))

    def get_key_frame_vt(self,key_frames,keys_dir=None):
        key_frame_vt=[]
        if keys_dir is None:
            keys_dir=self.keys_dir
        for frame_i in key_frames:
            vt_path=join(keys_dir,'fill_vt_{}.txt'.format(frame_i))
            key_frame_vt.append(read_vt(vt_path))
        return key_frame_vt

    # def get_key_frame_vt_from_disp(self,key_frames,keys_dir=None):
    #     key_frame_vt=[]
    #     base_vt=read_vt('../vt_groundtruth_div.txt')
    #     for frame_i in key_frames:
    #         disp_path=join(keys_dir,'fill_disp_{}.txt'.format(frame_i))
    #         key_frame_vt.append(np.loadtxt(disp_path)+base_vt)
    #     return key_frame_vt

    def get_square_blender_dir(self,out_dir,n_total_frames,use_key_frames=True):
        if not isdir(out_dir):
            os.makedirs(out_dir)
        n_key_frames=4
        with open(join(out_dir,'last_frame.txt'),'w') as f:
            f.write('{} {}\n'.format(n_key_frames,n_total_frames))
        if use_key_frames:
            key_frame_vt=self.get_key_frame_vt(range(n_key_frames))
        else:
            camera_pos_list=[]
        total_length=(2+2*np.sqrt(2))
        cam_path=join(self.sample_dir,'camera.txt')
        cameras=np.loadtxt(cam_path)

        for i in range(n_total_frames):
            t=i/n_total_frames*total_length
            if t<1:
                seg_t=t
                camera_pos=(1-seg_t)*cameras[0]+seg_t*cameras[1]
                w=np.array([1-seg_t,seg_t,0,0])
            elif t<1+np.sqrt(2):
                seg_t=(t-1)/np.sqrt(2)
                camera_pos=(1-seg_t)*cameras[1]+seg_t*cameras[2]
                w=np.array([seg_t*(1-seg_t),(1-seg_t)*(1-seg_t),seg_t*seg_t,(1-seg_t)*seg_t])
            elif t<2+np.sqrt(2):
                seg_t=t-(1+np.sqrt(2))
                camera_pos=(1-seg_t)*cameras[2]+seg_t*cameras[3]
                w=np.array([0,0,1-seg_t,seg_t])
            else:
                seg_t=(t-(2+np.sqrt(2)))/np.sqrt(2)
                camera_pos=(1-seg_t)*cameras[3]+seg_t*cameras[0]
                w=np.array([seg_t*seg_t,(1-seg_t)*seg_t,seg_t*(1-seg_t),(1-seg_t)*(1-seg_t)])

            if use_key_frames:
                vt=sum([key_frame_vt[i]*w[i] for i in range(4)])            
                out_path=join(out_dir,'frame_{}.obj'.format(i))
                print('write to',out_path)
                write_obj(Obj(v=self.pd_v,f=self.fcs,vt=vt),out_path)
            else:
                camera_pos_list.append(camera_pos)

            camera_transform=self.blender_pos(camera_pos),np.array([1,0,0,0])
            out_path=join(out_dir,'cam_{}.txt'.format(i))
            # print('write to',out_path)
            self.write_camera_transform(out_path,camera_transform)

        if not use_key_frames:
            n_threads=30
            def f(frame_i):
                self.get_fill_obj(camera_pos_list[frame_i],postfix=str(frame_i),frame=frame_i,check=False)
            pool=ProcessPool(nodes=n_threads)
            pool.map(f,range(n_total_frames))

    def get_square_tsnn_disp(self,keys_dir):
        if not isdir(keys_dir):
            os.makedirs(keys_dir)
        cmds=[]
        for i in range(4):
            cmds+=['cp /data/zhenglin/dataset_subdivision/tsnn_output/test_result_{}/displace_{:08d}.txt {}'.format(i+1,self.sample_id,join(keys_dir,'fill_disp_{}.txt'.format(i))),
                'python recover_texcoord.py -d {0}/fill_disp_{1}.txt -g ../vt_groundtruth_div.txt > {0}/fill_vt_{1}.txt'.format(keys_dir,i)]
        self.run_cmds(cmds)

    def get_grid_fill_objs(self,size):
        max_n_threads=40
        n_rows,n_cols=size
        n_cameras=n_rows*n_cols
        # print('n_cameras',n_cameras)
        n_threads=min(n_cameras,max_n_threads)
        cam_path=join(self.sample_dir,'camera.txt')
        cams=np.loadtxt(cam_path)
        def f(camera_i):
            row_i=camera_i//n_cols
            col_i=camera_i%n_cols
            y=row_i/(n_rows-1)
            x=col_i/(n_cols-1)
            camera_pos=(1-x)*(1-y)*cams[3]+x*(1-y)*cams[2]+(1-x)*y*cams[1]+x*y*cams[0]
            self.write_meshlab_camera(join(self.keys_dir,'meshlab_camera_{}.txt'.format(camera_i)),camera_pos,np.eye(3))
            self.get_fill_obj(camera_pos,postfix=str(camera_i),frame=camera_i,check=False)
        pool=ProcessPool(nodes=n_threads)
        pool.map(f,range(n_cameras))

    def get_grid_blender_dir(self,out_dir,size,keys_dir=None):
        if not isdir(out_dir):
            os.makedirs(out_dir)
        n_rows,n_cols=size
        n_cameras=n_rows*n_cols
        n_key_frames=4
        n_total_frames=n_cameras
        cam_path=join(self.sample_dir,'camera.txt')
        cams=np.loadtxt(cam_path)
        if keys_dir is None:
            key_frame_vt=self.get_key_frame_vt([n_rows*n_cols-1,n_rows*n_cols-n_cols,n_cols-1,0],keys_dir=keys_dir)
        else:
            key_frame_vt=self.get_key_frame_vt(range(4),keys_dir=keys_dir)
        with open(join(out_dir,'last_frame.txt'),'w') as f:
            f.write('{} {}\n'.format(n_key_frames,n_total_frames))
        for camera_i in range(n_cameras):
            row_i=camera_i//n_cols
            col_i=camera_i%n_cols
            y=row_i/(n_rows-1)
            x=col_i/(n_cols-1)
            w=np.array([x*y,(1-x)*y,x*(1-y),(1-x)*(1-y)])
            camera_pos=sum([cams[i]*w[i] for i in range(4)])
            vt=sum([key_frame_vt[i]*w[i] for i in range(4)])
            out_path=join(out_dir,'fill_vt_{}.txt'.format(camera_i))
            print('write to',out_path)
            np.savetxt(out_path,vt)
            out_path=join(out_dir,'frame_{}.obj'.format(camera_i))
            print('write to',out_path)
            write_obj(Obj(v=self.pd_v,f=self.fcs,vt=vt),out_path)

            camera_transform=self.blender_pos(camera_pos),np.array([1,0,0,0])
            out_path=join(out_dir,'cam_{}.txt'.format(camera_i))
            self.write_camera_transform(out_path,camera_transform)

    def get_grid_vt_error(self,gt_dir,interp_dir,out_path,size):
        n_rows,n_cols=size
        n_cameras=n_rows*n_cols
        error=np.zeros(size)
        for camera_i in range(n_cameras):
            row_i=camera_i//n_cols
            col_i=camera_i%n_cols
            gt_vt=read_vt(join(gt_dir,'fill_vt_{}.txt'.format(camera_i)))
            interp_vt=np.loadtxt(join(interp_dir,'fill_vt_{}.txt'.format(camera_i)))
            error[row_i][col_i]=np.sqrt(np.mean(np.sum((gt_vt-interp_vt)**2,axis=1)))
        print('write to',out_path)
        np.savetxt(out_path,error)

    def get_grid_uv_imgs(self,blender_dir,uv_type='gt'):
        if uv_type=='gt':
            in_pattern=join(abspath(self.sample_dir),'gt_{}_div.obj'.format(self.sample_id))
        elif uv_type=='pd':
            in_pattern=join(abspath(self.sample_dir),'pd_{}_div.obj'.format(self.sample_id))
        elif uv_type=='gt_interp':
            in_pattern=join(abspath(blender_dir),'frame_{}.obj')
        elif uv_type=='pd_interp':
            in_pattern=join(abspath(blender_dir),'tsnn/frame_{}.obj')
        else:
            assert(False)
        out_dir=join(blender_dir,'{}_imgs'.format(uv_type))
        if not isdir(out_dir):
            os.makedirs(out_dir)
        n_frames=self.read_n_frames(blender_dir)
        cwd=os.getcwd()
        cmds=['cd ../../figure_scripts']
        cmds.append('python draw_blender_dir.py -in_pattern {} -out_pattern {} -cam_pattern {} -n_frames {}'.format(in_pattern,join(abspath(out_dir),'img_{}.npy'),join(abspath(blender_dir),'cam_{}.txt'),n_frames))
        self.run_cmds(cmds)
        os.chdir(cwd)

    def get_grid_pix_error(self,blender_dir,size,uv_type='pd'):
        n_rows,n_cols=size
        n_cameras=n_rows*n_cols
        error=np.zeros(size)
        for camera_i in range(n_cameras):
            row_i=camera_i//n_cols
            col_i=camera_i%n_cols
            gt_img=np.load(join(blender_dir,'gt_imgs','img_{}.npy'.format(camera_i)))
            pd_img=np.load(join(blender_dir,'{}_imgs'.format(uv_type),'img_{}.npy'.format(camera_i)))
            error[row_i][col_i]=compute_pixel_error(gt_img,pd_img)
        out_path=join(blender_dir,'error_{}.txt'.format(uv_type))
        print('write to',out_path)
        np.savetxt(out_path,error)

    def read_n_frames(self,blender_dir):
        with open(join(blender_dir,'last_frame.txt')) as f:
            line=f.readline()
            parts=line.split()
            n_frames=int(parts[1])
        return n_frames

    def get_circle_key_frame_vt(self,n_key_frames):
        key_frame_vt=[]
        for frame_i in range(n_key_frames):
            deg=int(360*frame_i/n_key_frames)
            vt_path=join(self.frames_dir,'fill_vt_{}.txt'.format(deg))
            key_frame_vt.append(read_vt(vt_path))
        return key_frame_vt

    def get_blender_camera_transform(self,camera_pos,view_pos):
        camera_pos=self.blender_pos(camera_pos)
        view_pos=self.blender_pos(view_pos)

        z_axis=-(view_pos-camera_pos)
        z_axis=z_axis/norm(z_axis)
        # y_axis=np.array([0,1,0])
        y_axis=np.array([0,0,1])
        x_axis=np.cross(y_axis,z_axis)
        R=np.zeros((3,3))
        R[:,0]=x_axis
        R[:,1]=y_axis
        R[:,2]=z_axis
        q=Quaternion(matrix=R)
        return camera_pos,self.blender_rot(q)

    def get_circle_camera_transform(self,theta,start_camera_pos):
        cx,cy=start_camera_pos[0],start_camera_pos[1]
        cz=self.center[2]
        r=np.abs(start_camera_pos[2]-cz)
        x=r*np.sin(theta)+cx
        z=r*np.cos(theta)+cz
        camera_pos=np.array([x,cy,z])
        view_pos=np.array([cx,cy,cz])
        return self.get_blender_camera_transform(camera_pos,view_pos)

    def get_circle_blender_dir(self,out_dir,n_key_frames,n_total_frames,start_camera_pos):
        if not isdir(out_dir):
            os.makedirs(out_dir)
        with open(join(out_dir,'last_frame.txt'),'w') as f:
            f.write('{} {}\n'.format(n_key_frames,n_total_frames))
        key_frame_vt=self.get_circle_key_frame_vt(n_key_frames)
        for i in range(n_total_frames):
            ratio=i*n_key_frames/n_total_frames
            key_frame_0=int(ratio)
            key_frame_1=(key_frame_0+1)%n_key_frames
            t=ratio-key_frame_0
            vt_0=key_frame_vt[key_frame_0]
            vt_1=key_frame_vt[key_frame_1]
            vt=vt_0*(1-t)+vt_1*t
            # out_path=join(out_dir,'vt_{}.txt'.format(i))
            # print('write to',out_path)
            # write_vt(out_path,vt)
            # out_path=join(out_dir,'uv_{}.txt'.format(i))
            # print('write to',out_path)
            # write_uv(out_path,vt,self.fcs)
            out_path=join(out_dir,'frame_{}.obj'.format(i))
            print('write to',out_path)
            write_obj(Obj(v=self.pd_v,f=self.fcs,vt=vt),out_path)

            theta=i/n_total_frames*(2*np.pi)
            camera_transform=self.get_circle_camera_transform(theta,start_camera_pos)
            out_path=join(out_dir,'cam_{}.txt'.format(i))
            # print('write to',out_path)
            self.write_camera_transform(out_path,camera_transform)


if __name__=='__main__':
    test=CameraTest(sample_id=5983,camera_type='_square_gt')
    # camera_pos=np.array([0.0437965,0.705956,0.787657])
    # test.get_fill_obj(camera_pos,'0',0)
    # test.get_circle_fill_objs(camera_pos,36)
    # test.get_blender_dir(join(test.sample_dir,'blender_4_120'),4,120,camera_pos)

    # test.get_square_fill_objs()
    test.get_square_blender_dir(join(test.sample_dir,'blender_square_120_gt'),120,use_key_frames=False)
    # grid
    # blender_dir=join(test.sample_dir,'blender_grid10x10')
    # test.get_grid_fill_objs(test.grid_size)
    # test.get_grid_blender_dir(blender_dir,test.grid_size)
    # test.get_square_tsnn_disp(join(blender_dir,'tsnn_keys'))
    # test.get_grid_blender_dir(join(blender_dir,'tsnn'),test.grid_size,keys_dir=join(blender_dir,'tsnn_keys'))
    # test.get_grid_uv_imgs(blender_dir,uv_type='gt')
    # test.get_grid_uv_imgs(blender_dir,uv_type='pd')
    # test.get_grid_uv_imgs(blender_dir,uv_type='gt_interp')
    # test.get_grid_uv_imgs(blender_dir,uv_type='pd_interp')
    # test.get_grid_pix_error(blender_dir,test.grid_size,uv_type='pd')
    # test.get_grid_pix_error(blender_dir,test.grid_size,uv_type='gt_interp')
    # test.get_grid_pix_error(blender_dir,test.grid_size,uv_type='pd_interp')
    # test.get_grid_error(test.keys_dir,join(test.sample_dir,'blender_grid10x10'),join(test.sample_dir,'blender_grid10x10/grid_error.txt'),test.grid_size)
    # test.get_mean_camera()
