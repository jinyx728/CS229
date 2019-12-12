######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from PIL import Image
from obj_io import Obj, read_obj
# from scipy.misc import imresize
import cv2
import argparse
import sys
import gzip

TRY_FRAMEBUFFER = True
GENERATE_TEXTURE_ID_PROBLEM = False  # Follow this global variable to see the error in texture ID generation

SIDE = 512  # window size
# OUTPUT_SIZE=512
OUTPUT_SIZE=SIDE


# Utility functions
def float_size(n=1):
    return sizeof(ctypes.c_float) * n


def float_pointer_offset(n=0):
    return ctypes.c_void_p(float_size(n))

def get_color_map():
    color_map=np.zeros((256,3))
    red=np.array([1,0,0])
    blue=np.array([0,0,1])
    white=np.array([0.9,.9,0.9])
    for i in range(0,128):
        a=i/256*2
        color_map[i]=blue*(1-a)+white*a
    for i in range(128,256):
        a=i/256*2
        color_map[i]=red*(a-1)+white*(2-a)
    # color_map[0]=np.array([0,0,0])
    return (color_map*255).astype(np.uint8)

def save_png_img(img_path,data,mask=None):
    # data: range(0,1), size (H,W,D)
    img=np.clip((data*255).astype(np.uint8),0,255)
    img=np.flip(img,axis=0)
    H,W,D=img.shape
    img=np.transpose(img,(0,2,1)).reshape((H,W*D))
    color_map=get_color_map()
    img=color_map[img]
    if mask is not None:
        img=img*mask+255*(1-mask)
    Image.fromarray(img).save(img_path)

def get_vts_normalize_m(vts):
    xmin,xmax=np.min(vts[:,0]),np.max(vts[:,0])
    ymin,ymax=np.min(vts[:,1]),np.max(vts[:,1])
    ymin-=0.1
    ymax+=0.1
    xcenter=(xmin+xmax)/2
    ycenter=(ymin+ymax)/2
    m=np.array([[1,0,-xcenter],
                [0,1,-ycenter]])
    m=m*2/(ymax-ymin)
    return m

def normalize_vts(vts,m):
    vts=np.hstack([vts,np.ones((len(vts),1))])
    return vts.dot(m.T)

# def normalize_vts(vts):
#     xyzmin,xyzmax=np.min(vts,axis=0),np.max(vts,axis=0)
#     ymin,ymax=xyzmin[1],xyzmax[1]
#     ymin-=0.1
#     ymax+=0.1
#     xcenter=(xyzmin[0]+xyzmax[0])/2
#     normalized_vts=vts.copy()
#     normalized_vts[:,0]=(vts[:,0]-xcenter)/(ymax-ymin)*2
#     normalized_vts[:,1]=(vts[:,1]-ymin)/(ymax-ymin)*2-1
#     return normalized_vts

def get_offsets_minmax(offsets,separate_axis=False):
    if separate_axis:
        offsets_min,offsets_max=np.min(offsets,axis=0),np.max(offsets,axis=0)
    else:
        offsets_min,offsets_max=np.min(offsets),np.max(offsets)
    return offsets_min,offsets_max

def normalize_offsets(offsets,offsets_min,offsets_max):
    offsets=(offsets-offsets_min)/(offsets_max-offsets_min)
    return offsets

def denormalize_offsets(offsets,offsets_min,offsets_max):
    if offsets_min.size==3 and offsets_max.size==3:
        offsets_min_repeat=np.tile(offsets_min,2).astype(np.float32)
        offsets_max_repeat=np.tile(offsets_max,2).astype(np.float32)
        return offsets*(offsets_max_repeat-offsets_min_repeat)+offsets_min_repeat
    else:
        return offsets*(offsets_max-offsets_min)+offsets_min

def load_crpds(crpds_path):
    raw_crpds=np.loadtxt(crpds_path).astype(int)
    crpds=np.zeros_like(raw_crpds,dtype=int)
    for i in range(len(raw_crpds)):
        raw_c=raw_crpds[i]
        if raw_c[0]<raw_c[1]:
            crpds[i][0]=raw_c[0]
            crpds[i][1]=raw_c[1]
        else:
            crpds[i][0]=raw_c[1]
            crpds[i][1]=raw_c[0]
    return crpds

def get_png_mask(mask):
    H,W=mask.shape
    mask=np.flip(mask,axis=0)
    mask=np.tile(np.expand_dims(np.uint8(mask),2),(1,3,3))
    return mask

def get_png_mask_both_sides(mask):
    front_mask,back_mask=mask[:,:,0],mask[:,:,1]
    front_png_mask=get_png_mask(front_mask)
    back_png_mask=get_png_mask(back_mask)
    return np.concatenate([front_png_mask,back_png_mask],axis=1)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-test_case',choices=['lowres','midres','local','lowres_skin','midres_skin','lowres_tshirt','midres_uhat','midres_vhat','midres_nhat','midres_fcs','lowres_fcs','highres_tex'],default='local')
    parser.add_argument('-mask',action='store_true')
    parser.add_argument('-start',default=10,type=int)
    parser.add_argument('-end',default=10,type=int)
    parser.add_argument('-data_root_dir',default='/phoenix/yxjin/poses_v3')
    parser.add_argument('-offset_dir')
    parser.add_argument('-out_dir')
    parser.add_argument('-vlz',action='store_true')
    parser.add_argument('-no_pad',action='store_true')
    parser.add_argument('-use_uvn',action='store_true')
    parser.add_argument('-gzip',action='store_true')
    parser.add_argument('-side',type=int,default=512)
    args=parser.parse_args()
    test_case=args.test_case

    SIDE=args.side

    def create_shader(vertex_shader, fragment_shader):
        vs_id = GLuint(glCreateShader(GL_VERTEX_SHADER))  # shader id for vertex shader
        glShaderSource(vs_id, [vertex_shader], None)  # Send the code of the shader
        glCompileShader(vs_id)  # compile the shader code
        status = glGetShaderiv(vs_id, GL_COMPILE_STATUS)
        if status != 1:
            print('VERTEX SHADER ERROR')
            print(glGetShaderInfoLog(vs_id).decode())

        fs_id = GLuint(glCreateShader(GL_FRAGMENT_SHADER))
        glShaderSource(fs_id, [fragment_shader], None)
        glCompileShader(fs_id)
        status = glGetShaderiv(fs_id, GL_COMPILE_STATUS)
        if status != 1:
            print('FRAGMENT SHADER ERROR')
            print(glGetShaderInfoLog(fs_id).decode())

        # Link the shaders into a single program
        program_id = GLuint(glCreateProgram())
        glAttachShader(program_id, vs_id)
        glAttachShader(program_id, fs_id)
        glLinkProgram(program_id)
        status = glGetProgramiv(program_id, GL_LINK_STATUS)
        if status != 1:
            print('status', status)
            print('SHADER PROGRAM', glGetShaderInfoLog(program_id))

        glDeleteShader(vs_id)
        glDeleteShader(fs_id)

        return program_id


    # Initialize project

    def init_gl():
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH| GLUT_3_2_CORE_PROFILE)
        glutInitWindowSize(SIDE, SIDE)
        glutCreateWindow('GLUT Framebuffer Test')

        glDisable( GL_LINE_SMOOTH );
        glDisable( GL_POLYGON_SMOOTH );
        glDisable( GL_MULTISAMPLE );

        # OpenGL version info
        renderer = glGetString(GL_RENDERER)
        version = glGetString(GL_VERSION)
        print('Renderer:', renderer)  # Renderer: b'Intel Iris Pro OpenGL Engine'
        print('OpenGL version supported: ', version)  # OpenGL version supported:  b'4.1 INTEL-10.12.13'


    vertex_shader = """#version 410
    precision highp float;
    layout(location = 0) in highp vec3 pos;
    layout(location = 1) in highp vec3 col;
    out highp vec3 fg_color;
    void main () {
        fg_color = col;
        gl_Position =vec4(pos, 1.0f);
    }"""

    fragment_shader = """#version 410
    precision highp float;
    in highp vec3 fg_color;
    out highp vec4 color;
    void main () {
        color = vec4(fg_color, 1.);
    }"""

    # color = vec4(fg_color, 1.);


    def make_vbo(vts_3d,attr):
        data=np.concatenate([vts_3d,attr],axis=1).astype(np.float32)
        data=np.ascontiguousarray(data.reshape(-1),dtype=np.float32)
        vbo=GLuint()
        glGenBuffers(1,vbo)
        glBindBuffer(GL_ARRAY_BUFFER,vbo)
        glBufferData(GL_ARRAY_BUFFER, float_size(len(data)), data, GL_STATIC_DRAW)
        glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,float_size(6),float_pointer_offset(0))
        glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,float_size(6),float_pointer_offset(3))
        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)
        glBindBuffer(GL_ARRAY_BUFFER,0)
        return vbo

    def make_vao(vts_3d,attr):
        vao=GLuint()
        glGenVertexArrays(1, vao)
        glBindVertexArray(vao)
        vbo=make_vbo(vts_3d,attr)
        # ebo=make_ebo(fcs)
        # ebo=None
        glBindVertexArray(0)
        # return vao,vbo,ebo
        return vao

    def make_fbo():
        fbo = GLuint()
        glGenFramebuffers(1, fbo)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        return fbo

    def make_fbo_texture():
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, SIDE, SIDE, 0, GL_RGB, GL_FLOAT, None)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)
        return texture

    def read_img():
        data=glReadPixels(0,0,SIDE,SIDE,GL_RGB,GL_FLOAT)
        # data=imresize(data,size=(OUTPUT_SIZE,OUTPUT_SIZE,3),interp='bicubic')
        # data=cv2.resize(data,(OUTPUT_SIZE,OUTPUT_SIZE))
        return data

    init_gl()
    main_shader = create_shader(vertex_shader, fragment_shader)
    fbo=make_fbo()
    texture=make_fbo_texture()

    if not glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE:
        print('framebuffer binding failed')
        # exit()
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glBindTexture(GL_TEXTURE_2D, 0)

    # load vts_3d, colors, fcs
    if test_case.find("lowres")!=-1:
        shared_data_dir='../../shared_data'
    elif test_case.find('midres')!=-1:
        shared_data_dir='../../shared_data_midres'
    elif test_case.find('highres')!=-1:
        shared_data_dir='../../shared_data_highres'

    use_tex=False
    if test_case.find('_tex')!=-1:
        use_tex=True

    front_obj_path=os.path.join(shared_data_dir,'flat_tshirt_front.obj')
    front_obj=read_obj(front_obj_path)
    front_vts,front_fcs=front_obj.v,front_obj.f
    front_vts_normalize_m=get_vts_normalize_m(front_vts)
    if args.no_pad:
        front_vts=normalize_vts(front_vts[:,:2],front_vts_normalize_m)

    front_obj_w_pad_path=os.path.join(shared_data_dir,'front_tshirt_w_pad.obj')
    front_obj_w_pad=read_obj(front_obj_w_pad_path)
    front_vts_w_pad,front_fcs_w_pad=front_obj_w_pad.v,front_obj_w_pad.f
    front_vts_w_pad=normalize_vts(front_vts_w_pad[:,:2],front_vts_normalize_m)

    n_vts=len(front_vts)

    back_obj_path=os.path.join(shared_data_dir,'flat_tshirt_back.obj')
    back_obj=read_obj(back_obj_path)
    back_vts,back_fcs=back_obj.v,back_obj.f
    back_vts_normalize_m=get_vts_normalize_m(back_vts)
    if args.no_pad:
        back_vts=normalize_vts(back_vts[:,:2],back_vts_normalize_m)

    back_obj_w_pad_path=os.path.join(shared_data_dir,'back_tshirt_w_pad.obj')
    back_obj_w_pad=read_obj(back_obj_w_pad_path)
    back_vts_w_pad,back_fcs_w_pad=back_obj_w_pad.v,back_obj_w_pad.f
    back_vts_w_pad=normalize_vts(back_vts_w_pad[:,:2],back_vts_normalize_m)

    front_crpds_path=os.path.join(shared_data_dir,'front_vert_crpds.txt')
    front_crpds=load_crpds(front_crpds_path)
    back_crpds_path=os.path.join(shared_data_dir,'back_vert_crpds.txt')
    back_crpds=load_crpds(back_crpds_path)

    front_indices=np.ascontiguousarray(front_fcs.reshape(-1).astype(np.uint32))
    back_indices=np.ascontiguousarray(back_fcs.reshape(-1).astype(np.uint32))
    front_indices_w_pad=np.ascontiguousarray(front_fcs_w_pad.reshape(-1).astype(np.uint32))
    back_indices_w_pad=np.ascontiguousarray(back_fcs_w_pad.reshape(-1).astype(np.uint32))

    # test case default values
    if test_case=='local':
        offset_dir='offset_test'
        out_dir='offset_test'
    elif test_case=='lowres':
        if args.use_uvn:
            offset_dir=os.path.join(args.data_root_dir,'lowres_skin_tshirt_nuvs')
            out_dir=os.path.join(args.data_root_dir,'lowres_uvn_offset_imgs_{}'.format(SIDE))
        else:
            offset_dir=os.path.join(args.data_root_dir,'lowres_offset_npys')
            out_dir=os.path.join(args.data_root_dir,'lowres_offset_imgs_{}'.format(SIDE))
    elif test_case=='lowres_skin':
        offset_dir=os.path.join(args.data_root_dir,'lowres_skin_npys')
        out_dir=os.path.join(args.data_root_dir,'lowres_skin_imgs_{}'.format(SIDE))
    elif test_case=='lowres_tshirt':
        offset_dir=os.path.join(args.data_root_dir,'lowres_tshirt_npys')
        out_dir=os.path.join(args.data_root_dir,'lowres_tshirt_imgs_{}'.format(SIDE))
    elif test_case=='lowres_fcs':
        offset_dir=os.path.join(args.data_root_dir,'lowres_offset_npys') # not used, what ever...
        out_dir=shared_data_dir
    elif test_case=='midres':
        if args.use_uvn:
            offset_dir=os.path.join(args.data_root_dir,'midres_skin_tshirt_nuvs')
            out_dir=os.path.join(args.data_root_dir,'midres_uvn_offset_imgs_{}'.format(SIDE))
        else:
            offset_dir=os.path.join(args.data_root_dir,'midres_offset_npys')
            out_dir=os.path.join(args.data_root_dir,'midres_offset_imgs_{}'.format(SIDE))
    elif test_case=='midres_fcs':
        offset_dir=os.path.join(args.data_root_dir,'midres_offset_npys') # not used, what ever...
        out_dir=shared_data_dir
    elif test_case=='midres_skin':
        offset_dir=os.path.join(args.data_root_dir,'midres_skin_npys')
        out_dir=os.path.join(args.data_root_dir,'midres_skin_imgs_{}'.format(SIDE))
    elif test_case=='midres_uhat':
        offset_dir=os.path.join(args.data_root_dir,'midres_skin_tshirt_nuvs')
        out_dir=os.path.join(args.data_root_dir,'midres_uhat_imgs_{}'.format(SIDE))
    elif test_case=='midres_vhat':
        offset_dir=os.path.join(args.data_root_dir,'midres_skin_tshirt_nuvs')
        out_dir=os.path.join(args.data_root_dir,'midres_vhat_imgs_{}'.format(SIDE))
    elif test_case=='midres_nhat':
        offset_dir=os.path.join(args.data_root_dir,'midres_skin_tshirt_nuvs')
        out_dir=os.path.join(args.data_root_dir,'midres_nhat_imgs_{}'.format(SIDE))
    elif test_case=='highres_tex':
        offset_dir=os.path.join(args.data_root_dir,'highres_texture_txt')
        out_dir=os.path.join(args.data_root_dir,'highres_texture_imgs_{}'.format(SIDE))
        
    # overwrite from argparse
    if args.out_dir is not None:
        out_dir=args.out_dir
    if args.offset_dir is not None:
        offset_dir=args.offset_dir

    print('offset_dir',offset_dir,'out_dir',out_dir)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    mask=None
    if args.no_pad:
        mask_path=os.path.join(shared_data_dir,'offset_img_mask_no_pad_{}.npy'.format(SIDE))
    else:
        mask_path=os.path.join(shared_data_dir,'offset_img_mask_{}.npy'.format(SIDE))
    if os.path.isfile(mask_path):
        mask=np.load(mask_path)
    else:
        print(mask_path,'does not exist')

    def draw(vao,indices):
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        # glEnable(GL_DEPTH_TEST)  # https://www.khronos.org/opengles/sdk/docs/man/xhtml/glEnable.xml
        glClearColor(0., 0., 0., 0.)
        glClearDepth(1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClear(GL_COLOR_BUFFER_BIT)

        glBindVertexArray(vao)
        glUseProgram(main_shader)
        glDrawElements(GL_TRIANGLES, indices.size, GL_UNSIGNED_INT,indices)

        glBindVertexArray(0)


    def process_id(test_id,front_vts_w_pad,back_vts_w_pad,front_crpds,back_crpds,front_indices_w_pad,back_indices_w_pad,mask=None,use_uvn=False):
        front_vts_w_pad=np.hstack([front_vts_w_pad,np.zeros((len(front_vts_w_pad),1))])
        back_vts_w_pad=np.hstack([back_vts_w_pad,np.zeros((len(back_vts_w_pad),1))])

        if test_id==-1:# make map
            print('render mask')
            offsets=np.ones((len(front_vts_w_pad),3))
            front_offsets=np.ones((len(front_vts_w_pad),3))
            back_offsets=np.ones((len(back_vts_w_pad),3))
            offsets_min=1
            offsets_max=1
            mask=None
        elif use_uvn:            
            if test_id%500 == 0:
                print('render uvn',test_id)
            if test_case=='midres_uhat':
                raw_front_offsets=np.load(os.path.join(offset_dir,'front_uhats_{:08d}.npy'.format(test_id)))
                raw_back_offsets=np.load(os.path.join(offset_dir,'back_uhats_{:08d}.npy'.format(test_id)))
            elif test_case=='midres_vhat':
                raw_front_offsets=np.load(os.path.join(offset_dir,'front_vhats_{:08d}.npy'.format(test_id)))
                raw_back_offsets=np.load(os.path.join(offset_dir,'back_vhats_{:08d}.npy'.format(test_id)))
            elif test_case=='midres_nhat':
                raw_front_offsets=np.load(os.path.join(offset_dir,'front_nhats_{:08d}.npy'.format(test_id)))
                raw_back_offsets=np.load(os.path.join(offset_dir,'back_nhats_{:08d}.npy'.format(test_id)))
            else:
                raw_front_offsets=np.load(os.path.join(offset_dir,'front_uvn_offset_{:08d}.npy'.format(test_id)))
                raw_back_offsets=np.load(os.path.join(offset_dir,'back_uvn_offset_{:08d}.npy'.format(test_id)))

            front_offsets_min,front_offsets_max=get_offsets_minmax(raw_front_offsets)
            back_offsets_min,back_offsets_max=get_offsets_minmax(raw_back_offsets)
            offsets_min=np.min([front_offsets_min,back_offsets_min])
            offsets_max=np.max([front_offsets_max,back_offsets_max])
            raw_front_offsets=normalize_offsets(raw_front_offsets,offsets_min,offsets_max)
            raw_back_offsets=normalize_offsets(raw_back_offsets,offsets_min,offsets_max)

            front_offsets=np.zeros_like(front_vts_w_pad)
            front_offsets[:len(raw_front_offsets),:]=raw_front_offsets
            front_offsets[front_crpds[:,1]]=raw_front_offsets[front_crpds[:,0]]

            back_offsets=np.zeros_like(back_vts_w_pad)
            back_offsets[:len(raw_back_offsets),:]=raw_back_offsets
            back_offsets[back_crpds[:,1]]=raw_back_offsets[back_crpds[:,0]]

        else:
            if test_case=='lowres_skin' or test_case=='midres_skin':
                offset_path=os.path.join(offset_dir,'skin_{:08d}.npy'.format(test_id))
                offsets=np.load(offset_path)
                offsets_min,offsets_max=get_offsets_minmax(offsets,separate_axis=False)
            elif test_case=='lowres_tshirt':
                offset_path=os.path.join(offset_dir,'tshirt_{:08d}.npy'.format(test_id))
                offsets=np.load(offset_path)
                offsets_min,offsets_max=get_offsets_minmax(offsets,separate_axis=True)
            elif test_case=='highres_tex':
                offset_path=os.path.join(offset_dir,'displace_{:08d}.txt'.format(test_id))
                offsets=np.loadtxt(offset_path)
                n_vts=len(offsets)
                offsets=np.hstack([offsets,np.zeros((n_vts,1))])
                offsets_min,offsets_max=get_offsets_minmax(offsets)
            else:
                offset_path=os.path.join(offset_dir,'offset_{:08d}.npy'.format(test_id))
                offsets=np.load(offset_path)
                offsets_min,offsets_max=get_offsets_minmax(offsets)

            offsets=normalize_offsets(offsets,offsets_min,offsets_max)
            offsets=offsets.astype(np.float32)

            front_offsets=np.zeros_like(front_vts_w_pad)
            front_offsets[:len(offsets)]=offsets
            if len(front_crpds)>0:
                front_offsets[front_crpds[:,1]]=offsets[front_crpds[:,0]]

            back_offsets=np.zeros_like(back_vts_w_pad)
            back_offsets[:len(offsets)]=offsets
            if len(back_crpds)>0:
                back_offsets[back_crpds[:,1]]=offsets[back_crpds[:,0]]

        if test_case=="midres_fcs" or test_case=="lowres_fcs":
            n_front_fcs_w_pad=len(front_fcs_w_pad)
            front_vts_3d=front_vts_w_pad[front_fcs_w_pad.reshape(-1)]
            front_attr=np.array(range(n_front_fcs_w_pad)).reshape(-1,1)
            front_attr=np.tile(front_attr,(1,9)).reshape(-1,3)/n_front_fcs_w_pad
            front_indices_draw=np.array(range(3*n_front_fcs_w_pad)).reshape(-1,3)
            front_indices_draw=np.ascontiguousarray(front_indices_draw.astype(np.uint32).reshape(-1))
            front_vao=make_vao(front_vts_3d,front_attr)
            draw(front_vao,front_indices_draw)
            front_data=read_img()
            glBindFramebuffer(GL_FRAMEBUFFER, 0)

            n_back_fcs_w_pad=len(back_fcs_w_pad)
            back_vts_3d=back_vts_w_pad[back_fcs_w_pad.reshape(-1)]
            back_attr=np.array(range(n_back_fcs_w_pad)).reshape(-1,1)
            back_attr=np.tile(back_attr,(1,9)).reshape(-1,3)/n_back_fcs_w_pad
            back_indices_draw=np.array(range(3*n_back_fcs_w_pad)).reshape(-1,3)
            back_indices_draw=np.ascontiguousarray(back_indices_draw.astype(np.uint32).reshape(-1))
            back_vao=make_vao(back_vts_3d,back_attr)
            draw(back_vao,back_indices_draw)
            back_data=read_img()
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
        else:
            front_vao=make_vao(front_vts_w_pad,front_offsets)
            draw(front_vao,front_indices_w_pad)
            front_data=read_img()
            glBindFramebuffer(GL_FRAMEBUFFER, 0)

            back_vao=make_vao(back_vts_w_pad,back_offsets)
            draw(back_vao,back_indices_w_pad)
            back_data=read_img()
            glBindFramebuffer(GL_FRAMEBUFFER, 0)

        data=np.concatenate([front_data,back_data],axis=2)

        if mask is not None:
            front_mask=mask[:,:,0]
            back_mask=mask[:,:,1]

        if args.vlz:
            if mask is None:
                png_mask=None
            else:
                front_png_mask=get_png_mask(front_mask)
                back_png_mask=get_png_mask(back_mask)
                png_mask=np.concatenate([front_png_mask,back_png_mask],axis=1)

            print('vlz in test.png')
            save_png_img('test.png',data,mask=png_mask)

        if test_id==-1:#mask
            data=np.concatenate([data[:,:,0:1],data[:,:,3:4]],axis=2)
            data=np.around(data)
            if args.no_pad:
                mask_path=os.path.join(shared_data_dir,'offset_img_mask_no_pad_{}.npy'.format(SIDE))
            else:
                mask_path=os.path.join(shared_data_dir,'offset_img_mask_{}.npy'.format(SIDE))
            print('save to path',mask_path)
            np.save(mask_path,data)
        else:
            if test_case=="midres_fcs" or test_case=="lowres_fcs":
                data[:,:,0]=np.around(data[:,:,0]*n_front_fcs_w_pad)*front_mask
                data[:,:,3]=np.around(data[:,:,3]*n_back_fcs_w_pad)*back_mask
                # print('min',np.min(data[:,:,0]),np.min(data[:,:,1]),'max',np.max(data[:,:,0]),np.max(data[:,:,1]))
            else:            
                data=denormalize_offsets(data,offsets_min,offsets_max)

            data[:,:,:3]*=np.expand_dims(front_mask,axis=2)
            data[:,:,3:6]*=np.expand_dims(back_mask,axis=2)
            if use_tex:
                data=np.concatenate([data[:,:,:2],data[:,:,3:5]],axis=2)

            if test_case=='lowres_skin' or test_case=='midres_skin':
                save_path=os.path.join(out_dir,'skin_img_{:08d}.npy'.format(test_id))
            elif test_case=='lowres_tshirt':
                save_path=os.path.join(out_dir,'tshirt_img_{:08d}.npy'.format(test_id))
            elif test_case.find('hat')!=-1:
                save_path=os.path.join(out_dir,'hat_img_{:08d}.npy'.format(test_id))
            elif test_case.find('fcs')!=-1:
                save_path=os.path.join(out_dir,'fc_id_img_{}.npy'.format(SIDE))
            else:
                save_path=os.path.join(out_dir,'offset_img_{:08d}.npy'.format(test_id))
            if args.gzip:
                with gzip.open('{}.gz'.format(save_path),'wb') as f:
                    np.save(file=f,arr=data)
            else:
                np.save(save_path,data)

    def draw_glut():
        if args.mask:
            if args.no_pad:
                process_id(-1,front_vts,back_vts,np.empty([0,2]),np.empty([0,2]),front_indices,back_indices)
            else:
                process_id(-1,front_vts_w_pad,back_vts_w_pad,front_crpds,back_crpds,front_indices_w_pad,back_indices_w_pad)
        else:
            assert(mask is not None)
            for i in range(args.start,args.end+1):
                print(i)
                #if i%500 == 0:
                #    print(i)
                #else:
                #    print('.',end='')
                #    sys.stdout.flush()
                # process_id(i,front_vts_w_pad,back_vts_w_pad,front_crpds,back_crpds,front_indices_w_pad,back_indices_w_pad,mask=mask,use_uvn=args.use_uvn)
                try:
                    if args.no_pad:
                        process_id(i,front_vts,back_vts,np.empty([0,2]),np.empty([0,2]),front_indices,back_indices,mask=mask,use_uvn=args.use_uvn)
                    else:
                        process_id(i,front_vts_w_pad,back_vts_w_pad,front_crpds,back_crpds,front_indices_w_pad,back_indices_w_pad,mask=mask,use_uvn=args.use_uvn)
                except Exception as e:
                    print(e)
        exit()
        glutSwapBuffers()


    # glDepthFunc(GL_LESS)
    glutDisplayFunc(draw_glut)
    glutMainLoop()
