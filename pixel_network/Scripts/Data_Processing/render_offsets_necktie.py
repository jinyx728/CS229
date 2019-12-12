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

TRY_FRAMEBUFFER = True
GENERATE_TEXTURE_ID_PROBLEM = False  # Follow this global variable to see the error in texture ID generation

SIDE = 512  # window size
# 512 for tie
# OUTPUT_SIZE=512
OUTPUT_SIZE=SIDE

# Utility functions
def float_size(n=1):
    return sizeof(ctypes.c_float) * n

def float_pointer_offset(n=0):
    return ctypes.c_void_p(float_size(n))

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

    # OpenGL version info
    renderer = glGetString(GL_RENDERER)
    version = glGetString(GL_VERSION)
    print('Renderer:', renderer)  # Renderer: b'Intel Iris Pro OpenGL Engine'
    print('OpenGL version supported: ', version)  # OpenGL version supported:  b'4.1 INTEL-10.12.13'


vertex_shader = """#version 410
layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 col;
out vec3 fg_color;
void main () {
    fg_color = col;
    gl_Position = vec4(pos, 1.0f);
}"""

fragment_shader = """#version 410
in vec3 fg_color;
out vec4 color;
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
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, SIDE, SIDE, 0, GL_RGB, GL_FLOAT, None)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)
    return texture

def read_img():
    data=glReadPixels(0,0,SIDE,SIDE,GL_RGB,GL_FLOAT)
    # data=imresize(data,size=(OUTPUT_SIZE,OUTPUT_SIZE,3),interp='bicubic')
    # data=cv2.resize(data,(OUTPUT_SIZE,OUTPUT_SIZE))
    return data

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

def normalize_offsets(offsets):
    offsets_min,offsets_max=np.min(offsets),np.max(offsets)
    offsets=(offsets-offsets_min)/(offsets_max-offsets_min)
    return offsets,offsets_min,offsets_max

def denormalize_offsets(offsets,offsets_min,offsets_max):
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

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-mask',action='store_true')
    parser.add_argument('-start',default=10,type=int)
    parser.add_argument('-end',default=10,type=int)
    parser.add_argument('-offset_dir',default='offset_test')
    parser.add_argument('-out_dir',default='offset_test')
    parser.add_argument('-vlz',action='store_true')
    parser.add_argument('-no_pad',action='store_true')
    args=parser.parse_args()

    init_gl()
    main_shader = create_shader(vertex_shader, fragment_shader)
    fbo=make_fbo()
    texture=make_fbo_texture()

    if not glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE:
        print('framebuffer binding failed')
        exit()
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glBindTexture(GL_TEXTURE_2D, 0)

    # load vts_3d, colors, fcs
    shared_data_dir='../../shared_data_necktie'
    front_obj_path=os.path.join(shared_data_dir,'necktie_rest.obj')
    front_uv_path=os.path.join(shared_data_dir,'necktie_uvs.npy')
    front_uvs=np.load(front_uv_path)
    front_vts=np.zeros([front_uvs.shape[0],3])
    front_vts[:,:2]=front_uvs
    front_obj=read_obj(front_obj_path)
    front_fcs=front_obj.f
    #front_vts,front_fcs=front_obj.v,front_obj.f
    front_indices=np.ascontiguousarray(front_fcs.reshape(-1).astype(np.uint32))
    n_vts=len(front_vts)
    front_vts_normalize_m=get_vts_normalize_m(front_vts)
    if args.no_pad:
        front_vts=normalize_vts(front_vts[:,:2],front_vts_normalize_m)
    else:
        front_obj_w_pad_path=os.path.join(shared_data_dir,'necktie_padding_new.obj')
#         front_obj_w_pad_path=os.path.join(shared_data_dir,'necktie_padding.obj')
        front_obj_w_pad=read_obj(front_obj_w_pad_path)
        front_vts_w_pad,front_fcs_w_pad=front_obj_w_pad.v,front_obj_w_pad.f
        front_vts_w_pad=normalize_vts(front_vts_w_pad[:,:2],front_vts_normalize_m)
        front_crpds_path=os.path.join(shared_data_dir,'necktie_padding_correspondence_new.txt')
#         front_crpds_path=os.path.join(shared_data_dir,'necktie_padding_correspondence.txt')
        front_crpds=load_crpds(front_crpds_path)
        front_indices_w_pad=np.ascontiguousarray(front_fcs_w_pad.reshape(-1).astype(np.uint32))

    out_dir=args.out_dir
    offset_dir=args.offset_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print('offset_dir',offset_dir,'out_dir',out_dir)

    mask=None
    if args.no_pad:
        mask_path=os.path.join(shared_data_dir,'offset_img_mask_no_pad.npy')
    else:
        mask_path=os.path.join(shared_data_dir,'offset_img_mask.npy')
    if os.path.isfile(mask_path):
        mask=np.load(mask_path)
    else:
        print(mask_path,'does not exist')

    def process_id(test_id,front_vts_w_pad,front_crpds,front_indices_w_pad,mask=None):
        if test_id==-1:# make map
            print('render mask')
            offsets=np.ones((len(front_vts_w_pad),3))
            offsets_min=1
            offsets_max=1
            mask=None
        else:
            offset_path=os.path.join(offset_dir,'offset_{:08d}.npy'.format(test_id))
            offsets=np.load(offset_path)
            print('offsets',offsets.shape)
            offsets,offsets_min,offsets_max=normalize_offsets(offsets)
            offsets=offsets.astype(np.float32)

        # front
        front_vts_w_pad=np.hstack([front_vts_w_pad,np.zeros((len(front_vts_w_pad),1))])
        front_offsets=np.zeros_like(front_vts_w_pad)
        print('front_offsets',front_offsets.shape,'offsets',offsets.shape)
        front_offsets[:len(offsets)]=offsets
        if len(front_crpds)>0:
            front_offsets[front_crpds[:,1]]=offsets[front_crpds[:,0]]
        front_vao=make_vao(front_vts_w_pad,front_offsets)
        draw(front_vao,front_indices_w_pad)
        front_data=read_img()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        data=front_data

        if mask is not None:
            front_mask=mask[:,:,0]

        if args.vlz:
            if mask is None:
                png_mask=None
            else:
                front_png_mask=get_png_mask(front_mask)
                png_mask=front_png_mask

            print('vlz in test.png')
            save_png_img('test.png',data,mask=png_mask)

        if test_id==-1:#mask
            data=np.concatenate([data[:,:,0:1],data[:,:,3:4]],axis=2)
            data=np.around(data)
            if args.no_pad:
                mask_path=os.path.join(shared_data_dir,'offset_img_mask_no_pad.npy')
            else:
                mask_path=os.path.join(shared_data_dir,'offset_img_mask.npy')
            print('save to path',mask_path)
            np.save(mask_path,data)
        else:
            data=denormalize_offsets(data,offsets_min,offsets_max)
            data[:,:,:3]*=np.expand_dims(front_mask,axis=2)
            np.save(os.path.join(out_dir,'offset_img_{:08d}.npy'.format(test_id)),data)

    def draw_glut():
        if args.mask:
            if args.no_pad:
                process_id(-1,front_vts,np.empty([0,2]),front_indices)
            else:
                process_id(-1,front_vts_w_pad,front_crpds,front_indices_w_pad)
        else:
            assert(mask is not None)
            for i in range(args.start,args.end+1):
                print(i)
                try:
                    if args.no_pad:
                        process_id(i,front_vts,np.empty([0,2]),front_indices,mask=mask)
                    else:
                        process_id(i,front_vts_w_pad,front_crpds,front_indices_w_pad,mask=mask)
                except Exception as e:
                    print(e)
        exit()
        glutSwapBuffers()


    # glDepthFunc(GL_LESS)
    glutDisplayFunc(draw_glut)
    glutMainLoop()
