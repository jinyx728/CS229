from obj_io import read_obj,write_obj,Obj
import numpy as np
from numpy.linalg import norm
class DrawObjUtils:
    def __init__(self):
        self.reset()
        self.sphere_obj=read_obj('draw_primitives/sphere.obj')
        self.cylinder_obj=read_obj('draw_primitives/cylinder.obj')

    def reset(self):
        self.v=np.zeros((0,3))
        self.f=np.zeros((0,3))

    def add_sphere(self,center,r):
        v=self.sphere_obj.v*r+center

        n_vts=len(self.v)
        self.v=np.vstack([self.v,v])
        f=self.sphere_obj.f+n_vts
        self.f=np.vstack([self.f,f])

    def add_cylinder(self,p0,p1,w=0.01):
        v=self.cylinder_obj.v.copy()
        h=norm(p1-p0)
        v*=np.array([[w,h/2,w]])

        y=p1-p0
        y/=norm(y)
        z=np.array([0,0,1])
        z=z-np.inner(y,z)*y
        z/=norm(z)
        x=np.cross(y,z)
        RT=np.array([x,y,z])
        t=(p1+p0)/2
        v=v.dot(RT)+t.reshape((1,-1))

        n_vts=len(self.v)
        self.v=np.vstack([self.v,v])
        f=self.cylinder_obj.f+n_vts
        self.f=np.vstack([self.f,f])

    def write_obj(self,out_path):
        write_obj(Obj(v=self.v,f=self.f),out_path)
