######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import os
from os.path import join
from obj_io import read_obj,Obj,write_obj
class PatchIOManager:
	def __init__(self,res_ctx):
		self.shared_data_dir=res_ctx['shared_data']
		self.vt_offset_dir=res_ctx['vt_offset_dir']
		self.skin_dir=res_ctx['skin_dir']
		obj_path=join(shared_data_dir,'flat_tshirt.obj')
		self.fcs=read_obj(obj_path).f

		self.patch_path=join(self.shared_data_dir,'patches')
		patch_names_path=join(self.patch_path,'patch_names.txt')
		self.patch_names=self.load_patch_names(patch_names_path)
		self.n_patches=len(self.patch_names)

		self.all_patch_vt_ids=self.load_all_patch_vt_ids()

	def load_patch_vt_ids(self,patch_id):
	    patch_name=self.patch_names[patch_id]
	    patch_bdry_vt_ids=np.loadtxt(os.path.join(self.patch_path,'{}_boundary_vertices.txt'.format(patch_name))).astype(np.int32)-1
	    patch_intr_vt_ids=np.loadtxt(os.path.join(self.patch_path,'{}_interior_vertices.txt'.format(patch_name))).astype(np.int32)-1
	    patch_vt_ids=np.hstack([patch_bdry_vt_ids,patch_intr_vt_ids])
	    return patch_vt_ids

	def load_all_patch_vt_ids(self):
	    all_patch_vt_ids=[]
	    for patch_id in range(self.n_patches):
	        all_patch_vt_ids.append(self.load_patch_vt_ids(patch_id))
	    return all_patch_vt_ids

	def save_patch_pieces_obj(self,samples_dir,sample_id):
	    skin_path=os.path.join(self.skin_dir,'skin_{:08d}.npy'.format(sample_id))
	    skin=np.load(skin_path)
	    n_vts=len(skin)
	    sample_dir=os.path.join(samples_dir,'{:08d}'.format(sample_id))
	    vts=[]
	    fcs=[]
	    for patch_id in range(self.n_patches):
	        patch_name=self.patch_names[patch_id]
	        patch_offset_path=os.path.join(sample_dir,'patches/pd_offset_{}.npy'.format(patch_name))
	        patch_offset=np.load(patch_offset_path).reshape((-1,3))

	        patch_vt_ids=self.all_patch_vt_ids[patch_id]
	        patch_fcs=self.all_patch_intr_fcs[patch_id]
	        patch_vts=skin.copy()
	        patch_vts[patch_vt_ids]+=patch_offset
	        vts.append(patch_vts)
	        fcs.append(patch_fcs+patch_id*n_vts)
	        # break

	    vts=np.concatenate(vts,axis=0)
	    fcs=np.concatenate(fcs,axis=0)

	    obj_path=os.path.join(sample_dir,'pieces.obj')
	    obj=Obj(v=vts,f=fcs)
	    write_obj(obj,obj_path)

	def save_stitched_patches_obj(self,samples_dir,sample_id)
		skin_path=os.path.join(self.skin_dir,'skin_{:08d}.npy'.format(sample_id))
		skin=np.load(skin_path)
		n_vts=len(skin)
		sample_dir=os.path.join(samples_dir,'{:08d}'.format(sample_id))
		vt_offsets=[[] for i in range(n_vts)]
		for patch_id in range(self.n_patches):
		    patch_name=self.patch_names[patch_id]
		    patch_offset_path=os.path.join(sample_dir,'patches/pd_offset_{}.npy'.format(patch_name))
		    patch_offset=np.load(patch_offset_path).reshape((-1,3))

		    patch_vt_ids=self.all_patch_vt_ids[patch_id]
		    for i in range(len(patch_offset)):
		        vt_id=patch_vt_ids[i]
		        vt_offsets[vt_id].append(patch_offset[i])

		for i in range(n_vts):
		    vt_offsets[i]=np.mean(np.array(vt_offsets[i]),axis=0)

		vt_offsets=np.array(vt_offsets)
		obj_path=os.path.join(sample_dir,'stitched.obj')
		obj=Obj(v=skin+vt_offsets,f=self.fcs)
		write_obj(obj,obj_path)


if __name__=='__main__':
	import argparse
	parser=argparse.ArgumentParser()
	parser.add_argument('-test_case',choices=['lowres','midres'])
	parser.add_argument('-samples_dir')
	args=parser.parse_args()

	def get_res_ctx(res):
		data_root_dir='/data/zhenglin/poses_v3'
		res_ctx={
			'shared_data_dir':'../../shared_data_{}'.format(res),
			'vt_offset_dir':join(data_root_dir,'{}_offset_npys'.format(res)),
			'skin_dir':join(data_root_dir,'{}_skin_npys'.format(res))
		}
		return res_ctx

	res_ctx=get_res_ctx(args.test_case)
	patch_io_manager=PatchIOManager(res_ctx=res_ctx)
	dirs=os.listdir(args.samples_dir)
	for d in dirs:
		sample_id=int(d)
		print('sample_id',sample_id)
		patch_io_manager.save_stitched_patches_obj(samples_dir,sample_id)