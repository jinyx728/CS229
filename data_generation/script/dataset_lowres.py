######################################################################
# Copyright 2019. Yongxu Jin.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import argparse
import math
import numpy as np
import trimesh

def texcoord_est(gtfile, pdfile, texfile, camera_pos):

    # load obj from file
    gt_mesh = trimesh.load(gtfile, process=False)
    pd_mesh = trimesh.load(pdfile, process=False)
    
    # load default texture coordinate
    vtf = open(texfile, "r")
    vt = vtf.readlines()
    vtf.close()
    for i in range(len(vt)):
        split = vt[i].split(" ")
        vt[i] = [float(split[1]), float(split[2])]

    camera_ray(gt_mesh, pd_mesh, vt, camera_pos)




# infer tex coord from closest vertex method
# def closest_vertex(gt_mesh, pd_mesh):
#     pdv = pd_mesh.vertices
#     gtv = gt_mesh.vertices
#     gtvt = gt_mesh.visual.uv
#     for i in range(len(pdv)):
#         pd_pos = pdv[i]
#         dist_min = 100000 # float_max
#         index_min = -1
#         for j in range(len(gtv)):
#             gt_pos = gtv[j]
#             dist = np.linalg.norm(pd_pos - gt_pos)
#             if dist < dist_min:
#                 dist_min = dist
#                 index_min = j
#         print("vt {} {}".format(gtvt[index_min][0], gtvt[index_min][1]))



# infer tex coord from ray casting method
def camera_ray(gt_mesh, pd_mesh, gtvt, camera_pos):
    # fixed camera position
    pdv = pd_mesh.vertices
    gtv = gt_mesh.vertices
    gtf = gt_mesh.faces
    
    mvt = []

    # ray casting to ground truth cloth
    for i in range(len(pdv)):
        pd_pos = pdv[i]

        extrapolate = False
        # ray to predicted cloth
        locations_pd, _, _ = pd_mesh.ray.intersects_location(ray_origins=np.array([camera_pos]), ray_directions=np.array([pd_pos-camera_pos]))
        test_dist = np.linalg.norm(pd_pos-camera_pos) - 1e-10 # epsilon for numerical error
        for loc in locations_pd:
            leng = np.linalg.norm(loc-camera_pos)
            if leng < test_dist:
                extrapolate = True
                break

        # ray to overlaid ground truth cloth
        locations, _, index_tri = gt_mesh.ray.intersects_location(ray_origins=np.array([camera_pos]), ray_directions=np.array([pd_pos-camera_pos]))

        # extrapolate when missed or non-visible
        extrapolate = extrapolate or (len(locations) == 0)

        # extrapolate - use default vt
        if extrapolate:
            mvt.append([gtvt[i][0], gtvt[i][1]])

        # intersect
        else:
            # find the closest point to camera
            dist_min = 100000 # float_max
            index_min = -1
            for j in range(len(locations)):
                location = locations[j]
                dist = np.linalg.norm(camera_pos - location)
                if dist < dist_min:
                    dist_min = dist
                    index_min = j

            # barycentrically determine texture coordinate
            face = index_tri[index_min]
            intersect = locations[index_min]
            tri_index = gtf[face]
            tri_pos = np.array([gtv[tri_index[0]], gtv[tri_index[1]], gtv[tri_index[2]]])
            tri_tex = np.array([gtvt[tri_index[0]], gtvt[tri_index[1]], gtvt[tri_index[2]]])
            v0 = np.array(tri_pos[1] - tri_pos[0])
            v1 = np.array(tri_pos[2] - tri_pos[0])
            v2 = np.array(intersect - tri_pos[0])
            d00 = np.dot(v0, v0)
            d01 = np.dot(v0, v1)
            d11 = np.dot(v1, v1)
            d20 = np.dot(v2, v0)
            d21 = np.dot(v2, v1)
            denom = d00 * d11 - d01 * d01
            w1 = (d11 * d20 - d01 * d21) / denom
            w2 = (d00 * d21 - d01 * d20) / denom
            w0 = 1 - w1 - w2
            tex_coord = w0 * tri_tex[0] + w1 * tri_tex[1] + w2 * tri_tex[2]
            mvt.append([tex_coord[0], tex_coord[1]])

    # compute displacement
    for i in range(len(gtvt)):
        print("{} {}".format(mvt[i][0]-gtvt[i][0], mvt[i][1]-gtvt[i][1]))

    return

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--groundtruth", help="ground truth cloth model", required=True)
    parser.add_argument("-p", "--predict", help="predicted cloth model", required=True)
    parser.add_argument("-c", "--camera", help="camera position", required=True, nargs="+", type=float)
    parser.add_argument("-t", "--texture", help="default texture coordinate", required=True)
    args = parser.parse_args()
    if len(args.camera) != 3:
        print("ERROR in texcoord_est.py: invalid camera position!")
        exit(1)

    texcoord_est(args.groundtruth, args.predict, args.texture, args.camera)
        