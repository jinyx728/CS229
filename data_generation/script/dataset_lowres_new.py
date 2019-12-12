######################################################################
# Copyright 2019. Yongxu Jin.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import argparse
import math
import numpy as np
import trimesh

def texcoord_est(gtfile, pdfile, texfile, camera_pos, backfile):

    # load obj from file
    gt_mesh = trimesh.load(gtfile, process=False)
    pd_mesh = trimesh.load(pdfile, process=False)

    back_vtx = np.loadtxt(backfile)

    # load default texture coordinate
    vtf = open(texfile, "r")
    vt = vtf.readlines()
    vtf.close()
    for i in range(len(vt)):
        split = vt[i].split(" ")
        vt[i] = [float(split[1]), float(split[2])]

    camera_ray(gt_mesh, pd_mesh, vt, camera_pos, back_vtx)

def hash_gen(gt_mesh, back_vtx):
    hashmap = {}
    gtf = gt_mesh.faces
    for i in range(len(gtf)):
        if (gtf[i][0] in back_vtx) and (gtf[i][1] in back_vtx) and (gtf[i][2] in back_vtx):
            hashmap[i] = True
        else:
            hashmap[i] = False
    return hashmap

# infer tex coord from ray casting method
def camera_ray(gt_mesh, pd_mesh, gtvt, camera_pos, back_vtx):
    # fixed camera position
    pd_mesh.fix_normals()
    pdv = pd_mesh.vertices
    gtv = gt_mesh.vertices
    gtf = gt_mesh.faces
    pdf = pd_mesh.faces
    pdfn = pd_mesh.face_normals
    gtfn = gt_mesh.face_normals
    front_flag = np.zeros(len(pdf), dtype=bool)
    ints_flag = np.zeros(len(pdv), dtype=bool)

    mvt = []
    hashmap = hash_gen(gt_mesh, back_vtx)

    # first pass: find all front triangles
    for i in range(len(pdf)):
        centroid = (pdv[pdf[i][0]] + pdv[pdf[i][1]] + pdv[pdf[i][2]]) / 3
        ray = centroid - camera_pos
        if np.dot(ray, pdfn[i]) < 0:
            front_flag[i] = True

    # second pass: find vertices that need intersection
    for i in range(len(pdf)):
        if not front_flag[i]:
            continue
        for j in range(3):
            vidx = pdf[i][j]
            if ints_flag[vidx]:
                continue
            viz = True
            locations_test, _, _ = pd_mesh.ray.intersects_location(ray_origins=np.array([camera_pos]), ray_directions=np.array([pdv[vidx]-camera_pos]))
            test_dist = np.linalg.norm(pdv[vidx]-camera_pos) - 1e-7 # epsilon for numerical error
            for loc in locations_test:
                leng = np.linalg.norm(loc-camera_pos)
                if leng < test_dist:
                    viz = False
                    break
            ints_flag[vidx] = viz
        if ints_flag[pdf[i][0]] or ints_flag[pdf[i][1]] or ints_flag[pdf[i][2]]:
            ints_flag[pdf[i][0]] = True
            ints_flag[pdf[i][1]] = True
            ints_flag[pdf[i][2]] = True
        else: # check subdivided triangle vertices
            subdiv = np.zeros((6,3))
            subdiv[0] = (pdv[pdf[i][0]] + pdv[pdf[i][1]]) / 2
            subdiv[1] = (pdv[pdf[i][0]] + pdv[pdf[i][2]]) / 2
            subdiv[2] = (pdv[pdf[i][1]] + pdv[pdf[i][2]]) / 2
            subdiv[3] = (subdiv[0] + subdiv[1]) / 2
            subdiv[4] = (subdiv[0] + subdiv[2]) / 2
            subdiv[5] = (subdiv[1] + subdiv[2]) / 2
            hidden = True
            for j in range(6):
                v_pos = subdiv[j]
                viz = True
                locations_test, _, _ = pd_mesh.ray.intersects_location(ray_origins=np.array([camera_pos]), ray_directions=np.array([v_pos-camera_pos]))
                test_dist = np.linalg.norm(v_pos-camera_pos) - 1e-7 # epsilon for numerical error
                for loc in locations_test:
                    leng = np.linalg.norm(loc-camera_pos)
                    if leng < test_dist:
                        viz = False
                        break
                if viz:
                    hidden = False
                    break
            if not hidden:
                ints_flag[pdf[i][0]] = True
                ints_flag[pdf[i][1]] = True
                ints_flag[pdf[i][2]] = True

    # third pass: ray intersecting to selected vertices
    for i in range(len(pdv)):

        if not ints_flag[i]:
            mvt.append([gtvt[i][0], gtvt[i][1]])
            continue

        pd_pos = pdv[i]

        # ray to overlaid ground truth cloth
        locations, _, index_tri = gt_mesh.ray.intersects_location(ray_origins=np.array([camera_pos]), ray_directions=np.array([pd_pos-camera_pos]))

        # extrapolate - use default vt
        if len(locations) == 0:
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

            face = index_tri[index_min]

            # back triangle or back face triangle: use default vt
            back_tri = hashmap[face]
            back_face = False
            fn = gtfn[face]
            ray = pd_pos - camera_pos
            if np.dot(fn, ray) > 0:
                back_face = True
            if back_tri or back_face:
                mvt.append([gtvt[i][0], gtvt[i][1]])
                continue

            # barycentrically determine texture coordinate
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
    parser.add_argument("-b", "--back", help="back vertices index", required=True)
    parser.add_argument("-c", "--camera", help="camera position", required=True, nargs="+", type=float)
    parser.add_argument("-t", "--texture", help="default texture coordinate", required=True)
    args = parser.parse_args()
    if len(args.camera) != 3:
        print("ERROR in texcoord_est.py: invalid camera position!")
        exit(1)
    texcoord_est(args.groundtruth, args.predict, args.texture, args.camera, args.back)
        
    