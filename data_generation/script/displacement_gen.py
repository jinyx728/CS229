######################################################################
# Copyright 2019. Yongxu Jin.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import argparse

def displacement_gen(vt1file, vt2file):
    vt1f = open(vt1file, "r")
    vt2f = open(vt2file, "r")
    vt1 = vt1f.readlines()
    vt2 = vt2f.readlines()

    if len(vt1) != len(vt2):
        print("ERROR in desplacement_gen.py! Size not match")
        return

    for i in range(len(vt1)):
        line1 = vt1[i].split(" ")
        line2 = vt2[i].split(" ")
        print("{} {}".format(float(line1[1])-float(line2[1]), float(line1[2])-float(line2[2])))

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--texture", help="generated texture", required=True)
    parser.add_argument("-g", "--groundtruth", help="ground truth texture", required=True)
    args = parser.parse_args()
    displacement_gen(args.texture, args.groundtruth)