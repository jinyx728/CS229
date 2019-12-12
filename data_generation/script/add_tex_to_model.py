######################################################################
# Copyright 2019. Yongxu Jin.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import argparse

def add_texture(file, outfile, vtfile):

    with open(file, "r") as objf:

        # read texture coordinate
        vtf = open(vtfile, "r")
        vt = vtf.readlines()
        vtf.close()

        obj = objf.readlines()
        index = 0

        # add mtllib entry
        while index < len(obj):
            if obj[index] == "\n":           # for ground truth obj file
                obj.insert(index + 1, "mtllib tshirt.mtl\n")
                index += 2
                break
            elif index == 0 and obj[index][0] == "v":      # for predict obj file
                obj.insert(index, "mtllib tshirt.mtl\n")
                index += 1
                break
            index += 1

        # add texture coordinate and material entry
        while index < len(obj):
            if obj[index][0] == "f":
                obj.insert(index, "usemtl material0\n")
                obj.insert(index, "\n")
                obj.insert(index, "\n")
                obj = obj[0:index+1] + vt + obj[index+1:]
                index += len(vt) + 3
                break
            index += 1

        # fix face format
        while index < len(obj):
            l = obj[index].rstrip().split(" ")
            obj[index] = ("f {}/{} {}/{} {}/{}\n").format(l[1], l[1], l[2], l[2], l[3], l[3])
            index += 1
        
        outf = open(outfile, "w")
        outf.writelines(obj)
        outf.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input obj file", required=True)
    parser.add_argument("-o", "--output", help="output obj file", required=True)
    parser.add_argument("-t", "--texture", help="texture coordinate file", required=True)
    args = parser.parse_args()
    add_texture(args.input, args.output, args.texture)