######################################################################
# Copyright 2019. Yongxu Jin.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import argparse
import numpy as np

def replace_texture(inf, outf, vertexf):
    with open(inf, "r") as objf:
        # read texture coordinate
        vtf = open(vertexf, "r")
        vt = vtf.read().splitlines()
        vtf.close()

        obj = objf.readlines()
        index = 0

        while index < len(obj):
            if obj[index][0] == "f":
                break
            index += 1

        to_delete = []
        for i in range(index, len(obj)):
            ilist = obj[i].split(" ")
            ilist[3] = ilist[3].rstrip()
            for j in range(1,4):
                ilist[j] = str(int(ilist[j])-1)
            if not ((ilist[1] in vt) and (ilist[2] in vt) and (ilist[3] in vt)):
                to_delete.append(i)

        for idx in sorted(to_delete, reverse=True):
            del obj[idx]

        outfile = open(outf, "w")
        outfile.writelines(obj)
        outfile.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input obj file", required=True)
    parser.add_argument("-o", "--output", help="output obj file", required=True)
    parser.add_argument("-v", "--vertices", help="vertex index file", required=True)
    args = parser.parse_args()
    replace_texture(args.input, args.output, args.vertices)