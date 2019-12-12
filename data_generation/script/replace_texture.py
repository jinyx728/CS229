######################################################################
# Copyright 2019. Yongxu Jin.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import argparse

def replace_texture(inf, outf, texf):
    with open(inf, "r") as objf:
        # read texture coordinate
        vtf = open(texf, "r")
        vt = vtf.readlines()
        vtf.close()

        obj = objf.readlines()
        index = 0

        while index < len(obj):
            if obj[index][0] == "v" and obj[index][1] == "t":
                break
            index += 1

        for i in range(len(vt)):
            obj[index+i] = vt[i]
        
        outfile = open(outf, "w")
        outfile.writelines(obj)
        outfile.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input obj file", required=True)
    parser.add_argument("-o", "--output", help="output obj file", required=True)
    parser.add_argument("-t", "--texture", help="texture coordinate file", required=True)
    args = parser.parse_args()
    replace_texture(args.input, args.output, args.texture)