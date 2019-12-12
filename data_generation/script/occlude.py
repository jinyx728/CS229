######################################################################
# Copyright 2019. Yongxu Jin.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import imageio
import numpy as np
import argparse

def is_white(vec):
    if vec[0]==255 and vec[1]==255 and vec[2]==255:
        return True
    else:
        return False

def occlude(inputf, maskf, outputf):
    input_img = imageio.imread(inputf)
    mask_img = imageio.imread(maskf)
    if input_img.shape != mask_img.shape:
        print('ERROR! Shape not match!')
        return
    for i in range(input_img.shape[0]):
        for j in range(input_img.shape[1]):
            if is_white(mask_img[i][j]) and not is_white(input_img[i][j]):
                input_img[i][j] = np.array([0,0,255])
    imageio.imwrite(outputf, input_img)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input image", required=True)
    parser.add_argument("-m", "--mask", help="mask image", required=True)
    parser.add_argument("-o", "--output", help="output image", required=True)
    args = parser.parse_args()
    occlude(args.input, args.mask, args.output)