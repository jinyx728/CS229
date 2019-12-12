######################################################################
# Copyright 2019. Yongxu Jin.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
#!/bin/sh

# cloth model path
GROUNDTRUTH_PATH=/phoenix/jwu/cloth/lowres_tshirts
PREDICT_PATH=/euler/yxjin/PhysBAM/Private_Projects/cloth_on_virtual_body/Learning/rundir/lowres_normal/pd/eval_pred
OUTDIR=/phoenix/yxjin

# camera position
CAMERA_X=0.0437965
CAMERA_Y=0.705956
CAMERA_Z=0.787657

mkdir $OUTDIR/dataset/

for INDEX in $(seq -f "%08g" 0 29999)
do

    python dataset_lowres.py -g $GROUNDTRUTH_PATH/tshirt_$INDEX.obj -p $PREDICT_PATH/$INDEX/cloth_$INDEX.obj -t ../vt_groundtruth.txt -c $CAMERA_X $CAMERA_Y $CAMERA_Z >> $OUTDIR/dataset/displace_$INDEX.txt

    echo "generated data $INDEX"
done
