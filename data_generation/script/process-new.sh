######################################################################
# Copyright 2019. Yongxu Jin.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
#!/bin/sh

# cloth model path
GROUNDTRUTH_PATH=/phoenix/yxjin/dataset_subdivision/dataset_gt_subdiv
PREDICT_PATH=/phoenix/yxjin/dataset_subdivision/dataset_pd_subdiv
OUTDIR=/phoenix/yxjin/dataset_subdivision

# camera position
CAMERA_X=0.0437965
CAMERA_Y=0.705956
CAMERA_Z=0.787657

# parallel execution
NUM_PROCS=12
NUM_JOBS="\j"

mkdir $OUTDIR/displacement_subdiv/

for FILE in $PREDICT_PATH/*.obj
do
    while ((${NUM_JOBS@P} >= NUM_PROCS))
    do
        wait -n
    done

    TMP=${FILE##*/}
    python dataset_lowres_new.py -g $GROUNDTRUTH_PATH/tshirt_div_${TMP:7:8}_tex.obj -p $PREDICT_PATH/pd_div_${TMP:7:8}_tex.obj -t ../vt_groundtruth_div.txt -c $CAMERA_X $CAMERA_Y $CAMERA_Z >> $OUTDIR/displacement_subdiv/displacement_${TMP:7:8}.txt &
    echo "generated data ${TMP:7:8}"
done
