######################################################################
# Copyright 2019. Yongxu Jin.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
#!/bin/sh

# cloth model path
PREDICT_NEW_PATH=/phoenix/yxjin/test_infer/mocap_13_30
OUTDIR=/phoenix/yxjin
mkdir $OUTDIR/mocap_13_30_subdiv/

for INDEX in $(seq -f "%08g" 0 495)
do

    python subdivision.py -i $PREDICT_NEW_PATH/$INDEX.obj -o $OUTDIR/mocap_13_30_subdiv/pd_div_${INDEX}_tex.obj -t ../vt_groundtruth.txt

    echo "generated data $INDEX"
done
