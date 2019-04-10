#!/bin/sh
#--snapshot=$latest \

if ! test -f example/MobileNetSSD_train.prototxt ;then
	echo "error: example/MobileNetSSD_train.prototxt does not exist."
	echo "please use the gen_model.sh to generate your own model."
        exit 1
fi
latest=$(ls -t snapshot/*.caffemodel | head -n 1)
if test -z $latest; then
	exit 1
fi
mkdir -p snapshot
../../build/tools/caffe train -solver="solver_train.prototxt" \
--weights=$latest \
-gpu 0 
