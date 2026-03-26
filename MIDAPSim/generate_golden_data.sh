#!/bin/bash
date=0910
flags="-fl PAD -f 8 128 -q -fs"
networks=("mobilenet_v2" "test_network" "discogan" "unet_small" "mobilenet_v3_small" "mobilenet_v3_large" "efficientnet-b0" "yolov4_tiny" "mobilenet_ssd")
dirs=("mv2" "test" "discogan" "unet_s" "mv3_s" "mv3_l" "eff-b0" "yolov4_tiny" "mv_ssd")
save=(0 1 2 3 4 5 6 7 8)
# save=(4 6 7)
# save=()
gen=(0 1 2 3 4 5 6 7 8)
# gen=(4 6 7)
# gen=()
# rm -rf temp/${date}
rm -rf /share/golden/${date}

for i in ${save[*]}
do
    python tools/test_system.py ${flags} -n ${networks[$i]} -so -sd temp/${date} -sp ${dirs[$i]}
done

for i in ${gen[*]}
do
    python tools/generate_binary.py -d temp/${date} -p ${dirs[$i]} -itg -gen -od /share/golden/${date}/${dirs[$i]}
done
