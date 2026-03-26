#!/bin/bash
date=0927
flags="-fl PAD -f 8 128 -q -fs -d TEST_3D" #-nf 4 -cf" #-dws"
networks=("mobilenet_v2" "test_network" "discogan" "unet_small" "mobilenet_v3_small" "mobilenet_v3_large" "efficientnet-b0" "yolov4_tiny" "mobilenet_ssd" "deeplabv3+mv" "efficientnet-b4" "resnet50" "test_dilation" "se_resnet50" "resnet101" "resnet152" "inception_v3" "dcgan" "efficientnet-b2" "test_bilinear")
#          0              1              2          3            4                    5                    6                 7             8               9              10                11         12              13            14          15          16             17      18                19
dirs=("mv2" "test" "discogan" "unet_s" "mv3_s" "mv3_l" "eff-b0" "yolov4_tiny" "mv_ssd" "deeplabv3+mv" "eff-b4" "resnet50" "test_dilation" "se_resnet50" "resnet101" "resnet152" "inception_v3" "dcgan" "eff-b2" "test_bilinear")
save=(9)
# save=(4 6 7)
# save=()
gen=(9)
# gen=(4 6 7)
# gen=()
# rm -rf temp/${date}
for i in ${save[*]}
do
    if [ -d ~/midap_compile_test/${date}/${dirs[$i]} ]
    then
        echo "The directory ~/midap_compile_test/${date}/${dirs[$i]} exists"
        exit 1
    fi
    #rm -rf ~/midap_compile_test/${date}/${dirs[$i]}
done

for i in ${save[*]}
do
    python tools/test_system.py ${flags} -n ${networks[$i]} -so -sd ~/midap_compile_test/${date} -sp ${dirs[$i]}
done

for i in ${gen[*]}
do
    python tools/generate_binary.py -da -d ~/midap_compile_test/${date} -p ${dirs[$i]} -itg -gen -od ~/midap_compile_test/${date}/${dirs[$i]}
done
