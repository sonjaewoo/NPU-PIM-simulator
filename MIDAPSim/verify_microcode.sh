#!/bin/bash
date=1020
flags="-fl PAD -f 8 128 -q -fs -d TEST_3D" #-nf 2" #-dws
networks=("mobilenet_v2" "test_network" "discogan" "unet_small" "mobilenet_v3_small" "mobilenet_v3_large" "efficientnet-b0" "yolov4_tiny" "mobilenet_ssd" "deeplabv3+mv" "efficientnet-b4" "resnet50" "test_dilation" "se_resnet50" "resnet101" "resnet152" "inception_v3" "dcgan" "efficientnet-b2" "test_bilinear")
#          0              1              2          3            4                    5                    6                 7             8               9              10                11         12              13            14          15          16             17      18                19
dirs=("mv2" "test" "discogan" "unet_s" "mv3_s" "mv3_l" "eff-b0" "yolov4_tiny" "mv_ssd" "deeplabv3+mv" "eff-b4" "resnet50" "test_dilation" "se_resnet50" "resnet101" "resnet152" "inception_v3" "dcgan" "eff-b2" "test_bilinear")
#save=(0 1 2 3 4 5 6 7 8 9 11 13 14 15 16 17 18 19)
save=(18)
# save=(4 6 7)
# save=()
#gen=(0 1 2 3 4 5 6 7 8 9 11 13 14 15 16 17 18 19)
gen=()
# gen=(4 6 7)
# gen=()
# rm -rf temp/${date}
#rm -rf ~/midap_compile_test/${date}

verify_and_report() {
    gen_id=$1
    if [ $gen_id = ${gen[0]} ]
    then
        python tools/generate_binary.py -d temp/${date} -p ${dirs[$gen_id]} -itg -gen -v |& tee temp/${date}/${dirs[$gen_id]}/report.txt
    else
        python tools/generate_binary.py -d temp/${date} -p ${dirs[$gen_id]} -itg -gen -v &> temp/${date}/${dirs[$gen_id]}/report.txt
    fi
    echo "Verification for the network ${networks[$gen_id]} has been finished"
}

for i in ${save[*]}
do
    mkdir -p temp/${date}/${dirs[$i]}
    python tools/test_system.py ${flags} -n ${networks[$i]} -so -sd temp/${date} -sp ${dirs[$i]} #&> temp/${date}/${dirs[$i]}/compile.txt
done

for i in ${gen[*]}
do
    verify_and_report $i &
done
