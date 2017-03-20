cd /home/byeonghak/Hanyang_SSD
./build/tools/caffe train \
--solver="models/SSD_v2/KITTI/inha/SSD_V2600x150/solver.prototxt" \
--weights="models/ImageNet_inception_v2_8_preActRes_iter_100000.caffemodel" \
--gpu 0 2>&1 | tee jobs/SSD_v2/KITTI/inha/SSD_V2600x150/KITTI_SSD_V2600x150_2017-3-21-0:37:30.log
