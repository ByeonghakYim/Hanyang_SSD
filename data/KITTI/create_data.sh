#!/bin/bash

root_dir=$HOME/data/KITTI/
bash_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for dataset in train val trainval 
do
  dst_file=$bash_dir/$dataset.txt
  if [ -f $dst_file ]
  then
    rm -f $dst_file
  fi
  echo "Create list for $dataset..."
  dataset_file=$root_dir/$dataset.txt

  img_file=$bash_dir/$dataset"_img.txt"
  cp $dataset_file $img_file
  sed -i "s/^/img\//g" $img_file
  sed -i "s/$/.png/g" $img_file

  label_file=$bash_dir/$dataset"_label.txt"
  cp $dataset_file $label_file
  sed -i "s/^/Annotations\//g" $label_file
  sed -i "s/$/.xml/g" $label_file

  paste -d' ' $img_file $label_file >> $dst_file

  rm -f $label_file
  rm -f $img_file

  # Generate image name and size infomation.
  if [ $dataset == "val" ]
  then
    $bash_dir/../../build/tools/get_image_size $root_dir $dst_file $bash_dir/$dataset"_name_size.txt"
  fi

done

cur_dir=$(cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
root_dir=$cur_dir/../..

cd $root_dir

redo=1
data_root_dir="$HOME/data/KITTI"
dataset_name="KITTI"
mapfile="$root_dir/data/$dataset_name/labelmap_kitti.prototxt"
anno_type="detection"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0

python $root_dir/data/KITTI/kitti2voc.py --data-root-dir=$data_root_dir --duplicate-car=1 --duplicate-ped=1 --duplicate-cyc=1
if [ -d $root_dir/examples/$dataset_name ]
then
	rm -r $root_dir/examples/$dataset_name
fi

extra_cmd="--encode-type=png --encoded"
if [ $redo ]
then
  extra_cmd="$extra_cmd --redo"
fi
for subset in train val trainval 
do
  python $root_dir/scripts/create_annoset.py --anno-type=$anno_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir $root_dir/data/$dataset_name/$subset.txt $data_root_dir/$db/$dataset_name"_"$subset"_"$db examples/$dataset_name
done

echo "Preparing LMDB Done!"