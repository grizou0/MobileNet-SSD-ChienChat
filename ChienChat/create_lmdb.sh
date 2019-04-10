#!/bin/bash

data_root_dir=$HOME/data/ChienChat
bash_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "path data=$data_root_dir"
echo "path work=$bash_dir"
# Affiche  path data=/home/jp/data/ChienChat et path work=//opt/movidius/ssd-caffe/examples/ChienChat

name=""
echo "create_list"
echo "-----------"
for dataset in trainval test  #boucle pour test.txt et trainval.txt
do
  dst_file=$bash_dir/$dataset.txt   #destination file
  if [ -f $dst_file ]               #efface si existe
  then
    rm -f $dst_file
  fi
    echo "Create list for $dataset..."
    dataset_file=$data_root_dir/$dataset.txt
    echo "dataset_file=$dataset_file"
    img_file=$bash_dir/$dataset"_img.txt"
    echo "img_file=$img_file"
    cp $dataset_file $img_file   #copy data trainval -> ssd-caffe trainval_img.txt
    sed -i "s/^/$name\/JPEGImages\//g" $img_file
    sed -i "s/$/.jpg/g" $img_file
    label_file=$bash_dir/$dataset"_label.txt"
    cp $dataset_file $label_file
    sed -i "s/^/$name\/Annotations\//g" $label_file
    sed -i "s/$/.xml/g" $label_file
    paste -d' ' $img_file $label_file >> $dst_file

    rm -f $label_file
    rm -f $img_file       #delete trainval_img.txt
  # Generate image name and size infomation.
  if [ $dataset == "test" ]
  then
    $bash_dir/../../build/tools/get_image_size $data_root_dir $dst_file $bash_dir/$dataset"_name_size.txt"
  fi

  # Shuffle trainval file.
  if [ $dataset == "trainval" ]
  then
    rand_file=$dst_file.random
    cat $dst_file | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' > $rand_file
    mv $rand_file $dst_file
  fi
done
# Create_data
#----------------------------------------------------------------------------------------
echo "Create_data"
echo "-----------"

root_dir=$bash_dir/../../../../../   #return to root 

cd $root_dir
rm -Rf $bash_dir/"test_lmdb"
rm -Rf $bash_dir/"trainval_lmdb"
redo=1
anno_type="detection"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0

extra_cmd="--encode-type=jpg --encoded"
if [ $redo ]
then
  extra_cmd="$extra_cmd --redo"
fi
for subset in test trainval
do
python opt/movidius/ssd-caffe/scripts/create_annoset.py --anno-type=$anno_type \
--label-map-file=$bash_dir/"labelmap.prototxt" \
--min-dim=$min_dim \
--max-dim=$max_dim \
--resize-width=$width \
--resize-height=$height \
--check-label $extra_cmd  \
$data_root_dir  \
$bash_dir/$subset".txt" \
$bash_dir/$subset"_lmdb" \
$bash_dir/
done




