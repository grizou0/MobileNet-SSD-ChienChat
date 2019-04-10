#!/bin/bash

root_dir=$HOME/data/ChienChat
bash_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "path data=$root_dir"
echo "path work=$bash_dir"
name=""
for dataset in trainval test
do
  dst_file=$bash_dir/$dataset.txt   #destination file
  if [ -f $dst_file ]               #efface si existe
  then
    rm -f $dst_file
  fi
    echo "Create list for $dataset..."
    dataset_file=$root_dir/$dataset.txt
    echo "dataset_file=$dataset_file"
    img_file=$bash_dir/$dataset"_img.txt"
    echo "img_file=$img_file"
    cp $dataset_file $img_file   #copy data trainval -> ssd-caffe trainval_img.txt
    sed -i "s/^/$name\/JPEGImages\//g" $img_file
    sed -i "s/$/.jpg/g" $img_file

#    sed -i "s/^/$name\/JPEGImages\//g" $img_file
#    sed -i "s/$/.jpg/g" $img_file

    label_file=$bash_dir/$dataset"_label.txt"
    cp $dataset_file $label_file
    sed -i "s/^/$name\/Annotations\//g" $label_file
    sed -i "s/$/.xml/g" $label_file
#    sed -i "s/^/$name\/Annotations\//g" $label_file
#    sed -i "s/$/.xml/g" $label_file
    paste -d' ' $img_file $label_file >> $dst_file

    rm -f $label_file
    rm -f $img_file       #delete trainval_img.txt
  # Generate image name and size infomation.
  if [ $dataset == "test" ]
  then
    $bash_dir/../../build/tools/get_image_size $root_dir $dst_file $bash_dir/$dataset"_name_size.txt"
  fi

  # Shuffle trainval file.
  if [ $dataset == "trainval" ]
  then
    rand_file=$dst_file.random
    cat $dst_file | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' > $rand_file
    mv $rand_file $dst_file
  fi
done
