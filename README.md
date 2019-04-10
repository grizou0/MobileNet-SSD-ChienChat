# MobileNet-SSD-ChienChat
Examples deep learning MobileNetSSD classe 101 avec 2 classes classifiée
------------------------------------------------------------------------

Les images se trouvent dans le répertoire home/xxx/data/ChientChat.
ChienChat
----------Annotations
---------------------chat
-------------------------000011.xml......
---------------------chien
-------------------------000018.xml .....
----------JPEGImages
---------------------chat
--------------------------000011.jpg .....
---------------------chien
--------------------------000018.jpg
-----------test.txt
-----------trainval.txt
test.txt et trainval.txt comprennent les noms des xml développés par labelImg.
chat/000011
.....
chien/000018
......
-------------------------------------------------------------------------
# 1-Creation lmdb
On place les fichiers dans le répertoire ssd-caffe, soit:
opt/movidius/ssd-caffe/examples/ChienChat.
On se place dans ce répertoire.
./create_lmdb.sh     
Ce fichier va créer un fichier test.txt, trainval.txt et test_name_size.txt
et les répertoires test_lmdb et trainval_lmdb

# 2-Creation fichier prototxt
On lance ./gen_model
# 3-Train
On lance ./train.sh
# 4-Essai
On lance:
python demo.py

