# MobileNet-SSD-ChienChat.
---------------------------

Examples de depp learning MobileNet SSD classe 101 avec 2 classes calculées.
Les images se trouvent dans le répertoire home/xxx/data/ChientChat.
# ChienChat
```
----------Annotations     
---------------------chat     
-------------------------000011.xml......    
---------------------chien    
-------------------------000018.xml .....        
----------JPEGImages                         
---------------------chat                        
--------------------------000011.jpg .....                 
---------------------chien                    
--------------------------000018.jpg .....              
-----------test.txt                         
-----------trainval.txt                   
```
test.txt et trainval.txt comprennent les noms des xml développés par labelImg.
chat/000011
.....
chien/000018
......
Fichier labelmap.prototxt
```
item {
  name: "none_of_the_above"
  label: 0
  display_name: "background"
}
item {
  name: "chat"
  label: 1
  display_name: "chat"
}
item {
  name: "chien"
  label: 2
  display_name: "chien"
}
```
-------------------------------------------------------------------------
# 1-Creation lmdb
On place les fichiers dans le répertoire ssd-caffe, soit:
opt/movidius/ssd-caffe/examples/ChienChat.
On se place dans ce répertoire.
```./create_lmdb.sh     ```
Ce fichier va créer un fichier test.txt, trainval.txt et test_name_size.txt
et les répertoires test_lmdb et trainval_lmdb

Si on a une erreur dans le train, il faut vérifier le bon format dans le fichier test_name__size.


# 2-Creation fichier prototxt
On lance ```./gen_model```
Ce fichier va générer 3 files prototxt dans le répertoire example.

# 3-Train
On lance ```./train.sh```

Dans le cas ou on aurait une erreur memory out Cuda, cela provient du type de carte.



On modifie le fichier MobileNetSSD_train.prototxt le nombre batch_size.
Dans mon cas, je passe à 10.
```
  data_param {
    source: "trainval_lmdb/"
    batch_size: 8
    backend: LMDB
```
Le model commence à être utilisable vers 3000 iteration. (se trouvant dans le répertoire snapshot).
# 4-Essai
On lance:
```python demo.py```
Demo utilise le caffemodel dans snapshot.
Les images se trouvent dans image/*.jpg
Celui-ci étant calculé par le train.

```python test.py ``` est une demo avec la webcam.
On specifie le prototxt et caffemodel utilisé.
La base de data doit être définie.

# 5-Raspberry
myssd et make file sont des fichiers C++ donnant un example de caméra sur raspberry pi 3b+.
Les librairies ncsdk (movidius)  et opencv doivent être installée avant.
