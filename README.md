# Implémentation de BLIS 

## 1. Installation de BLIS 

Installation d'Anaconda : 
```
cd /tmp
url -O https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
bash Anaconda3-2019.10-Linux-x86_64.sh
source ~/.bashrc

``` 

Machines utilisées .... avec ... 

```
git clone https://github.com/flame/blis.git
``` 

Installation d'une version plus récente de gcc :
```
ldconfig -p | grep libpthread
cd ~/blis
sudo apt install build-essential
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install gcc-6 g++-6
sudo update-alternatives --install /usr/bin/gcc gcc /usr//bin//gcc-6 20 --slave /usr/bin//g++ g++ /usr/bin/g++-6
```

 Configuration de BLIS
```
./configure --enable-threading=openmp --enable-cblas  auto
make -j 4
make check
export BLIS_INSTALL_PATH=/usr/local;
sudo make install
``` 


## 2. Installation de numpy adossée à BLIS

Installation de numpy : 
```
git clone https://github.com/numpy/numpy

``` 
Changement de la configuration : 

``` 
cd numpy
cp site.cfg.example site.cfg
nano site.cfg 
``` 

Le fichier site.cfg doit être modifié de la manière suivante : 

``` 
[blis]
libraries = blis
library_dirs = /home/ubuntu/blis/lib/haswell
include_dirs = /home/ubuntu/blis/frame/compat/cblas/src 
runtime_library_dirs = /home/ubuntu/blis/lib/haswell  
``` 
Déclarer l'ordre : 
``` 
export NPY_BLAS_ORDER=blis
export NPY_LAPACK_ORDER='' 
``` 
Verification : 
``` 
python setup.py config 
``` 
Installation : 
``` 
python setup.py install 
``` 
























