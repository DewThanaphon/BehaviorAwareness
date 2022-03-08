# BehaviorAwareness

My Master Degree Research.

Author: Thanaphon Rianthong

Contact: dew.thanaphon523@gmail.com

## How to install the OpenPose in a Jetson Nano

Firstly, getting start the Jetson Nano following this url: https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#intro

Then, install the OpenPose following this url: https://spyjetson.blogspot.com/2019/10/jetsonnano-human-pose-estimation-using.html

or following these commands:

check the cmake version
```bash
$ cmake --version
```
The cmake version should be 3.12.2 or higher

If the cmake version is lower than 3.12.2, please check the lastest version at **[here](https://github.com/Kitware/CMake/releases)**

At this time (2022/01/14), the lastest version is v3.22.1.

Then, following these commands for preparing to install OpenPose:

```bash
$ sudo apt-get update
$ sudo apt-get install -y libssl-dev libcurl4-openssl-dev qt5-default build-essential libboost-all-dev libboost-dev libhdf5-dev libatlas-base-dev python3-dev python3-pip
$ sudo apt-get remove -y cmake
$ cd /usr/local/src
$ sudo wget https://github.com/Kitware/CMake/releases/download/v3.22.1/cmake-3.22.1.tar.gz
$ sudo tar -xvzf cmake-3.22.1.tar.gz
$ cd cmake-3.22.1
$ sudo ./bootstrap --qt-gui
$ sudo make -j4
$ sudo make install
```

Then, you must restart your ssh session.

Next, install the OpenPose following these commands:

```bash
$ sudo apt-get install -y libprotobuf-dev protobuf-compiler libgflags-dev libgoogle-glog-dev
$ cd /usr/local/src
$ sudo wget https://github.com/CMU-Perceptual-Computing-Lab/openpose/archive/v1.7.0.tar.gz
$ sudo tar -xvzf v1.7.0.tar.gz
$ cd openpose-1.7.0/3rdparty
$ sudo git clone https://github.com/CMU-Perceptual-Computing-Lab/caffe.git
$ sudo git clone https://github.com/pybind/pybind11.git
```

Open cmake-gui:
```bash
$ cd ..
$ mkdir build
$ sudo cmake-gui
```
cmake-gui:

Set the directorys and Configure following a image:

![270865410_633347834579314_4176813386005096523_n](https://user-images.githubusercontent.com/92207106/149400197-a415637a-e65e-42bb-81be-b89a795e352f.png)

Set and Finish following a image:

![271485156_464680985260430_2617461527686066176_n](https://user-images.githubusercontent.com/92207106/149400472-e3fdd050-f4b3-4c4f-ab52-ec0a35d431b1.png)

Wait the configure process done.

Enable BUILD_PYTHON OPTION:

![269712800_422121446327661_4547882833180041178_n](https://user-images.githubusercontent.com/92207106/149400617-08e63a38-f8b0-42bb-a2ce-7de9e3edaa03.png)

Enable DOWNLOAD_BODY_COCO_MODEL & DOWNLOAD_BODY_MPI_MODEL:

![270682121_328922122422171_2132494908907320135_n](https://user-images.githubusercontent.com/92207106/149400763-2dc654b0-1479-442e-a72c-6190c7909c6d.png)

Disable USE_CUDNN option:

![271465204_659265595255142_5952516023144944446_n](https://user-images.githubusercontent.com/92207106/149400815-926ba05e-2616-4606-b208-d60f1eb16db9.png)

Generate:

![271465204_659265595255142_5952516023144944446_n](https://user-images.githubusercontent.com/92207106/149401014-02b5a984-6611-4091-ab40-db745ffa06b0.png)

Wait the generate process done and close the cmake-gui.

Then, following these commands:
```bash
$ cd build
$ sudo make -j4
$ sudo make install
$ cd python
$ sudo make -j4
$ sudo make install
$ sudo cp -r openpose /usr/lib/python3/dist-packages
```

You can test the completed install by:
```bash
$ python3
>>> import openpose
```

***If you can import the openpose with no error, it will be use.***

## Install necessary python3 library

```bash
$ pip3 install Cython
$ pip3 install numpy
$ pip3 install pandas
$ pip3 install scipy
$ pip3 install -U scikit-learn
$ pip3 install joblib
$ pip3 install dill
$ pip3 install opencv-python
```

## Run a program

```bash
$ cd "your directory of the program"
$ python3 "your python flie name"
```
