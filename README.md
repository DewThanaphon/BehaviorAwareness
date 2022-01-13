# BehaviorAwareness
My Master Degree Research.

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
