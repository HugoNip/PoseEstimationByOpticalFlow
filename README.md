## Introduction
This project shows how to use optical flow and direct method to estimate pose of camera.
## Requirements
### OpenCV
#### Required Packages
OpenCV  
OpenCV Contrib

### Eigen Package (Version >= 3.0.0)
#### Source
http://eigen.tuxfamily.org/index.php?title=Main_Page

#### Compile and Install
```
cd [path-to-Eigen]
mkdir build
cd build
cmake ..
make 
sudo make install 
```

#### Search Installing Location
```
sudo updatedb
locate eigen3
```

default location "/usr/include/eigen3"

### Sophus Package
#### Download
https://github.com/HugoNip/Sophus

#### Compile and Install
```
cd [path-to-pangolin]
mkdir build
cd build
cmake ..
make 
sudo make install 
```


## Compile this Project
```
mkdir build
cd build
cmake ..
make 
```

## Run
### Keypoint Matching by using ORB features

### Pose Estimation by Optical Flow
```
./optical_flow
```
### Pose Estimation by Direct Method
```
./direct_method
```

## Reference
[Source](https://github.com/HugoNip/slambook2/tree/master/ch8)
