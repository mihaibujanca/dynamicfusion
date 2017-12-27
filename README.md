DynamicFusion
============
Implementation of [Newcombe et al. 2015 DynamicFusion paper](http://grail.cs.washington.edu/projects/dynamicfusion/papers/DynamicFusion.pdf).

#### This project is still in active development and does not yet reproduce the results of the paper accurately.

The code is based on this [KinectFusion implemenation](https://github.com/Nerei/kinfu_remake)

## Building instructions:

### Ubuntu 16.04
Clone dynamicfusion and dependencies. 
```
git clone https://github.com/mihaibujanca/dynamicfusion --recursive
```

Install NVIDIA drivers.
- Enable NVidia drivers (Search / Additional Drivers) selecting:
	"Using NVIDIA binary driver - version 375.66 from nvidia-375 (proprietary, tested)"
	"Using processor microcode firmware for Intel CPUs from intel-microcode (proprietary)"
- Restart pc to complete installation

Alternatively a good tutorial with some common issues covered can be found [here](
              https://askubuntu.com/a/61433/167689).

For fresh installs (this assumes you cloned your project in your home directory!):
```
chmod +x build.sh
./build.sh
```

If you are not on a fresh install, check `build.sh` for building instructions and dependencies.

If you want to build the tests as well, set `-DBUILD_TESTS=ON`.
To save frames showing the reconstruction progress, pass `-DSAVE_RECONSTRUCTION_FRAMES=ON`. The frames will be saved in <project_root>/output

To build documentation, go to the project root directory and execute
```
doxygen -g
doxygen Doxyfile
```


### Running
```
./download_data 
./build/bin/dynamicfusion data/umbrella
```

### Windows
Dependencies:
* CUDA 5.0 or higher
* OpenCV 2.4.8 or higher (modules opencv_core, opencv_highgui, opencv_calib3d, opencv_imgproc, opencv_viz). Make sure that WITH_VTK flag is enabled in CMake during OpenCV configuration.
* Boost (libraries system, filesystem and program options. Only used in the demo. Tested with [1.64.0](http://www.boost.org/users/history/version_1_64_0.html))
* Ceres solver (Tested with version [1.13.0](http://ceres-solver.org/ceres-solver-1.13.0.tar.gz))

Implicit dependency (needed by opencv_viz):
* VTK 5.8.0 or higher
* SuiteSparse, BLAS and LAPACK for ceres
Optional dependencies:
* GTest for testing
* Doxygen for documentation
* OpenNI v1.5.4 for getting input straight from a kinect device.

[Install NVIDIA drivers](https://www.geforce.com/drivers) and [CUDA](https://developer.nvidia.com/cuda-downloads)
* [Install LAPACK](http://icl.cs.utk.edu/lapack-for-windows/lapack/).
* [Install VTK](http://www.vtk.org/download/) (download and build from source)
* [Install OpenCV](http://docs.opencv.org/3.2.0/d3/d52/tutorial_windows_install.html).  
* [Install Boost](http://www.boost.org/users/download/)

 
Optionals:
* [Doxygen](http://www.stack.nl/~dimitri/doxygen/download.html)
* [GTest](https://github.com/google/googletest) 
* [OpenNI]( http://pointclouds.org/downloads/windows.html)

[Download the dataset](http://lgdv.cs.fau.de/uploads/publications/data/innmann2016deform/umbrella_data.zip).\
Create a `data` folder inside the project root directory. \
Unzip the archive into `data` and remove any files that are not .png. \
Inside `data`, create directories `color` and `depth`, and move color and depth frames to their corresponding folders.

To use with .oni captures or straight from a kinect device, use `./build/bin/dynamicfusion_kinect <path-to-oni>` or `./build/bin/dynamicfusion_kinect <device_id>` 

---
Note: currently, the frame rate is too low (10s / frame) to be able to cope with live inputs, so it is advisable that you capture your input first.

## References
[DynamicFusion project page](http://grail.cs.washington.edu/projects/dynamicfusion/)

```
@InProceedings{Newcombe_2015_CVPR,
author = {Newcombe, Richard A. and Fox, Dieter and Seitz, Steven M.},
title = {DynamicFusion: Reconstruction and Tracking of Non-Rigid Scenes in Real-Time},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2015}
}
```

The example dataset is taken from the [VolumeDeform project](http://lgdv.cs.fau.de/publications/publication/Pub.2016.tech.IMMD.IMMD9.volume_6/).
```
@inbook{innmann2016volume,
author = "Innmann, Matthias and Zollh{\"o}fer, Michael and Nie{\ss}ner, Matthias and Theobalt, Christian 
         and Stamminger, Marc",
editor = "Leibe, Bastian and Matas, Jiri and Sebe, Nicu and Welling, Max",
title = "VolumeDeform: Real-Time Volumetric Non-rigid Reconstruction",
bookTitle = "Computer Vision -- ECCV 2016: 14th European Conference, Amsterdam, The Netherlands,
            October 11-14, 2016, Proceedings, Part VIII",
year = "2016",
publisher = "Springer International Publishing",
address = "Cham",
pages = "362--379",
isbn = "978-3-319-46484-8",
doi = "10.1007/978-3-319-46484-8_22",
url = "http://dx.doi.org/10.1007/978-3-319-46484-8_22"
}
```
