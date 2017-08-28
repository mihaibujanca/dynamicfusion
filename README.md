DynamicFusion
============
Based on https://github.com/Nerei/kinfu_remake

This is lightweight, reworked and optimized version of Kinfu that was originally shared in PCL in 2011. 

Key changes/features:
* Performance has been improved by 1.6x factor (Fermi-tested)
* Code size is reduced drastically. Readability improved. 
* No hardcoded algorithm parameters! All of them can be changed at runtime (volume size, etc.)
* The code is made independent from OpenCV GPU module and PCL library. 

Dependencies:
* Fermi or Kepler or newer
* CUDA 5.0 or higher
* OpenCV 2.4.8 or higher (modules opencv_core, opencv_highgui, opencv_calib3d, opencv_imgproc, opencv_viz required). Make sure that WITH_VTK flag is enabled in CMake during OpenCV configuration. You will need to build OpenCV from source
* OpenNI v1.5.4 (for Windows can download and install from http://pointclouds.org/downloads/windows.html)
* GTest for testing
* Nanoflann (included in the repository)
* Boost (libraries system and filesystem, only used in the demo. Tested with [1.64.0](http://www.boost.org/users/history/version_1_64_0.html))
* Ceres solver

Implicit dependency (needed by opencv_viz):
* VTK 5.8.0 or higher. (apt-get install on linux, for windows please download and compile from www.vtk.org)

Building instructions:

Use cmake to build the project. Set BUILD_TESTS to ON to build tests as well.
To run, go to <build_directory>/bin/demo and pass a path to an .oni file.

For linux users, go to the root of the project and run
`chmod +x download_data` then `./download_data`. To run, use `./demo <project_root>/data/umbrella`