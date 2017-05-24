KinFu remake
============

This is lightweight, reworked and optimized version of Kinfu that was originally shared in PCL in 2011. 

Key changes/features:
* Performance has been improved by 1.6x factor (Fermi-tested)
* Code size is reduced drastically. Readability improved. 
* No hardcoded algorithm parameters! All of them can be changed at runtime (volume size, etc.)
* The code is made independent from OpenCV GPU module and PCL library. 

Dependencies:
* Fermi or Kepler or newer
* CUDA 5.0 or higher
* OpenCV 2.4.9 with new Viz module (only opencv_core, opencv_highgui, opencv_imgproc, opencv_viz modules required). Make sure that WITH_VTK flag is enabled in CMake during OpenCV configuration.
* OpenNI v1.5.4 (for Windows can download and install from http://pointclouds.org/downloads/windows.html)

Implicit dependency (needed by opencv_viz):
* VTK 5.8.0 or higher. (apt-get install on linux, for windows please download and compile from www.vtk.org)

Screenshot:
* https://raw.githubusercontent.com/Nerei/kinfu_remake/master/perf-39.5fps-Tesla-C2070.png

