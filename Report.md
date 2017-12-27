# Project Report

##### [Project repository](https://github.com/mihaibujanca/dynamicfusion) | [Commits list](https://github.com/search?utf8=%E2%9C%93&q=repo%3Amihaibujanca%2Fdynamicfusion+committer%3Amihaibujanca+author-date%3A%3A%3C2017-08-30&type=Commits) | [PR to OpenCV]()

### Brief
The aim of this project was to implement [DynamicFusion](http://grail.cs.washington.edu/projects/dynamicfusion/papers/DynamicFusion.pdf), published in 2015 by Newcombe et al., which introduces an RGBD SLAM system capable of reconstructing 3D models of non-rigidly deforming scenes.
The method represents the scene as a canonical model and a warp field that transforms the model at each frame, in order
to distinguish between already observed data and data that needs to be added to the model. 
The system employs KinectFusion [Newcombe et al., 2011] as a 3D reconstruction engine, with an altered pipeline:

* Get a new frame 
* Estimate warp field parameters (align warp field with frame) 
* Apply warp on the canonical model to transform it into the frame 
* Perform data association 
* Update geometry with the new data
* Add new nodes to the warp field if needed 

### Work structure
The implementation is based off [an available implementation of KinectFusion](https://github.com/Nerei/kinfu_remake)

My work required the implementation of four main components, plus helper classes and putting everything in a pipeline:
##### 1. Implementation of the warp field structure, along with the warp function (and functions depending on it) - implemented

Includes the warp function that transforms vertices from the canonical model (the 3d reconstruction of the model) into the live frame,
the quaternion and dual quaternion classes and other related functionality.
##### 2. Implementation of the surface fusion algorithm, to add new data to the model - implemented

Includes functionality related to data association (classifying incoming vertices into new data and old data), and updating geometry of the model
 
##### 3. Estimation of the warp field parameters - partially implemented

This is one of the most important components and involves estimating the state of the warp field (adapting the warp field to match incoming frames).
This step involves an optimisation routine consisting of a data term and a regularisation term.
There is a significant amount of code for estimating the data term and updating the 
warp field with the new estimation, but no code for the regularisation term.

##### 4. Extending the warp field - stubbed out functionality
This step involves extending the warp field to cover the canonical model geometry. The warp field holds a 
subsampled version of the canonical model. As the canonical model grows with new data being fused in, the warp
field needs to grow as well, to support it.

The code for this step should be reasonably easy to write and is not expected to take long.

### Next steps

* Finalise implementation of the data term in warp field estimation. After this is done, 
the code should be able to reconstruct deforming scenes reasonably well, but is expected to break
when there are major changes between frames, or large portions of perviously occluded objects come into view rapidly
* Implement the regularisation term estimation routine
* Estimate the total energy using the data term and the regularisation term
* Improve speed of the code by rewriting CPU code for the GPU
* Better testing coverage
* Improve documentation
* Restructure the code into interfaces for incoming frames, pose estimation etc, to be able to work with 
devices other than kinect (e.g. Google Project Tango, that provides pose estimation from the IMU)
* Add code to color the models
* Export the reconstructions to .ply or .obj files.


---
I would like to thank my mentors Zhe Zhang and Reza Amayeh for all their support throughout GSoC. I have 
certainly learned a lot through this and have enjoyed working on the project.