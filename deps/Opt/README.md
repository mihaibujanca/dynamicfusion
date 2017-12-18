Opt
---

Opt (optlang.org) is a new language in which a user simply writes energy functions over image- or graph-structured unknowns, and a compiler automatically generates state-of-the-art GPU optimization kernels. Real-world energy functions compile directly into highly optimized GPU solver implementations with performance competitive with the best published hand-tuned, application-specific GPU solvers.

This is an alpha release of the software to get feedback on the expressiveness of the language. We are interested in seeing what problems can be expressed and what features will be necessary to support more problems.

As an alpha release, there are some things that are not complete that will be improved over time.

* Error reporting is limited and may be difficult to understand at times. Please report any confusing error message either as github issues or through e-mail.
* Somewhat sparse documentation. This file provides a good overview of the Opt system, and detailed comments can be found in Opt.h, but rigorous documentation is under active development. If you report particular places where the documentation is sparse we can focus there first!
* Code can only run on NVIDIA GPUs with a relatively modern version of CUDA (7.5)
* The library of built-in math functions is somewhat limited. For instance, it include vectors and mat3xvec3 multplication but doesn't include 4x4 matrix operations.

These issues will improve over time, but if you run into issues, just send us an email:

mmara at cs dot stanford dot edu

or open an issue on github (https://github.com/niessner/Opt); or if you feel like getting your hands dirty, submit a pull request with some changes!

See the [roadmap](https://github.com/niessner/Opt/blob/master/ROADMAP.md) for near-future changes.

See the [changelog](https://github.com/niessner/Opt/blob/master/CHANGELOG.md) for changes since the initial release.

### Prerequisites ###

Opt and all of its examples require a recent version of [Terra](https://github.com/zdevito/terra)/, and [CUDA 7.5](https://developer.nvidia.com/cuda-75-downloads-archive). On Windows we use Visual Studio 2013 for development/compilation, though other versions may also work. 

Download and unzip the [terra binary release for your platform, release-2016-03-25](https://github.com/zdevito/terra/releases). Add terra/bin to your $PATH environment variable (if you want to run the examples). *On windows, this binary release searches for the CUDA compiler binaries using the CUDA_PATH environment variable. The default installation of the CUDA 7.5 Toolkit sets these paths, but if you did a nonstandard installation or subsequently installed a different version of CUDA, you will need to (re)set the path yourself.* 

Our recommended directory structure for development is:

- /Optlang/Opt (this repo)
- /Optlang/terra (renamed from the fully qualified directory name from the release download)

If you change this repo structure, you must update Opt/API/buildOpt.bat's second argument on line 2 to point to the terra repo. If you compiled terra from scratch, its internal directory structure might also be different from the release, and you will need to update Opt/API/optMake.bat's line 3 to point to the terra binary.

The examples should run on all platforms. Use the included Visual Studio .sln files on Windows. On OS X and linux, use the included Makefiles. On OS X and linux, before running examples, you will need to build Opt itself. Make sure to download and install CUDA 7.5 before starting ([Linux](http://developer.download.nvidia.com/compute/cuda/7.5/Prod/docs/sidebar/CUDA_Installation_Guide_Linux.pdf),[OS X](http://developer.download.nvidia.com/compute/cuda/7.5/Prod/docs/sidebar/CUDA_Installation_Guide_Mac.pdf))

Example script to get Opt (and terra) on a fresh Ubuntu install:
    
    # Get Opt
    git clone https://github.com/niessner/Opt.git
    # Get the prerequisite utilities to download and unzip and compile if you don't already have them
    sudo apt-get install wget zip clang
    # Get terra
    wget https://github.com/zdevito/terra/releases/download/release-2016-03-25/terra-Linux-x86_64-332a506.zip
    unzip terra-Linux-x86_64-332a506.zip 
    mv terra-Linux-x86_64-332a506 terra
    
Example script to get and compile Opt on OS X:
    

    # Get Opt
    git clone https://github.com/niessner/Opt.git
    # Get terra
    wget https://github.com/zdevito/terra/releases/download/release-2016-03-25/terra-OSX-x86_64-332a506.zip
    unzip terra-OSX-x86_64-332a506.zip 
    mv terra-OSX-x86_64-332a506 terra

Build and run an example on either platform

    # Build Opt itself
    cd Opt/API/;make
    # Compile and run the image_warping example
    cd ../examples/image_warping/;make;./image_warping

Overview
========

Opt is composed of a library `libOpt.a` (`Opt.lib` under windows) and a header file `Opt.h`. An application links Opt and uses its API to define and solve optimization problems. Opt's high-level energy functions behave like shaders in OpenGL. They are loaded as your application runs using the `Opt_ProblemDefine` API.

See the Makefiles in the examples for instructions on how to link Opt into your applications. In particular, on OSX, you will need to add the following linker flags:

    # osx only
    OSXFLAGS += -pagezero_size 10000 -image_base 100000000
    
    clang++ main.cpp -o main.cpp $(OSXFLAGS) -std=c++11 -L$(OPTHOME)/lib -L$(OPTHOME)/include -lOpt -lterra -ldl -pthread

Using the Opt C/C++ API
=======================

    Opt_State* Opt_NewState(Opt_InitializationParameters params);
    
Allocate a new independant context for Opt. This takes a small parameter struct as input that can effect global Opt state, such as the precision it uses internally and for unknowns (float or double), amount of timing information gathered, and verbosity level.

*Note:* The default implementation of Opt uses a slow double-precision atomicAdd() implementation that is guaranteed to work on all hardware that Opt runs on. If you have a Maxwell-class GPU (or later), and wish to have significantly higher performance, Opt has an internal implementation of double-precision atomicAdd that is significantly faster; open a github issue or contact the developers if this is a high-priority want; it is straightforward to change, but is not considered a priority at the moment.
    
---
    
    Opt_Problem* Opt_ProblemDefine(Opt_State* state, const char* filename, const char* solverkind);

Load the energy specification from 'filename' and initialize a solver of type 'solverkind' (currently only two related solvers are supported: 'gaussNewtonGPU' and 'LMGPU', for Gauss-Newton and Levenberg-Marquadt solvers (with parallel PCG for the inner solves)).
See writing energy specifications for how to describe energy functions.

---

    void Opt_ProblemDelete(Opt_State* state, Opt_Problem* problem);

Delete memory associated with the Problem object.

---

    Opt_Plan* Opt_ProblemPlan(Opt_State* state, Opt_Problem* problem, unsigned int* dimensions);

Allocate intermediate arrays necessary to run 'problem' on the dimensions listed in 'dimensions'
How the dimensions are used is based on the problem specification (see 'binding values' in 'writing energy specifications')

---

    void Opt_PlanFree(Opt_State * state, Opt_Plan* plan);

Delete the memory associated with the plan.

---

    void Opt_SetSolverParameter(Opt_State* state, Opt_Plan* plan, const char* name, void* value);

Set a solver-specific variable by name. The value parameter must be a pointer to the actual value.
For now, these values are "locked-in" after Opt_ProblemInit()
Consult the solver-specific documentation for valid values and names.

Example:

    float linearIterationCount = 25;
    Opt_SetSolverParameter(m_state, m_plan, "lIterations", (void*)&linearIterationCount);

---

    Opt_ProblemSolve(Opt_State* state, Opt_Plan* plan, void** problemparams);

Run the solver until completion using the plan 'plan'. 'problemparams' are the problem-specific inputs 
and outputs that define the problem, including arrays, graphs, and problem paramaters
(see 'writing problem specifications').

---

    void Opt_ProblemInit(Opt_State* state, Opt_Plan* plan, void** problemparams);
    int Opt_ProblemStep(Opt_State* state, Opt_Plan* plan, void** problemparams);

Use these two functions to control the outer solver loop on your own. The arguments are the same as `Opt_ProblemSolve` but
the `Step` function returns between iterations of the solver. Problem parameters can be inspected and updated between calls to Step.
A zero return value indicates that the solver is finished according to its parameters.

    Opt_ProblemInit(...);
    while(Opt_ProblemStep(...) != 0) {
        // inspect and update problem state as desired.
    }
    // solver finished

___

    double Opt_ProblemCurrentCost(Opt_State* state, Opt_Plan* plan);

Return the result of the cost function evaluated on the current unknowns.
If the solver is initialized to not use double precision, the return value
will be upconverted from a float before being returned.

Useful for user-land evaluation of the convergence of the solve.



Writing Energy Specifications
==============================

Specifications of the energy are written using an API embedded in Lua. 
Similar to SymPy or Mathematica, objects with overloaded operators in Lua are used to build up a symbolic expression of the energy. There are Lua functions to declare objects: dimensions, arrays (including the unknowns to be solved for), and graphs.

These objects can be used to create residuals functions defined per-pixel in an array or per-edge in graph. The mathematical expressions of energy are built using overloaded operators defined on these objects. The 'Energy' function adds an expression to the overall energy of the system. 

A simple laplacian smoothing energy in this system would have the form:


    W = Dim("W",0) 
    H = Dim("H",1)
    X = Array("X",float,{W,H},0) 
    A = Array("A",float,{W,H},1)
    
    w_fit,w_reg = .1,.9
    
    -- overloaded operators allow you to defined mathematical expressions as energies
    fit = w_fit*(X(0,0) - A(0,0))
    
    -- register fitting energy
    Energy(fit) --fitting
    
    -- Energy function can be called multiple times or with multiple arguments
    -- to add more residual terms to the energy
    Energy(w_reg*(X(0,0) - X(1,0)),
           w_reg*(X(0,0) - X(0,1)))


The functions are described in more details below.

## Declaring the inputs/outputs of an energies ##
 
    dimension = Dim(name,dimensions_position)
    
Create a new dimension used to describe the size of Arrays. `dimensions_position` is the 0-based offset into the `dimensions` argument to `Opt_ProblemPlan` that will be bound to this value. See 'Binding Values'.

    local W =
    H = Dim("W",0), Dim("H",1)
    
---

    array = Array(name,type,dimlist,problemparams_position)
    array = Unknown(name,type,dimlist,problemparams_position)
    
Declare a new input to the problem (`Array`), or an unknown value to be solved for `Unknown`. Both return an Array object that can be used to formulate energies.

`name` is the name of the object, used for debugging 
`type` can be float, float2, float3, ...
`dimlist` is a Lua array of dimensions (e.g., `{W,H}`). Arrays can be 1, 2, or 3 dimensional.
`problemparams_position` is the 0-based offset into the `problemparams` argument to `Opt_ProblemSolve` that will be bound to this value. 

Examples:

    local Angle = Unknown("Angle",float, {W,H},1)
    local UrShape = Array("UrShape", float2,{W,H},2)
    
---

    graph = Graph(name, problemparams_position_of_graph_size,
                 {vertexname, dimlist, problemparams_position_of_indices}*)

Declare a new graph that connects arrays together through hyper-edges.

`name` is a string for debugging.
`problemparams_position_of_graph_size` is the 0-based offset into the `problemparams` argument to `Opt_ProblemSolve` that will determine the number of edges in the graph.

The remaining arguments are used to define vertices in the hyper-edge of the graph.
Each vertex requires the following arguments:

     vertexname, dimlist, problemparams_position_of_indices
     
`vertexname` is the name of the vertex used in the energy specification.
`dimlist` is a Lua array of dimensions (e.g., `{W,H}`). Arrays can be 1, 2, or 3 dimensional. This vertex will be a pointer into any array of this dimension.
`problemparams_position_of_indices` is the 0-based offset into the `problemparams` argument to `Opt_ProblemSolve` that is an array of indexes the size of the number of edges in the graph, where each entry is an index into the dimension specified in `dimlist`. For 2- or 3- dimensional arrays the indices for both dimensions are listed sequentially `(int,int)`.
    
Example:

    N = Dim("N",0)
    local Angle = Unknown("Angle", float3,{N},0)
    local G =  Graph("G", 1, "head", {N}, 2,
                             "tail", {N}, 3)
                             
    Energy(Angle(G.v0) - Angle(G.v1))

---

## Writing Energies ##

Energies are described using a mathematical expressions constructed using Lua object overloaded.

Values can be read from the arrays created with the `Array` or `Unknown` constructors. 

### Accessing values with Stencils or Graphs ###

    value = Angle(0,0) -- value of the 'Angle' array at the centered pixel
    value = Angle(1,0) -- value of the 'Angle' array at the pixel to the right of the centered pixel
    value = Angle(0,2) -- value of the 'Angle' array at the pixel two pixels above the centered pixel
    ...

Each expression is implicitly defined over an entire array or entire set of edges. 
Expressions are implicitly squared and summed over all domains since our solver is for non-linear least squared problems. Energies are described per-pixel or per-edge with, e.g., `Angle(0,0)`, as the centered pixel. Other constant offsets can be given to select neighbors.

To access values at graph locations you use the name of the vertex as the index into the array:

    N = Dim("N",0)
    local Angle = Unknown("Angle", float3,{N},0)
    local G =  Graph("G", 1, "head", {N}, 2,
                             "tail", {N}, 3)  

    value = Angle(G.head)
    value2 = Angle(G.tail)
    
### Math Operators ###

Generic math operators are usable on any value or vector:

    +
    -
    *
    /
    abs
    acos
    acosh
    and_
    asin
    asinh
    atan
    atan2
    classes
    cos
    cosh
    div
    eq
    exp
    greater
    greatereq
    less
    lesseq
    log
    log10
    mul
    not_
    or_
    pow
    prod
    sin
    sinh
    sqrt
    tan
    tanh
    Select(condition,truevalue,falsevalue) -- piecewise conditional operator, if condition ~= 0, it is truevalue, otherwise it is falsevalue
    scalar = All(vector) -- true if all values in the vector are true
    scalar = Any(vector) -- true of any value in the vector is true
    Rotate2D(angle, vector2)
    Rotate3D(angle3, vector3)


All operators apply elementwise to `Vector` objects.

Because Lua does not allow generic overloading of comparison ( `==` , '<=', ... ), you must use the functions we have provided instead for comparisions:
`eq(a,b)`, `lesseq(a,b)`, etc.


### Defining Energies ###

    `Energy(energy1,energy2,...)`
    
Add the terms `energy1`, ... to the energy of the whole problem. Energy terms are implicitly squared and summed over the entire domain (array or graph) on which they are defined.  Each channel of a `Vector` passed as an energy is treated as a separate energy term.


### Boundaries ###

For energies defined on arrays, it is possible to control how the energy behaves on the boundaries.  Any energy term has a particular pattern of data it reads from neighboring pixels in the arrays, which we call its `stencil`. By default, residual values are only defined for pixels in the array where the whole stencil has defined values. For a 3x3 stencil, for instance, this means that the 1-pixel border of an image will not evaluate this energy term (or equivalently, this term contributes 0 to the overall energy).

If you do not want the default behavior, you can use the `InBounds(x,y)` functions along with the `Select` function to describe custom behavior:

    customvalue = Select(InBounds(1,0),value_in_bounds,value_on_the_border) 

`InBounds` is true only when the relative offet `(1,0)` is in-bounds for the centered pixel. Any energy that uses `InBounds` will be evaluated at _every_ pixel including the border region, and it is up to the user to choose what to do about boundaries.

It is also possible to exclude arbitrary pixels from the solve using the `Exclude(exp)` method. When `exp` is true, unknowns defined at these pixels will not be updated and residuals at these pixels will not be evaluated.


### ComputedArray ###

Energy functions for neighboring pixels can share expensive-to-compute expressions. For instance,
our [shape-from-shading example](https://github.com/niessner/Opt/blob/master/examples/shape_from_shading/shape_from_shading.t#L67) uses an expensive lighting calculation that is shared by neighboring pixels. We allow the user to turn these calculations into computed arrays, which behave
like arrays when used in energy functions, but are defined as an expression of other arrays:

    -- B_I is previously defined as a function that uses 
    -- a large number of images. After this line, B_I can
    -- be accessed like a regular Array or Image
    B_I = ComputedArray("B_I", {W,H}, B_I(0,0))

Computed arrays can include computations using unknowns, and are recalculated as necessary during the optimization. Similar to scheduling annotations in Halide, they allow the user to balance recompute with locality at a high-level.


### Vectors ###

    vector = Vector(a,b,c)
    vector2 = vector:dot(vector)
    scalar = vector:sum()
    numelements = vector:size() -- 3 for this vector
    vector3 = vector + vector -- elementwise addition

Objects of type `float2`, `float3` ... are vectors. The function `Vector` constructs them from individual elements. All math is done elementwise to vectors, including functions like `abs`.


### Binding Values for the C/C++ API ###

To connect values passed in from C/C++ API to values in the energy specification, the functions  `Array`, `Unknown`, `Dim`, and `Graph` have an argument (e.g., `problemparams_position`) that binds the object in the energy specification to the argument at that numeric offset in the parameters passed to Opt. 

API Example:

    uint32_t dims[] = { width, height };
	Plan * m_plan = Opt_ProblemPlan(m_optimizerState, m_problem, dims);
	
Energy Specification:

    local W,H = Dim("W",0), Dim("H",1)
    
API Example:

	int   nLinearIterations = 8;
	int   nNonLinearIterations = 8;
    
	float weightFitSqrt = sqrt(weightFit);
	float weightRegSqrt = sqrt(weightReg);
	
	float * d_x = ... //raw image data for x in (H,W,channel) order
	float * d_a = ...
	float * d_urshape = ...
	float * d_constraints = ...
	float * d_mask = ...
	
	void* problemParams[] = { d_x, d_a, d_urshape, d_constraints, d_mask, &weightFitSqrt, &weightRegSqrt };
		
	Opt_SetSolverParameter(m_optimizerState, m_plan, "nIterations", (void*)&nNonLinearIterations);
	Opt_SetSolverParameter(m_optimizerState, m_plan, "lIterations", (void*)&nLinearIterations);
	Opt_ProblemSolve(m_optimizerState, m_plan, problemParams);
    
Energy Specification:
    
    local Offset = Unknown("Offset",float2,{W,H},0)
    local Angle = Unknown("Angle",float,{W,H},1)
    local UrShape = 	Array("UrShape", float2,{W,H},2)		
    local Constraints = Array("Constraints", float2,{W,H},3)	
    local Mask = 		Array("Mask", float, {W,H},4)	
    local w_fitSqrt = Param("w_fitSqrt", float, 5)
    local w_regSqrt = Param("w_regSqrt", float, 6)


### Helpers ###

    for x,y in Stencil { {1,0},{-1,0},{0,1},{0,-1} } do
        Energy(X(0,0) - X(x,y)) -- laplacian regularization
    end
    
    -- equivalent to
    Energy(X(0,0) - X(1,0)) -- laplacian regularization
    Energy(X(0,0) - X(-1,0)) -- laplacian regularization
    Energy(X(0,0) - X(0,1)) -- laplacian regularization
    Energy(X(0,0) - X(0,-1)) -- laplacian regularization
    
The function `Stencil` is a Lua iterator that makes it easy to define similar energy functions for a set of neighboring offsets.

There currently is no equivalent iterator for nodes in a graph hyperedge, but for the time being you can roll your own using standard Lua functionality. For example:

	G = Graph("G", 3, "v0", 4, "v1", 5, "v2", 6, "v3", 7)
	nodes = {"v0","v1","v2","v3"}
	local result = 1.0
	for _,n in ipairs(nodes) do
		result = result * X(G[n])
	end
	Energy(result)


Solver Parameters
=================

Solver parameters are set using the Opt_SetSolverParameter() function.

    float linearIterationCount = 25;
    Opt_SetSolverParameter(m_state, m_plan, "lIterations", (void*)&linearIterationCount);

Both 'gaussNewtonGPU' and 'LMGPU' solvers have two integer (int) parameters. Here we list them, along
with default values

    nIterations = 10 // the number of non-linear iterations
    lIterations = 10 // the number of linear iterations in each non-linear step

The 'LMGPU' solver is based off of Ceres's (http://ceres-solver.org/) version of Levenberg-Marquadt 
and borrows the rest of its parameter names from that. All LMGPU-exclusive parameters are floats. We
list them all here for completeness, consult the [Ceres documentation](http://ceres-solver.org/) for
more details until we flesh out this section of the documentation.

    min_relative_decrease = 1e-3,
    min_trust_region_radius = 1e-32,
    max_trust_region_radius = 1e16,
    q_tolerance = 0.0001,
    function_tolerance = 0.000001,
    trust_region_radius = 1e4,
    radius_decrease_factor = 2.0,
    min_lm_diagonal = 1e-6,
    max_lm_diagonal = 1e32,

Initial Guess
=================
Opt uses the values of the unknown array you pass into it as the initial guess for the solve. Since nonlinear least-square solvers only find a local minimum, it is best if you provide Opt with a reasonable initial guess. In the absence of outside information, at least memset the values of the unknown so that Opt doesn't start with garbage data for the initial guess, which may contain infinities or even NaNs!

### Common Problems ###

1. All examples/tests immediately quit with an error message about cudalib being nil: 
	- Your Terra installation likely does not have CUDA enabled. You can double check by running the Terra tests, and seeing if the cuda tests fail. Remember to install CUDA 7.5 before running Opt. On windows, the binary release searches for the CUDA compiler binaries using the CUDA_PATH environment variable. The default installation of the CUDA 7.5 Toolkit sets these paths, but if you did a nonstandard installation or subsequently installed a different version of CUDA, you will need to (re)set the path yourself.
2. "The program can't start because cudart64_75.dll is missing from your computer. Try reinstalling the program to fix this problem.": 
	- CUDA 7.5 is not on your path. Perhaps you didn't install it, or haven't closed Visual Studio since installing it. If you have done both, you'll need to manually add it to your path environment variable. By default, the path will be "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\bin"
3. "The program can't start because terra.dll is missing from your computer. Try reinstalling the program to fix this problem." 
	- Terra is not on your path. Perhaps you didn't add it to your $PATH environment variable, or haven't closed Visual Studio since adding it.
4. My initial cost is NaN (or infinity) and Opt doesn't improve it.
	- You may have provided garbage initialization to Opt, see the "Initial Guess" section above.
5. When trying to compile the graph-examples, I get an 'undefined reference' error for some functions in the OpenMesh library.
	- There may be some incompatibility with the distributed OpenMesh binary and your architecture ([this has been reported on some variant of linux](https://github.com/niessner/Opt/issues/105)). While we will try and find a more suitable fix, you should be able to work around it for now by compiling your own version of [OpenMesh](https://www.openmesh.org/).
6. My problem size changes at runtime, and recompiling every solve is too slow for my application.
	- We are working on a change for this (see the roadmap), but cannot guarantee it will be available by the time you need it. In the meantime, consider compiling once for a "maximum" edge size, making a dummy node in your graph that indexes into a small image that encodes whether the edge is valid or not, and multiplying your energy term with 0 when the edge is invalid.
