#pragma once

typedef struct Opt_State 	Opt_State;
typedef struct Opt_Plan 	Opt_Plan;
typedef struct Opt_Problem 	Opt_Problem;

// Parameters that are set once per initialization of Opt
// A zeroed-out version of this struct is a good default 
// for maximum speed on well-behaved problems
struct Opt_InitializationParameters {
	// If true (nonzero), all intermediate values and unknowns, are double-precision
	// On platforms without double-precision float atomics, this 
	// can be a drastic drag of performance.
	int doublePrecision;

	// Valid Values: 
    //  0. no verbosity 
    //  1. verbose solve 
    //  2. verbose initialization, autodiff and solve
    //  3. full verbosity (includes ptx dump)
	int verbosityLevel;

	// If true (nonzero), a cuda timer is used to collect per-kernel timing information
	// while the solver is running. This adds a small amount of overhead to every kernel.
	int collectPerKernelTimingInfo;

	// Default block size for kernels (in threads). 
	// Must be a positive multiple of 32; if not, will default to 256.
	int threadsPerBlock;
};

typedef struct Opt_InitializationParameters 	Opt_InitializationParameters;

// Allocate a new independant context for Opt
Opt_State* Opt_NewState(Opt_InitializationParameters params);

// load the problem specification including the energy function from 'filename' and
// initializer a solver of type 'solverkind' (currently only 'gaussNewtonGPU' 
// and 'LMGPU' are supported)
Opt_Problem* Opt_ProblemDefine(Opt_State* state, const char* filename, const char* solverkind);
void Opt_ProblemDelete(Opt_State* state, Opt_Problem* problem);


// Allocate intermediate arrays necessary to run 'problem' on the dimensions listed in 'dimensions'
// how the dimensions are used is based on the problem specification (see 'writing problem specifications')
Opt_Plan* Opt_ProblemPlan(Opt_State* state, Opt_Problem* problem, unsigned int* dimensions);
void Opt_PlanFree(Opt_State * state, Opt_Plan* plan);

// Set a solver-specific variable by name. For now, these values are "locked-in" after ProblemInit()
// Consult the solver-specific documentation for valid values and names
void Opt_SetSolverParameter(Opt_State* state, Opt_Plan* plan, const char* name, void* value);

// Run the solver until completion using the plan 'plan'. 'problemparams' are the problem-specific inputs 
// and outputs that define the problem, including images, graphs, and problem paramaters
// (see 'writing problem specifications').
void Opt_ProblemSolve(Opt_State* state, Opt_Plan* plan, void** problemparams);


// use these two functions to control the outer solver loop on your own. In between iterations,
// problem parameters can be inspected and updated.

// run just the initialization for a problem, but do not do any outer steps.
void Opt_ProblemInit(Opt_State* state, Opt_Plan* plan, void** problemparams);
// perform one outer iteration of the solver loop and return to the user.
// a zero return value indicates that the solver is finished according to its parameters
int Opt_ProblemStep(Opt_State* state, Opt_Plan* plan, void** problemparams);

// Return the result of the cost function evaluated on the current unknowns
// If the solver is initialized to not use double precision, the return value
// will be upconverted from a float before being returned
double Opt_ProblemCurrentCost(Opt_State* state, Opt_Plan* plan);
