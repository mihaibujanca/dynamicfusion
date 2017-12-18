-- Switch to double to check for precision issues in the solver
-- using double incurs bandwidth, compute, and atomic performance penalties
if _opt_double_precision then
	opt_float =  double
else
	opt_float =  float
end