local N = Dim("N",0)

local w_fitSqrt         = Param("w_fitSqrt", float, 0)
local w_regSqrt         = Param("w_regSqrt", float, 1)
local w_confSqrt        = 0.1
local Offset            = Unknown("Offset", opt_float3,{N},2)			--vertex.xyz, rotation.xyz <- unknown
local Angle             = Unknown("Angle", opt_float3,{N},3)		    --vertex.xyz, rotation.xyz <- unknown		
local RobustWeights     = Unknown("RobustWeights", opt_float,{N},4)	
local UrShape           = Array("UrShape", opt_float3, {N},5)		    --urshape: vertex.xyz
local Constraints       = Array("Constraints", opt_float3,{N},6)	    --constraints
local ConstraintNormals = Array("ConstraintNormals", opt_float3,{N},7)
local G                 = Graph("G", 8, "v0", {N}, 9, "v1", {N}, 10)
UsePreconditioner(true)

local robustWeight = RobustWeights(0)
--fitting
local e_fit = robustWeight*ConstraintNormals(0):dot(Offset(0) - Constraints(0))
local validConstraint = greatereq(Constraints(0), -999999.9)
Energy(w_fitSqrt*Select(validConstraint, e_fit, 0.0))

--RobustWeight Penalty
local e_conf = 1-(robustWeight*robustWeight)
e_conf = Select(validConstraint, e_conf, 0.0)
Energy(w_confSqrt*e_conf)

--regularization
local ARAPCost = (Offset(G.v0) - Offset(G.v1)) - Rotate3D(Angle(G.v0),UrShape(G.v0) - UrShape(G.v1))
Energy(w_regSqrt*ARAPCost)