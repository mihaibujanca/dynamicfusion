local N = opt.Dim("N",0)
local w_fitSqrt =   Param("w_fitSqrt", float, 0)
local w_regSqrt =   Param("w_regSqrt", float, 1)
local Offset =      Unknown("Offset", opt_float3,{N},2)            --vertex.xyz, rotation.xyz <- unknown
local Angle = 	    Unknown("Angle",opt_float3,{N},3)	
local UrShape =     Array("UrShape",opt_float3,{N},4)        --original position: vertex.xyz
local Constraints = Array("Constraints",opt_float3,{N},5)    --user constraints
local G = Graph("G", 6, "v0", {N}, 7, "v1", {N}, 8)
UsePreconditioner(true)

--fitting
local e_fit = Offset(0) - Constraints(0)
local valid = greatereq(Constraints(0,0), -999999.9)
Energy(Select(valid,w_fitSqrt*e_fit,0))

--regularization
local ARAPCost = (Offset(G.v0) - Offset(G.v1)) - Rotate3D(Angle(G.v0),UrShape(G.v0) - UrShape(G.v1))
Energy(w_regSqrt*ARAPCost)