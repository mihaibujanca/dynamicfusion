local W,H,D = Dim("W",0), Dim("H",1), Dim("D",2)

local Offset =      Unknown("Offset",   opt_float3,{W,H,D},0)            --vertex.xyz, rotation.xyz <- unknown
local Angle = 	    Unknown("Angle",    opt_float3,{W,H,D},1)	
local UrShape =     Array("UrShape",    opt_float3,{W,H,D},2)        --original position: vertex.xyz
local Constraints = Array("Constraints",opt_float3,{W,H,D},3)    --user constraints
local w_fitSqrt =   Param("w_fitSqrt",  float, 4)
local w_regSqrt =   Param("w_regSqrt",  float, 5)
UsePreconditioner(true)

--fitting
local e_fit = Offset(0,0,0) - Constraints(0,0,0)
local valid = greatereq(Constraints(0,0,0)(0), -999999.9)
Energy(Select(valid,w_fitSqrt*e_fit,0))


for i,j,k in Stencil { {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}} do
	local ARAPCost = (Offset(0,0,0) - Offset(i,j,k)) - Rotate3D(Angle(0,0,0),UrShape(0,0,0) - UrShape(i,j,k))
	local ARAPCostF = Select(InBounds(0,0,0),	Select(InBounds(i,j,k), ARAPCost, 0.0), 0.0)
	Energy(w_regSqrt*ARAPCostF)
end