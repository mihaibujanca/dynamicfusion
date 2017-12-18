local W,H = Dim("W",0), Dim("H",1)
local Offset = Unknown("Offset",opt_float2,{W,H},0)
local Angle = Unknown("Angle",opt_float,{W,H},1)			
local UrShape = Array("UrShape", opt_float2,{W,H},2) --original mesh position
local Constraints = Array("Constraints", opt_float2,{W,H},3) -- user constraints
local Mask = Array("Mask", opt_float, {W,H},4) -- validity mask for mesh
local w_fitSqrt = Param("w_fitSqrt", float, 5)
local w_regSqrt = Param("w_regSqrt", float, 6)

UsePreconditioner(true)
Exclude(Not(eq(Mask(0,0),0)))

--regularization
for x,y in Stencil { {1,0}, {-1,0}, {0,1}, {0, -1} } do
    local e_reg = w_regSqrt*((Offset(0,0) - Offset(x,y)) 
                             - Rotate2D(Angle(0,0),(UrShape(0,0) - UrShape(x,y))))          
    local valid = InBounds(x,y) * eq(Mask(x,y),0) * eq(Mask(0,0),0)
    Energy(Select(valid,e_reg,0))
end
--fitting
local e_fit = (Offset(0,0)- Constraints(0,0))
local valid = All(greatereq(Constraints(0,0),0))
Energy(w_fitSqrt*Select(valid, e_fit , 0.0))