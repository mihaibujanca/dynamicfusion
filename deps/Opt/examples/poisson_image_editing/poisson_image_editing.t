local W,H = Dim("W",0), Dim("H",1)
local X = Unknown("X", opt_float4,{W,H},0) -- unknown, initialized to base image
local T = Array("T", opt_float4,{W,H},1) -- inserted image
local M = Array("M", opt_float, {W,H},2) -- mask, excludes parts of base image
UsePreconditioner(false)

-- do not include unmasked pixels in the solve
Exclude(Not(eq(M(0,0),0)))

for x,y in Stencil { {1,0},{-1,0},{0,1},{0,-1} } do
    local e = (X(0,0) - X(x,y)) - (T(0,0) - T(x,y))
    Energy(Select(InBounds(x,y),e,0))
end
