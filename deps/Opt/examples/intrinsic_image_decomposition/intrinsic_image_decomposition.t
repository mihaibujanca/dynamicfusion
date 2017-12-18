W,H = opt.Dim("W",0), opt.Dim("H",1)
local w_fitSqrt         = Param("w_fitSqrt", float, 0)
local w_regSqrtAlbedo   = Param("w_regSqrtAlbedo", float, 1)
local w_regSqrtShading  = Param("w_regSqrtShading", float, 2)
local pNorm             = Param("pNorm", opt_float, 3)
local r                 = Unknown("r", opt_float3,{W,H},4)
local r_const           = Array("r_const", opt_float3,{W,H},4) -- A constant view of the unknown
local i                 = Array("i", opt_float3,{W,H},5)
local s                 = Unknown("s", opt_float,{W,H},6)

-- reg Albedo
for x,y in Stencil { {1,0}, {-1,0}, {0,1}, {0,-1} } do
	local diff = (r(0,0) - r(x,y))
	local diff_const = (r_const(0,0) - r_const(x,y))
    -- The helper L_p function takes diff_const, raises it's length to the (p-2) power, 
    -- and stores it in a computed array, so its value remains constant during the nonlinear iteration,
    -- then multiplies it with diff and returns
    local laplacianCost = L_p(diff, diff_const, pNorm, {W,H})
    local laplacianCostF = Select(InBounds(0,0),Select(InBounds(x,y), laplacianCost,0),0)
    Energy(w_regSqrtAlbedo*laplacianCostF)
end

-- reg Shading
for x,y in Stencil { {1,0}, {-1,0}, {0,1}, {0,-1} } do
    local diff = (s(0,0) - s(x,y))
    local laplacianCostF = Select(InBounds(0,0),Select(InBounds(x,y), diff,0),0)
    Energy(w_regSqrtShading*laplacianCostF)
end

-- fit
local fittingCost = r(0,0)+s(0,0)-i(0,0)
Energy(w_fitSqrt*fittingCost)