local W,H = Dim("W",0), Dim("H",1)
local w_fitSqrt = Param("w_fit", float, 0)
local w_regSqrt = Param("w_reg", float, 1)
local X = Unknown("X", opt_float2,{W,H},2)
local I = Array("I",opt_float,{W,H},3)
local I_hat_im = Array("I_hat",opt_float,{W,H},4)
local I_hat_dx = Array("I_hat_dx",opt_float,{W,H},5)
local I_hat_dy = Array("I_hat_dy",opt_float,{W,H},6)
local I_hat = SampledImage(I_hat_im,I_hat_dx,I_hat_dy)

local i,j = Index(0), Index(1)
UsePreconditioner(false)
-- fitting
local e_fit = w_fitSqrt*(I(0,0) - I_hat(i + X(0,0,0),j + X(0,0,1)))
Energy(e_fit)
-- regularization
for nx,ny in Stencil { {1,0}, {-1,0}, {0,1}, {0,-1} } do
	local e_reg = w_regSqrt*(X(0,0) - X(nx,ny))
    Energy(Select(InBounds(nx,ny),e_reg,0))
end