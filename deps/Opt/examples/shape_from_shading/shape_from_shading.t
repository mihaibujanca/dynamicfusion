local DEPTH_DISCONTINUITY_THRE = 0.01
local W,H 	= Dim("W",0), Dim("H",1)

local w_p	    = sqrt(Param("w_p",float,0))-- Fitting weight
local w_s	    = sqrt(Param("w_s",float,1))-- Regularization weight
local w_g	    = sqrt(Param("w_g",float,2))-- Shading weight
local f_x	    = Param("f_x",float,3)
local f_y	    = Param("f_y",float,4)
local u_x 	    = Param("u_x",float,5)
local u_y 	    = Param("u_y",float,6)
local L = {}
for i=1,9 do -- lighting model parameters
	L[i] = Param("L_" .. i .. "",float,6+i)
end
local X 	    = Unknown("X",opt_float, {W,H},16) -- Refined Depth
local D_i 	    = Array("D_i",opt_float, {W,H},17) -- Depth input
local Im 	    = Array("Im",opt_float, {W,H},18) -- Target Intensity
local edgeMaskR = Array("edgeMaskR",uint8, {W,H},19) -- Edge mask. 
local edgeMaskC = Array("edgeMaskC",uint8, {W,H},20) -- Edge mask. 


local posX,posY = Index(0),Index(1)

-- equation 8
function p(offX,offY) 
    local d = X(offX,offY)
    local i = offX + posX
    local j = offY + posY
    return Vector(((i-u_x)/f_x)*d, ((j-u_y)/f_y)*d, d)
end

-- equation 10
function normalAt(offX, offY)
	local i = offX + posX -- good
    local j = offY + posY -- good
    
    local n_x = X(offX, offY - 1) * (X(offX, offY) - X(offX - 1, offY)) / f_y
    local n_y = X(offX - 1, offY) * (X(offX, offY) - X(offX, offY - 1)) / f_x
    local n_z = (n_x * (u_x - i) / f_x) + (n_y * (u_y - j) / f_y) - (X(offX-1, offY)*X(offX, offY-1) / (f_x*f_y))
    local sqLength = n_x*n_x + n_y*n_y + n_z*n_z
    local inverseMagnitude = Select(greater(sqLength, 0.0), 1.0/sqrt(sqLength), 1.0)
    return inverseMagnitude * Vector(n_x, n_y, n_z)
end

function B(offX, offY)
	local normal = normalAt(offX, offY)
	local n_x = normal[0]
	local n_y = normal[1]
	local n_z = normal[2]

	return           L[1] +
					 L[2]*n_y + L[3]*n_z + L[4]*n_x  +
					 L[5]*n_x*n_y + L[6]*n_y*n_z + L[7]*(-n_x*n_x - n_y*n_y + 2*n_z*n_z) + L[8]*n_z*n_x + L[9]*(n_x*n_x-n_y*n_y)
end

function I(offX, offY)
	return Im(offX,offY)*0.5 + 0.25*(Im(offX-1,offY)+Im(offX,offY-1))
end

local function DepthValid(x,y) return greater(D_i(x,y),0) end
 
local function B_I(x,y)
    local bi = B(x,y) - I(x,y)
    local valid = DepthValid(x-1,y)*DepthValid(x,y)*DepthValid(x,y-1)
    return Select(InBoundsExpanded(0,0,1)*valid,bi,0)
end
B_I = ComputedArray("B_I", {W,H}, B_I(0,0))


-- do not include unknowns for where the depth is invalid
Exclude(Not(DepthValid(0,0)))

-- fitting term
local E_p = X(0,0) - D_i(0,0)
Energy(Select(DepthValid(0,0),w_p*E_p,0))

-- shading term
local E_g_h = (B_I(0,0) - B_I(1,0))*edgeMaskR(0,0)
local E_g_v = (B_I(0,0) - B_I(0,1))*edgeMaskC(0,0)
Energy(Select(InBoundsExpanded(0,0,1),w_g*E_g_h,0))
Energy(Select(InBoundsExpanded(0,0,1),w_g*E_g_v,0))

-- regularization term
local function Continuous(x,y) return less(abs(X(0,0) - X(x,y)),DEPTH_DISCONTINUITY_THRE) end
local valid = DepthValid(0,0)*DepthValid(0,-1)*DepthValid(0,1)*DepthValid(-1,0)*DepthValid(1,0)*
                  Continuous(0,-1)*Continuous(0,1)*Continuous(-1,0)*Continuous(1,0)*InBoundsExpanded(0,0,1)
local validArray = ComputedArray("valid", {W,H},valid)
valid = eq(validArray(0,0),1)
local E_s = 4.0*p(0,0) - (p(-1,0) + p(0,-1) + p(1,0) + p(0,1)) 
Energy(Select(valid,w_s*E_s,0))