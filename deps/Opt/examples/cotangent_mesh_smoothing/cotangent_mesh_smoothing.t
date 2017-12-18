local N = Dim("N",0)

local w_fitSqrt = Param("w_fit", float, 0)
local w_regSqrt = Param("w_reg", float, 1)
local X = Unknown("X", opt_float3,{N},2)
local A = Array("A", opt_float3,{N},3)
local G = Graph("G", 4, "v0", {N}, 5, --current vertex
                            "v1", {N}, 6, --neighboring vertex
                            "v2", {N}, 7, --prev neighboring vertex
                            "v3", {N}, 8) --next neighboring vertex

UsePreconditioner(true)

function cot(v0, v1) 
	local adotb = Dot3(v0, v1)
	local disc = Dot3(v0, v0)*Dot3(v1, v1) - adotb*adotb
	disc = Select(greater(disc, 0.0), disc,  0.0001)
	return Dot3(v0, v1) / Sqrt(disc)
end

-- fit energy
Energy(w_fitSqrt*(X(0) - A(0)))

local a = normalize(X(G.v0) - X(G.v2)) --float3
local b = normalize(X(G.v1) - X(G.v2))	--float3
local c = normalize(X(G.v0) - X(G.v3))	--float3
local d = normalize(X(G.v1) - X(G.v3))	--float3

--cotangent laplacian; Meyer et al. 03
local w = 0.5*(cot(a,b) + cot(c,d))
w = Sqrt(Select(greater(w, 0.0), w, 0.0001))
Energy(w_regSqrt*w*(X(G.v1) - X(G.v0)))

