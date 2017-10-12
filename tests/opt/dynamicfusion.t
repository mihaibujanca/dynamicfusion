local K = Dim("K",0)
local WARP_SIZE = Dim("WARP_SIZE",1)

local live_vertex = Param("live_vertex", float3, 0)
local live_normal = Param("live_normal", float3, 1)
local canonical_vertex = Param("canonical_vertex", float3, 2)
local canonical_normal = Param("canonical_normal", float3, 3)

UsePreconditioner(true)

function tukey(x, c) -- use 0.1 for c
    if lesseq(abs(x),c) then
        return x * pow(1.0 - (x * x) / (c * c), 2)
    else
        return 0.0
    end
end
total_translation = 0
Energy(tukey(total_translation, 0.1))