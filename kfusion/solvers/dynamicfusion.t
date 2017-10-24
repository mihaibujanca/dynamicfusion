local D,N = Dim("D",0), Dim("N",1)

local RotationDeform = Unknown("RotationDeform", opt_float3,{D},0)
local TranslationDeform = Unknown("TranslationDeform", opt_float3,{D}, 1)

local CanonicalVertices = Array("CanonicalVertices", opt_float3,{N}, 2)
local LiveVertices = Array("LiveVertices", opt_float3,{N},3)

local CanonicalNormals = Array("CanonicalNormals", opt_float3,{N}, 4)
local LiveNormals = Array("LiveNormals", opt_float3,{N},5)

local Weights = Array("Weights", opt_float8, {N}, 6)

local G = Graph("DataG", 7,
                    "v", {N}, 8,
                    "n0", {D}, 9,
                    "n1", {D}, 10,
                    "n2", {D}, 11,
                    "n3", {D}, 12,
                    "n4", {D}, 13,
                    "n5", {D}, 14,
                    "n6", {D}, 15,
                    "n7", {D}, 16)

weightedTranslation = 0

nodes = {0,1,2,3,4,5,6,7}

for _,i in ipairs(nodes) do
    weightedTranslation = weightedTranslation + Weights(G.v)(i) * TranslationDeform(G["n"..i])
end

Energy(RotationDeform(0))
Energy(LiveVertices(G.v) - CanonicalVertices(G.v) - weightedTranslation)
