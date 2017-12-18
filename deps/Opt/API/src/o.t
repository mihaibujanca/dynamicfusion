opt = {} --anchor it in global namespace, otherwise it can be collected
local S = require("std")
local ffi = require("ffi")
local util = require("util")
local optlib = require("lib")
ad = require("ad")
require("version")
require("precision")
local A = ad.classes

local C = util.C

local use_pitched_memory = true
local use_split_sums = true
local use_condition_scheduling = true
local use_register_minimization = true
local use_conditionalization = true
local use_contiguous_allocation = false
local use_bindless_texture = true and (not use_contiguous_allocation)
local use_cost_speculate = false -- takes a lot of time and doesn't do much

if false then
    local fileHandle = C.fopen("crap.txt", 'w')
    C._close(1)
    C._dup2(C._fileno(fileHandle), 1)
    C._close(2)
    C._dup2(C._fileno(fileHandle), 2)
end

-- constants
local verboseSolver = _opt_verbosity > 0
local verboseAD 	= _opt_verbosity > 1

local vprintfname = ffi.os == "Windows" and "vprintf" or "cudart:vprintf"
local vprintf = terralib.externfunction(vprintfname, {&int8,&int8} -> int)

if verboseSolver then
    print("Using Opt "..opt_version_string)
end

local function createbuffer(args)
    local Buf = terralib.types.newstruct()
    for i,e in ipairs(args) do
        local typ = e:gettype()
        local field = "_"..tonumber(i)
        typ = typ == opt_float and double or typ
        table.insert(Buf.entries,{field,typ})
    end
    return quote
        var buf : Buf
        escape
            for i,e in ipairs(args) do
                emit quote
                   buf.["_"..tonumber(i)] = e
                end
            end
        end
    in
        [&int8](&buf)
    end
end

printf = macro(function(fmt,...)
    local buf = createbuffer({...})
    return `vprintf(fmt,buf) 
end)
local dprint

if verboseSolver then
	logSolver = macro(function(fmt,...)
		local args = {...}
		return `C.printf(fmt, args)
	end)
else
	logSolver = macro(function(fmt,...)
		return 0
	end)
end

if verboseAD then
	logAD = macro(function(fmt,...)
		local args = {...}
		return `C.printf(fmt, args)
	end)
	dprint = print
else
	logAD = macro(function(fmt,...)
		return 0
	end)
	dprint = function() end
end


local GPUBlockDims = {{"blockIdx","ctaid"},
              {"gridDim","nctaid"},
              {"threadIdx","tid"},
              {"blockDim","ntid"}}
for i,d in ipairs(GPUBlockDims) do
    local a,b = unpack(d)
    local tbl = {}
    for i,v in ipairs {"x","y","z" } do
        local fn = cudalib["nvvm_read_ptx_sreg_"..b.."_"..v] 
        tbl[v] = `fn()
    end
    _G[a] = tbl
end

__syncthreads = cudalib.nvvm_barrier0

local gaussNewtonGPU = require("solverGPUGaussNewton")

local ffi = require('ffi')

local problems = {}

-- this function should do anything it needs to compile an optimizer defined
-- using the functions in tbl, using the optimizer 'kind' (e.g. kind = gradientdecesnt)
-- it should generate the field makePlan which is the terra function that 
-- allocates the plan

local function compilePlan(problemSpec, kind)
    assert(kind == "gaussNewtonGPU" or kind == "LMGPU" ,"expected solver kind to be gaussNewtonGPU or LMGPU")
    return gaussNewtonGPU(problemSpec)
end

struct opt.Plan(S.Object) {
    init : {&opaque,&&opaque} -> {} -- plan.data,params
    setsolverparameter : {&opaque,rawstring,&opaque} -> {} -- plan.data,name,param
    free : {&opaque} -> {} -- plan.data
    step : {&opaque,&&opaque} -> int
    cost : {&opaque} -> double
    data : &opaque
}

struct opt.Problem {} -- just used as an opaque type, pointers are actually just the ID
local function problemDefine(filename, kind, pid)
    local problemmetadata = { filename = ffi.string(filename), kind = ffi.string(kind), id = #problems + 1 }
    problems[problemmetadata.id] = problemmetadata
    pid[0] = problemmetadata.id
end
problemDefine = terralib.cast({rawstring, rawstring, &int} -> {}, problemDefine)


local function problemDelete(pid)
    -- TODO: Some other value?
    problems[pid] = "DELETED"
end
problemDelete = terralib.cast({int} -> {}, problemDelete)

local List = terralib.newlist
A:Extern("ExpLike",function(x) return ad.Exp:isclassof(x) or ad.ExpVector:isclassof(x) end)
A:Define [[
Dim = (string name, number size, number? _index) unique
IndexSpace = (Dim* dims) unique
Index = Offset(number* data) unique
      | GraphElement(any graph, string element) unique
ImageType = (IndexSpace ispace, TerraType scalartype, number channelcount) unique
ImageLocation = ArgumentLocation(number idx) | UnknownLocation | StateLocation
Image = (string name, ImageType type, boolean scalar, ImageLocation location)
ImageVector = (Image* images)
ProblemParam = ImageParam(ImageType imagetype, boolean isunknown)
             | ScalarParam(TerraType type)
             | GraphParam(TerraType type)
             attributes (string name, any idx)

VarDef =  ImageAccess(Image image,  Shape _shape, Index index, number channel) unique
       | BoundsAccess(Offset min, Offset max) unique
       | IndexValue(number dim, number shift_) unique
       | ParamValue(string name,TerraType type) unique
Graph = (string name)

FunctionKind = CenteredFunction(IndexSpace ispace) unique
             | GraphFunction(string graphname) unique

ResidualTemplate = (Exp expression, ImageAccess* unknowns)
EnergySpec = (FunctionKind kind, ResidualTemplate* residuals)

FunctionSpec = (FunctionKind kind, string name, string* arguments, ExpLike* results, Scatter* scatters, EnergySpec? derivedfrom)

Scatter = (Image image,Index index, number channel, Exp expression, string kind)
Condition = (IRNode* members)
IRNode = vectorload(ImageAccess value, number count) # another one
       | sampleimage(Image image, number count, IRNode* children)
       | reduce(string op, IRNode* children)
       | vectorconstruct(IRNode* children)
       | vectorextract(IRNode* children, number channel)
       | load(ImageAccess value)
       | intrinsic(VarDef value)
       | const(number value)
       | vardecl(number constant)
       | varuse(IRNode* children)
       | apply(string op, function generator, IRNode * children, number? const)
         attributes (TerraType type, Shape shape, Condition? condition)
ProblemSpec = ()
ProblemSpecAD = ()
SampledImage = (table op)
GradientImage = (ImageAccess unknown, Exp expression, Image image)
UnknownType = (ImageParam* images)

ProblemFunctions = (FunctionKind typ, table functionmap)
]]
local Dim,IndexSpace,Index,Offset,GraphElement,ImageType,Image,ImageVector,ProblemParam,ImageParam,ScalarParam,GraphParam,VarDef,ImageAccess,BoundsAccess,IndexValue,ParamValue,Graph,GraphFunctionSpec,Scatter,Condition,IRNode,ProblemSpec,ProblemSpecAD,SampledImage, GradientImage,UnknownType = 
      A.Dim,A.IndexSpace,A.Index,A.Offset,A.GraphElement,A.ImageType,A.Image,A.ImageVector,A.ProblemParam,A.ImageParam,A.ScalarParam,A.GraphParam,A.VarDef,A.ImageAccess,A.BoundsAccess,A.IndexValue,A.ParamValue,A.Graph,A.GraphFunctionSpec,A.Scatter,A.Condition,A.IRNode,A.ProblemSpec,A.ProblemSpecAD,A.SampledImage,A.GradientImage,A.UnknownType

opt.PSpec = ProblemSpec
local PROBLEM_STAGES  = { inputs = 0, functions = 1 }
function opt.ProblemSpec()
    local ps = ProblemSpec()
    ps.parameters = terralib.newlist() -- ProblemParam*
    ps.names = {} -- name -> index in parameters list
    ps.functions = List() -- ProblemFunctions*
    ps.maxStencil = 0
    ps.stage = "inputs"
    ps.usepreconditioner = false
    ps.problemkind = opt.problemkind
    return ps
end

function ProblemSpec:UsePreconditioner(v)
    self:Stage "inputs"
    self.usepreconditioner = v
end
function ProblemSpec:Stage(name)
    assert(PROBLEM_STAGES[self.stage] <= PROBLEM_STAGES[name], "all inputs must be specified before functions are added")
    self.stage = name
end

function ProblemSpec:registername(name)
    assert(not self.names[name],string.format("name %s already in use",name))
    self.names[name] = #self.parameters + 1
end

function ProblemParam:terratype() return self.type end
function ImageParam:terratype() return self.imagetype:terratype() end


function ProblemSpec:MaxStencil()
    self:Stage "functions"
    return self.maxStencil
end

function ProblemSpec:Stencil(stencil) 
    self:Stage "inputs"
    self.maxStencil = math.max(stencil, self.maxStencil)
end

function ProblemSpec:newparameter(p)
    assert(ProblemParam:isclassof(p))
    self:registername(p.name)
    self.parameters:insert(p)
end

function ProblemSpec:ParameterType()
    self:Stage "functions"
    if not self.ProblemParameters then
        self.ProblemParameters = terralib.types.newstruct("ProblemParameters")
        self.ProblemParameters.entries:insert { "X" , self:UnknownType():terratype() }
        for i,p in ipairs(self.parameters) do
            local n,t = p.name,p:terratype()
            if not p.isunknown then self.ProblemParameters.entries:insert { n, t } end
        end
    end
    return self.ProblemParameters
end

function ProblemSpec:UnknownType()
    self:Stage "functions"
    if not self._UnknownType then
        local images = List()
        for _,p in ipairs(self.parameters) do
            if p.isunknown then images:insert(p) end
        end
        self._UnknownType = UnknownType(images)
    end
    return self._UnknownType
end

function A.CenteredFunction:__tostring() return tostring(self.ispace) end
function A.GraphFunction:__tostring() return tostring(self.graphname) end

function ProblemSpec:Functions(ft, functions)
    self:Stage "functions"
    for k,v in pairs(functions) do
        if k ~= "derivedfrom" then
            v:gettype() -- check they typecheck now
        end
    end
    
    -- support by-hand interface
    if type(ft) == "string" then
        ft = A.GraphFunction(ft)
    elseif IndexSpace:isclassof(ft) then
        ft = A.CenteredFunction(ft)
    end
    
    assert(A.FunctionKind:isclassof(ft))
    
    if ft.kind == "GraphFunction" then
        local idx = assert(self.names[ft.graphname],"graph not defined")
        assert(self.parameters[idx].kind == "GraphParam","expected a valid graph name")
        self.usesgraphs = true
    end
    if not functions.exclude then
        functions.exclude = macro(function() return `false end)
    end
    self.functions:insert(A.ProblemFunctions(ft, functions))
end
function ProblemSpec:UsesGraphs() return self.usesgraphs or false end

function ProblemSpec:Param(name,typ,idx)
    self:Stage "inputs"
    self:newparameter(ScalarParam(typ,name,idx))
end

function ProblemSpec:UsesLambda() return self.problemkind:match("LM") ~= nil end


function Dim:__tostring() return "Dim("..self.name..")" end

function opt.Dim(name, idx)
    idx = assert(tonumber(idx), "expected an index for this dimension")
    local size = tonumber(opt.dimensions[idx])
    return Dim(name,size,idx)
end

function IndexSpace:cardinality()
    local c = 1
    for i,d in ipairs(self.dims) do
        c = c * d.size
    end
    return c
end
function IndexSpace:init()
    self._string = self.dims:map(function(x) return x.name end):concat("_")
end
function IndexSpace:__tostring() return self._string end

function IndexSpace:ZeroOffset()
    if self._zerooffset then return self._zerooffset end
    local zeros = terralib.newlist()
    for i = 1,#self.dims do
        zeros:insert(0)
    end
    self._zerooffset = Offset(zeros)
    return self._zerooffset
end

function IndexSpace:indextype()
    if self._terratype then return self._terratype end
    local dims = self.dims
    assert(#dims > 0, "index space must have at least 1 dimension")
    local struct Index {}
    self._terratype = Index
    
    local params,params2 = List(),List()
    local fieldnames = List()
    for i = 1,#dims do
        local n = "d"..tostring(i-1)
        params:insert(symbol(int,n))
        params2:insert(symbol(int,n))
        fieldnames:insert(n)
        Index.entries:insert { n, int }
    end

    terra Index.metamethods.__apply(self : &Index, [params])
        var rhs : Index
        escape
            for i = 1,#dims do
                emit quote  
                    rhs.[fieldnames[i]] = self.[fieldnames[i]] + [params[i]]
                end 
            end
        end
        return rhs
    end
    local function genoffset(self)
        local s = 1
        local offset = `self.d0
        for i = 2,#dims do
            s = s * dims[i-1].size
            offset = `s*self.[fieldnames[i]] + offset
        end
        return offset
    end
    terra Index:tooffset()
        return [genoffset(self)]
    end
    local function genbounds(self,bmins,bmaxs)
        local valid
        for i = 1, #dims do
            local n = fieldnames[i]
            local bmin,bmax = 0,0
            if bmins then
                bmin = assert(bmins[i])
            end
            if bmaxs then
                bmax = assert(bmaxs[i])
            end
            local v = `self.[n] >= -[bmin] and self.[n] < [dims[i].size] - [bmax]
            if valid then
                valid = `valid and v
            else
                valid = v
            end
        end
        return valid
    end
    terra Index:InBounds() return [ genbounds(self) ] end
    
    terra Index:InBoundsExpanded([params],[params2]) return [ genbounds(self,params,params2) ] end
    
    if #dims <= 3 then
        local dimnames = "xyz"
        terra Index:initFromCUDAParams() : bool
            escape
                local lhs,rhs = terralib.newlist(),terralib.newlist()
                local valid = `true
                for i = 1,#dims do
                    local name = dimnames:sub(i,i)
                    local l = `self.[fieldnames[i]]
                    local r = `blockDim.[name] * blockIdx.[name] + threadIdx.[name]
                    lhs:insert(l)
                    rhs:insert(r)
                    valid = `valid and l < [dims[i].size]
                end
                emit quote
                    [lhs] = [rhs]
                    return valid
                end
            end  
        end
    end
    return Index
end

function ImageType:usestexture() -- texture, 2D texture
    local c = self.channelcount
    if use_bindless_texture and self.scalartype == float and 
       (c == 1 or c == 2 or c == 4) then
       if use_pitched_memory and #self.ispace.dims == 2 then
            local floatstride = self.ispace.dims[1].size*c
            local m = floatstride % 32
            if m ~= 0 then
                print(string.format("***** falling back to linear texture (width in floats %d %% 32 == %d)", floatstride , m))
            end
            return true,m == 0
       end
       return true, false 
    end
    return false, false 
end

local cd = macro(function(apicall) 
    local apicallstr = tostring(apicall)
    local filename = debug.getinfo(1,'S').source
    return quote
        var str = [apicallstr]
        var r = apicall
        if r ~= 0 then  
            C.printf("Cuda reported error %d: %s\n",r, C.cudaGetErrorString(r))
            C.printf("In call: %s", str)
            C.printf("In file: %s\n", filename)
            C.exit(r)
        end
    in
        r
    end end)

local terra wrapBindlessTexture(data : &uint8, channelcount : int, width : int, height : int) : C.cudaTextureObject_t
    var res_desc : C.cudaResourceDesc
    C.memset(&res_desc, 0, sizeof(C.cudaResourceDesc))

    res_desc.res.linear.devPtr = data
    res_desc.res.linear.desc.f = C.cudaChannelFormatKindFloat
    res_desc.res.linear.desc.x = 32  -- bits per channel
    if  channelcount > 1 then
        res_desc.res.linear.desc.y = 32
    end
    if channelcount == 4 then
        res_desc.res.linear.desc.z = 32
        res_desc.res.linear.desc.w = 32
    end

    if height ~= 0 then
        res_desc.resType = C.cudaResourceTypePitch2D
        res_desc.res.pitch2D.width = width
        res_desc.res.pitch2D.height = height
        res_desc.res.pitch2D.pitchInBytes = sizeof(float)*width*channelcount
    else
        res_desc.resType = C.cudaResourceTypeLinear
        res_desc.res.linear.sizeInBytes = sizeof(float)*width*channelcount
    end

    var tex_desc : C.cudaTextureDesc
    C.memset(&tex_desc, 0, sizeof(C.cudaTextureDesc))
    tex_desc.readMode       = C.cudaReadModeElementType

    var tex : C.cudaTextureObject_t = 0;
    cd(C.cudaCreateTextureObject(&tex, &res_desc, &tex_desc, nil))
    return tex
end

function ImageType:ElementType() return util.Vector(self.scalartype,self.channelcount) end
function ImageType:LoadAsVector() return self.channelcount == 2 or self.channelcount == 4 end
function ImageType:terratype()
    if self._terratype then return self._terratype end
    local scalartype = self.scalartype
    local vectortype = self:ElementType()
    local struct Image {
        data : &vectortype
        tex  : C.cudaTextureObject_t;
    }
    self._terratype = Image
    local channelcount = self.channelcount
    local textured,pitched = self:usestexture()
    local Index = self.ispace:indextype()
    function Image.metamethods.__typename()
      return string.format("Image(%s,%s,%d)",tostring(self.scalartype),tostring(self.ispace),channelcount)
    end

    local VT = &vector(scalartype,channelcount)    
    -- reads
    if pitched then
        terra Image.metamethods.__apply(self : &Image, idx : Index) : vectortype
            var read = terralib.asm([tuple(float,float,float,float)],
                "tex.2d.v4.f32.s32  {$0,$1,$2,$3}, [$4,{$5,$6}];",
                "=f,=f,=f,=f,l,r,r",false, self.tex, idx.d0,idx.d1)
            return @[&vectortype](&read)
        end
    elseif textured then         
        terra Image.metamethods.__apply(self : &Image, idx : Index) : vectortype
            var read = terralib.asm([tuple(float,float,float,float)],
                "tex.1d.v4.f32.s32  {$0,$1,$2,$3}, [$4,{$5}];",
                "=f,=f,=f,=f,l,r",false, self.tex,idx:tooffset())
            return @[&vectortype](&read)
        end
    elseif self:LoadAsVector() then
        terra Image.metamethods.__apply(self : &Image, idx : Index) : vectortype
            var a = VT(self.data)[idx:tooffset()]
            return @[&vectortype](&a)
        end
    else    
        terra Image.metamethods.__apply(self : &Image, idx : Index) : vectortype
            return self.data[idx:tooffset()]
        end
    end
    -- writes
    if self:LoadAsVector() then
        terra Image.metamethods.__update(self : &Image, idx : Index, v : vectortype)
            VT(self.data)[idx:tooffset()] = @VT(&v)
        end
    else
        terra Image.metamethods.__update(self : &Image, idx : Index, v : vectortype)
            self.data[idx:tooffset()] = v
        end
    end

    if scalartype == opt_float then    
        terra Image:atomicAddChannel(idx : Index, c : int32, v : scalartype)
            var addr : &scalartype = &self.data[idx:tooffset()].data[c]
            util.atomicAdd(addr,v)
        end
        terra Image:atomicAdd(idx : Index, v : vectortype) -- only for hand written stuff
            for i = 0,channelcount do
                self:atomicAddChannel(idx,i,v(i))
            end
        end
    end

    terra Image:get(idx : Index)
        var v : vectortype = 0.f
        if idx:InBounds() then
            v = self(idx)
        end
        return v
    end
    -- lerps for 2D images only
    if 2 == #self.ispace.dims then
        local terra lerp(v0 : vectortype, v1 : vectortype, t : opt_float)
            return (opt_float(1.) - t)*v0 + t*v1
        end
        terra Image:sample(x : opt_float, y : opt_float)
            var x0 : int, x1 : int = opt.math.floor(x),opt.math.ceil(x)
            var y0 : int, y1 : int = opt.math.floor(y),opt.math.ceil(y)
            var xn,yn = x - x0,y - y0
            var u = lerp(self:get( Index {x0,y0} ),self:get( Index {x1,y0} ),xn)
            var b = lerp(self:get( Index {x0,y1} ),self:get( Index {x1,y1} ),xn)
            return lerp(u,b,yn)
        end
    end
    local cardinality = self.ispace:cardinality()
    terra Image:totalbytes() return sizeof(vectortype)*cardinality end
    if textured then
        local W,H = cardinality,0
        if pitched then
            W,H = self.ispace.dims[1].size,self.ispace.dims[2].size
        end
        terra Image:setGPUptr(ptr : &uint8)
            if [&uint8](self.data) ~= ptr then
                if self.data ~= nil then
                    cd(C.cudaDestroyTextureObject(self.tex))
                end
                self.tex = wrapBindlessTexture(ptr, channelcount, W, H)
            end
            self.data = [&vectortype](ptr)
        end
        terra Image:freeData()
            if self.data ~= nil then
                cd(C.cudaDestroyTextureObject(self.tex))
                cd(C.cudaFree([&opaque](self.data)))
                self.data = nil
            end
        end
    else
        terra Image:setGPUptr(ptr : &uint8) self.data = [&vectortype](ptr) end
        terra Image:freeData()
            if self.data ~= nil then
                cd(C.cudaFree([&opaque](self.data)))
                self.data = nil
            end
        end
    end
    terra Image:initFromGPUptr( ptr : &uint8 )
        self.data = nil
        self:setGPUptr(ptr)
    end
    terra Image:initGPU()
        var data : &uint8
        cd(C.cudaMalloc([&&opaque](&data), self:totalbytes()))
        cd(C.cudaMemset([&opaque](data), 0, self:totalbytes()))
        self:initFromGPUptr(data)
    end
    return Image
end

local function MapAndGroupBy(list,fn,...)
    local groups,map = List(),{}
    for _,l in ipairs(list) do
        local g,v = fn(l,...)
        if not map[g] then
            map[g] = List()
            groups:insert(g)
        end
        map[g]:insert(v)
    end
    return groups,map
end

function UnknownType:init()
    self.ispaces,self.ispacetoimages = MapAndGroupBy(self.images, function(ip)
        if opt_float == float then
            assert(ip.imagetype.scalartype == float, "Unknowns must be floats when doublePrecision = false, but "..ip.name.." was declared as "..tostring(ip.imagetype.scalartype))
        else
            assert(opt_float == double, "Invalid unknown type "..tostring(opt_float))
            assert(ip.imagetype.scalartype == double, "Unknowns must be doubles when doublePrecision = true, but "..ip.name.." was declared as "..tostring(ip.imagetype.scalartype))
        end
        return ip.imagetype.ispace, ip
    end)
    self.ispacesizes = {}
    for _,ispace in ipairs(self.ispaces) do
        local N = 0
        for _,ip in ipairs(self.ispacetoimages[ispace]) do
            N = N + ip.imagetype.channelcount
        end
        self.ispacesizes[ispace] = N
    end
end
function UnknownType:IndexSpaces()
    return self.ispaces
end
function UnknownType:VectorSizeForIndexSpace(ispace) return assert(self.ispacesizes[ispace],"unused ispace") end 
function UnknownType:VectorTypeForIndexSpace(ispace)
    return util.Vector(opt_float,self:VectorSizeForIndexSpace(ispace))
end
function UnknownType:UnknownIteratorForIndexSpace(ispace)
    local images = self.ispacetoimages[ispace]
    local i,j,c = 0,1,0
    return function()
        if c >= images[j].imagetype.channelcount then
            j,c = j+1,0
        end
        if j > #images then return nil end
        i,c = i + 1,c + 1
        --print(i-1,images[j].name,c-1)
        return i - 1, images[j].name,c - 1
    end
end

function UnknownType:terratype()
    if self._terratype then return self._terratype end
    self._terratype = terralib.types.newstruct("UnknownType")
    local T = self._terratype
    local images = self.images
    for i,ip in ipairs(images) do
        T.entries:insert { ip.name, ip.imagetype:terratype() }
    end
    if use_contiguous_allocation then
        T.entries:insert { "_contiguousallocation", &opaque }
        terra T:initGPU()
            var size = 0
            escape
                for i,ip in ipairs(images) do
                    emit quote 
                        size = size + self.[ip.name]:totalbytes()
                    end
                end
            end
            var data : &uint8
            cd(C.cudaMalloc([&&opaque](&data), size))
            self._contiguousallocation = data
            cd(C.cudaMemset([&opaque](data), 0, size))
            size = 0
            escape
                for i,ip in ipairs(images) do
                    emit quote 
                        self.[ip.name]:initFromGPUptr(data+size)
                        size = size + self.[ip.name]:totalbytes() 
                    end
                end
            end
        end
        terra T:freeData()
            cd(C.cudaFree(self._contiguousallocation))
        end
    else
        terra T:initGPU()
            escape
                for i,ip in ipairs(images) do
                    emit quote self.[ip.name]:initGPU() end
                end
            end
        end
        terra T:freeData()
            escape
                for i,ip in ipairs(images) do
                    emit quote self.[ip.name]:freeData() end
                end
            end
        end
    end
    for _,ispace in ipairs(self:IndexSpaces()) do   
        local Index = ispace:indextype()
        local ispaceimages = self.ispacetoimages[ispace]
        local VT = self:VectorTypeForIndexSpace(ispace)
        terra T.metamethods.__apply(self : &T, idx : Index) : VT
            var r : VT
            escape
                local off = 0
                for _,im in ipairs(ispaceimages) do
                    emit quote
                        var d = self.[im.name](idx)
                        for i = 0,im.imagetype.channelcount do
                            r.data[off+i] = d.data[i]
                        end
                    end
                    off = off + im.imagetype.channelcount
                end
            end
            return r
        end
        terra T.metamethods.__update(self : &T, idx : Index, v : VT)
            escape
                local off = 0
                for _,im in ipairs(ispaceimages) do
                    emit quote
                        var d : im.imagetype:ElementType()
                        for i = 0,im.imagetype.channelcount do
                            d.data[i] = v.data[off+i]
                        end
                        self.[im.name](idx) = d
                    end
                    off = off + im.imagetype.channelcount
                end
            end
        end
    end

    return self._terratype
end

local unity = Dim("1",1)
local function todim(d)
    return Dim:isclassof(d) and d or d == 1 and unity
end

local function tovalidimagetype(typ)
    if not terralib.types.istype(typ) then return nil end
    if util.isvectortype(typ) then
        return typ.metamethods.type, typ.metamethods.N
    elseif typ:isarithmetic() then
        return typ, 1
    end
end

function ProblemSpec:ImageType(typ,ispace)
    local scalartype,channelcount = tovalidimagetype(typ,"expected a number or an array of numbers")
    assert(scalartype,"expected a number or an array of numbers")
    return ImageType(ispace,scalartype,channelcount) 
end

local function toispace(ispace)
    if not IndexSpace:isclassof(ispace) then -- for handwritten API
        assert(#ispace > 0, "expected at least one dimension")
        ispace = IndexSpace(List(ispace)) 
    end
    return ispace
end


function ProblemSpec:Image(name,typ,ispace,idx,isunknown)
    self:Stage "inputs"
    isunknown = isunknown and true or false
    self:newparameter(ImageParam(self:ImageType(typ,toispace(ispace)),isunknown,name,idx))
end
function ProblemSpec:Unknown(name,typ,ispace,idx) return self:Image(name,typ,ispace,idx,true) end


function ProblemSpec:Graph(name, idx, ...)
    self:Stage "inputs"
    local GraphType = terralib.types.newstruct(name)
    GraphType.entries:insert ( {"N",int32} )
    
    local mm = GraphType.metamethods
    mm.idx = idx -- the index into the graph size table
    mm.elements = terralib.newlist()
    for i = 1, select("#",...),3 do
        local name,dims,didx = select(i,...) --TODO: we don't bother to track the dimensions of these things now
        local ispace = toispace(dims)
        local Index = ispace:indextype()
        GraphType.entries:insert {name, &Index}
        mm.elements:insert( { name = name, type = Index, idx = assert(tonumber(didx))} )
    end
    self:newparameter(GraphParam(GraphType,name,idx))
end

local activePlans = terralib.newlist()

errorPrint = rawget(_G,"errorPrint") or print

function opt.problemSpecFromFile(filename)
    local file, errorString = terralib.loadfile(filename)
    if not file then
        error(errorString, 0)
    end
    local P = ad.ProblemSpec()
    local libinstance = optlib(P)
    setfenv(file,libinstance)
    local result = file()
    if ProblemSpec:isclassof(result) then
        return result
    end
    return libinstance.Result()
end

local function printCurrentBytes()
    collectgarbage()
    collectgarbage()
    print(collectgarbage("count"))
end

local function problemPlan(id, dimensions, pplan)
    local success,p = xpcall(function()  
        local problemmetadata = assert(problems[id])
        opt.dimensions = dimensions
        opt.math = problemmetadata.kind:match("GPU") and util.gpuMath or util.cpuMath
        opt.problemkind = problemmetadata.kind
        local b = terralib.currenttimeinseconds()
        local tbl = opt.problemSpecFromFile(problemmetadata.filename)
        assert(ProblemSpec:isclassof(tbl))
        local result = compilePlan(tbl,problemmetadata.kind)
        local e = terralib.currenttimeinseconds()
        print("compile time: ",e - b)
        pplan[0] = result()
        activePlans[tostring(pplan[0])] = result
        print("problem plan complete")
		if _opt_verbosity > 0 then
	        util.reportGPUMemoryUse()
	        printCurrentBytes()
		end
    end,function(err) errorPrint(debug.traceback(err,2)) end)
end
problemPlan = terralib.cast({int,&uint32,&&opt.Plan} -> {}, problemPlan)

local function planFree(pplan)
    local success,p = xpcall(function()
        activePlans[tostring(pplan)] = nil
        if _opt_verbosity > 0 then
			util.reportGPUMemoryUse()
		end
        print("plan free complete")
        collectgarbage()
        collectgarbage()
		if _opt_verbosity > 0 then
        	util.reportGPUMemoryUse()
		end
    end,function(err) errorPrint(debug.traceback(err,2)) end)
end
planFree = terralib.cast({&opt.Plan} -> {}, planFree)

function Offset:__tostring() return string.format("(%s)",self.data:map(tostring):concat(",")) end
function GraphElement:__tostring() return ("%s_%s"):format(tostring(self.graph), self.element) end

function VarDef:asvar() return ad.v[self] end

function ImageAccess:__tostring()
    local r = ("%s_%s_%s"):format(self.image.name,tostring(self.index),self.channel)
    if self:shape() ~= ad.scalar then
        r = r .. ("_%s"):format(tostring(self:shape()))
    end
    return r
end
function BoundsAccess:__tostring() return ("bounds_%s_%s"):format(tostring(self.min),self.min == self.max and "p" or tostring(self.max)) end
function IndexValue:__tostring() return ({[0] = "i","j","k"})[self.dim] end
function ParamValue:__tostring() return "param_"..self.name end

function ImageAccess:shape() return self._shape end -- implementing AD's API for keys

local emptygradient = {}
function ImageAccess:gradient()
    if self.image.gradientimages then
        assert(Offset:isclassof(self.index),"NYI - support for gradient images of graphs")
        local gt = {}
        for i,im in ipairs(self.image.gradientimages) do
            local k = im.unknown:shift(self.index)
            local v = ad.Const:isclassof(im.expression) and im.expression or im.image(self.index)
            gt[k] = v
        end
        return gt
    end
    return emptygradient
 end
 
function ad.Index(d) return IndexValue(d,0):asvar() end
 

function ad.ProblemSpec()
    local ps = ProblemSpecAD()
    ps.P,ps.nametoimage,ps.precomputed,ps.extraarguments,ps.excludeexps = opt.ProblemSpec(), {}, List(), List(), List()
    if ps.P:UsesLambda() then
        ps.trust_region_radius = ps:Param("trust_region_radius",opt_float,-1)
        ps.radius_decrease_factor = ps:Param("radius_decrease_factor",opt_float,-1)
        ps.min_lm_diagonal = ps:Param("min_lm_diagonal",opt_float,-1)
        ps.max_lm_diagonal = ps:Param("max_lm_diagonal",opt_float,-1)
    end
    return ps
end
function ProblemSpecAD:UsesLambda() return self.P:UsesLambda() end
function ProblemSpecAD:UsePreconditioner(v)
    self.P:UsePreconditioner(v)
end

function ProblemSpecAD:Image(name,typ,dims,idx,isunknown)
    if not terralib.types.istype(typ) then
        typ,dims,idx,isunknown = opt_float,typ,dims,idx --shift arguments left
    end
    isunknown = isunknown and true or false
    local ispace = toispace(dims)
    assert( (type(idx) == "number" and idx >= 0) or idx == "alloc", "expected an index number") -- alloc indicates that the solver should allocate the image as an intermediate
    self.P:Image(name,typ,ispace,idx,isunknown)
    local r = Image(name,self.P:ImageType(typ,ispace),not util.isvectortype(typ),isunknown and A.UnknownLocation or A.StateLocation)
    self.nametoimage[name] = r
    return r
end
function ProblemSpecAD:Unknown(name,typ,dims,idx) return self:Image(name,typ,dims,idx,true) end

function ProblemSpecAD:UnknownArgument(argpos)
    if not self.extraarguments[argpos] then
        local r = {}
        for _,ip in ipairs(self.P:UnknownType().images) do
            local template = self.nametoimage[ip.name]
            r[ip.name] = Image(ip.name,template.type,template.scalar,A.ArgumentLocation(argpos))
        end
        self.extraarguments[argpos] = r
    end
    return self.extraarguments[argpos]
end

function ProblemSpecAD:ImageTemporary(name,ispace)
    self.P:Image(name,opt_float,ispace,"alloc",false)
    local r = Image(name,self.P:ImageType(opt_float,ispace),true,A.StateLocation)
    self.nametoimage[name] = r
    return r
end

function ProblemSpecAD:ImageWithName(name)
    return assert(self.nametoimage[name],"unknown image name?")
end

function Image:__tostring() return self.name end

local function bboxforexpression(ispace,exp)
    local usesbounds = false
    local bmin,bmax = ispace:ZeroOffset(),ispace:ZeroOffset()
    exp:visit(function(a)
        if ImageAccess:isclassof(a) then
            assert(Offset:isclassof(a.index,"bbox not defined for graphs"))
            if a.image.gradientimages then
                local shiftedbbox = a.image.bbox:shift(a.index)
                bmin,bmax = bmin:Min(shiftedbbox.min),bmax:Max(shiftedbbox.max)
            else
                bmin,bmax = bmin:Min(a.index),bmax:Max(a.index)
            end
        elseif BoundsAccess:isclassof(a) then
            usesbounds = true
        end
    end)
    if usesbounds then 
        bmin,bmax = ispace:ZeroOffset(),ispace:ZeroOffset()
    end
    return BoundsAccess(bmin,bmax)
end

function ProblemSpecAD:ComputedImage(name,dims,exp)
    if ad.ExpVector:isclassof(exp) then
        local imgs = terralib.newlist()
        for i,e in ipairs(exp:expressions()) do
            imgs:insert(self:ComputedImage(name.."_"..tostring(i-1),dims,e))
        end
        return ImageVector(imgs)
    end
    exp = assert(ad.toexp(exp),"expected a math expression")
    local unknowns = terralib.newlist()
    local seen = {}
    exp:visit(function(a)
        if ImageAccess:isclassof(a) and a.image.location == A.UnknownLocation then
            assert(Offset:isclassof(a.index),"NYI - support for precomputed graphs")
            if not seen[a] then
                seen[a] = true
                unknowns:insert(a)
            end
        end
    end)
    local ispace = toispace(dims)
    local im = self:ImageTemporary(name,ispace)
    local gradients = exp:gradient(unknowns:map(function(x) return ad.v[x] end))
    im.gradientimages = terralib.newlist()
    for i,g in ipairs(gradients) do
        local u = unknowns[i]
        local gim = self:ImageTemporary(name.."_d_"..tostring(u),ispace)
        im.gradientimages:insert(GradientImage(u,g,gim))
    end
    im.expression = exp
    im.bbox = bboxforexpression(ispace,exp)
    self.precomputed:insert(im)
    return im
end

function Graph:__tostring() return self.name end
function ProblemSpecAD:Graph(name,idx,...)
    self.P:Graph(name,idx,...)
    local g = Graph(name)
    for i = 1, select("#",...),3 do
        local name,dims,didx = select(i,...)
        local ge = GraphElement(g,name) 
        ge.ispace = toispace(dims)
        g[name] = ge
    end
    return g
end

function ProblemSpecAD:Param(name,typ,idx)
    self.P:Param(name,typ,idx)
    return ParamValue(name,typ):asvar()
end

function Image:DimCount() return #self.type.ispace.dims end
function Image:__call(first,...)
    local index,c
    if GraphElement:isclassof(first) or Offset:isclassof(first) then
        index = first
        c = ...
    else
        local o = terralib.newlist { (assert(first,"no arguments?")) }
        for i = 1,self:DimCount() - 1 do
            o:insert((select(i,...)))
        end
        index = Offset(o)
        c = select(self:DimCount(), ...)
    end
    if GraphElement:isclassof(index) then    
        assert(index.ispace == self.type.ispace,"Graph element is in a different index space from image")
    end
    c = tonumber(c)
    assert(not c or c < self.type.channelcount, "Channel outside of range")
    if self.scalar or c then
        return ImageAccess(self,ad.scalar,index,c or 0):asvar()
    else
        local r = {}
        for i = 1,self.type.channelcount do
            r[i] = ImageAccess(self,ad.scalar,index,i-1):asvar()
        end
        return ad.Vector(unpack(r))
    end
end
 -- wrapper for many images in a vector, just implements the __call methodf for Images Image:
function ImageVector:__call(...)
    local args = {...}
    local channelindex = self.images[1]:DimCount() + 1
    if #args == channelindex then
        local c = args[channelindex]
        assert(c < #self.images, "Channel outside of range")
        return self.images[c+1](unpack(args,1,channelindex-1))
    end
    local result = self.images:map(function(im) return im(unpack(args)) end)
    return ad.Vector(unpack(result))
end

function opt.InBounds(...)
    local offset = Offset(List{...})
	return BoundsAccess(offset,offset):asvar()
end
function opt.InBoundsExpanded(...)
    local args = {...}
    local expand = args[#args]
    args[#args] = nil
    local min,max = List(),List()
    for i,a in ipairs(args) do
        min[i],max[i] = a - expand, a + expand
    end
    return BoundsAccess(Offset(min),Offset(max)):asvar()
end
function BoundsAccess:type() return bool end --implementing AD's API for keys


function VarDef:shift(o) return self end
function BoundsAccess:shift(o)
    return BoundsAccess(self.min:shift(o),self.max:shift(o))
end
function ImageAccess:shift(o)
    assert(Offset:isclassof(self.index), "Cannot shift graph accesses!")
    return ImageAccess(self.image,self:shape(),self.index:shift(o),self.channel)
end
function IndexValue:shift(o)
    return IndexValue(self.dim,self.shift_ + assert(o.data[self.dim+1],"dim of index not in shift"))
end

local function shiftexp(exp,o)
    local function rename(a)
        return ad.v[a:shift(o)]
    end
    return exp:rename(rename)
end 

function Offset:IsZero()
    for i,o in ipairs(self.data) do
        if o ~= 0 then return false end
    end
    return true
end
function Offset:MaxValue()
    local m = 0
    for i, o in ipairs(self.data) do
        if o > m then m = o end
    end
    return m
end
function Offset:Invert()
    local r = terralib.newlist()
    for i,o in ipairs(self.data) do
        r[i] = -o
    end
    return Offset(r)
end
function Offset:Min(rhs)
    assert(Offset:isclassof(rhs) and #self.data == #rhs.data)
    local r = List()
    for i = 1,#self.data do
        r[i] = math.min(self.data[i],rhs.data[i])
    end
    return Offset(r)
end
function Offset:Max(rhs)
    assert(Offset:isclassof(rhs) and #self.data == #rhs.data)
    local r = List()
    for i = 1,#self.data do
        r[i] = math.max(self.data[i],rhs.data[i])
    end
    return Offset(r)
end
function Offset:shift(o)
    assert(Offset:isclassof(o) and #o.data == #self.data)
    local ns = terralib.newlist()
    for i = 1,#self.data do
        ns[i] = self.data[i] + o.data[i]
    end
    return Offset(ns)
end
local function removeboundaries(exp)
    if ad.ExpVector:isclassof(exp) or terralib.islist(exp) then return exp:map(removeboundaries) end
    local function nobounds(a)
        if BoundsAccess:isclassof(a) and a.min:IsZero() and a.max:IsZero() then return ad.toexp(1)
        else return ad.v[a] end
    end
    return exp:rename(nobounds)
end

local nextirid = 0
function IRNode:init()
    self.id,nextirid = nextirid,nextirid+1
end
function Condition:create(members)
    local function cmp(a,b)
        if a.kind == "intrinsic" and b.kind ~= "intrinsic" then return true
        elseif a.kind ~= "intrinsic" and b.kind == "intrinsic" then return false
        else return a.id < b.id end
    end
    table.sort(members,cmp)
    return Condition(members)
end

function Condition:Intersect(rhs)
    local lhsmap = {}
    for i,m in ipairs(self.members) do
        lhsmap[m] = true
    end
    local r = terralib.newlist()
    for i,m in ipairs(rhs.members) do
        if lhsmap[m] then
            r:insert(m)
        end
    end
    return Condition:create(r)
end

function Condition:Union(rhs)
    local lhsmap = {}
    local r = terralib.newlist()
    for i,m in ipairs(self.members) do
        lhsmap[m] = true
        r:insert(m)
    end
    for i,m in ipairs(rhs.members) do
        if not lhsmap[m] then
            r:insert(m)
        end
    end
    return Condition:create(r)
end

local function createfunction(problemspec,name,Index,arguments,results,scatters)
    results = removeboundaries(results)
    
    local imageload = terralib.memoize(function(imageaccess)
        return A.vectorload(imageaccess,0,imageaccess.image.type:ElementType(),imageaccess:shape())
    end)
    local imagesample = terralib.memoize(function(image, shape, x, y)
        return A.sampleimage(image,0,List{x,y},image.scalar and image.type.scalartype or image.type:ElementType(),shape)
    end)
    local irmap
    
    local function tofloat(ir,exp)
        if ir.type ~= opt_float then
            return `opt_float(exp)
        else
            return exp
        end
    end
    local function createreduce(op,vardecl,n)
        local cond
        if op == "sum" and n.kind == "Apply" and n.op.name == "prod" then
            local conditions = terralib.newlist()
            local factors = terralib.newlist()
            for i,c in ipairs(n:children()) do
                if c:type() == bool then
                    conditions:insert(irmap(c))
                else
                    factors:insert(c)
                end
            end
            n = ad.prod(n.const,unpack(factors))
            cond = Condition:create(conditions)
        end
        return A.reduce(op,List{vardecl,irmap(n)},opt_float,vardecl.shape,cond)
    end
    irmap = terralib.memoize(function(e)
        if ad.ExpVector:isclassof(e) then
            return A.vectorconstruct(e.data:map(irmap),util.Vector(opt_float,#e.data),ad.scalar)
        elseif "Var" == e.kind then
            local a = e:key()
            if "ImageAccess" == a.kind then
                if not a.image.scalar then
                    local loadvec = imageload(ImageAccess(a.image,a:shape(),a.index,0))
                    loadvec.count = loadvec.count + 1
                    return A.vectorextract(List {loadvec}, a.channel, e:type(), a:shape())
                else
                    return A.load(a,e:type(),a:shape()) 
                end 
            else
                return A.intrinsic(a,e:type(),ad.scalar)
            end
        elseif "Const" == e.kind then
            return A.const(e.v,e:type(),ad.scalar)
        elseif "Apply" == e.kind then
            if use_split_sums and (e.op.name == "sum") and #e:children() > 2 then
                local vardecl = A.vardecl(e.const,opt_float,e:shape())
                local children = List { vardecl }
                local varuse = A.varuse(children,opt_float,e:shape())
                for i,c in ipairs(e:children()) do
                    children:insert(createreduce(e.op.name,vardecl,c))
                end
                return varuse
            end
            local children = e:children():map(irmap)
            if e.op.name:match("^sampleimage") then
                local sm = imagesample(e.op.imagebeingsampled,e:shape(),children[1],children[2])
                sm.count = sm.count + 1
                if not util.isvectortype(sm.image.type) then
                    return sm
                end
                return A.vectorextract(List {sm}, e.const, e:type(), e:shape()) 
            end
            local fn,gen = opt.math[e.op.name]
            if fn then
                function gen(args)
                    local nargs = terralib.newlist()
                    for i,a in ipairs(args) do
                        nargs[i] = tofloat(children[i],a)
                    end
                    return `fn(nargs) 
                end
            else
                function gen(args) return e.op:generate(e,args) end
            end
            return A.apply(e.op.name,gen,children,e.const,e:type(),e:shape()) 
        elseif "Reduce" == e.kind then
            local vardecl = A.vardecl(0,e:type(),e:shape()) 
            local arg = e.args[1]
            local red = A.reduce("sum",List { vardecl, irmap(arg) }, vardecl.type, arg:shape()) 
            local children = List { vardecl, red }
            local varuse = A.varuse(children,vardecl.type,e:shape())
            return varuse
        end
    end)
    
    local irroots = results:map(irmap)
    for i,s in ipairs(scatters) do
        irroots:insert(irmap(s.expression))
    end
    
    local function  linearizedorder(irroots)
        local visited = {}
        local linearized = terralib.newlist()
        local function visit(ir)
            if visited[ir] then return end
            visited[ir] = true
            if ir.children then
                for i,c in ipairs(ir.children) do visit(c) end
            end
            if ir.condition then
                for i,c in ipairs(ir.condition.members) do visit(c) end
            end
            linearized:insert(ir)
        end
        for i,r in ipairs(irroots) do
            visit(r)
        end
        return linearized
    end
    
    -- tighten the conditions under which ir nodes execute
    local linearized = linearizedorder(irroots)
    
    for i = #linearized,1,-1 do
        local ir = linearized[i]
        if not ir.condition then
            ir.condition = Condition:create(List{})
        end
        local function applyconditiontolist(condition,lst)
            for i,c in ipairs(lst) do
                if not c.condition then
                    c.condition = condition
                elseif c.kind == "reduce" then -- single use is this node, so the condition is the already established condition plus any that the variable use imposes
                    c.condition = c.condition:Union(condition)
                else
                    c.condition = c.condition:Intersect(condition)
                end
            end
        end
        if use_conditionalization then
            if ir.children then applyconditiontolist(ir.condition,ir.children) end
        end
        if ir.kind == "reduce" then applyconditiontolist(Condition:create(List{}), ir.condition.members) end
    end
    
    local function calculateusesanddeps(roots)
        local uses,deps = {},{}
        local function visit(parent,ir)
            if not deps[ir] then assert(not uses[ir])
                uses[ir],deps[ir] = terralib.newlist(),terralib.newlist()
                local function visitlist(lst)
                    for i,c in ipairs(lst) do
                        deps[ir]:insert(c)
                        visit(ir,c)
                    end
                end
                if ir.children then visitlist(ir.children) end
                if ir.condition then visitlist(ir.condition.members) end
            end
            if parent then
                uses[ir]:insert(parent)
            end
        end
        for i, r in ipairs(roots) do
            visit(nil,r)
        end
        return uses,deps
    end
    
    local uses,deps = calculateusesanddeps(irroots)
     
    local function prefixsize(a,b)
        for i = 1,math.huge do
            if a[i] ~= b[i] or a[i] == nil then return i - 1 end
        end
    end
    local function conditiondiff(current,next)
        local i = prefixsize(current.members,next.members)
        local uplevels,downlevels = #current.members - i, #next.members - i
        return uplevels,downlevels
    end
    local function conditioncost(current,next)
        local uplevels,downlevels = conditiondiff(current,next)
        return uplevels*1000 + downlevels
    end
    local function shapecost(current,next)
        return current ~= next and 1 or 0
    end
        
    local function schedulebackwards(roots,uses)
        
        local state = nil -- ir -> "ready" or ir -> "scheduled"
        local readylists = terralib.newlist()
        local currentcondition,currentshape = Condition:create(List{}), ad.scalar
        local function enter()
            state = setmetatable({}, {__index = state})
            readylists:insert(terralib.newlist())
        end
        enter() --initial root level for non-speculative moves
        
        for i,r in ipairs(roots) do
            if not state[r] then -- roots may appear in list more than once
                -- It is possible for a member of the irroots list to
                -- not actually be a root of the DAG; prune those out
                if #uses[r] == 0 then
                    state[r] = "ready"
                    readylists[#readylists]:insert(r)
                end
            end
        end
        
        local function leave()
            readylists:remove()
            state = assert(getmetatable(state).__index,"pop!")    
        end
        
        local function registersreleased(ir)
            if ir.kind == "const" then return 0
            elseif ir.kind == "vectorload" or ir.kind == "sampleimage" then return ir.count
            elseif ir.kind == "vectorextract" then return 0
            elseif ir.kind == "varuse" then return 0
            elseif ir.kind == "vardecl" then return 1
            elseif ir.kind == "reduce" then return 0 
            else return 1 end
        end
        local function registersliveonuse(ir)
            if ir.kind == "const" then return 0
            elseif ir.kind == "vectorload" then return 0
            elseif ir.kind == "sampleimage" then return util.isvectortype(ir.type) and 0 or 1
            elseif ir.kind == "vectorextract" then return 1
            elseif ir.kind == "varuse" then return 1
            elseif ir.kind == "reduce" then return 0
            elseif ir.kind == "vardecl" then return 0
            else return 1 end
        end
        local function netregisterswhenscheduled(ir)
            local n = -registersreleased(ir)
            local newlive = {}
            for i,c in ipairs(deps[ir]) do
                newlive[c] = true
            end
            for k,_ in pairs(newlive) do
                if not state[k] then
                    n = n + registersliveonuse(k)
                end
            end
            return n
        end
        local function checkandmarkready(ir)
            if state[ir] ~= "ready" then
                for i,u in ipairs(uses[ir]) do
                    if state[u] ~= "scheduled" then return end -- not ready
                end            
                readylists[#readylists]:insert(ir)
                state[ir] = "ready"
            end
        end
        local function markscheduled(ir)
            state[ir] = "scheduled"
            for i,c in ipairs(deps[ir]) do 
                if not state[c] then
                    state[c] = "used"
                end
                checkandmarkready(c)
            end
        end
        
        local function vardeclcost(ir)
            return ir.kind == "vardecl" and 0 or 1
        end

        local function costspeculate(depth,ir)
            local c = netregisterswhenscheduled(ir)
            if depth > 0 then
                local minr = math.huge
                enter() -- start speculation level
                markscheduled(ir)
                
                for _,rl in ipairs(readylists) do
                    for _,candidate in ipairs(rl) do
                        if state[candidate] == "ready" then -- might not be ready because an overlay already scheduled it and we don't track the deletions
                            minr = math.min(minr,costspeculate(depth-1,candidate))
                        end
                    end
                end
                
                leave()
                if minr ~= math.huge then
                    c = c*10 + minr
                end
            end
            return c
        end

        local function cost(idx,ir)
            local c =  { shapecost(currentshape,ir.shape) }
            if use_condition_scheduling then
                table.insert(c, conditioncost(currentcondition,ir.condition))
            end
            if use_register_minimization then
                table.insert(c, vardeclcost(ir))
                if use_cost_speculate then
                    table.insert(c, costspeculate(1,ir))
                else
                    table.insert(c, costspeculate(0,ir))
                end
            end
            return c
        end
        
        local function costless(n,a,b)
            for i,ac in ipairs(a) do
                local bc = b[i]
                if ac ~= bc then return ac < bc end
            end
            return false
        end
        local ready = readylists[1] -- the true ready list is the first one, the rest are the speculative lists
        local function choose()
            --print("---------------------")
            local best = cost(1,assert(ready[1]))
            local bestidx = 1
            for i = 2,#ready do
                local ci = cost(i,ready[i])
                if costless(i,ci,best) then
                    bestidx = i
                    best = ci
                end
            end
            --print("choose",bestidx)
            return table.remove(ready,bestidx)
        end
        
        local instructions = terralib.newlist()
        local regcounts = terralib.newlist()
        local currentregcount = 1
        while #ready > 0 do
            local ir = choose()
            instructions:insert(1,ir)
            regcounts:insert(1,currentregcount)
            currentregcount = currentregcount + netregisterswhenscheduled(ir)
            markscheduled(ir)
            currentcondition,currentshape = ir.condition,ir.shape
        end
        return instructions,regcounts
    end
    
    local instructions,regcounts = schedulebackwards(irroots,uses)
    
    local function printschedule(W,instructions,regcounts)
        W:write(string.format("schedule for %s -----------\n",name))
        local emittedpos = {}
        local function formatchildren(children)
            local cs = terralib.newlist()
            for i,c in ipairs(children) do
                cs:insert("r"..tostring(emittedpos[c]))
            end
            return cs:concat(",")
        end
    
        local function formatinst(inst)
            local fs = terralib.newlist()
            fs:insert(inst.kind.." ")
            for k,v in pairs(inst) do
                if k ~= "kind" and k ~= "children" and type(v) ~= "function" and k ~= "id" and k ~= "condition" and k ~= "type" then
                    fs:insert(tostring(v))
                    fs:insert(" ")
                end
            end
            if inst.children then
                fs:insert("{")
                fs:insert(formatchildren(inst.children))
                fs:insert("}")
            end
            return fs:concat()
        end
        local function formatcondition(c)
            local fs = terralib.newlist()
            fs:insert("[")
            fs:insert(formatchildren(c.members))
            fs:insert("]")
            local r = fs:concat()
            return r .. (" "):rep(4*(1+#c.members) - #r)
        end
        for i,ir in ipairs(instructions) do
            emittedpos[ir] = i
            W:write(("[%d]%sr%d : %s%s = %s\n"):format(regcounts[i],formatcondition(ir.condition),i,tostring(ir.type),tostring(ir.shape),formatinst(ir)))
            if instructions[i+1] and conditioncost(ir.condition,instructions[i+1].condition) ~= 0 then
                W:write("---------------------\n")
            end
        end
        W:write("----------------------\n")
    end
    
    if verboseAD then
        local W = io.open("log.txt","a")
        printschedule(W,instructions,regcounts)
        W:close()
    end
    
    local P = symbol(problemspec.P:ParameterType(),"P")
    local idx = symbol(Index,"idx")
    local midx = symbol(Index,"midx")
    
    local statementstack = terralib.newlist { terralib.newlist() } 
    local statements = statementstack[1]
    local TUnknownType = problemspec.P:UnknownType():terratype()
    local extraarguments = arguments:map(function(a) return symbol(TUnknownType,a) end)
    
    local emit
    local function emitconditionchange(current,next)
        local u,d = conditiondiff(current,next)
        for i = 0,u - 1 do
            local c = current.members[#current.members - i]
            local ce = emit(c)
            local stmts = statementstack:remove()
            statementstack[#statementstack]:insert quote
                if ce then
                    [stmts]
                end
            end
        end
        for i = 1,d do
            statementstack:insert(terralib.newlist())
        end
        statements = statementstack[#statementstack]
    end
    local currentidx
    local function boundcoversload(ba,off)
        --print("Bound Covers? ",ba,off)
        assert(#off.data == #ba.min.data)
        for i = 1,#off.data do
            local o,bmin,bmax = off.data[i],ba.min.data[i],ba.max.data[i]
            if o < bmin or o > bmax then
                --print("no")
                return false
            end
        end
        --print("yes")
        return true
    end
    local function conditioncoversload(condition,off)
        if off:IsZero() then return true end
        for i,ir in ipairs(condition.members) do
            assert(ir.type == bool)
            if ir.kind == "intrinsic" and ir.value.kind == "BoundsAccess" and boundcoversload(ir.value,off) then
                return true
            end
        end
        return false
    end
    local function imageref(image)
        if image.location == A.StateLocation then
            return `P.[image.name]
        elseif image.location == A.UnknownLocation then
            return `P.X.[image.name]
        else
            local sym = assert(extraarguments[image.location.idx],"Unknown extra image")
            return `sym.[image.name]
        end
    end
    local function graphref(ge)
        return `P.[ge.graph.name].[ge.element][idx]
    end
    local function createexp(ir)        
        if "const" == ir.kind then
            return `opt_float(ir.value)
        elseif "intrinsic" == ir.kind then
            local a = ir.value
            if "BoundsAccess" == a.kind then--bounds calculation
                return `midx:InBoundsExpanded([a.min.data],[a.max.data])
            elseif "IndexValue" == a.kind then
                local n = "d"..tostring(a.dim)
                return `idx.[n] + a.shift_ 
            else assert("ParamValue" == a.kind)
                return `opt_float(P.[a.name])
            end
        elseif "load" == ir.kind then
            local a = ir.value
            local im = imageref(a.image)
            if Offset:isclassof(a.index) then
                if conditioncoversload(ir.condition,a.index) then
                   return `im(midx(a.index.data))(0) 
                else
                   return `im:get(midx(a.index.data))(0)
                end
            else
                local gr = graphref(a.index)
                return `im(gr)(0)
            end
        elseif "vectorload" == ir.kind then
            local a = ir.value
            local im = imageref(a.image)
            local s = symbol(a.image.type:ElementType(),("%s_%s"):format(a.image.name,tostring(a.index)))
            if Offset:isclassof(a.index) then
                if conditioncoversload(ir.condition,a.index) then
                    statements:insert(quote
                        var [s] = im(midx(a.index.data))
                    end)
                else 
                    statements:insert(quote
                        var [s] = 0.f
                        if midx(a.index.data):InBounds() then
                            [s] = im(midx(a.index.data))
                        end
                    end)
                end
            else
                local gr = graphref(a.index)
                statements:insert(quote
                    var [s] = im(gr)
                end)
            end
            return s
        elseif "vectorextract" == ir.kind then
            local v = emit(ir.children[1])
            return `v(ir.channel)
        elseif "vectorconstruct" == ir.kind then
            local exps = ir.children:map(emit)
            return `[util.Vector(opt_float,#exps)]{ array(exps) }
        elseif "sampleimage" == ir.kind then
            local im = imageref(ir.image)
            local exps = ir.children:map(emit)
            local r = `im:sample(exps)
            if ir.image.scalar then
                r = `r(0)
            end
            return r
        elseif "apply" == ir.kind then
            local exps = ir.children:map(emit)
            return ir.generator(exps)
        elseif "vardecl" == ir.kind then
            return `opt_float(ir.constant)
        elseif "varuse" == ir.kind then
            local children = ir.children:map(emit)
            return children[1] -- return the variable declaration, which is the first child
        elseif "reduce" == ir.kind then
            local children = ir.children:map(emit)
            local vd, exp = children[1], tofloat(ir.children[2],children[2])
            local op
            if ir.op == "sum" then
                op = quote [vd] = [vd] + [exp] end
            else
                op = quote [vd] = [vd] * [exp] end
            end
            statements:insert(op)
            return children[1]
        end
    end
    
    local emitted,emitteduse = {},{}
    
    function emit(ir)
        assert(ir)
        return assert(emitted[ir],"Use before def")
    end

    local basecondition = Condition:create(List{})
    local currentcondition = basecondition
    local currentshape = ad.scalar
    
    local function emitshapechange(current,next)
        if current == next then return end
        emitconditionchange(currentcondition,basecondition) -- exit all conditions
        currentcondition = basecondition
        while not current:isprefixof(next) do
            local stmts = statementstack:remove()
            local a = current.keys[#current.keys]
            statementstack[#statementstack]:insert quote
                error("NYI - shapeiteration")
            end
            current = current:fromreduction()
       end
       for i = 1,#next.keys - #current.keys do
            statementstack:insert(terralib.newlist())
       end
       statements = statementstack[#statementstack]
    end
    
    local declarations = terralib.newlist()
    for i,ir in ipairs(instructions) do
        currentidx = i
        emitshapechange(currentshape,ir.shape)
        currentshape = ir.shape
        
        emitconditionchange(currentcondition,ir.condition)
        currentcondition = ir.condition
        
        if false then -- dynamically check dependencies are initialized before use, very slow, only use for debugging
            local ruse = symbol(bool,"ruse"..tostring(i))
            declarations:insert quote var [ruse] = false end
            statements:insert quote [ruse] = true end
            emitteduse[ir] = ruse
            for _,u in ipairs(deps[ir]) do
                if ir.kind ~= "varuse" or ir.children[1] == u then
                    local ruse = assert(emitteduse[u])
                    local str = ("%s r%s used %s which is not initialized\n"):format(name,tostring(i),tostring(ruse))
                    statements:insert quote
                        if not ruse then
                            printf(str)
                        end
                    end
                end
            end
        end
        
        local r
        if ir.kind == "const" or ir.kind == "varuse" or ir.kind == "reduce" then 
            r = assert(createexp(ir),"nil exp") 
        else
            r = symbol(ir.type,"r"..tostring(i))
            declarations:insert quote var [r] end
            local exp = assert(createexp(ir),"nil exp")
            statements:insert(quote
                [r] = exp
            end)
        end
        emitted[ir] = r
    end
    
    emitshapechange(currentshape,ad.scalar) -- also blanks condition
    assert(#statementstack == 1)
    
    local expressions = irroots:map(emit)
    local resultexpressions,scatterexpressions = {unpack(expressions,1,#results)},{unpack(expressions,#results+1)}
        
    local scatterstatements = terralib.newlist()
    local function toidx(index)
        if Offset:isclassof(index) then return `midx(index.data)
        else return graphref(index) end
    end
    for i,s in ipairs(scatters) do
        local image,exp = imageref(s.image),scatterexpressions[i]
        local index = toidx(s.index)
        local stmt
        if s.kind == "add" then
            assert(s.channel, "no channel on scatter?")
            stmt = `image:atomicAddChannel(index, s.channel, exp)
        else
            assert(s.kind == "set" and s.channel == 0, "set only works for single channel images")
            stmt = quote 
                image(index) = exp
            end
        end
        scatterstatements:insert(stmt)
    end

    local terra generatedfn([idx], [P], [extraarguments])
        var [midx] = idx
        [declarations]
        [statements]
        [scatterstatements]
        return [resultexpressions]
    end
    
    generatedfn:setname(name)
    if verboseAD then
        generatedfn:printpretty(false, false)
    end
    return generatedfn
end

local noscatters = terralib.newlist()

function ProblemSpecAD:CompileFunctionSpec(functionspec)
    local Index = functionspec.kind.kind == "GraphFunction" and int or functionspec.kind.ispace:indextype()
    return createfunction(self,functionspec.name,Index,functionspec.arguments,functionspec.results,functionspec.scatters)
end

function ProblemSpecAD:AddFunctions(functionspecs)
    local kind_to_functionmap = {}
    local kinds = List()
    for i,fs in ipairs(functionspecs) do -- group by unique function kind to pass to ProblemSpec:Functions call
        local fm = kind_to_functionmap[fs.kind]
        if not fm then
            fm = {}
            kind_to_functionmap[fs.kind] = fm
            kinds:insert(fs.kind)
        end
        assert(not fm[fs.name],"function already defined!")
        fm[fs.name] = self:CompileFunctionSpec(fs)
        if fm.derivedfrom and fs.derivedfrom then
            assert(fm.derivedfrom == fs.derivedfrom, "not same energy spec?")
        end
        fm.derivedfrom = fm.derivedfrom or fs.derivedfrom
    end
    for _,k in ipairs(kinds) do
        local fm = kind_to_functionmap[k]
        self.P:Functions(k,fm)
    end
end


local function classifyexpression(exp) -- what index space, or graph is this thing mapped over
    local classification
    local seenunknown = {}
    local unknownaccesses = terralib.newlist()
    local function addunknown(u)
        if not seenunknown[u] then
            unknownaccesses:insert(u)
            seenunknown[u] = true
        end
    end
    exp:visit(function(a)
        if ImageAccess:isclassof(a) then -- assume image X is unknown
            if a.image.location == A.UnknownLocation then
                addunknown(a)
            elseif a.image.gradientimages then
                for i,im in ipairs(a.image.gradientimages) do
                    assert(Offset:isclassof(a.index),"NYI - precomputed with graphs")
                    addunknown(im.unknown:shift(a.index))
                end
            end
            local aclass = Offset:isclassof(a.index) and A.CenteredFunction(a.image.type.ispace) or A.GraphFunction(a.index.graph.name)
            assert(nil == classification or aclass == classification, "residual contains image reads from multiple domains")
            classification = aclass
        end
    end)
    local template = A.ResidualTemplate(exp,unknownaccesses)
    if not classification then
        error("residual must actually use some image")
    end
    if classification.kind == "CenteredFunction" then
        exp:visit(function(a)
            if BoundsAccess:isclassof(a) and #a.min.data ~= #classification.ispace.dims then
                error(string.format("%s does not match index space %s",a,classification.ispace.dims))
            end
        end)
        -- by default zero-out any residual computation that uses out-of-bounds things
        -- users can opt for per-residual custom behavior using the InBounds checks
        local bbox = bboxforexpression(classification.ispace,exp)
        template.expression = ad.select(bbox:asvar(),exp,0)
    end
    return classification,template
end

local function extract_unused_unknown_ispaces(kinds,kind_to_templates)
    local used_nongraph_unknown_ispaces = {}
    local all_unknown_ispaces = {}
    for _,k in ipairs(kinds) do
        for _,template in ipairs(kind_to_templates[k]) do
            for _,u in ipairs(template.unknowns) do
                if k.kind == "CenteredFunction" then
                    used_nongraph_unknown_ispaces[u.image.type.ispace] = true
                end
                all_unknown_ispaces[u.image.type.ispace] = true
            end
        end
    end
    local unused_unknown_ispaces = terralib.newlist()
    for _,k in ipairs(table.keys(all_unknown_ispaces)) do
        if not used_nongraph_unknown_ispaces[k] then
            unused_unknown_ispaces:insert(k)
        end
    end
    return unused_unknown_ispaces
end

local function insert_dummy_energies_for_unused_unknown_ispaces(kinds,kind_to_templates,unused_unknown_ispaces)
    for _,ispace in ipairs(unused_unknown_ispaces) do
        local kind = A.CenteredFunction(ispace)
        if not kind_to_templates[kind] then
            kinds:insert(kind)
            kind_to_templates[kind] = terralib.newlist()
        end
        local template = A.ResidualTemplate(ad.toexp(0),terralib.newlist())
        kind_to_templates[kind]:insert(template)
    end
end

local function handle_unused_unknown_ispaces(kinds,kind_to_templates)
    local unused_unknown_ispaces = extract_unused_unknown_ispaces(kinds,kind_to_templates)
    if #unused_unknown_ispaces > 0 then
        local message = "No unknownwise residuals for ispaces(s) "..tostring(unused_unknown_ispaces[1])
        for i=2,#unused_unknown_ispaces do
            message = message..", "..tostring(unused_unknown_ispaces[i])
        end
        print(message..". Creating zero-valued stand-ins.")
    end
    insert_dummy_energies_for_unused_unknown_ispaces(kinds,kind_to_templates,unused_unknown_ispaces)
end

local function toenergyspecs(Rs)    
    local kinds,kind_to_templates = MapAndGroupBy(Rs,classifyexpression)
    -- We need to execute kernels for unknowns even if they have no unknownwise residuals,
    -- to initialize values for graph residuals and to generally just do PCG/GN bookkeeping
    handle_unused_unknown_ispaces(kinds,kind_to_templates)
    return kinds:map(function(k) return A.EnergySpec(k,kind_to_templates[k]) end)
end

--given that the residual at (0,0) uses the variables in 'unknownsupport',
--what is the set of residuals will use variable X(0,0).
--this amounts to taking each variable in unknown support and asking which residual is it
--that makes that variable X(0,0)
local function residualsincludingX00(unknownsupport,unknown,channel)
    assert(channel)
    local r = terralib.newlist()
    for i,u in ipairs(unknownsupport) do
        assert(Offset:isclassof(u.index),"unexpected graph access")
        if u.image == unknown and u.channel == channel then
            r:insert(u.index:Invert())
        end
    end
    return r
end
local function unknownsforresidual(r,unknownsupport)
    return unknownsupport:map("shift",r)
end

local function createzerolist(N)
    local r = terralib.newlist()
    for i = 1,N do
        r[i] = ad.toexp(0)
    end
    return r
end
    
local function lprintf(ident,fmt,...)
    if true then return end 
    local str = fmt:format(...)
    ident = (" "):rep(ident*4)
    str = ident..str:gsub('\n', "\n"..ident)
    return print(str) 
end

local EMPTY = List()

local function createjtjcentered(PS,ES)
    local UnknownType = PS.P:UnknownType()
    local ispace = ES.kind.ispace
    local N = UnknownType:VectorSizeForIndexSpace(ES.kind.ispace)
    local P = PS:UnknownArgument(1)
    local CtC = PS:UnknownArgument(2)
    --local Pre = PS:UnknownArgument(3)
    local P_hat_c = {}
    local conditions = terralib.newlist()
    for rn,residual in ipairs(ES.residuals) do
        local F,unknownsupport = residual.expression,residual.unknowns
        lprintf(0,"\n\n\n\n\n##################################################")
        lprintf(0,"r%d = %s",rn,F)
        for idx,unknownname,chan in UnknownType:UnknownIteratorForIndexSpace(ispace) do 
            local unknown = PS:ImageWithName(unknownname) 
            local x = unknown(ispace:ZeroOffset(),chan)
            local residuals = residualsincludingX00(unknownsupport,unknown,chan)
            for _,r in ipairs(residuals) do
                local rexp = shiftexp(F,r)
                local condition,drdx00 = ad.splitcondition(rexp:d(x))
                lprintf(1,"instance:\ndr%d_%s/dx00[%d] = %s",rn,tostring(r),chan,tostring(drdx00))
                local unknowns = unknownsforresidual(r,unknownsupport)
                for _,u in ipairs(unknowns) do
                    local uv = ad.v[u]
                    local condition2, drdx_u = ad.splitcondition(rexp:d(uv))
                    local exp = drdx00*drdx_u

                    lprintf(2,"term:\ndr%d_%s/dx%s[%d] = %s",rn,tostring(r),tostring(u.index),u.chan,tostring(drdx_u))
                    local conditionmerged = condition*condition2
                    if not P_hat_c[conditionmerged] then
                        conditions:insert(conditionmerged)
                        P_hat_c[conditionmerged] = createzerolist(N)
                    end
                    P_hat_c[conditionmerged][idx+1] = P_hat_c[conditionmerged][idx+1] + P[u.image.name](u.index,u.channel)*exp
                end
            end
        end
    end
    local P_hat = createzerolist(N)
    for _,c in ipairs(conditions) do
        for i = 1,N do
            P_hat[i] = P_hat[i] + c*P_hat_c[c][i]
        end
    end
    for i,p in ipairs(P_hat) do
        P_hat[i] = 1.0 * p
    end
    if PS:UsesLambda() then
        for idx,unknownname,chan in UnknownType:UnknownIteratorForIndexSpace(ispace) do
            local unknown = PS:ImageWithName(unknownname) 
            local u = unknown(ispace:ZeroOffset(),chan)
            P_hat[idx+1] = P_hat[idx+1] + CtC[unknownname](ispace:ZeroOffset(),chan)*P[unknownname](ispace:ZeroOffset(),chan)
        end
    end
    dprint("JTJ[nopoly] = ", ad.tostrings(P_hat))
    P_hat = ad.polysimplify(P_hat)
    dprint("JTJ[poly] = ", ad.tostrings(P_hat))
    local r = ad.Vector(unpack(P_hat))
    local result = A.FunctionSpec(ES.kind,"applyJTJ", List {"P", "CtC"}, List{r}, EMPTY,ES)
    return result
end


local function createjtjgraph(PS,ES)
    local P,Ap_X = PS:UnknownArgument(1),PS:UnknownArgument(2)

    local result = ad.toexp(0)
    local scatters = List() 
    local scattermap = {}
    local function addscatter(u,exp)
        local s = scattermap[u]
        if not s then
            s =  Scatter(Ap_X[u.image.name],u.index,u.channel,ad.toexp(0),"add")
            scattermap[u] = s
            scatters:insert(s)
        end
        s.expression = s.expression + exp
    end
    for i,term in ipairs(ES.residuals) do
        local F,unknownsupport = term.expression,term.unknowns
        local unknownvars = unknownsupport:map(function(x) return ad.v[x] end)
        local partials = F:gradient(unknownvars)
        local Jp = ad.toexp(0)
        for i,partial in ipairs(partials) do
            local u = unknownsupport[i]
            assert(GraphElement:isclassof(u.index))
            Jp = Jp + partial*P[u.image.name](u.index,u.channel)
        end
        for i,partial in ipairs(partials) do
            local u = unknownsupport[i]
            local jtjp = 1.0 * Jp*partial
            result = result + P[u.image.name](u.index,u.channel)*jtjp
            addscatter(u,jtjp)
        end
    end

    return A.FunctionSpec(ES.kind,"applyJTJ", List {"P", "Ap_X"}, List { result }, scatters, ES)
end


local function createjtfcentered(PS,ES)
   local UnknownType = PS.P:UnknownType()
   local ispace = ES.kind.ispace
   local N = UnknownType:VectorSizeForIndexSpace(ispace)

   local F_hat = createzerolist(N) --gradient
   local P_hat = createzerolist(N) --preconditioner
    
    for ridx,residual in ipairs(ES.residuals) do
        local F, unknownsupport = residual.expression,residual.unknowns
        lprintf(0,"-------------")
        lprintf(1,"R[%d] = %s",ridx,tostring(F))

        for idx,unknownname,chan in UnknownType:UnknownIteratorForIndexSpace(ispace) do
            local unknown = PS:ImageWithName(unknownname) 
            local x = unknown(ispace:ZeroOffset(),chan)
            
            local residuals = residualsincludingX00(unknownsupport,unknown,chan)

            local sum = 0
            for _,f in ipairs(residuals) do
                local F_x = shiftexp(F,f)
                local dfdx00 = F_x:d(x)		-- entry of J^T
                local dfdx00F = dfdx00*F_x	-- entry of \gradF == J^TF
                F_hat[idx+1] = F_hat[idx+1] + dfdx00F			-- summing it up to get \gradF

                local dfdx00Sq = dfdx00*dfdx00	-- entry of Diag(J^TJ)
                P_hat[idx+1] = P_hat[idx+1] + dfdx00Sq			-- summing the pre-conditioner up
                lprintf(2,"dR[%d]_%s/dx[%d] = %s",ridx,tostring(f),chan,tostring(dfdx00F))
            end

        end
    end
	for i = 1,N do
	    if not PS.P.usepreconditioner then
		    P_hat[i] = ad.toexp(1.0)
	    else
		    P_hat[i] = ad.polysimplify(P_hat[i])
	    end
	    F_hat[i] = ad.polysimplify(1.0 * F_hat[i])
	end
	dprint("JTF =", ad.tostrings({F_hat[1], F_hat[2], F_hat[3]}))
    return A.FunctionSpec(ES.kind,"evalJTF", EMPTY, List{ ad.Vector(unpack(F_hat)), ad.Vector(unpack(P_hat)) }, EMPTY,ES)
end

local function createmodelcost(PS,ES)
    local UnknownType = PS.P:UnknownType()
    local ispace = ES.kind.ispace
    local N = UnknownType:VectorSizeForIndexSpace(ispace)


    local Delta = PS:UnknownArgument(1)
    local result = 0.0 --model residuals squared (and summed among unknowns...)
    
    for ridx,residual in ipairs(ES.residuals) do
        local F, unknownsupport = residual.expression,residual.unknowns
        lprintf(0,"-------------")
        lprintf(1,"R[%d] = %s",ridx,tostring(F))
        local unknownvars = unknownsupport:map(function(x) return ad.v[x] end)
        local partials = F:gradient(unknownvars)

        local JTdelta = 0.0

        for i,partial in ipairs(partials) do
            local u = unknownsupport[i]
            local delta = Delta[u.image.name](u.index,u.channel)
            JTdelta = JTdelta + (partial * delta)
        end
        local residual_m = F + JTdelta
        result = result + (residual_m*residual_m)
    end
    result = ad.polysimplify(0.5*result)
    return A.FunctionSpec(ES.kind,"modelcost", List {"Delta"}, List{ result }, EMPTY,ES)
end

local function createmodelcostgraph(PS,ES)
    local Delta = PS:UnknownArgument(1)
    local result = 0.0 --model residuals squared (and summed among unknowns...)
    for i,term in ipairs(ES.residuals) do
        local F,unknownsupport = term.expression,term.unknowns
        local unknownvars = unknownsupport:map(function(x) return ad.v[x] end)
        local partials = F:gradient(unknownvars)

        local JTdelta = 0.0

        for i,partial in ipairs(partials) do
            local u = unknownsupport[i]
            assert(GraphElement:isclassof(u.index))
            local delta = Delta[u.image.name](u.index,u.channel)
            JTdelta = JTdelta + (partial * delta)
        end
        local residual_m = F + JTdelta
        result = result + (residual_m*residual_m)
    end
    result = ad.polysimplify(0.5*result)
    return A.FunctionSpec(ES.kind, "modelcost", List { "Delta" }, List{ result }, EMPTY,ES)
end


local function createjtfgraph(PS,ES)
    local R,Pre = PS:UnknownArgument(1),PS:UnknownArgument(2)
    local scatters = List() 
    local scattermap = { [R] = {}, [Pre] = {}}
    local function addscatter(im,u,exp)
        local s = scattermap[im][u]
        if not s then
            s =  Scatter(im[u.image.name],u.index,u.channel,ad.toexp(0),"add")
            scattermap[im][u] = s
            scatters:insert(s)
        end
        s.expression = s.expression + exp
    end
    for i,term in ipairs(ES.residuals) do
        local F,unknownsupport = term.expression,term.unknowns
        local unknownvars = unknownsupport:map(function(x) return ad.v[x] end)
        local partials = F:gradient(unknownvars)
        for i,partial in ipairs(partials) do
            local u = unknownsupport[i]
            assert(GraphElement:isclassof(u.index))
            addscatter(R,u,-1.0*partial*F)
            addscatter(Pre,u,partial*partial)
        end
    end
    return A.FunctionSpec(ES.kind, "evalJTF", List { "R", "Pre" }, EMPTY, scatters,ES)
end

local function computeCtCcentered(PS,ES)
   local UnknownType = PS.P:UnknownType()
   local ispace = ES.kind.ispace
   local N = UnknownType:VectorSizeForIndexSpace(ispace)
   local D_hat = createzerolist(N) --gradient
    
    for ridx,residual in ipairs(ES.residuals) do
        local F, unknownsupport = residual.expression,residual.unknowns
        lprintf(0,"-------------")
        lprintf(1,"R[%d] = %s",ridx,tostring(F))

        for idx,unknownname,chan in UnknownType:UnknownIteratorForIndexSpace(ispace) do
            local unknown = PS:ImageWithName(unknownname) 
            local x = unknown(ispace:ZeroOffset(),chan)

            local residuals = residualsincludingX00(unknownsupport,unknown,chan)
            local sum = 0
            for _,f in ipairs(residuals) do
                local F_x = shiftexp(F,f)
                local dfdx00 = F_x:d(x)     -- entry of J^T
                local dfdx00Sq = dfdx00*dfdx00  -- entry of Diag(J^TJ)

                local inv_radius = 1.0 / PS.trust_region_radius
                local D_entry = dfdx00Sq*inv_radius 
                D_hat[idx+1] = D_hat[idx+1] + D_entry
            end

        end
    end
    for i = 1,N do
        D_hat[i] = ad.polysimplify(D_hat[i])
    end
    return A.FunctionSpec(ES.kind,"computeCtC", List { }, List{ ad.Vector(unpack(D_hat)) }, EMPTY,ES)
end

local function computeCtCgraph(PS,ES)
    local CtC = PS:UnknownArgument(1),PS:UnknownArgument(2)
    local scatters = List() 
    local scattermap = { [CtC] = {}}

    local function addscatter(im,u,exp)
        local s = scattermap[im][u]
        if not s then
            s =  Scatter(im[u.image.name],u.index,u.channel,ad.toexp(0),"add")
            scattermap[im][u] = s
            scatters:insert(s)
        end
        s.expression = s.expression + exp
    end
    for i,term in ipairs(ES.residuals) do
        local F,unknownsupport = term.expression,term.unknowns
        local unknownvars = unknownsupport:map(function(x) return ad.v[x] end)
        local partials = F:gradient(unknownvars)
        for i,partial in ipairs(partials) do
            local u = unknownsupport[i]
            assert(GraphElement:isclassof(u.index))
            local inv_radius = 1.0 / PS.trust_region_radius
            addscatter(CtC,u,partial*partial*inv_radius)
        end
    end
    return A.FunctionSpec(ES.kind, "computeCtC", List { "CtC" }, EMPTY, scatters, ES)
end

local function createdumpjcentered(PS,ES)
   local UnknownType = PS.P:UnknownType()
   local ispace = ES.kind.ispace
   local N = UnknownType:VectorSizeForIndexSpace(ispace)

    local outputs = List{}

    for ridx,residual in ipairs(ES.residuals) do
        local F, unknownsupport = residual.expression,residual.unknowns
        lprintf(0,"-------------")
        lprintf(1,"R[%d] = %s",ridx,tostring(F))
        for i,unknown in ipairs(unknownsupport) do
            outputs:insert(F:d(ad.v[unknown]))
        end
    end
    return A.FunctionSpec(ES.kind,"dumpJ", EMPTY, outputs, EMPTY,ES)
end
local function createdumpjgraph(PS,ES)
    local outputs = List{}
    for i,term in ipairs(ES.residuals) do
        local F,unknownsupport = term.expression,term.unknowns
        for i,unknown in ipairs(unknownsupport) do
            outputs:insert(F:d(ad.v[unknown]))
        end
    end
    return A.FunctionSpec(ES.kind, "dumpJ", EMPTY, outputs, EMPTY,ES)
end

local lastTime = nil
function timeSinceLast(name)
    local currentTime = terralib.currenttimeinseconds()
    if (lastTime) then
        local deltaTime = currentTime-lastTime
        print(string.format("%s: %f sec\n",name,deltaTime))
    end
    lastTime = currentTime
end

local function creategradient(unknown,costexp)
    local unknownvars = unknowns(costexp)
    local gradient = costexp:gradient(unknownvars)

    dprint("grad expression")
    local names = table.concat(unknownvars:map(function(v) return tostring(v:key()) end),", ")
    dprint(names.." = "..ad.tostrings(gradient))
    
    local gradientsgathered = createzerolist(unknown.type.channelcount)
    for i,u in ipairs(unknownvars) do
        local a = u:key()
        local shift = shiftexp(gradient[i],a.index:Invert())
        gradientsgathered[a.channel+1] = gradientsgathered[a.channel+1] + shift
    end
    dprint("grad gather")
    dprint(ad.tostrings(gradientsgathered))
    return ad.Vector(gradientsgathered)
end
    
local function createcost(ES)
    local function sumsquared(terms)
        local sum = ad.toexp(0)
        for i,t in ipairs(terms) do
            sum = sum + t*t
        end
        return 0.5*sum
    end
    local exp = sumsquared(ES.residuals:map("expression"))
    return A.FunctionSpec(ES.kind,"cost", EMPTY, List{exp}, EMPTY,ES) 
end

function createprecomputed(self,precomputedimages)

    local ispaces,image_map = MapAndGroupBy(precomputedimages,function(im) return im.type.ispace,im end)

    local precomputes = List()
    for _,ispace in ipairs(ispaces) do
        local scatters = List()
        local zoff = ispace:ZeroOffset()
        for _,im in ipairs(image_map[ispace]) do
            local expression = ad.polysimplify(im.expression)
            scatters:insert(Scatter(im, zoff, 0, im.expression, "set"))
            for _,gim in ipairs(im.gradientimages) do
                local gradientexpression = ad.polysimplify(gim.expression)
                if not ad.Const:isclassof(gradientexpression) then
                    scatters:insert(Scatter(gim.image, zoff, 0, gradientexpression, "set"))
                end
            end
        end
        local pc = A.FunctionSpec(A.CenteredFunction(ispace),"precompute", EMPTY, EMPTY, scatters)
        precomputes:insert(pc)
    end
    return precomputes
end
local function extractresidualterms(...)
    local exp = terralib.newlist {}
    for i = 1, select("#",...) do
        local e = select(i,...)
        if ad.ExpVector:isclassof(e) then
            for i,t in ipairs(e:expressions()) do
                t = assert(ad.toexp(t), "expected an ad expression")
                exp:insert(t)
            end
        else
            exp:insert((assert(ad.toexp(e), "expected an ad expression")))
        end
    end
    return exp
end
function ProblemSpecAD:Cost(...)
    local terms = extractresidualterms(...)
    local functionspecs = List()
    local energyspecs = toenergyspecs(terms)
    for _,energyspec in ipairs(energyspecs) do
        functionspecs:insert(createcost(energyspec))          
        if energyspec.kind.kind == "CenteredFunction" then
            functionspecs:insert(createjtjcentered(self,energyspec))
            functionspecs:insert(createjtfcentered(self,energyspec))
            functionspecs:insert(createdumpjcentered(self,energyspec))
            
            if self.P:UsesLambda() then
                functionspecs:insert(computeCtCcentered(self,energyspec))
                functionspecs:insert(createmodelcost(self,energyspec))
            end
        else
            functionspecs:insert(createjtjgraph(self,energyspec))
            functionspecs:insert(createjtfgraph(self,energyspec))
            functionspecs:insert(createdumpjgraph(self,energyspec))
            
            if self.P:UsesLambda() then
                functionspecs:insert(computeCtCgraph(self,energyspec))
                functionspecs:insert(createmodelcostgraph(self,energyspec))         
            end
        end
    end
    functionspecs:insertall(createprecomputed(self,self.precomputed))
    for i,exclude in ipairs(self.excludeexps) do
        local class = classifyexpression(exclude)
        functionspecs:insert(A.FunctionSpec(class, "exclude", EMPTY,List{exclude}, EMPTY))
    end
    
    self:AddFunctions(functionspecs)
    self.P.energyspecs = energyspecs
    return self.P
end

function ProblemSpecAD:Exclude(exp)
    exp = assert(ad.toexp(exp), "expected a AD expression")
    self.excludeexps:insert(exp)
end

function SampledImage:__call(x,y,c)
    if c or self.op.imagebeingsampled.type.channelcount == 1 then
        assert(not c or c < self.op.imagebeingsampled.type.channelcount, "index out of bounds")
        return self.op(c or 0,x,y)
    else
        local r = {}
        for i = 0,self.op.imagebeingsampled.type.channelcount - 1 do
            r[i+1] = self.op(i,x,y)
        end
        return ad.Vector(unpack(r))
    end
end
local function tosampledimage(im)
    if Image:isclassof(im) then
        assert(im:DimCount() == 2, "sampled images must be 2D")
        return ad.sampledimage(im)
    end
    return SampledImage:isclassof(im) and im or nil
end
function ad.sampledimage(image,imagedx,imagedy)
    if imagedx then
        imagedx = assert(tosampledimage(imagedx), "expected an image or a sampled image as a derivative")
        imagedy = assert(tosampledimage(imagedy), "expected an image or a sampled image as a derivative")
    end
    local op = ad.newop("sampleimage_"..image.name)
    op.imagebeingsampled = image --not the best place to store this but other ways are more cumbersome
    op.hasconst = true
    function op:generate(exp,args) error("sample image is not implemented directly") end
    function op:getpartials(exp)
        assert(imagedx and imagedy, "image derivatives are not defined for this image and cannot be used in autodiff")
        local x,y = unpack(exp:children())
        return terralib.newlist { imagedx(x,y,exp.const), imagedy(x,y,exp.const) }
    end
    return SampledImage(op)
end

for i = 2,12 do
    opt["float"..tostring(i)] = util.Vector(float,i)
    opt["double"..tostring(i)] = util.Vector(double,i)
    if opt_float == float then
        opt["opt_float"..tostring(i)] = opt["float"..tostring(i)]
    else
        opt["opt_float"..tostring(i)] = opt["double"..tostring(i)]
    end
end


opt.Dot = util.Dot
opt.toispace = toispace

-- C API implementation functions
-- WARNING: if you change these you need to update release/Opt.h

-- define just stores meta-data right now. ProblemPlan does all compilation for now
terra opt.ProblemDefine(filename : rawstring, kind : rawstring)
    var id : int
    problemDefine(filename, kind, &id)
    return [&opt.Problem](id)
end 
terra opt.ProblemDelete(p : &opt.Problem)
    var id = int(int64(p))
    problemDelete(id)
end
terra opt.ProblemPlan(problem : &opt.Problem, dimensions : &uint32) : &opt.Plan
    var p : &opt.Plan = nil 
    problemPlan(int(int64(problem)),dimensions,&p)
    return p
end

terra opt.PlanFree(plan : &opt.Plan)
    plan.free(plan.data)
    planFree(plan)
    plan:delete()
end

terra opt.ProblemInit(plan : &opt.Plan, params : &&opaque) 
    return plan.init(plan.data, params)
end
terra opt.ProblemStep(plan : &opt.Plan, params : &&opaque) : int
    return plan.step(plan.data, params)
end
terra opt.ProblemSolve(plan : &opt.Plan, params : &&opaque)
   opt.ProblemInit(plan, params)
   while opt.ProblemStep(plan, params) ~= 0 do end
end
terra opt.ProblemCurrentCost(plan : &opt.Plan) : double
    return plan.cost(plan.data)
end

terra opt.SetSolverParameter(plan : &opt.Plan, name : rawstring, value : &opaque) 
    return plan.setsolverparameter(plan.data, name, value)
end

return opt
