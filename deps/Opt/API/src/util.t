local S = require("std")
require("precision")
local util = {}
local verbosePTX = _opt_verbosity > 2

util.C = terralib.includecstring [[
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#ifdef _WIN32
	#include <io.h>
#endif
]]
local C = util.C

--[[ rPrint(struct, [limit], [indent])   Recursively print arbitrary data. 
    Set limit (default 100) to stanch infinite loops.
    Indents tables as [KEY] VALUE, nested tables as [KEY] [KEY]...[KEY] VALUE
    Set indent ("") to prefix each line:    Mytable [KEY] [KEY]...[KEY] VALUE
--]]
function util.rPrint(s, l, i) -- recursive Print (structure, limit, indent)
    l = (l) or 100; i = i or "";    -- default item limit, indent string
    local ts = type(s);
    if (l<1) then print (i,ts," *snip* "); return end;
    if (ts ~= "table") then print (i,ts,s); return end
    print (i,ts);           -- print "table"
    for k,v in pairs(s) do  -- print "[KEY] VALUE"
        util.rPrint(v, l-1, i.."\t["..tostring(k).."]");
    end
end 

function table.keys(tab)
    local result = terralib.newlist()
    for k,_ in pairs(tab) do
        result:insert(k)
    end
    return result
end

local cuda_compute_version = 30
local libdevice = terralib.cudahome..string.format("/nvvm/libdevice/libdevice.compute_%d.10.bc",cuda_compute_version)

local terra toYesNo(pred : int32)
    if pred ~= 0 then return "Yes" else return  "No" end
end

local terra toEnDisabled(pred : int32)
    if pred ~= 0 then return "Enabled" else return  "Disabled" end
end

local terra printCudaDeviceProperties()
    var dev : int32
    var driverVersion = 0
    var runtimeVersion = 0;
    C.cudaGetDevice(&dev)
    var deviceProp : C.cudaDeviceProp
    C.cudaGetDeviceProperties(&deviceProp, dev)

    C.printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name)

    C.cudaDriverGetVersion(&driverVersion)
    C.cudaRuntimeGetVersion(&runtimeVersion)
    C.printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
    C.printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor)

    C.printf("  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
            [float](deviceProp.totalGlobalMem)/1048576.0f, [uint64](deviceProp.totalGlobalMem))
    -- TODO: compute # "cores"
    C.printf("  (%2d) Multiprocessors\n",deviceProp.multiProcessorCount)
    C.printf("  GPU Max Clock rate:                            %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

    C.printf("  Memory Clock rate:                             %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);
    C.printf("  Memory Bus Width:                              %d-bit\n",   deviceProp.memoryBusWidth);

    if (deviceProp.l2CacheSize ~= 0) then
        C.printf("  L2 Cache Size:                                 %d bytes\n", deviceProp.l2CacheSize);
    end

    C.printf("  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n",
           deviceProp.maxTexture1D   , deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
           deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
    C.printf("  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
           deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
    C.printf("  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d layers\n",
           deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);

    C.printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
    C.printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
    C.printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
    C.printf("  Warp size:                                     %d\n", deviceProp.warpSize);
    C.printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
    C.printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
    C.printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
           deviceProp.maxThreadsDim[0],
           deviceProp.maxThreadsDim[1],
           deviceProp.maxThreadsDim[2]);
    C.printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
           deviceProp.maxGridSize[0],
           deviceProp.maxGridSize[1],
           deviceProp.maxGridSize[2]);
    C.printf("  Maximum memory pitch:                          %lu bytes\n", deviceProp.memPitch);
    C.printf("  Texture alignment:                             %lu bytes\n", deviceProp.textureAlignment);
    C.printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n", toYesNo(deviceProp.deviceOverlap), deviceProp.asyncEngineCount);
    C.printf("  Run time limit on kernels:                     %s\n", toYesNo(deviceProp.kernelExecTimeoutEnabled))
    C.printf("  Integrated GPU sharing Host Memory:            %s\n", toYesNo(deviceProp.integrated))
    C.printf("  Support host page-locked memory mapping:       %s\n", toYesNo(deviceProp.canMapHostMemory))
    C.printf("  Alignment requirement for Surfaces:            %s\n", toYesNo(deviceProp.surfaceAlignment))
    C.printf("  Device has ECC support:                        %s\n", toEnDisabled(deviceProp.ECCEnabled))
    C.printf("  Device supports Unified Addressing (UVA):      %s\n", toYesNo(deviceProp.unifiedAddressing))
    C.printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n", deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);
end


local pascalOrBetterGPU = false 

terra deviceMajorComputeCapability()
    var deviceID : int32
    C.cudaGetDevice(&deviceID)
    var majorComputeCapability : int32
    C.cudaDeviceGetAttribute(&majorComputeCapability, C.cudaDevAttrComputeCapabilityMajor, deviceID)
    return majorComputeCapability
end
local computeC = deviceMajorComputeCapability()
if computeC >= 6 then
    pascalOrBetterGPU = true
end
if _opt_verbosity > 1 then
    printCudaDeviceProperties()
end
if opt_float == double and (not pascalOrBetterGPU) then
    print("Warning: double precision on GPUs with compute capability < 6.0 (before Pascal) have no native double precision atomics, so we must use slow software emulation instead. This has a large performance impact on graph energies.")
end

local extern = terralib.externfunction
if terralib.linkllvm then
    local obj = terralib.linkllvm(libdevice)
    function extern(...) return obj:extern(...) end
else
    terralib.linklibrary(libdevice)
end
local mathParamCount = {sqrt = 1,
cos  = 1,
acos = 1,
sin  = 1,
asin = 1,
tan  = 1,
atan = 1,
ceil = 1,
floor = 1,
log = 1,
exp = 1,
pow  = 2,
fmod = 2,
fmax = 2,
fmin = 2
}

util.gpuMath = {}
util.cpuMath = {}
for k,v in pairs(mathParamCount) do
	local params = {}
	for i = 1,v do
		params[i] = opt_float
	end
    if (opt_float == float) then
        util.gpuMath[k] = extern(("__nv_%sf"):format(k), params -> opt_float)
        util.cpuMath[k] = C[k.."f"]
    else
        util.gpuMath[k] = extern(("__nv_%s"):format(k), params -> opt_float)
        util.cpuMath[k] = C[k]
    end
end
if (opt_float == float) then
    util.cpuMath["abs"] = C["fabsf"]
else
    util.cpuMath["abs"] = C["fabs"]
end
util.gpuMath["abs"] = terra (x : opt_float)
	if x < 0 then
		x = -x
	end
	return x
end

local Vectors = {}
function util.isvectortype(t) return Vectors[t] end
util.Vector = terralib.memoize(function(typ,N)
    N = assert(tonumber(N),"expected a number")
    local ops = { "__sub","__add","__mul","__div" }
    local struct VecType { 
        data : typ[N]
    }
    Vectors[VecType] = true
    VecType.metamethods.type, VecType.metamethods.N = typ,N
    VecType.metamethods.__typename = function(self) return ("%s_%d"):format(tostring(self.metamethods.type),self.metamethods.N) end
    for i, op in ipairs(ops) do
        local i = symbol(int,"i")
        local function template(ae,be)
            return quote
                var c : VecType
                for [i] = 0,N do
                    c.data[i] = operator(op,ae,be)
                end
                return c
            end
        end
        local terra opvv(a : VecType, b : VecType) [template(`a.data[i],`b.data[i])]  end
        local terra opsv(a : typ, b : VecType) [template(`a,`b.data[i])]  end
        local terra opvs(a : VecType, b : typ) [template(`a.data[i],`b)]  end
        
        local doop
        if terralib.overloadedfunction then
            doop = terralib.overloadedfunction("doop",{opvv,opsv,opvs})
        else
            doop = opvv
            doop:adddefinition(opsv:getdefinitions()[1])
            doop:adddefinition(opvs:getdefinitions()[1])
        end
        
       VecType.metamethods[op] = doop
    end
    terra VecType.metamethods.__unm(self : VecType)
        var c : VecType
        for i = 0,N do
            c.data[i] = -self.data[i]
        end
        return c
    end
    terra VecType:abs()
       var c : VecType
       for i = 0,N do
	  -- TODO: use opt.abs
	  if self.data[i] < 0 then
	     c.data[i] = -self.data[i]
	  else
	     c.data[i] = self.data[i]
	  end
       end
       return c
    end
    terra VecType:sum()
       var c : typ = 0
       for i = 0,N do
	  c = c + self.data[i]
       end
       return c
    end
    terra VecType:dot(b : VecType)
        var c : typ = 0
        for i = 0,N do
            c = c + self.data[i]*b.data[i]
        end
        return c
    end
    terra VecType:max()
        var c : typ = 0
        if N > 0 then
            c = self.data[0]
        end
        for i = 1,N do
            if self.data[i] > c then
                c = self.data[i]
            end
        end
        return c
    end
	terra VecType:size()
        return N
    end
    terra VecType.methods.FromConstant(x : typ)
        var c : VecType
        for i = 0,N do
            c.data[i] = x
        end
        return c
    end
    VecType.metamethods.__apply = macro(function(self,idx) return `self.data[idx] end)
    VecType.metamethods.__cast = function(from,to,exp)
        if from:isarithmetic() and to == VecType then
            return `VecType.FromConstant(exp)
        end
        error(("unknown vector conversion %s to %s"):format(tostring(from),tostring(to)))
    end
    return VecType
end)

function Array(T,debug)
    local struct Array(S.Object) {
        _data : &T;
        _size : int32;
        _capacity : int32;
    }
    function Array.metamethods.__typename() return ("Array(%s)"):format(tostring(T)) end
    local assert = debug and S.assert or macro(function() return quote end end)
    terra Array:init() : &Array
        self._data,self._size,self._capacity = nil,0,0
        return self
    end
    terra Array:reserve(cap : int32)
        if cap > 0 and cap > self._capacity then
            var oc = self._capacity
            if self._capacity == 0 then
                self._capacity = 16
            end
            while self._capacity < cap do
                self._capacity = self._capacity * 2
            end
            self._data = [&T](S.realloc(self._data,sizeof(T)*self._capacity))
        end
    end
    terra Array:initwithcapacity(cap : int32) : &Array
        self:init()
        self:reserve(cap)
        return self
    end
    terra Array:__destruct()
        assert(self._capacity >= self._size)
        for i = 0ULL,self._size do
            S.rundestructor(self._data[i])
        end
        if self._data ~= nil then
            C.free(self._data)
            self._data = nil
        end
    end
    terra Array:size() return self._size end
    
    terra Array:get(i : int32)
        assert(i < self._size) 
        return &self._data[i]
    end
    Array.metamethods.__apply = macro(function(self,idx)
        return `@self:get(idx)
    end)
    
    terra Array:insertNatlocation(idx : int32, N : int32, v : T) : {}
        assert(idx <= self._size)
        self._size = self._size + N
        self:reserve(self._size)

        if self._size > N then
            var i = self._size
            while i > idx do
                self._data[i - 1] = self._data[i - 1 - N]
                i = i - 1
            end
        end

        for i = 0ULL,N do
            self._data[idx + i] = v
        end
    end
    terra Array:insertatlocation(idx : int32, v : T) : {}
        return self:insertNatlocation(idx,1,v)
    end
    terra Array:insert(v : T) : {}
        return self:insertNatlocation(self._size,1,v)
    end
    terra Array:remove(idx : int32) : T
        assert(idx < self._size)
        var v = self._data[idx]
        self._size = self._size - 1
        for i = idx,self._size do
            self._data[i] = self._data[i + 1]
        end
        return v
    end
    if not T:isstruct() then
        terra Array:indexof(v : T) : int32
            for i = 0LL,self._size do
                if (v == self._data[i]) then
                    return i
                end
            end
            return -1
        end
        terra Array:contains(v : T) : bool
            return self:indexof(v) >= 0
        end
    end
	
    return Array
end

local Array = S.memoize(Array)

local warpSize = 32
util.warpSize = warpSize

util.symTable = function(typ, N, name)
	local r = terralib.newlist()
	for i = 1, N do
		r[i] = symbol(typ, name..tostring(i-1))
	end
	return r
end

util.ceilingDivide = terra(a : int32, b : int32)
	return (a + b - 1) / b
end

struct util.TimingInfo {
	startEvent : C.cudaEvent_t
	endEvent : C.cudaEvent_t
	duration : float
	eventName : rawstring
}
local TimingInfo = util.TimingInfo

util.TimerEvent = C.cudaEvent_t

struct util.Timer {
	timingInfo : &Array(TimingInfo)
}
local Timer = util.Timer

terra Timer:init() 
	self.timingInfo = [Array(TimingInfo)].alloc():init()
end

terra Timer:cleanup()
    for i = 0,self.timingInfo:size() do
        var eventInfo = self.timingInfo(i);
        C.cudaEventDestroy(eventInfo.startEvent);
        C.cudaEventDestroy(eventInfo.endEvent);
    end
    self.timingInfo:delete()
end 


terra Timer:startEvent(name : rawstring,  stream : C.cudaStream_t, endEvent : &C.cudaEvent_t)
    var timingInfo : TimingInfo
    timingInfo.eventName = name
    C.cudaEventCreate(&timingInfo.startEvent)
    C.cudaEventCreate(&timingInfo.endEvent)
    C.cudaEventRecord(timingInfo.startEvent, stream)
    self.timingInfo:insert(timingInfo)
    @endEvent = timingInfo.endEvent
end
terra Timer:endEvent(stream : C.cudaStream_t, endEvent : C.cudaEvent_t)
    C.cudaEventRecord(endEvent, stream)
end

terra isprefix(pre : rawstring, str : rawstring) : bool
    if @pre == 0 then return true end
    if @str ~= @pre then return false end
    return isprefix(pre+1,str+1)
end
terra Timer:evaluate()
	if ([_opt_verbosity > 0]) then
		var aggregateTimingInfo = [Array(tuple(float,int))].salloc():init()
		var aggregateTimingNames = [Array(rawstring)].salloc():init()
		for i = 0,self.timingInfo:size() do
			var eventInfo = self.timingInfo(i);
			C.cudaEventSynchronize(eventInfo.endEvent)
	    	C.cudaEventElapsedTime(&eventInfo.duration, eventInfo.startEvent, eventInfo.endEvent);
	    	var index =  aggregateTimingNames:indexof(eventInfo.eventName)
	    	if index < 0 then
	    		aggregateTimingNames:insert(eventInfo.eventName)
	    		aggregateTimingInfo:insert({eventInfo.duration, 1})
	    	else
	    		aggregateTimingInfo(index)._0 = aggregateTimingInfo(index)._0 + eventInfo.duration
	    		aggregateTimingInfo(index)._1 = aggregateTimingInfo(index)._1 + 1
	    	end
	    end

		C.printf(		"--------------------------------------------------------\n")
	    C.printf(		"        Kernel        |   Count  |   Total   | Average \n")
		C.printf(		"----------------------+----------+-----------+----------\n")
	    for i = 0, aggregateTimingNames:size() do
	    	C.printf(	"----------------------+----------+-----------+----------\n")
			C.printf(" %-20s |   %4d   | %8.3fms| %7.4fms\n", aggregateTimingNames(i), aggregateTimingInfo(i)._1, aggregateTimingInfo(i)._0, aggregateTimingInfo(i)._0/aggregateTimingInfo(i)._1)
	    end
        C.printf(		"--------------------------------------------------------\n")
        C.printf("TIMING ")
        for i = 0, aggregateTimingNames:size() do
	        var n = aggregateTimingNames(i)
            if isprefix("PCGInit1",n) or isprefix("PCGStep1",n) or isprefix("overall",n) then
                C.printf("%f ",aggregateTimingInfo(i)._0)
            end
        end
        C.printf("\n")
        -- TODO: Refactor timing code
        var linIters = 0
        var nonLinIters = 0
        for i = 0, aggregateTimingNames:size() do
            var n = aggregateTimingNames(i)
            if isprefix("PCGInit1",n) then
                linIters = aggregateTimingInfo(i)._1
            end
            if isprefix("PCGStep1",n) then
                nonLinIters = aggregateTimingInfo(i)._1
            end
        end
        var linAggregate : float = 0.0f
        var nonLinAggregate : float = 0.0f
        for i = 0, aggregateTimingNames:size() do
            var n = aggregateTimingInfo(i)._1
            if n == linIters then
                linAggregate = linAggregate + aggregateTimingInfo(i)._0
            end
            if n == nonLinIters then
                nonLinAggregate = nonLinAggregate + aggregateTimingInfo(i)._0
            end
        end
        C.printf("Per-iter times ms (nonlinear,linear): %7.4f\t%7.4f\n", linAggregate, nonLinAggregate)
	end
    
end


local terra laneid()
    var laneid : int;
    laneid = terralib.asm(int,"mov.u32 $0, %laneid;","=r", true)
    return laneid;
end
util.laneid = laneid

__syncthreads = cudalib.nvvm_barrier0


local __shfl_down 

if opt_float == float then

    local terra atomicAdd(sum : &float, value : float)
    	terralib.asm(terralib.types.unit,"red.global.add.f32 [$0],$1;","l,f", true, sum, value)
    end
    util.atomicAdd = atomicAdd

    terra __shfl_down(v : float, delta : uint, width : int)
    	var ret : float;
        var c : int;
    	c = ((warpSize-width) << 8) or 0x1F;
    	ret = terralib.asm(float,"shfl.down.b32 $0, $1, $2, $3;","=f,f,r,r", true, v, delta, c)
    	return ret;
    end
else
    struct ULLDouble {
        union {
            a : uint64;
            b : double;
        }
    }

    struct uint2 {
        x : uint32;
        y : uint32;
    }

    struct uint2Double {
        union {
            u2 : uint2;
            d: double;
        }
    }

    local terra __double_as_ull(v : double)
        var u : ULLDouble
        u.b = v;

        return u.a;
    end

    local terra __ull_as_double(v : uint64)
        var u : ULLDouble
        u.a = v;

        return u.b;
    end

    if pascalOrBetterGPU then
        local terra atomicAdd(sum : &double, value : double)
            var address_as_i : uint64 = [uint64] (sum);
            terralib.asm(terralib.types.unit,"red.global.add.f64 [$0],$1;","l,d", true, address_as_i, value)
        end
        util.atomicAdd = atomicAdd
    else
        local terra atomicAdd(sum : &double, value : double)
            var address_as_i : &uint64 = [&uint64] (sum);
            var old : uint64 = address_as_i[0];
            var assumed : uint64;

            repeat
                assumed = old;
                old = terralib.asm(uint64,"atom.global.cas.b64 $0,[$1],$2,$3;", 
                    "=l,l,l,l", true, address_as_i, assumed, 
                    __double_as_ull( value + __ull_as_double(assumed) )
                    );
            until assumed == old;

            return __ull_as_double(old);
        end
        util.atomicAdd = atomicAdd
    end

    terra __shfl_down(v : double, delta : uint, width : int)
        var ret : uint2Double;
        var init : uint2Double;
        init.d = v
        var c : int;
        c = ((warpSize-width) << 8) or 0x1F;
        ret.u2.x = terralib.asm(uint32,"shfl.down.b32 $0, $1, $2, $3;","=f,f,r,r", true, init.u2.x, delta, c)
        ret.u2.y = terralib.asm(uint32,"shfl.down.b32 $0, $1, $2, $3;","=f,f,r,r", true, init.u2.y, delta, c)
        return ret.d;
    end
end

-- Using the "Kepler Shuffle", see http://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
local terra warpReduce(val : opt_float) 

  var offset = warpSize >> 1
  while offset > 0 do 
    val = val + __shfl_down(val, offset, warpSize);
    offset =  offset >> 1
  end
-- Is unrolling worth it?
  return val;

end
util.warpReduce = warpReduce

-- Straightforward implementation of: http://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
-- sdata must be a block of 128 bytes of shared memory we are free to trash
local terra blockReduce(val : opt_float, sdata : &opt_float, threadIdx : int, threadsPerBlock : uint)
	var lane = laneid()
  	var wid = threadIdx / 32 -- TODO: check if this is right for 2D domains

  	val = warpReduce(val); -- Each warp performs partial reduction

  	if (lane==0) then
  		sdata[wid]=val; -- Write reduced value to shared memory
  	end

  	__syncthreads();   -- Wait for all partial reductions

  	--read from shared memory only if that warp existed
  	val = 0
  	if (threadIdx < threadsPerBlock / 32) then
  		val = sdata[lane]
  	end

  	if (wid==0) then
		val = warpReduce(val); --Final reduce within first warp
  	end
end
util.blockReduce = blockReduce


util.max = terra(x : double, y : double)
	return terralib.select(x > y, x, y)
end

local function noHeader(pd)
	return quote end
end

local function noFooter(pd)
	return quote end
end

util.initParameters = function(self, ProblemSpec, params, isInit)
    local stmts = terralib.newlist()
	for _, entry in ipairs(ProblemSpec.parameters) do
		if entry.kind == "ImageParam" then
		    if entry.idx ~= "alloc" then
                local function_name = isInit and "initFromGPUptr" or "setGPUptr"
                local loc = entry.isunknown and (`self.X.[entry.name]) or `self.[entry.name]
                stmts:insert quote
                    loc:[function_name]([&uint8](params[entry.idx]))
                end
            end
		else
            local rhs
            if entry.kind == "GraphParam" then
                local graphinits = terralib.newlist { `@[&int](params[entry.idx]) }
                for i,e in ipairs(entry.type.metamethods.elements) do
                    graphinits:insert( `[&e.type](params[e.idx]) )
                end
                rhs = `entry.type { graphinits }
            elseif entry.kind == "ScalarParam" and entry.idx >= 0 then
                rhs = `@[&entry.type](params[entry.idx])
            end
            if rhs then
                stmts:insert quote self.[entry.name] = rhs end
            end
        end
	end
	return stmts
end

util.initPrecomputedImages = function(self, ProblemSpec)
    local stmts = terralib.newlist()
	for _, entry in ipairs(ProblemSpec.parameters) do
		if entry.kind == "ImageParam" and entry.idx == "alloc" then
            stmts:insert quote
    		    self.[entry.name]:initGPU()
    		end
    	end
    end
    return stmts
end

util.freePrecomputedImages = function(self, ProblemSpec)
    local stmts = terralib.newlist()
    for _, entry in ipairs(ProblemSpec.parameters) do
        if entry.kind == "ImageParam" and entry.idx == "alloc" then
            stmts:insert quote
                self.[entry.name]:freeData()
            end
        end
    end
    return stmts
end




util.getValidUnknown = macro(function(pd,pw,ph)
	return quote
		@pw,@ph = blockDim.x * blockIdx.x + threadIdx.x, blockDim.y * blockIdx.y + threadIdx.y
	in
		 @pw < pd.parameters.X:W() and @ph < pd.parameters.X:H() 
	end
end)
util.getValidGraphElement = macro(function(pd,graphname,idx)
	graphname = graphname:asvalue()
	return quote
		@idx = blockDim.x * blockIdx.x + threadIdx.x
	in
		 @idx < pd.parameters.[graphname].N 
	end
end)

local positionForValidLane = util.positionForValidLane

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


local checkedLaunch = macro(function(kernelName, apicall)
    local apicallstr = tostring(apicall)
    local filename = debug.getinfo(1,'S').source
    return quote
        var name = [kernelName]
        var r = apicall
        if r ~= 0 then  
            C.printf("Kernel %s, Cuda reported error %d: %s\n", name, r, C.cudaGetErrorString(r))
            C.exit(r)
        end
    in
        r
    end end)

-- Assumes x is (nonnegative) power of 2
local function iLog2(x)
    local result = 0
    while x > 1 do
        result = result + 1
        x = x / 2
    end
    return result
end

-- Get the block dimensions that best approximate a square/cube 
-- in 2 or 3 dimensions while only using power of 2 side lengths.
local function getBlockDims(blockSize)
    local LOG_BLOCK_SIZE = iLog2(blockSize)
    local dim2x = math.ceil(LOG_BLOCK_SIZE/2)
    local dim2y = LOG_BLOCK_SIZE - dim2x

    local dim3x = math.ceil(LOG_BLOCK_SIZE/3)
    local dim3y = math.ceil((LOG_BLOCK_SIZE - dim3x)/2)
    local dim3z = LOG_BLOCK_SIZE - dim3x - dim3y

    return { {blockSize,1,1}, {math.pow(2,dim2x),math.pow(2,dim2y),1}, {math.pow(2,dim3x),math.pow(2,dim3y),math.pow(2,dim3z)} }
end

local BLOCK_SIZE = _opt_threads_per_block
assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE should be a multiple of the warp size (32), but is "..tostring(BLOCK_SIZE))

local BLOCK_DIMS = getBlockDims(BLOCK_SIZE)


local function makeGPULauncher(PlanData,kernelName,ft,compiledKernel)
    kernelName = kernelName.."_"..tostring(ft)
    local kernelparams = compiledKernel:gettype().parameters
    local params = terralib.newlist {}
    for i = 3,#kernelparams do --skip GPU launcher and PlanData
        params:insert(symbol(kernelparams[i]))
    end
    local function createLaunchParameters(pd)
        if ft.kind == "CenteredFunction" then
            local ispace = ft.ispace
            local exps = terralib.newlist()
            for i = 1,3 do
               local dim = #ispace.dims >= i and ispace.dims[i].size or 1
                local bs = BLOCK_DIMS[#ispace.dims][i]
                exps:insert(dim)
                exps:insert(bs)
            end
            return exps
        else
            return {`pd.parameters.[ft.graphname].N,BLOCK_DIMS[1][1],1,1,1,1}
        end
    end
    local terra GPULauncher(pd : &PlanData, [params])
        var xdim,xblock,ydim,yblock,zdim,zblock = [ createLaunchParameters(pd) ]
            
        var launch = terralib.CUDAParams { (xdim - 1) / xblock + 1, (ydim - 1) / yblock + 1, (zdim - 1) / zblock + 1, 
                                            xblock, yblock, zblock, 
                                            0, nil }
        var stream : C.cudaStream_t = nil
        var endEvent : C.cudaEvent_t 
        if ([_opt_collect_kernel_timing]) then
            pd.timer:startEvent(kernelName,nil,&endEvent)
        end

        checkedLaunch(kernelName, compiledKernel(&launch, @pd, params))
        
        if ([_opt_collect_kernel_timing]) then
            pd.timer:endEvent(nil,endEvent)
        end

        cd(C.cudaGetLastError())
    end
    return GPULauncher
end

function util.makeGPUFunctions(problemSpec, PlanData, delegate, names)
    -- step 1: compile the actual cuda kernels
    local kernelFunctions = {}
    local key = tostring(os.time())
    local function getkname(name,ft)
        return string.format("%s_%s_%s",name,tostring(ft),key)
    end
    
    for _,problemfunction in ipairs(problemSpec.functions) do
        if problemfunction.typ.kind == "CenteredFunction" then
           local ispace = problemfunction.typ.ispace
           local dimcount = #ispace.dims
	       assert(dimcount <= 3, "cannot launch over images with more than 3 dims")
           local ks = delegate.CenterFunctions(ispace,problemfunction.functionmap)
           for name,func in pairs(ks) do
                kernelFunctions[getkname(name,problemfunction.typ)] = { kernel = func , annotations = { {"maxntidx", BLOCK_DIMS[dimcount][1]}, {"maxntidy", BLOCK_DIMS[dimcount][2]}, {"maxntidz", BLOCK_DIMS[dimcount][3]}, {"minctasm",1} } }
           end
        else
            local graphname = problemfunction.typ.graphname
            local ks = delegate.GraphFunctions(graphname,problemfunction.functionmap)
            for name,func in pairs(ks) do            
                kernelFunctions[getkname(name,problemfunction.typ)] = { kernel = func , annotations = { {"maxntidx", BLOCK_DIMS[1][1]}, {"minctasm",1} } }
            end
        end
    end
    if verbosePTX then
        print("Compiling kernels!")
    end
    local kernels = terralib.cudacompile(kernelFunctions, verbosePTX)
    
    -- step 2: generate wrapper functions around each named thing
    local grouplaunchers = {}
    for _,name in ipairs(names) do
        local args
        local launches = terralib.newlist()
        for _,problemfunction in ipairs(problemSpec.functions) do
            local kname = getkname(name,problemfunction.typ)
            local kernel = kernels[kname]
            if kernel then -- some domains do not have an associated kernel, (see _Finish kernels in GN which are only defined for 
                local launcher = makeGPULauncher(PlanData, name, problemfunction.typ, kernel)
                if not args then
                    args = launcher:gettype().parameters:map(symbol)
                end
                launches:insert(`launcher(args))
            else
                --print("not found: "..name.." for "..tostring(problemfunction.typ))
            end
        end
        local fn
        if not args then
            fn = macro(function() return `{} end) -- dummy function for blank groups occur for things like precompute and _Graph when they are not present
        else
            fn = terra([args]) launches end
            fn:setname(name)
            fn:gettype()
        end
        grouplaunchers[name] = fn 
    end
    return grouplaunchers
end


function util.reportGPUMemoryUse()
    local terra reportMem()
        var free_byte : C.size_t
        var total_byte : C.size_t

        cd(C.cudaMemGetInfo( &free_byte, &total_byte ))


        var free_db = [double](free_byte)

        var total_db = [double](total_byte)

        var used_db = total_db - free_db ;

        C.printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",

            used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
    end
    reportMem()
end

return util
