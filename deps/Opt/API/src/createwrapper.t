local libraryname, sourcedirectory, main, headerfile, outputname, embedsource = ...
embedsource = "true" == embedsource or false

local ffi = require("ffi")

terralib.includepath = terralib.terrahome.."/include;."
local C,CN = terralib.includecstring[[ 
    #define _GNU_SOURCE
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    #ifndef _WIN32
    #include <dlfcn.h>
    #include <libgen.h>
    #include <signal.h>
    sig_t SIG_DFL_fn() { return SIG_DFL; }
    #else
    #define NOMINMAX
    #include <windows.h>
    #include <Shlwapi.h>
    #endif
    #include "terra/terra.h"
]]

local LUA_GLOBALSINDEX = -10002

local tabsolutepath,setupsigsegv

if ffi.os == "Windows" then
    terra setupsigsegv(L : &C.lua_State) end
    terra tabsolutepath(rel : rawstring)
        var buf : rawstring = rawstring(C.malloc(C.MAX_PATH))
        C.GetFullPathNameA(rel,C.MAX_PATH,buf,nil)
        return buf
    end  
else

    local sigactionwrapper = ffi.os == "Linux" and "__sigaction_handler" or "__sigaction_u"
    local sigactionstruct = ffi.os == "Linux" and "sa_sigaction" or "__sa_sigaction"
    local terratraceback = global(&opaque -> {})
    
    local terra sigsegv(sig : int, info : &C.siginfo_t, uap : &opaque)
        C.signal(sig,C.SIG_DFL_fn())  --reset signal to default, just in case traceback itself crashes
        terratraceback(uap)
        C.raise(sig)
    end
    terra setupsigsegv(L : &C.lua_State)
        C.lua_getfield(L, LUA_GLOBALSINDEX,"terralib");
        C.lua_getfield(L, -1, "traceback");
        var tb = C.lua_topointer(L,-1);
        if tb == nil then return end
        terratraceback = @[&(&opaque -> {})](tb)
        var sa : CN.sigaction
        sa.sa_flags = [terralib.constant(uint32,C.SA_RESETHAND)] or C.SA_SIGINFO
        C.sigemptyset(&sa.sa_mask)
        sa.[sigactionwrapper].[sigactionstruct] = sigsegv
        C.sigaction(C.SIGSEGV, &sa, nil)
        C.sigaction(C.SIGILL, &sa, nil)
        C.lua_settop(L,-3)
    end
    
    terra tabsolutepath(path : rawstring)
        return C.realpath(path,nil)
    end
end
absolutepath = function(p) return ffi.string(tabsolutepath(p)) end

local Header = terralib.includec(headerfile)
local statename = libraryname.."_State"
local LibraryState = Header[statename]
assert(terralib.types.istype(LibraryState) and LibraryState:isstruct())

local apifunctions = terralib.newlist()
local apimatch = ("%s_(.*)"):format(libraryname)
for k,v in pairs(Header) do
    local name = k:match(apimatch)
    if name and terralib.isfunction(v) and name ~= "NewState" then
        local type = v:gettype()
        apifunctions[name] = {unpack(type.parameters,2)} -> type.returntype
    end
end

local wrappers = {}

struct LibraryState { L : &C.lua_State }

-- Must match Opt.h
struct Opt_InitializationParameters {
    -- If true, all intermediate values and unknowns, are double-precision
    -- On platforms without double-precision float atomics, this 
    -- can be a drastic drag of performance.
    doublePrecision : int

    -- Valid Values: 
    --  0. no verbosity 
    --  1. verbose solve 
    --  2. verbose initialization, autodiff and solve
    --  3. full verbosity (includes ptx dump)
    verbosityLevel : int

    -- If true, a cuda timer is used to collect per-kernel timing information
    -- while the solver is running. This adds a small amount of overhead to every kernel.
    collectPerKernelTimingInfo : int

    -- Default block size for kernels (in threads). 
    -- Must be a positive multiple of 32; if not, will default to 256.
    threadsPerBlock : int
}

for name,type in pairs(apifunctions) do
    LibraryState.entries:insert { name, type }
end

local terra doerror(L : &C.lua_State)
    C.printf("%s\n",C.luaL_checklstring(L,-1,nil))
    C.lua_getfield(L,LUA_GLOBALSINDEX,"os")
    C.lua_getfield(L,-1,"exit")
    C.lua_pushnumber(L,1)
    C.lua_call(L,1,0)
    return nil
end

local sourcepath = absolutepath(sourcedirectory).."/?.t"
local terra NewState(params : Opt_InitializationParameters) : &LibraryState
    var S = [&LibraryState](C.malloc(sizeof(LibraryState)))
    var L = C.luaL_newstate();
    S.L = L
    if L == nil then return doerror(L) end
    C.luaL_openlibs(L)
    var o  = C.terra_Options { verbose = 0, debug = 1, usemcjit = 1 }
    
    if C.terra_initwithoptions(L,&o) ~= 0 then
        doerror(L)
    end
    setupsigsegv(L)

    -- Set global variables from Opt_InitializationParameters
    C.lua_pushboolean(L,params.doublePrecision);
    C.lua_setfield(L,LUA_GLOBALSINDEX,"_opt_double_precision")

    var verbosityLevel : C.lua_Number = params.verbosityLevel
    C.lua_pushnumber(L,verbosityLevel);
    C.lua_setfield(L,LUA_GLOBALSINDEX,"_opt_verbosity")

    C.lua_pushboolean(L,params.collectPerKernelTimingInfo);
    C.lua_setfield(L,LUA_GLOBALSINDEX,"_opt_collect_kernel_timing")

    var threadsPerBlock = params.threadsPerBlock
    if threadsPerBlock <= 0 or threadsPerBlock % 32 ~= 0 then
        threadsPerBlock = 256
    end

    C.lua_pushnumber(L,threadsPerBlock);
    C.lua_setfield(L,LUA_GLOBALSINDEX,"_opt_threads_per_block")

    C.lua_getfield(L,LUA_GLOBALSINDEX,"package")

    -- C.lua_setfield(L,LUA_GLOBALSINDEX,)
    escape 
        if embedsource then
            emit quote C.lua_getfield(L,-1,"preload") end
			
			local command = ""
			if ffi.os == "Windows" then
				command = "cmd /c dir /b "
			else
				command = "ls "
			end 

			print(command..sourcedirectory)
			
            for line in io.popen(command..sourcedirectory):lines() do
                local name = line:match("(.*)%.t")
                if name then
                    local content = io.open(sourcedirectory.."/"..line,"r"):read("*all")
                    emit quote
                        if 0 ~= C.terra_loadbuffer(L,content,[#content],["@"..line]) then doerror(L) end
                        C.lua_setfield(L,-2,name)
                    end
                end
            end
        else
            emit quote 
                C.lua_getfield(L,-1,"terrapath")
                C.lua_pushstring(L,";")
                C.lua_pushstring(L,sourcepath)
                C.lua_concat(L,3)
                C.lua_setfield(L,-2,"terrapath")
            end
        end
    end
    
    C.lua_getfield(L,LUA_GLOBALSINDEX,"require")
    C.lua_pushstring(L,main)
    if C.lua_pcall(L,1,1,0) ~= 0 then return doerror(L) end
    
    escape
        for k,type in pairs(apifunctions) do
            emit quote
                C.lua_getfield(L,-1,k)
                C.lua_getfield(L,-1,"getpointer")
                C.lua_insert(L,-2)
                C.lua_call(L,1,1) 
                S.[k] = @[&type](C.lua_topointer(L,-1))
                C.lua_settop(L, -2)
            end
        end
    end
    return S
end
wrappers[libraryname.."_NewState"] =  NewState

for k,type in pairs(apifunctions) do
    local syms = type.type.parameters:map(symbol)
    local terra wfn(state : &LibraryState, [syms]) : type.type.returntype
        return state.[k]([syms])
    end
    wrappers[libraryname.."_"..k] = wfn 
end 

local flags = {}
if ffi.os == "Windows" then
    flags = terralib.newlist { string.format("/IMPLIB:%s.lib",libraryname),terralib.terrahome.."\\lib\\terra.lib",terralib.terrahome.."\\lib\\lua51.lib","Shlwapi.lib" }
    
    for k,_ in pairs(wrappers) do
        flags:insert("/EXPORT:"..k)
    end
end
terralib.saveobj(outputname,wrappers,flags)
