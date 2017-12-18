
local ffi = require("ffi")
errorPrint = print -- make global
if false and ffi.os == "Windows" then
    debugOutput = terralib.externfunction("OutputDebugStringA", rawstring->{})
    errorPrint = function (rawstring) 
        local correctedString = rawstring:gsub(":([0-9]+):", "(%1):")
        debugOutput(correctedString)
    end
end

local success,p = xpcall(function() 
		return require("o")
    end,function(err) errorPrint(debug.traceback(err,2)) end)
if not success then 
	error()
else
	return p
end
