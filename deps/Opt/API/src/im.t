--[[
This file is for reading and writing our ad-hoc floating-point image format.
The only commonly-used image format that has full support for uncompressed 32-bit
floating point images is OpenEXR, and it is not widely supported (and more relevantly for us,
   would require importing a large library that may or may not go well)

So our .imagedump format is as follows

width : int32
height : int32
channelCount : int32
datatype : int32 (0 means 32-bit floating point, all other values are reserved in case we need to support other types)
pixelData (row-major, no padding between rows, takes up width*height*channelCount*sizeof(type indicated by datatype)

]]

local util = require("util")
local C = util.C
local im = {}


-- Up to the user to free the pointer in imPtr
terra im.initImage(imPtr : &&float, width : int, height : int, numChannels : int)
   var numBytes : int = sizeof(float)*width*height*numChannels
   @imPtr = [&float](C.malloc(numBytes))
end

terra im.imageWrite(imPtr : &float, width : int, height : int, channelCount : int, filename : rawstring)
   var datatype : int = 0 -- floating point
   var fileHandle = C.fopen(filename, 'wb') -- b for binary
   C.fwrite(&width, sizeof(int), 1, fileHandle)
   C.fwrite(&height, sizeof(int), 1, fileHandle)
   C.fwrite(&channelCount, sizeof(int), 1, fileHandle)
   C.fwrite(&datatype, sizeof(int), 1, fileHandle)
   
   C.fwrite(imPtr, sizeof(float), [uint32](width*height*channelCount), fileHandle)
   C.fclose(fileHandle)
end


-- mallocs the ptr to the image data, and stores the ptr in the imPtr handle
terra im.imageRead(imPtr : &&float, width : &int, height : &int, numChannels : &int, filename : rawstring)
   var fileHandle = C.fopen(filename, 'rb')
   var datatype : int = 0
   C.fread(width, sizeof(int), 1, fileHandle)
   C.fread(height, sizeof(int), 1, fileHandle)
   C.fread(numChannels, sizeof(int), 1, fileHandle)
   C.fread(&datatype, sizeof(int), 1, fileHandle)
   --TODO: assert
   @imPtr = [&float](C.malloc(sizeof(float)*@width*@height*@numChannels))
   C.fread(@imPtr, sizeof(float), [uint32](@width*@height*@numChannels), fileHandle)
   C.fclose(fileHandle)
end

return im
