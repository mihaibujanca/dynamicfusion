echo Remaking Opt.dll
set TERRAHOME=%1
set TERRA=%TERRAHOME%\bin\terra
set OPT_DIR=%2
set OPT_FULL_BUILD=%3

%TERRA% %OPT_DIR%\src\createWrapper.t Opt %OPT_DIR%\src o %OPT_DIR%\release\include\Opt.h .\Opt.dll %OPT_FULL_BUILD%
MOVE "./Opt.dll" "%OPT_DIR%\release\bin\"
MOVE "./Opt.exp" "%OPT_DIR%\release\bin\"
MOVE "./Opt.lib" "%OPT_DIR%\release\lib\"