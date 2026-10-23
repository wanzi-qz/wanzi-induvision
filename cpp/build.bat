@echo off
rem W4-2 C++ build script. Does NOT call vcvars64.bat: it internally runs
rem reg.exe which is on the WorkBuddy command blacklist. We set the exact
rem MSVC/SDK environment ourselves (paths are fixed for this machine).
rem usage: cpp\build.bat [clean]
setlocal
set REPO=%~dp0..
set CMAKE=W:\Anaconda\Scripts\cmake.exe
set TRT_INC=%REPO%\..\_vendor\TensorRT-v10.16\include
set TRT_LIB=%REPO%\cpp\trt_lib
set CUDA_INC=W:\Anaconda\envs\induvision\Library\include
set CUDA_LIB=W:\Anaconda\envs\induvision\Library\lib\x64
set MSVC=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207
set SDK=C:\Program Files (x86)\Windows Kits\10
set SDKVER=10.0.26100.0

set PATH=%MSVC%\bin\Hostx64\x64;%SDK%\bin\%SDKVER%\x64;%SDK%\bin\%SDKVER%\ucrt\x64;W:\Anaconda\Scripts;%PATH%
set INCLUDE=%MSVC%\include;%SDK%\Include\%SDKVER%\ucrt;%SDK%\Include\%SDKVER%\um;%SDK%\Include\%SDKVER%\shared;%SDK%\Include\%SDKVER%\winrt;%SDK%\Include\%SDKVER%\cppwinrt
set LIB=%MSVC%\lib\x64;%SDK%\Lib\%SDKVER%\um\x64;%SDK%\Lib\%SDKVER%\ucrt\x64

if "%1"=="clean" if exist "%REPO%\cpp\build" rmdir /S /Q "%REPO%\cpp\build"

where cl.exe >nul 2>&1 || (echo cl not on PATH & exit /b 1)
where rc.exe  >nul 2>&1 || (echo rc  not on PATH & exit /b 1)

"%CMAKE%" -S "%REPO%\cpp" -B "%REPO%\cpp\build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DTENSORRT_INCLUDE_DIR="%TRT_INC%" -DTENSORRT_LIBRARY_DIR="%TRT_LIB%" -DCUDA_INCLUDE_DIR="%CUDA_INC%" -DCUDA_LIB_DIR="%CUDA_LIB%"
if errorlevel 1 exit /b 1
"%CMAKE%" --build "%REPO%\cpp\build"
if errorlevel 1 exit /b 1
echo BUILD_OK
