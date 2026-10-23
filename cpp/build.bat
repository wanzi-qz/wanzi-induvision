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
set NVCC=W:\Anaconda\envs\induvision\Library\bin\nvcc.exe
set CV_INC=C:\Users\11646\cppenv\Library\include
set CV_LIB=C:\Users\11646\cppenv\Library\lib
set MSVC=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207
set SDK=C:\Program Files (x86)\Windows Kits\10
set SDKVER=10.0.26100.0

set PATH=%MSVC%\bin\Hostx64\x64;%SDK%\bin\%SDKVER%\x64;%SDK%\bin\%SDKVER%\ucrt\x64;W:\Anaconda\Scripts;%PATH%
set INCLUDE=%MSVC%\include;%SDK%\Include\%SDKVER%\ucrt;%SDK%\Include\%SDKVER%\um;%SDK%\Include\%SDKVER%\shared;%SDK%\Include\%SDKVER%\winrt;%SDK%\Include\%SDKVER%\cppwinrt
set LIB=%MSVC%\lib\x64;%SDK%\Lib\%SDKVER%\um\x64;%SDK%\Lib\%SDKVER%\ucrt\x64

if "%1"=="clean" if exist "%REPO%\cpp\build" rmdir /S /Q "%REPO%\cpp\build"

where cl.exe >nul 2>&1 || (echo cl not on PATH & exit /b 1)
where rc.exe  >nul 2>&1 || (echo rc  not on PATH & exit /b 1)

if not exist "%REPO%\cpp\build" mkdir "%REPO%\cpp\build"

rem -- preproc.cu -> preproc.obj (nvcc; host compiler = cl on PATH).
rem -- -arch=native: build for the local GPU; for CI/other machines pass
rem    e.g. -arch=sm_86 explicitly. We do NOT enable CMake CUDA language:
rem    find_package(CUDAToolkit) guesses wrong on conda layouts, one explicit
rem    nvcc call for one translation unit is more honest (see CMakeLists.txt).
"%NVCC%" -c "%REPO%\cpp\src\preproc.cu" -o "%REPO%\cpp\build\preproc.obj" -O3 -arch=native -Xcompiler /utf-8 -Xcompiler /MD
if errorlevel 1 (echo BUILD_FAILED & exit /b 1)

"%CMAKE%" -S "%REPO%\cpp" -B "%REPO%\cpp\build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DTENSORRT_INCLUDE_DIR="%TRT_INC%" -DTENSORRT_LIBRARY_DIR="%TRT_LIB%" -DCUDA_INCLUDE_DIR="%CUDA_INC%" -DCUDA_LIB_DIR="%CUDA_LIB%" -DOPENCV_INCLUDE_DIR="%CV_INC%" -DOPENCV_LIB_DIR="%CV_LIB%"
if errorlevel 1 (echo BUILD_FAILED & exit /b 1)
"%CMAKE%" --build "%REPO%\cpp\build"
if errorlevel 1 (echo BUILD_FAILED & exit /b 1)
echo BUILD_OK
