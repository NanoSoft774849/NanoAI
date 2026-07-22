@echo off

REM Disable logo, use all cores, aggressive optimizations
set CL=/nologo /MP /O2 /Ob2 /Oi /Ot /GT /GL /fp:fast

REM Use Ninja if available (MUCH faster than MSBuild)
if exist "../build6/build.ninja" (
    cmake --build build -j %NUMBER_OF_PROCESSORS%
) else (
    REM Use MSBuild with all cores
    cmake --build build --config Release --verbose -j %NUMBER_OF_PROCESSORS% -- /p:CL_MPCount=%NUMBER_OF_PROCESSORS% /maxcpucount
)

pause