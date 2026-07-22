@echo off
@REM -DCMAKE_PREFIX_PATH=F:/Qt6/6.11.1/msvc2022_64
cmake . -B build 

if %errorlevel% neq 0 pause

pause