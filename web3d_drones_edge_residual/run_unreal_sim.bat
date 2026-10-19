@echo off
rem Starts the packaged Unreal game (viewer only) and then flies the trained brain in it.
rem Extra arguments are passed to ue_bridge.py, e.g.  run_unreal_sim.bat --weather gale --hz 30
set GAME=E:\Softwares\UnrealProjects\MyProject2\Packaged\Windows\MyProject2.exe
cd /d "%~dp0"
start "" "%GAME%" -RCWebControlEnable -windowed -ResX=1280 -ResY=720
python ue_bridge.py %*
