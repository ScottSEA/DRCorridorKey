@echo off
REM ═══════════════════════════════════════════════════════════════════
REM  Install the CorridorKey Fuse into DaVinci Resolve (Windows)
REM
REM  Copies CorridorKey.fuse to the Fusion Fuses directory so it
REM  appears as a node in the Fusion page.  Restart Resolve after
REM  running this script.
REM ═══════════════════════════════════════════════════════════════════

setlocal

echo.
echo  ╔═══════════════════════════════════════════════╗
echo  ║   CorridorKey — Fuse Installer (Windows)      ║
echo  ╚═══════════════════════════════════════════════╝
echo.

set "FUSE_SRC=%~dp0Fuses\CorridorKey.fuse"
set "FUSE_DIR=%APPDATA%\Blackmagic Design\DaVinci Resolve\Support\Fusion\Fuses"

REM Check source exists
if not exist "%FUSE_SRC%" (
    echo [ERROR] Cannot find CorridorKey.fuse at:
    echo         %FUSE_SRC%
    pause
    exit /b 1
)

REM Create destination directory if needed
if not exist "%FUSE_DIR%" (
    echo [INFO] Creating Fuses directory...
    mkdir "%FUSE_DIR%"
)

REM Copy the Fuse file
echo [INFO] Installing CorridorKey.fuse to:
echo        %FUSE_DIR%
copy /Y "%FUSE_SRC%" "%FUSE_DIR%\CorridorKey.fuse"

if errorlevel 1 (
    echo [ERROR] Copy failed. Check permissions.
    pause
    exit /b 1
)

echo.
echo [OK] CorridorKey Fuse installed successfully.
echo      Restart DaVinci Resolve to see it in the Fusion page.
echo      Look for it under: Add Tool ^> Fuses ^> Keying ^> CorridorKey
echo.
pause
