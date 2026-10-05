@echo off
REM Earthquake Simulator -- Windows launcher
REM Copyright (C) 2026 Earthquake Simulator contributors
REM
REM This program is free software: you can redistribute it and/or modify
REM it under the terms of the GNU General Public License as published by
REM the Free Software Foundation, either version 3 of the License, or
REM (at your option) any later version.
REM
REM This program is distributed in the hope that it will be useful,
REM but WITHOUT ANY WARRANTY; without even the implied warranty of
REM MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
REM GNU General Public License for more details.
REM
REM You should have received a copy of the GNU General Public License
REM along with this program.  If not, see <https://www.gnu.org/licenses/>.
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (set PY=py -3) else (set PY=python)

%PY% -c "import panda3d" >nul 2>nul
if not %errorlevel%==0 (
    echo Installing dependencies, one moment...
    %PY% -m pip install -r requirements.txt
)

%PY% earthquake.py %*
if not %errorlevel%==0 pause
endlocal
