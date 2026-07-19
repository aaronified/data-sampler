@echo off
rem Launch the data-sampler TUI (Windows). Double-click or run from cmd.
rem Usage: run-tui.bat [file.csv] [--sheet NAME]

where data-sampler >nul 2>nul
if %errorlevel%==0 (
    data-sampler --tui %*
    goto :done
)

python -c "import data_sampler" >nul 2>nul
if %errorlevel%==0 (
    python -m data_sampler --tui %*
    goto :done
)

echo data-sampler is not installed. Install it with:
echo   pip install https://github.com/aaronified/data-sampler/releases/download/v3.0.1/data_sampler-3.0.1-py3-none-any.whl
pause
exit /b 1

:done
