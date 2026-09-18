@echo off
REM batch_html.bat — пакетная конвертация всех PDF каталога в HTML (без перевода).
REM
REM Использование:
REM   batch_html.bat [КАТАЛОГ]
REM
REM   КАТАЛОГ — каталог с PDF-файлами (по умолчанию: .\pdfs)
REM
REM Выход: рядом с каждым PDF создаётся файл *_raw.html.
REM Алгоритм — модульный pipeline (main.py --raw-html), портированный из pdf2html.py.

setlocal EnableDelayedExpansion

set "CATALOG=%~1"
if "%CATALOG%"=="" set "CATALOG=.\pdfs"

if not exist "%CATALOG%" (
    echo Ошибка: каталог '%CATALOG%' не существует
    exit /b 1
)

set "SCRIPT_DIR=%~dp0"
set "MAIN_PY=%SCRIPT_DIR%main.py"

if not exist "%MAIN_PY%" (
    echo Ошибка: %MAIN_PY% не найден
    exit /b 1
)

set COUNT=0
for %%F in ("%CATALOG%\*.pdf") do (
    set /a COUNT+=1
    echo [!COUNT!] Конвертация %%~nxF ...
    python "%MAIN_PY%" --raw-html "%%F"
    if errorlevel 1 (
        echo Ошибка при обработке %%~nxF
    ) else (
        echo   -^> готово
    )
)

if %COUNT%==0 (
    echo Ошибка: в каталоге '%CATALOG%' нет PDF-файлов
    exit /b 1
)

echo Все задачи выполнены. Обработано %COUNT% файлов.
endlocal
