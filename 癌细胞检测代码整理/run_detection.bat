@echo off
REM 癌细胞检测批处理脚本
echo ========================================
echo    CHIEF 癌细胞检测
echo ========================================
echo.

echo 1. 结肠癌检测
echo 2. 乳腺癌检测
echo 3. 检查环境
echo 4. 退出
echo.

set /p choice="请选择操作 (1-4): "

if "%choice%"=="1" (
    echo 开始结肠癌检测...
    cd Cancer_Cell_Detection
    python classification_eval.py --config_path configs/colon.yaml --dataset_name Dataset_PT
    cd ..
) else if "%choice%"=="2" (
    echo 开始乳腺癌检测...
    cd Cancer_Cell_Detection
    python classification_eval.py --config_path configs/breast.yaml --dataset_name DROID_breast
    cd ..
) else if "%choice%"=="3" (
    echo 检查环境...
    python quick_start.py --check-env
) else if "%choice%"=="4" (
    echo 退出
    exit /b
) else (
    echo 无效选择
)

pause
