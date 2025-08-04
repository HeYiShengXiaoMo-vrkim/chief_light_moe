@echo off
REM 轻量化癌细胞检测脚本 - 适用于6GB显存/8GB内存
echo ============================================
echo    CHIEF 轻量化癌细胞检测
echo    适配: 6GB显存 + 8GB内存
echo ============================================
echo.

echo 💡 轻量化优化特性:
echo   ✓ 模型参数减少80%% (768→64维度)
echo   ✓ 批量大小降低75%% (32→8)
echo   ✓ 内存使用优化 (混合精度+动态清理)
echo   ✓ patch数量限制 (最多512个)
echo.

echo 请选择检测类型:
echo 1. 结肠癌检测 (轻量化)
echo 2. 乳腺癌检测 (轻量化)
echo 3. 检查系统环境
echo 4. 显存/内存监控
echo 5. 退出
echo.

set /p choice="请输入选择 (1-5): "

if "%choice%"=="1" (
    echo.
    echo 🚀 开始结肠癌轻量化检测...
    echo 配置: colon_light.yaml
    echo 预计显存使用: ~4GB
    echo 预计内存使用: ~6GB
    echo.
    cd Cancer_Cell_Detection
    python classification_eval_light.py --config_path configs/colon_light.yaml --dataset_name Dataset_PT --use_mixed_precision
    cd ..
    echo.
    echo ✅ 检测完成! 结果保存在 results/colon_Dataset_PT_light/evaluation/
    
) else if "%choice%"=="2" (
    echo.
    echo 🚀 开始乳腺癌轻量化检测...
    echo 配置: breast_light.yaml  
    echo 预计显存使用: ~4GB
    echo 预计内存使用: ~6GB
    echo.
    cd Cancer_Cell_Detection
    python classification_eval_light.py --config_path configs/breast_light.yaml --dataset_name DROID_breast --use_mixed_precision
    cd ..
    echo.
    echo ✅ 检测完成! 结果保存在 results/DROID_breast_light/evaluation/
    
) else if "%choice%"=="3" (
    echo.
    echo 🔍 检查系统环境...
    python quick_start.py --check-env
    echo.
    echo 📊 轻量化模型需求:
    echo   - Python 3.8+
    echo   - PyTorch (CUDA 11.0+)
    echo   - 显存: 4-6GB
    echo   - 内存: 6-8GB
    echo   - 存储: 5GB+ 可用空间
    
) else if "%choice%"=="4" (
    echo.
    echo 📊 系统资源监控...
    echo.
    echo GPU信息:
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits 2>nul || echo "无法获取GPU信息"
    echo.
    echo 内存信息:
    wmic OS get TotalVisibleMemorySize,FreePhysicalMemory /format:list | findstr "="
    echo.
    echo 💡 建议:
    echo   - 关闭其他占用GPU的程序
    echo   - 确保至少有4GB空闲显存
    echo   - 确保至少有6GB空闲内存
    
) else if "%choice%"=="5" (
    echo 👋 再见!
    exit /b
    
) else (
    echo ❌ 无效选择，请输入1-5
)

echo.
echo 🔧 遇到问题? 尝试以下解决方案:
echo   1. 显存不足: 减小max_bag_size (512→256)
echo   2. 内存不足: 关闭其他程序
echo   3. 速度慢: 使用SSD存储数据
echo   4. 精度下降: 调整dropout (0.3→0.2)
echo.
pause
