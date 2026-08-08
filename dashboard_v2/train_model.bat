@echo off
echo ====================================================================
echo             AGRI-SHIELD DEEP LEARNING MODEL TRAINING PIPELINE
echo ====================================================================
echo Ingesting data, building PyTorch tensors, and training on local RTX GPU...
echo.

:: Run training script using UV package manager
uv run --with pandas --with numpy --with scikit-learn --with torch train_deep_learning.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Training failed. Retrying with PyTorch CUDA index source...
    echo.
    uv run --index-url https://download.pytorch.org/whl/cu121 --with pandas --with numpy --with scikit-learn --with torch train_deep_learning.py
)

echo.
echo ====================================================================
echo Training complete! Metrics and model weights have been exported.
echo Refresh your web dashboard to see the deep learning comparison charts.
echo ====================================================================
echo.
pause
