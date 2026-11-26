@echo off
echo ====================================================================
echo             AGRI-SHIELD DEEP LEARNING MODEL TRAINING PIPELINE
echo ====================================================================
echo Ingesting data, building PyTorch tensors, and training on local RTX GPU...
echo.

:: Install exact pinned dependencies
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Pinned dependency installation failed.
    pause
    exit /b 1
)

python train_deep_learning.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Training failed. Retrying with PyTorch CUDA index source...
    echo.
    python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.12.1
    python train_deep_learning.py
)

echo.
echo ====================================================================
echo Training complete! Metrics and model weights have been exported.
echo Refresh your web dashboard to see the deep learning comparison charts.
echo ====================================================================
echo.
pause
