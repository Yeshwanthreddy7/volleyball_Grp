@echo off
setlocal
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
if not exist %PY% set PY=python

if not exist "fyp\volleyball_best.pt" (
    echo Put the Colab-trained weights at fyp\volleyball_best.pt first.
    echo Run colab_train_volleyball_yolov11.ipynb to produce it.
    pause
    exit /b 1
)

echo ================================================================
echo  FULL CHAIN with your custom YOLOv11 volleyball detector
echo ================================================================
echo.
echo  STEP 1/4  Verify weights + class-id resolution
%PY% fyp\train_detector.py --verify-only fyp\volleyball_best.pt
if errorlevel 1 goto :fail

echo.
echo  STEP 2/4  Re-extract features from all 508 clips with best.pt  [1.5-3 h CPU]
%PY% fyp\prepare_training_data.py "dataset\dataset" --output-dir training_csv --yolo-model fyp\volleyball_best.pt --clean-output --tracker botsort
if errorlevel 1 goto :fail

echo.
echo  STEP 3/4  Retrain Mamba v2 on the re-extracted CSVs  [10-30 min CPU]
%PY% fyp\train_mamba.py training_csv --augment --epochs 80 --checkpoint mamba_checkpoint_v2.pt --history_csv training_history_v2.csv
if errorlevel 1 goto :fail

echo.
echo  STEP 4/4  Inference report
%PY% fyp\infer_mamba.py mamba_checkpoint_v2.pt training_csv --output_csv preds_v2.csv
if errorlevel 1 goto :fail

echo.
echo  ALL DONE. Deploy with:
echo    .venv\Scripts\python fyp\pipeline.py "videoplayback (4).mp4" mamba_checkpoint_v2.pt --yolo-model fyp\volleyball_best.pt --tracker botsort --auto-court --cmc
pause
exit /b 0

:fail
echo.
echo  A STEP FAILED - scroll up and read the first error line.
pause
exit /b 1
