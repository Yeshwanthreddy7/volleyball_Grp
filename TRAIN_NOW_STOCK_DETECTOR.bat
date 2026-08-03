@echo off
setlocal
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
if not exist %PY% set PY=python

echo ================================================================
echo  TRAIN NOW  (stock COCO detector yolo11n.pt)  -  FULL FINAL OUTPUT
echo  1) extract 508 clips  2) train Mamba  3) report CSV
echo  4) annotated demo video via the full pipeline
echo  Total: roughly 2-4 hours on CPU. Leave this window open.
echo  NOTE: after Colab gives volleyball_best.pt, TRAIN_AFTER_COLAB.bat
echo  supersedes this run.
echo ================================================================
echo.
echo  STEP 1/4  Extract features from 508 clips  [roughly 1.5-3 h on CPU]
%PY% fyp\prepare_training_data.py "dataset\dataset" --output-dir training_csv_stock --yolo-model yolo11n.pt --clean-output
if errorlevel 1 goto :fail

echo.
echo  STEP 2/4  Train Mamba on the extracted CSVs  [10-30 min on CPU]
%PY% fyp\train_mamba.py training_csv_stock --augment --epochs 80 --checkpoint mamba_checkpoint_v2_stock.pt --history_csv training_history_v2_stock.csv
if errorlevel 1 goto :fail

echo.
echo  STEP 3/4  Inference report over the training CSVs
%PY% fyp\infer_mamba.py mamba_checkpoint_v2_stock.pt training_csv_stock --output_csv preds_v2_stock.csv
if errorlevel 1 goto :fail

echo.
echo  STEP 4/4  Final annotated demo video (first ~1 min of match)  [10-25 min]
%PY% fyp\pipeline.py "videoplayback (4).mp4" mamba_checkpoint_v2_stock.pt --yolo-model yolo11n.pt --tracker botsort --auto-court --cmc --max-frames 1800 --output-video final_annotated_v2_stock.mp4 --output-csv predictions_seq_v2_stock.csv
if errorlevel 1 goto :fail

echo.
echo ================================================================
echo  ALL DONE - FINAL OUTPUTS:
echo    mamba_checkpoint_v2_stock.pt        trained tactical model
echo    preds_v2_stock.csv                  per-clip predictions + entropy
echo    training_history_v2_stock.csv       learning curves
echo    final_annotated_v2_stock.mp4        annotated demo video + HUD
echo    predictions_seq_v2_stock.csv        sequence-level pipeline output
echo ================================================================
pause
exit /b 0

:fail
echo.
echo  A STEP FAILED - scroll up and read the first error line.
pause
exit /b 1
