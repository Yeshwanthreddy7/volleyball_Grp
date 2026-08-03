@echo off
setlocal
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
if not exist %PY% set PY=python

if not exist "fyp\volleyball_best.pt" (
    echo Missing fyp\volleyball_best.pt - it is still needed, but ONLY for the ball.
    pause
    exit /b 1
)

echo ================================================================
echo  RETRAIN after the detector fix  (technical review section 13)
echo ================================================================
echo.
echo  WHY THIS IS MANDATORY, NOT OPTIONAL
echo    The current mamba_checkpoint_v2.pt was trained on CSVs extracted
echo    with the OLD detector path (fine-tune for players, imgsz 640,
echo    geometric team split). That path is now known to have produced
echo    near-empty and mixed-team formations. The fixed pipeline feeds the
echo    model a different feature distribution - measured mean spacing
echo    ~196 cm now vs ~330 cm in the old CSVs - so serving the old
echo    checkpoint is a train/serve mismatch. It is exactly the class of
echo    bug logged as L6 and section 12.8, and it shows up as every window
echo    being flagged ANOMALY with confidence near chance.
echo.
echo  The extraction below uses the SAME flags the demo uses. Do not change
echo  one without changing the other.
echo.
pause

echo.
echo  STEP 1/4  Verify the ball weights still resolve their class ids
%PY% fyp\train_detector.py --verify-only fyp\volleyball_best.pt
if errorlevel 1 goto :fail

echo.
echo  STEP 2/4  Re-extract features from all clips  [2-4 h CPU at imgsz 1280]
%PY% fyp\prepare_training_data.py "dataset\dataset" --output-dir training_csv_v3 --yolo-model yolo11n.pt --ball-model fyp\volleyball_best.pt --imgsz 1280 --team-split colour --auto-court --court-coords linear --clean-output --tracker botsort
if errorlevel 1 goto :fail

echo.
echo  STEP 3/4  Retrain Mamba on the re-extracted CSVs  [10-30 min CPU]
%PY% fyp\train_mamba.py training_csv_v3 --augment --epochs 80 --checkpoint mamba_checkpoint_v3.pt --history_csv training_history_v3.csv
if errorlevel 1 goto :fail

echo.
echo  STEP 4/4  Honest cross-video evaluation (leave-one-video-out)
echo    Quote BOTH the random-split and the LOVO numbers; the gap between
echo    them is the per-video-signature finding from section 12.6.
%PY% fyp\train_mamba.py training_csv_v3 --augment --epochs 80 --test-video "(1)" --checkpoint mamba_lovo_1.pt
%PY% fyp\train_mamba.py training_csv_v3 --augment --epochs 80 --test-video "(3)" --checkpoint mamba_lovo_3.pt

echo.
echo  ALL DONE. Render the demo with the matching flags:
echo    MAKE_DEMO_VIDEO.bat        (already uses them)
echo  or explicitly:
echo    %PY% fyp\pipeline.py "videoplayback (4).mp4" mamba_checkpoint_v3.pt ^
echo      --yolo-model yolo11n.pt --ball-model fyp\volleyball_best.pt ^
echo      --imgsz 1280 --team-split colour --tracker botsort ^
echo      --auto-court --court-coords linear
pause
exit /b 0

:fail
echo.
echo  A STEP FAILED - scroll up and read the first error line.
pause
exit /b 1
