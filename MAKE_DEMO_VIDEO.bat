@echo off
setlocal
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
if not exist %PY% set PY=python

if not exist "fyp\volleyball_best.pt" (
    echo Missing fyp\volleyball_best.pt - copy it from FINAL_OUTPUTS.zip first.
    pause
    exit /b 1
)
if not exist "mamba_checkpoint_v2.pt" (
    echo Missing mamba_checkpoint_v2.pt - copy it from FINAL_OUTPUTS.zip first.
    pause
    exit /b 1
)

echo Open "videoplayback (4).mp4" in a video player and find a minute
echo where an actual rally is being played (players visible on court).
echo.
set /p START_MIN="Start at which minute of the video? (e.g. 6): "
if not defined START_MIN set START_MIN=6
set /a START_FRAME=%START_MIN%*1800

echo.
echo Rendering 60 seconds starting at minute %START_MIN%  [20-40 min on CPU]
echo   players : yolo11n.pt      (stock COCO - domain-robust on unseen courts)
echo   ball    : volleyball_best.pt  (the fine-tune, which only the ball needs)
echo   imgsz   : 1280            (a 15px ball does not survive the 640 default)
echo   teams   : jersey colour + per-track vote
echo Court mask ON (drops coaches/refs/bench) + LINEAR coords (training parity).
%PY% fyp\pipeline.py "videoplayback (4).mp4" mamba_checkpoint_v2.pt --yolo-model yolo11n.pt --ball-model fyp\volleyball_best.pt --imgsz 1280 --team-split colour --tracker botsort --auto-court --court-coords linear --start-frame %START_FRAME% --max-frames 1800 --output-video final_annotated_v2.mp4 --output-csv predictions_seq_v2.csv
if errorlevel 2 (
    echo.
    echo PREFLIGHT ABORTED THE RUN - read the FATAL line above; it names the fix.
    echo Nothing was rendered, on purpose: a report built on that input would
    echo have been meaningless.
    pause
    exit /b 2
)
if errorlevel 1 (
    echo FAILED - read the error above.
    pause
    exit /b 1
)
echo.
echo DONE:  final_annotated_v2.mp4  +  predictions_seq_v2.csv
pause
