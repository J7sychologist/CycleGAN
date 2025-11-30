@echo off
cd /d "E:\Other\python_folder\neuro\ct_neuro\"
@REM Стандартный запуск
@REM start cmd /k "python -m p2ch15.request_batching_server data/p1ch2/horse2zebra_0.4.0.pth && exit"
@REM Версия с JIT
start cmd /k "python -m p2ch15.request_batching_jit_server data/p1ch2/horse2zebra_jit.pt && exit"