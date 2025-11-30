cd /d "E:\Other\python_folder\neuro\ct_neuro\"
curl -X PUT -T data/p1ch2/horse.jpg http://localhost:8000/image --output E:\Other\python_folder\neuro\ct_neuro\res.jpg

@REM Проверка готовности
@REM curl http://localhost:8000/ready
@REM Проверка здоровья
@REM curl http://localhost:8000/health