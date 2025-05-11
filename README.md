# Bachelor_diploma
 
Перед тем как запустить DeepSORT, выполните следующие шаги:

1. Клонируйте репозиторий.

2. Запустите программу с нужными параметрами: имя файла для трекинга, тип видео (visible/infrared) и фов камеры.
   ```bash
   python .\deep_sort.py  ".\\test\\visible.mp4" visible --fov 60
   ```

Для запуска NanoTrack повторите п.1 и перейдите в нужную папку:
   ```bash
   cd NanoTrack
   ```
2. Далее выполните:
    ```bash
   python .bin\test.py --vis  --fov 60
   ```
Где --vis - флаг визуализации, --fov - фов камеры.