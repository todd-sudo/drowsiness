# Drowsiness

### Стек:
1. Python3.10
2. Numpy
3. OpenCV
4. Mediapipe

<br>


### О проекте

Обнаружение сонливости водителя

<br>

### Запуск:

Создать виртуальное окружение
```bash
python3.10 -m venv venv
```
Активируем его
```bash
source venv/bin/activate
```
Устанавливаем зависимости
```bash
pip install -r requirements.txt
```
Запускаем
```bash
python drowsiness.py
```

<br>

### Билд бинарника:

1. Первоначальный билд бинарника:
```bash
pyinstaller --onefile --windowed --noconsole drowsiness.py
```

2. Добавить в `drowsiness.spec`:

```python
def get_mediapipe_path():
    import mediapipe
    mediapipe_path = mediapipe.__path__[0]
    return mediapipe_path

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

mediapipe_tree = Tree(get_mediapipe_path(), prefix='mediapipe', excludes=["*.pyc"])
a.datas += mediapipe_tree
a.binaries = filter(lambda x: 'mediapipe' not in x[0], a.binaries)
```

Чтобы вышло примерно так:

```python
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['drowsiness.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

def get_mediapipe_path():
    import mediapipe
    mediapipe_path = mediapipe.__path__[0]
    return mediapipe_path

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

mediapipe_tree = Tree(get_mediapipe_path(), prefix='mediapipe', excludes=["*.pyc"])
a.datas += mediapipe_tree
a.binaries = filter(lambda x: 'mediapipe' not in x[0], a.binaries)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='drowsiness',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

```

3. После этого запустить билд бинарника 
(последующие ребилды выполнять именно этой командой):

```bash
pyinstaller drowsiness.spec
```

<br>

### Дополнительный материал:

1. [Документация Mediapipe](https://google.github.io/mediapipe/)
2. [Примеры использования Mediapipe](https://github.com/google/mediapipe/tree/master/docs/solutions)
3. [Определение цвета с картинки](https://ru.inettools.net/image/opredelit-tsvet-piksela-na-kartinke-onlayn)
4. [Онлайн RGB редактор](https://www.rapidtables.com/web/color/RGB_Color.html)
5. [Документация OpenCV Python](https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_tutorials.html)
6. [pyinstaller mediapipe build](https://python.tutorialink.com/issues-compiling-mediapipe-with-pyinstaller-on-macos/)
7. [pyinstaller mediapipe build 2](https://stackoverflow.com/questions/71804849/py-to-exe-error-filenotfounderror-the-path-does-not-exist)
8. [Конвертер изображения в иконку](https://convertio.co/ru/jpg-ico/)
