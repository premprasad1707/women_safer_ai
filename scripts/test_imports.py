import importlib
import sys
import os

# ensure project root (one level up) is on sys.path so `src` imports work
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

mods = ['src.alert_system', 'src.data_generator', 'src.map_utils', 'src.predict',
        'src.preprocessing', 'src.train_model', 'src.utils']

for m in mods:
    try:
        importlib.import_module(m)
        print(m + ' OK')
    except Exception as e:
        print(m + ' ERROR: ' + str(e))
        sys.exit(1)

print('All imports OK')
