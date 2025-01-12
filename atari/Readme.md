# Atari-100k

## Requirements
We assume you have habanalabs-installer.sh with version of 1.18.0.

```
export HABANALABS_VIRTUAL_DIR="/your/directory/to/venv" # Recommend /project_dir/atari/atari_env
./habanalabs-installer.sh install -t dependencies
./habanalabs-installer.sh install --type pytorch --venv
```

then activate your venv.
```
source ./atari_env/bin/activate
```

Install requirements.txt
```
pip install -r requirements.txt
```

## Installing Atari environment

### Download Rom dataset
```
python
import urllib.request
urllib.request.urlretrieve('http://www.atarimania.com/roms/Roms.rar','Roms.rar')
```

### Connect Rom dataset to atari_py library
```
apt-get install unrar
unrar x Roms.rar
mkdir rars
mv HC\ ROMS rars
mv ROMS rars
python -m atari_py.import_roms rars
``` 

## Instructions

To run a single run, activate your env, then use the `run.py` script
```
PT_HPU_LAZY_MODE=0 python run.py # Eager mode
PT_HPU_LAZY_MODE=1 python run.py # Lazy mode
```

To run the Atari-100k benchmark (26 games with 5 random sees), use `run_parallel.py` script
```
PT_HPU_LAZY_MODE=0 python run_parallel.py # Eager mode
PT_HPU_LAZY_MODE=1 python run_parallel.py # Lazy mode
```

To reproduce the performance of PLASTIC* or PLASTIC, use scripts inside the `script`.
```
PT_HPU_LAZY_MODE=0 bash scripts/NeurIPS2023/drq_plastic_dagger_rr2.sh # Eager mode
PT_HPU_LAZY_MODE=1 bash scripts/NeurIPS2023/drq_plastic_dagger_rr2.sh # Lazy mode


PT_HPU_LAZY_MODE=0 bash scripts/NeurIPS2023/drq_plastic_rr2.sh # Eager mode
PT_HPU_LAZY_MODE=1 bash scripts/NeurIPS2023/drq_plastic_rr2.sh # Lazy mode
```



