tar -xvf tucker.tar.xz

python3 -m venv .venv
source .venv/bin/activate
which python3

pip install -r requirements.txt
pip install .

./run.sh scripts/00_setup_ts.py
./run.sh scripts/02_full_benchmark.py


