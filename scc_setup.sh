
module load python3/3.10.12

[ ! -d "env" ] && python -m venv env

source env/bin/activate
pip install -r requirements.txt

