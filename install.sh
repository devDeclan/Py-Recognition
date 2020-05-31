sudo apt-get update
sudo apt-get install -y git ffmpeg

python3 -m virtualenv venv

source venv/bin/activate
python -m pip install --upgrade pip
pip install wheel
pip install -r requirements.txt
