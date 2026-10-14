apt update && apt install -y python3-pip

mkdir -p /workspace
cd /workspace/
git clone https://github.com/jakebraunberger/sn-gamestate.git
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.9
source .venv/bin/activate
cd /workspace/sn-gamestate
uv pip install -e .
apt-get update && apt-get install -y libgl1 libglib2.0-0
apt-get update && apt-get install -y aria2

uv run mim install mmcv==2.0.1

mkdir -p /workspace/sn-gamestate/pretrained_models/reid/
cd /workspace/sn-gamestate/pretrained_models/reid/
aria2c -x 16 -s 16 "https://zenodo.org/records/10653453/files/prtreid-soccernet-baseline.pth.tar?download=1"
cd /workspace/sn-gamestate/
mkdir -p /workspace/sn-gamestate/pretrained_models/calibration/
cd pretrained_models/calibration/ && aria2c -x 16 -s 16 "https://zenodo.org/records/12626395/files/SV_kp?download=1"
aria2c -x 16 -s 16 "https://zenodo.org/records/12626395/files/SV_lines?download=1"
cd /workspace/sn-gamestate/
echo "Finished..." > /workspace/finished