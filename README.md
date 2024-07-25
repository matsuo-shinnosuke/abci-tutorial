# ABCI Tutorial

## Setup
Create an environment using
```
ssh abci
cd abci-tutorial
conda create -n abci python=3.9
conda activate abci
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
## Job Exection
```
qsub -g gcf51214 run.sh
```