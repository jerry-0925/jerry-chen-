cd /data3/yangkaixing/CustomDance/YF/Code/AudioGeneration
conda activate /data3/yangkaixing/CustomDance/Reference/Env/mamba

python train_fsq.py
nohup python train_fsq.py > train_fsq.log 2>&1 &

python train_gpt.py
nohup python train_gpt.py > train_gpt.log 2>&1 &

python demo_gpt.py