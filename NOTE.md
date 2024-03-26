## Train
nohup python run.py --conf confs/base.conf --dataname 3e5e4ff60c151baee9d84a67fdc5736 --dir 3e5e4ff60c151baee9d84a67fdc5736_0315_2w512 > logs/train_3e5e_0315_2w512_log 2>&1 &

nohup python run.py --conf confs/base.conf --dataname 7b00e029725c0c96473f10e6caaeca56 --dir 7b00e029725c0c96473f10e6caaeca56_0315_4w5121l > logs/train_7b00_0315_4w5121l_log 2>&1 &

nohup python run.py --conf confs/base.conf --dataname 1a56d596c77ad5936fa87a658faf1d26 --dir 1a56d596c77ad5936fa87a658faf1d26_0315_4w5121l > logs/train_1a56_0315_4w5121l_log 2>&1 &

nohup python run.py --conf confs/base.conf --dataname 2aa870a4f4645401a9ced22d91ad7027 --dir 2aa870a4f4645401a9ced22d91ad7027_0315_4w5121l > logs/train_2aa8_0315_4w5121l_log 2>&1 &

nohup python run.py --conf confs/base.conf --dataname 1005ca47e516495512da0dbf3c68e847 --dir 1005ca47e516495512da0dbf3c68e847_0315_4w5121l > logs/train_1005_0315_4w5121l_log 2>&1 &

nohup python run.py --conf confs/base.conf --dataname 22c0b90fbda00bb9a3a61aa922ccc66 --dir 22c0b90fbda00bb9a3a61aa922ccc66_0315_4w5121l > logs/train_22c0_0315_4w5121l_log 2>&1 &

nohup python run.py --conf confs/base.conf --dataname 35d473527fa9bd8cbdb24a67dc08c308 --dir 35d473527fa9bd8cbdb24a67dc08c308_4w5121l > logs/train_35d4_4w5121l_log 2>&1 &

nohup python run.py --conf confs/base.conf --dataname 43a723b6845f6f90b1eebe42821a51d7 --dir 43a723b6845f6f90b1eebe42821a51d7_4w5121l > logs/train_43a7_4w5121l_log 2>&1 &

nohup python run.py --conf confs/base.conf --dataname 5675ff1d006c56009b2acbfd8323f804 --dir 5675ff1d006c56009b2acbfd8323f804_4w5121l > logs/train_5675_4w5121l_log 2>&1 &

nohup python run.py --conf confs/base.conf --dataname 5dcbc04a1ce783eb73f41773bda9db5c --dir 5dcbc04a1ce783eb73f41773bda9db5c_4w5121l > logs/train_5dcb_4w5121l_log 2>&1 &

nohup python run.py --conf confs/base.conf --dataname 88000bd0d8997374350937622ac92802 --dir 88000bd0d8997374350937622ac92802_4w5121l > logs/train_8800_4w5121l_log 2>&1 &

## Test
nohup python evaluation/shapenetCars/eval_mesh.py --conf confs/base.conf --dataname 3e5e4ff60c151baee9d84a67fdc5736 --dir 3e5e4ff60c151baee9d84a67fdc5736_0315_2w512 > eval_results/test_3e5e_0315_2w512_log 2>&1 &

nohup python evaluation/shapenetCars/eval_mesh.py --conf confs/base.conf --dataname 7b00e029725c0c96473f10e6caaeca56 --dir 7b00e029725c0c96473f10e6caaeca56_0315_4w5121l > eval_results/test_7b00_0315_4w5121l_log 2>&1 &

nohup python evaluation/shapenetCars/eval_mesh.py --conf confs/base.conf --dataname 1a56d596c77ad5936fa87a658faf1d26 --dir 1a56d596c77ad5936fa87a658faf1d26_0315_4w5121l > eval_results/test_1a56_0315_4w5121l_log 2>&1 &

nohup python evaluation/shapenetCars/eval_mesh.py --conf confs/base.conf --dataname 2aa870a4f4645401a9ced22d91ad7027 --dir 2aa870a4f4645401a9ced22d91ad7027_0315_4w5121l > eval_results/test_2aa8_0315_4w5121l_log 2>&1 &

nohup python evaluation/shapenetCars/eval_mesh.py --conf confs/base.conf --dataname 1005ca47e516495512da0dbf3c68e847 --dir 1005ca47e516495512da0dbf3c68e847_0315_4w5121l > eval_results/test_1005_0315_4w5121l_log 2>&1 &

nohup python evaluation/shapenetCars/eval_mesh.py --conf confs/base.conf --dataname 22c0b90fbda00bb9a3a61aa922ccc66 --dir 22c0b90fbda00bb9a3a61aa922ccc66_0315_4w5121l > eval_results/test_22c0_0315_4w5121l_log 2>&1 &

nohup python evaluation/shapenetCars/eval_mesh.py --conf confs/base.conf --dataname 35d473527fa9bd8cbdb24a67dc08c308 --dir 35d473527fa9bd8cbdb24a67dc08c308_4w5121l > eval_results/test_35d4_4w5121l_log 2>&1 &

nohup python evaluation/shapenetCars/eval_mesh.py --conf confs/base.conf --dataname 43a723b6845f6f90b1eebe42821a51d7 --dir 43a723b6845f6f90b1eebe42821a51d7_4w5121l > eval_results/test_43a7_4w5121l_log 2>&1 &

nohup python evaluation/shapenetCars/eval_mesh.py --conf confs/base.conf --dataname 5675ff1d006c56009b2acbfd8323f804 --dir 5675ff1d006c56009b2acbfd8323f804_4w5121l > eval_results/test_5675_4w5121l_log 2>&1 &

nohup python evaluation/shapenetCars/eval_mesh.py --conf confs/base.conf --dataname 5dcbc04a1ce783eb73f41773bda9db5c --dir 5dcbc04a1ce783eb73f41773bda9db5c_4w5121l > eval_results/test_5dcb_4w5121l_log 2>&1 &

nohup python evaluation/shapenetCars/eval_mesh.py --conf confs/base.conf --dataname 88000bd0d8997374350937622ac92802 --dir 88000bd0d8997374350937622ac92802_4w5121l > eval_results/test_8800_4w5121l_log 2>&1 &

## tensorboard
```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('loss_log')
writer.add_scalar("loss",loss, self.iter_step)
writer.close()
```

## nvidia-smi
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv

nvidia-smi --query-compute-apps=pid,used_memory --format=csv
