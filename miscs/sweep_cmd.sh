nohup python -u miscs/run.py --gpu_id 0 --mode_id 0 --label_id 0 > logs/000.log 2>&1 & 
nohup python -u miscs/run.py --gpu_id 1 --mode_id 0 --label_id 1 > logs/101.log 2>&1 & 
nohup python -u miscs/run.py --gpu_id 2 --mode_id 0 --label_id 2 > logs/202.log 2>&1 & 
nohup python -u miscs/run.py --gpu_id 3 --mode_id 1 --label_id 0 > logs/310.log 2>&1 & 
nohup python -u miscs/run.py --gpu_id 4 --mode_id 1 --label_id 1 > logs/411.log 2>&1 & 
nohup python -u miscs/run.py --gpu_id 5 --mode_id 1 --label_id 2 > logs/512.log 2>&1 & 


nohup python -u miscs/run.py --gpu_id 0 --mode_id 2 --label_id 0 > logs/020.log 2>&1 & 
nohup python -u miscs/run.py --gpu_id 1 --mode_id 2 --label_id 1 > logs/121.log 2>&1 & 
nohup python -u miscs/run.py --gpu_id 2 --mode_id 2 --label_id 2 > logs/222.log 2>&1 & 
nohup python -u miscs/run.py --gpu_id 3 --mode_id 3 --label_id 0 > logs/330.log 2>&1 & 
nohup python -u miscs/run.py --gpu_id 4 --mode_id 3 --label_id 1 > logs/431.log 2>&1 & 
nohup python -u miscs/run.py --gpu_id 5 --mode_id 3 --label_id 2 > logs/532.log 2>&1 & 