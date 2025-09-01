python train.py wandb.run_name=xd1 task.future_traj_steps=1
python train.py wandb.run_name=rpy task.use_rpy_obs=true
python train.py wandb.run_name=wind task.wind=true

#python train.py wandb.run_name=action1 task.use_action_history=true task.action_history_step=1
#python train.py wandb.run_name=action5 task.use_action_history=true task.action_history_step=5


#python train.py wandb.run_name=delay1-3 task.latency_step=3 task.random_latency=true
#python train.py wandb.run_name=delay1-2 task.latency_step=2 task.random_latency=true
#python train.py wandb.run_name=delay0-1 task.latency=true task.latency_step=1 task.random_latency=true
#python train.py 


#python train.py wandb.run_name=xd4 task.future_traj_steps=4
