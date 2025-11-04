# python train.py wandb.run_name=diffusion_Hf_HM task.diffusion=true task.use_HM=true
python train.py wandb.run_name=Hf_TD task.diffusion=true task.use_HM=false

python train.py wandb.run_name=wind0.03 task.wind=true task.randomization.wind.train.intensity=[0,0.03]

python train.py wandb.run_name=br+Hf0.1 task.diffusion=true task.use_HM=false task.use_action_history=true task.action_history_step=1 task.only_use_bodyrates_history=

python train.py wandb.run_name=action_ct task.use_action_history=true task.action_history_step=1 task.only_use_thrust_history=true
# python train.py wandb.run_name=xd1 task.future_traj_steps=1
# python train.py wandb.run_name=rpy task.use_rpy_obs=true
# python train.py wandb.run_name=wind task.wind=true

#python train.py wandb.run_name=action1 task.use_action_history=true task.action_history_step=1
#python train.py wandb.run_name=action5 task.use_action_history=true task.action_history_step=5


#python train.py wandb.run_name=delay1-3 task.latency_step=3 task.random_latency=true
#python train.py wandb.run_name=delay1-2 task.latency_step=2 task.random_latency=true
#python train.py wandb.run_name=delay0-1 task.latency=true task.latency_step=1 task.random_latency=true
#python train.py 


#python train.py wandb.run_name=xd4 task.future_traj_steps=4
