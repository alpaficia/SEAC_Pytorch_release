from SEAC import SEACAgent
from ReplayBuffer import RandomBuffer
from Adapter import *
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from ReplayBuffer import device

import numpy as np
import gymnasium
import torch
import os
import shutil
import argparse

from gymnasium.envs.registration import register

import time


def str2bool(v):
    # transfer str to bool for argparse
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True', 'true', 'TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


"""""""""
If you wanna change the weather, please go to the env/dynamic_world.py file and change the self.weather_info.
list of weather info
0.0 presents sunny
1.0 presents rain
2.0 presents snow without freezing land
3.0 presents snow with freezing land 
"""""""""

'''Hyper Parameters Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=2250000, help='which model to load')
parser.add_argument('--seed', type=int, default=1995, help='seed for training')

parser.add_argument('--total_steps', type=int, default=int(3e6), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(1e4), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(1e3), help='Model evaluating interval, in steps.')
parser.add_argument('--eval_turn', type=int, default=5, help='Model evaluating times, in episode.')
parser.add_argument('--update_every', type=int, default=50, help='Training Frequency, in steps')
parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=256, help='Hidden net width')
parser.add_argument('--a_lr', type=float, default=2e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=2e-4, help='Learning rate of critic')

parser.add_argument('--batch_size', type=int, default=256, help='Batch Size')
parser.add_argument('--alpha', type=float, default=0.12, help='Entropy coefficient')
parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='Use adaptive_alpha or Not')
# Set it True to enable the SAC V2

parser.add_argument('--fixed_freq', type=float, default=0.0, help='if 0.0, not use fixed frequency')
parser.add_argument('--obs_freq', type=float, default=5.0, help='fixed obs frequency setting by user, should not be 0')
parser.add_argument('--energy_per_step', type=float, default=1.0, help='energy to compute one step, in J, if you want '
                                                                       'to change this parameter, you need to change '
                                                                       'the env file also')
parser.add_argument('--min_time', type=float, default=0.01, help='min time of taking one action, should not be 0')
parser.add_argument('--ep_max_length', type=int, default=500, help='Usr define the maximum control frequency')

parser.add_argument('--time_benchmark', type=str2bool, default=False, help='bench mark the time cost')


opt = parser.parse_args()
print(opt)
print(device)


def evaluate_policy(env, model, render, max_action_t, max_action_f, min_time, obs_freq, fixed_freq, energy_per_step):
    scores = 0
    total_time = 0
    total_energy = 0
    turns = opt.eval_turn
    for j in range(turns):
        current_step_eval = 0
        ep_r = 0
        dead = False
        obs, info = env.reset()
        agent_obs = obs['agent_pos']
        obstacle = obs['obstacle']
        target = obs['target']
        speed = obs['speed']
        time_last_step = obs['time']
        force = obs['force']
        if fixed_freq:
            s = np.concatenate([agent_obs, obstacle, target, speed, time_last_step, force], axis=0)
        else:
            s = np.concatenate([agent_obs, obstacle, target, speed, time_last_step, force], axis=0)
        time_epoch = 0
        while not dead:
            current_step_eval += 1
            # Take deterministic actions at test time
            if current_step_eval < opt.ep_max_length:
                last_step = np.zeros(1,)
            else:
                last_step = np.ones(1,)
            a = model.select_action(s, deterministic=True, with_logprob=False)
            if fixed_freq:
                a_f_eval = a[0:2]
                act_f_eval = Action_adapter(a_f_eval, max_action_f)
                act_t_eval = np.array([1.0 / obs_freq])
            else:
                a_t_eval = a[0]
                a_f_eval = a[1:3]
                act_f_eval = Action_adapter(a_f_eval, max_action_f)
                act_t_eval = Action_t_relu6_adapter(a_t_eval, max_action_t)
                if act_t_eval <= min_time:
                    act_t_eval = min_time
                act_t_eval = np.array([act_t_eval])
            act = np.concatenate([act_t_eval, act_f_eval, last_step], axis=0)
            obs, r, terminated, truncated, info = env.step(act)
            agent_obs = obs['agent_pos']
            obstacle = obs['obstacle']
            target = obs['target']
            speed = obs['speed']
            time_last_step = obs['time']
            force = obs['force']
            s_prime = np.concatenate([agent_obs, obstacle, target, speed, time_last_step, force], axis=0)
            s = s_prime
            time_epoch += act_t_eval
            if terminated or truncated:
                dead = True
            ep_r += r
            if render:
                env.render()
        energy = current_step_eval * energy_per_step
        total_energy += energy
        scores += ep_r
        total_time += time_epoch
    return float(scores / turns), float(total_time / turns), float(total_energy / turns)


def main():
    write = opt.write  # Use SummaryWriter to record the training.
    render = opt.render
    seed = opt.seed
    env_with_dead = True
    steps_per_epoch = opt.ep_max_length
    register(
        id="DynamicWorld-nodelay",
        entry_point="envs:DynamicWorldNDEnv",
        max_episode_steps=steps_per_epoch,
    )
    env = gymnasium.make('DynamicWorld-nodelay')
    fixed_freq = opt.fixed_freq
    state_dim = 11
    if fixed_freq:
        action_dim = 2
    else:
        action_dim = 3
    max_action_t = 1.0
    max_action_f = 100.0
    min_time = opt.min_time

    time_benchmark = opt.time_benchmark
    obs_freq = opt.obs_freq
    energy_per_step = opt.energy_per_step

    # Interaction config:
    start_steps = 5 * steps_per_epoch  # in steps
    update_after = 2 * steps_per_epoch  # in steps
    update_every = opt.update_every
    total_steps = opt.total_steps
    eval_interval = opt.eval_interval  # in steps
    save_interval = opt.save_interval  # in steps

    # SummaryWriter config:
    if write:
        time_now = str(datetime.now())[0:-10]
        time_now = ' ' + time_now[0:13] + '_' + time_now[-2::]
        write_path = 'runs/SEAC_time{}'.format("dynamicworld-nodelay") + time_now
        if os.path.exists(write_path):
            shutil.rmtree(write_path)
        writer = SummaryWriter(log_dir=write_path)
    else:
        writer = None

    # Model hyperparameter config:
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "gamma": opt.gamma,
        "hid_shape": (opt.net_width, opt.net_width),
        "a_lr": opt.a_lr,
        "c_lr": opt.c_lr,
        "batch_size": opt.batch_size,
        "alpha": opt.alpha,
        "adaptive_alpha": opt.adaptive_alpha
    }

    model = SEACAgent(**kwargs)
    if not os.path.exists('model'):
        os.mkdir('model')
    if opt.Loadmodel:
        model.load(opt.ModelIdex)

    replay_buffer = RandomBuffer(state_dim, action_dim, env_with_dead, max_size=int(1e6))

    current_steps = 0
    obs, info = env.reset()
    agent_obs = obs['agent_pos']
    obstacle = obs['obstacle']
    target = obs['target']
    speed = obs['speed']
    time_last_step = obs['time']
    force = obs['force']
    s = np.concatenate([agent_obs, obstacle, target, speed, time_last_step, force], axis=0)
    fixed_freq = np.array([fixed_freq])
    tricker = 0
    time_old = 0.0
    for t in range(total_steps):
        current_steps += 1
        if render:
            env.render()
        if t < start_steps:
            # Random explore for start_steps, but first 10 step with certainty moving speed
            act = env.action_space.sample()
            act_t = act[0]
            act_f = act[1:3]
            act_t = Act_t_correction(act_t)  # to make sure that the time should be positive
            act_t = max_action_t * (act_t / max_action_f)  # fixed the range of time from [0,-0.1] to [0, 1]
            if act_t <= min_time:
                act_t = min_time  # We don't want the time goes to 0, which makes many troubles
            act_t = np.array([act_t])
            if fixed_freq:
                act_t = np.array([1.0/obs_freq])
            if current_steps < steps_per_epoch:
                last_step = np.zeros(1,)
            else:
                last_step = np.ones(1,)
            act = np.concatenate([act_t, act_f, last_step], axis=0)
            a_f = Action_adapter_reverse(act_f, max_action_f)
            a_t = Action_t_relu6_adapter_reverse(act_t, max_action_t)
            if fixed_freq:
                a = a_f
            else:
                a = np.concatenate([a_t, a_f], axis=0)
        else:
            a = model.select_action(s, deterministic=False, with_logprob=False)
            if fixed_freq:
                a_f = a[0:2]
                act_f = Action_adapter(a_f, max_action_f)
                act_t = np.array([1.0 / obs_freq])
            else:
                a_f = a[1:3]
                a_t = a[0]
                act_f = Action_adapter(a_f, max_action_f)
                act_t = Action_t_relu6_adapter(a_t, max_action_t)
                if act_t <= min_time:
                    act_t = min_time  # We don't want the time goes to 0, which makes many troubles
                act_t = np.array([act_t])
            if current_steps < steps_per_epoch:
                last_step = np.zeros(1,)
            else:
                last_step = np.ones(1,)
            act = np.concatenate([act_t, act_f, last_step], axis=0)
        obs, rew, terminated, truncated, info = env.step(act)
        agent_obs = obs['agent_pos']
        obstacle = obs['obstacle']
        target = obs['target']
        speed = obs['speed']
        time_last_step = obs['time']
        force = obs['force']
        s_prime = np.concatenate([agent_obs, obstacle, target, speed, time_last_step, force], axis=0)
        s_prime_t = torch.tensor(np.float32(s_prime))
        if terminated or truncated:
            dead = True
        else:
            dead = False
        s_t = torch.tensor(np.float32(s))
        a_t = torch.tensor(a)
        replay_buffer.add(s_t, a_t, rew, s_prime_t, dead)
        s = s_prime
        if (t+1) % 500 == 0:
            print('CurrentPercent:', ((t + 1)*100.0)/total_steps, '%')
            if tricker == 0:
                time_old = time.time()
            else:
                time_new = time.time()
                time_diff = time_new - time_old
                if time_benchmark:
                    print("Predicted Completion Time：", time_diff * ((total_steps-t)/500), "in seconds")
                time_old = time_new
            tricker += 1

        # 50 environment steps company with 50 gradient steps.
        # Stabler than 1 environment step company with 1 gradient step.
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                model.train(replay_buffer)

        '''save model'''
        if (t + 1) % save_interval == 0:
            model.save(t + 1)

        '''record & log'''
        if (t + 1) % eval_interval == 0:
            score, average_time, average_energy_cost = evaluate_policy(env, model, False, max_action_t, max_action_f,
                                                                       min_time, obs_freq, fixed_freq, energy_per_step)
            if write:
                writer.add_scalar('ep_r', score, global_step=t + 1)
                writer.add_scalar('alpha', model.alpha, global_step=t + 1)
                writer.add_scalar('average_time', average_time, global_step=t + 1)
                writer.add_scalar('average_energy_cost', average_energy_cost, global_step=t + 1)
            print('EnvName: dynamicworld-nodelay', 'TotalSteps:', t + 1, 'score:', score, 'average_time:', average_time,
                  'average_energy_cost:', average_energy_cost)
        if dead:
            current_steps = 0
            obs, info = env.reset()
            agent_obs = obs['agent_pos']
            obstacle = obs['obstacle']
            target = obs['target']
            speed = obs['speed']
            time_last_step = obs['time']
            force = obs['force']
            s = np.concatenate([agent_obs, obstacle, target, speed, time_last_step, force], axis=0)

    writer.close()
    env.close()


if __name__ == '__main__':
    main()
