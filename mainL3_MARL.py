import csv
import datetime
import os

# 以允许共享使用不同的MKL库，避免出现不兼容的错误
import traci

from agents.nstep_pdqn import PDQNNStepAgent

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import matplotlib.pyplot as plt
import click
import gym
import pandas as pd

from my_EnvCluster import LaneChangePredict
import numpy as np

@click.command()
@click.option('--seed', default=0, help='Random seed.', type=int)
@click.option('--episodes', default=2000, help='Number of epsiodes.', type=int)
@click.option('--evaluation-episodes', default=1000, help='Episodes over which to evaluate after training.', type=int)
@click.option('--batch-size', default=128, help='Minibatch size.', type=int)
@click.option('--gamma', default=0.99, help='Discount factor.', type=float)
@click.option('--update-ratio', default=0.1, help='Ratio of updates to samples.', type=float)
@click.option('--inverting-gradients', default=True,
              help='Use inverting gradients scheme instead of squashing function.', type=bool)
@click.option('--initial-memory-threshold', default=1000, help='Number of transitions required to start learning.',
              type=int)
@click.option('--replay-memory-size', default=5000, help='Replay memory size in transitions.', type=int)
@click.option('--epsilon-start', default=0.95, help='Initial epsilon value.', type=float)
@click.option('--epsilon-steps', default=1000, help='Number of episodes over which to linearly anneal epsilon.',
              type=int)
@click.option('--epsilon-final', default=0.02, help='Final epsilon value.', type=float)
@click.option('--learning-rate-actor', default=0.00001, help="Actor network learning rate.", type=float)
@click.option('--learning-rate-actor-param', default=0.00001, help="Critic network learning rate.", type=float)
@click.option('--clip-grad', default=1., help="Gradient clipping.", type=float)
@click.option('--beta', default=0.2, help='Averaging factor for on-policy and off-policy targets.', type=float)
@click.option('--scale-actions', default=True, help="Scale actions.", type=bool)
@click.option('--save-freq', default=0, help='How often to save models (0 = never).', type=int)
@click.option('--save-dir', default="results/soccer", help='Output directory.', type=str)
@click.option('--title', default="PDQN", help="Prefix of output files", type=str)

def run(seed, episodes, batch_size, gamma, inverting_gradients, initial_memory_threshold, replay_memory_size,
        epsilon_steps, learning_rate_actor, learning_rate_actor_param, title, epsilon_start, epsilon_final, clip_grad,
        beta,
        scale_actions, evaluation_episodes, update_ratio, save_freq, save_dir):
    N_Vehicle = 3

    if save_freq > 0 and save_dir:
        save_dir = os.path.join(save_dir, title + "{}".format(str(seed)))
        os.makedirs(save_dir, exist_ok=True)

    env = LaneChangePredict()
    dir = os.path.join(save_dir, title)
    np.random.seed(seed)
    agent_class = PDQNNStepAgent

    agents = [agent_class(
        env.n_features, env.n_actions,
        actor_kwargs={
            'activation': "relu", },
        actor_param_kwargs={
            'activation': "relu", },
        batch_size=batch_size,
        learning_rate_actor=learning_rate_actor,
        learning_rate_actor_param=learning_rate_actor_param,
        epsilon_initial=epsilon_start,
        epsilon_steps=epsilon_steps,
        epsilon_final=epsilon_final,
        gamma=gamma,  # 0.99
        clip_grad=clip_grad,
        beta=beta,
        initial_memory_threshold=initial_memory_threshold,
        replay_memory_size=replay_memory_size,
        inverting_gradients=inverting_gradients,
        seed=seed + i) for i in range(N_Vehicle)]
    print(agents)
    max_steps = 230

    max_episode_data = None
    max_avg_reward = -np.inf

    for i_eps in range(episodes):
        env.reset()
        all_reward = np.array([])
        transitions = [np.array([], dtype=np.float32).tolist() for _ in range(N_Vehicle)]
        action_all = np.zeros([N_Vehicle, 2], dtype=np.float32)
        act_all = np.zeros((N_Vehicle, 1), dtype=np.float32)
        state_all = np.zeros((N_Vehicle, 16), dtype=np.float32)
        action_all_idx = 0
        all_action_par = np.zeros((N_Vehicle, N_Vehicle), dtype=np.float32)
        for i, agent in enumerate(agents):
            state = env._findstate(i)
            state = np.array(state, dtype=np.float32, copy=False)
            act, act_param, all_action_parameters = agent.act(state)
            action = env.find_action(act)

            action_all[action_all_idx, 0] = action
            action_all[action_all_idx, 1] = act_param
            act_all[action_all_idx, 0] = act

            all_action_par[action_all_idx, :] = all_action_parameters.cpu().numpy()

            state_all[action_all_idx, :] = state

            action_all_idx += 1

        current_episode_data = {i: {'speedControl': [], 'ttc': []} for i in range(N_Vehicle)}

        for i_step in range(max_steps):
            next_state_all = np.zeros((N_Vehicle, 16), dtype=np.float32)
            reward_all = np.zeros((N_Vehicle, 1), dtype=np.float32)
            next_action_all = np.zeros((N_Vehicle, 2), dtype=np.float32)
            next_act_all = np.zeros((N_Vehicle, 1), dtype=np.float32)
            all_terminal = np.array([], dtype=np.float32)
            next_all_action_par = np.zeros((N_Vehicle, N_Vehicle), dtype=np.float32)


            for i, agent in enumerate(agents):
                action = action_all[i, 0]
                act_param = action_all[i, 1]
                env._findCurrentState(i)
                next_state, reward, terminal = env.step(action, act_param, i)

                if i_eps >= episodes - 100:
                    ttc = env.getFinaTCC()
                    current_episode_data[i]['speedControl'].append((np.tanh(act_param) + 1) * 15)
                    current_episode_data[i]['ttc'].append(ttc)

                next_state = np.array(next_state, dtype=np.float32, copy=False)
                next_act, next_act_param, next_all_action_parameters = agent.act(next_state)
                next_action = env.find_action(next_act)

                next_act_all[i, 0] = next_act
                next_state_all[i, :] = next_state
                next_action_all[i, 0] = next_action
                next_action_all[i, 1] = next_act_param
                next_all_action_par[i, :] = next_all_action_parameters.cpu().numpy()
                reward_all[i, 0] = reward
                all_reward = np.append(all_reward, reward)
                all_terminal = np.append(all_terminal, terminal)
                transitions[i].append(
                    [state_all[i, :], np.concatenate(([act_all[i, 0]], all_action_par[i, :])).ravel(),
                     reward_all[i, 0], next_state_all[i, :], np.concatenate(([next_act_all[i, 0]],
                                                       next_all_action_par[i, :])).ravel(), terminal])

            for i in range(N_Vehicle):
                act_all[i, 0], action_all[i, 1], all_action_par[i, :] = next_act_all[i, 0], next_action_all[i, 1], next_all_action_par[i, :]
                action_all[i, 0] = next_action_all[i, 0]
                state_all[i, :] = next_state_all[i, :]

            terminal = int(all(element == 1 for element in all_terminal))
            if terminal:
                print("Complete！！")
                break

        episode_rewards = sum(all_reward)

        avg_reward = episode_rewards / max(1, i_step)

        if (i_eps >= episodes - 100) and (avg_reward > max_avg_reward):
            max_avg_reward = avg_reward
            max_episode_data = current_episode_data

        for _, agent in enumerate(agents):
            agent.end_episode()
        print(i_eps, "average_episode_reward：", episode_rewards/i_step)

        for i, agent in enumerate(agents):
            n_step_returns = compute_n_step_returns(transitions[i], gamma)
            for t, nsr in zip(transitions[i], n_step_returns):
                t.append(nsr)
                agent.replay_memory.append(state=t[0], action_with_param=t[1], reward=t[2], next_state=t[3],
                                           done=t[5], n_step_return=nsr)

            n_updates = int(update_ratio * i_step)
            for _ in range(n_updates):
                agent._optimize_td_loss()

            if i_eps % 2 == 0:
                agent.actor_target.load_state_dict(agent.actor.state_dict())
                agent.actor_param_target.load_state_dict(agent.actor_param.state_dict())

        df = pd.DataFrame({'reward': episode_rewards/i_step}, index=[0])
        df.to_csv('../result/normal/reward.csv', index=False, mode='a', header=False)

    for i in range(N_Vehicle):
        df1 = pd.DataFrame({'speedControl': max_episode_data[i]['speedControl']})
        df2 = pd.DataFrame({'ttc': max_episode_data[i]['ttc']})
        df1.to_csv(f'../result/normal/agent{i + 1}_speedControl.csv', index=False, header=False)
        df2.to_csv(f'../result/normal/agent{i + 1}_ttc.csv', index=False, header=False)

def compute_n_step_returns(episode_transitions, gamma):
    n = len(episode_transitions)
    n_step_returns = np.zeros((n,))
    n_step_returns[n - 1] = episode_transitions[n - 1][2]  # Q-value is just the final reward
    for i in range(n - 2, 0, -1):
        reward = episode_transitions[i][2]
        target = n_step_returns[i + 1]
        n_step_returns[i] = reward + gamma * target
    return n_step_returns

if __name__ == '__main__':
    sumo_binary = "sumo-gui"
    sumocfg_file = "data/Lane3/StraightRoad.sumocfg"
    csvfile = 'data_trainFirst.csv'

    sumo_cmd = [sumo_binary, "-c", sumocfg_file, "--start", "--delay", "100", "--scale", "1"]
    traci.start(sumo_cmd)
    with open(csvfile, 'w', newline='') as csvfile:
        simulation_time = 0
        simulation_duration = 6
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            vehicles = traci.vehicle.getIDList()
            for vehicle_id in vehicles:
                time = traci.simulation.getTime()
                vehicle_speed = traci.vehicle.getSpeed(vehicle_id)
                vehicle_accler = traci.vehicle.getAcceleration(vehicle_id)

                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([time, vehicle_id, vehicle_speed, vehicle_accler])

            simulation_time = time

            if simulation_time >= simulation_duration:
                break
    run()



