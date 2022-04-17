from pickle import TRUE
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt
import numpy as np
from multi_env import Env

EPISODES = 1000
GAMMA = 0.9
EPSILON = 0
BATCH = 32
EPS_DEC = 9e-7
LR = 1e-3
dir = 'chkpt/mult_dqn'

from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    """Set normal_render = True to see the environment at normal speed"""

    writer = SummaryWriter(f'{dir}/runs_mult')

    env = Env(normal_render=False)
    agent = DQNAgent(GAMMA, 
                    EPSILON, 
                    env.observation_space.shape[0], 
                    env.action_size, 
                    BATCH, 
                    eps_dec=EPS_DEC, 
                    name = f'lr{LR}_b{BATCH}_dec{EPS_DEC}_g{GAMMA}', 
                    lr=LR, 
                    chkpt_dir= dir)

    print(f'Training parameters: {EPISODES=}, {GAMMA=}, {BATCH=}, {EPS_DEC=}, {LR=}')

    best_avg_score = best_score = -np.inf
    load_checkpoint = True

    if load_checkpoint:
        agent.load_models()

    global_step = 0
    scores = []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()

        while not done:
            global_step += 1

            action = agent.choose_action(state)
            # action = int(input("Enter next action: "))
            next_state, reward, done, info = env.step(action)
            
            score += reward

            if not load_checkpoint:
                agent.store_transition(state, action, reward, next_state, done)
                agent.train()

            state = next_state

        print(info)
        writer.add_scalar('reward', score, e)
        writer.add_scalar('hours', info['count']/10, e)
        writer.add_scalar('waiting time', np.mean(info['waiting_time']), e)
        writer.add_scalar('users', info['served_users'], e)
        writer.add_scalar('charging', info['to_charge'], e)
        cs_info = {}
        for k, v in info['cs'].items():
            cs_info[str(k+1)] = v
        writer.add_scalars('Charging Stations', cs_info, e)

        scores.append(score)

        avg_score = np.mean(scores[-100:])

        print('episode: ', e,'score: ', score, 'best score: %.2f' % best_score,
            ' avg score: %.1f' % avg_score, 'best avg score: %.2f' % best_avg_score,
            'epsilon: %.2f' % agent.epsilon, 'steps:', global_step, 'ep duration:', info['count'])

        if score > best_score:
            best_score = score

        if avg_score > best_avg_score:
            if not load_checkpoint:
                agent.save_models()
            best_avg_score = avg_score