import numpy as np
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt
from multi_env import Env

from torch.utils.tensorboard import SummaryWriter

EPISODES = 1000
dir = 'chkpt/mult'
writer = SummaryWriter(f'{dir}/runs')

if __name__ == "__main__":
    """Set normal_render = True to see the environment at normal speed"""
    env = Env(normal_render=False)

    best_avg_score = best_score = -np.inf

    global_step = 0
    scores, avg_scores = np.zeros(shape=EPISODES), []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()

        while not done:
            global_step += 1
            action = np.random.randint(env.action_size)
            next_state, reward, done, info = env.step(action)
            
            score += reward
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

        scores[e] = score

        avg_score = np.mean(scores[-100:])

        print('episode: ', e,'score: ', score, 'best score: %.2f' % best_score,
            ' avg score: %.1f' % avg_score, 'best avg score: %.2f' % best_avg_score, 
            'steps:', global_step, 'ep duration:', info['count'])

        if score > best_score:
            best_score = score

        if avg_score > best_avg_score:
            best_avg_score = avg_score