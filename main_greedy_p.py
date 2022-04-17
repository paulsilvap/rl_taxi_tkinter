import numpy as np
from multi_env import Env
import networkx as nx

from torch.utils.tensorboard import SummaryWriter

def greedy_agent(state, graph, user_info, cs_info, env):
    action = 4
    pass_list = np.empty(shape=len(user_info))
    cs_list = np.empty(shape=len(cs_info))
    if state[0] <= 10 * 0.4:
        for k,v in cs_info.items():
            _, r_car_to_cs = nx.single_source_dijkstra(graph, state[1], env.state_to_node(v['position']), weight ='state')
            cs_list[k] = len(r_car_to_cs[1:])
        action = 10 + np.where(cs_list == np.amin(cs_list))[0][0] 
    else:
        for k,v in user_info.items():
            l_pass_to_dest, _ = nx.single_source_dijkstra(graph, env.state_to_node(v['source']), env.state_to_node(v['destination']), weight ='state')
            pass_list[k] = l_pass_to_dest
        action = 5 + np.where(pass_list == np.amin(pass_list))[0][0]

    return action
    
EPISODES = 1000
dir = 'chkpt/mult'
writer = SummaryWriter(f'{dir}/runs_greedy_p')

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
            action = greedy_agent(state, env.graph, env.user_info, env.cs_info, env)
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

