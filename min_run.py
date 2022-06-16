import numpy as np
from dispatch_env import GymEnv

n_ev = 1  

env = GymEnv(
    normal_render = True, 
    show=True,
    days = 2,
    hours_per_day = 24,
    minutes_per_hour = 60, 
    n_passengers = 2, 
    n_ev = n_ev,
    min_threshold = 10,     
    min_soc = 0,
    max_soc = 39,           
    charge_time = 60,       
    discharge_rate = 0.2,
    time_step_duration = 1,
    alpha = 25,
    beta = 10,
    soc_threshold = 20/100,
    occurrence_rate = 7/10,
    max_number_of_requests = 110
    )
obs = env.reset()
episodes = 1

best_avg_score = best_score = -np.inf

global_step = 0
scores = []

for e in range(episodes):
    done = False
    score = 0
    state = env.reset()

    while not done:
        global_step += 1
        next_state, reward, done, info = env.step(env.action_space.sample())

        score += reward
        state = next_state

    print(info)

    scores.append(score)

    avg_score = np.mean(scores[-100:])

    print('episode: ', e,'score: ', score, 'best score: %.2f' % best_score,
    ' avg score: %.1f' % avg_score, 'best avg score: %.2f' % best_avg_score, 
    'steps:', global_step, 'ep duration:', info['count'])

    if score > best_score:
        best_score = score

    if avg_score > best_avg_score:
        best_avg_score = avg_score