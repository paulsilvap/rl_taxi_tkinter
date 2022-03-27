import numpy as np
from environment import Env
from dqn_agent import DQNAgent

EPISODES = 3000
GAMMA = 0.7
EPSILON = 1.0
BATCH = 32
EPS_DEC = 5e-6
LR = 1e-3

if __name__ == "__main__":
    env = Env()
    agent = DQNAgent(GAMMA, EPSILON, env.observation_space.shape[0], env.action_size, BATCH, eps_dec=EPS_DEC, 
        name = f'lr_{LR}_batch_{BATCH}_dec_{EPS_DEC}_{EPISODES}_gamma_{GAMMA}', lr=LR)

    best_avg_score = best_score = -np.inf
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()

    global_step = 0
    scores, eps_history, step = [], [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()

        while not done:
            global_step += 1

            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            
            score += reward

            if not load_checkpoint:
                agent.store_transition(state, action, reward, next_state, done)
                agent.train()

            state = next_state

        scores.append(score)
        step.append(global_step)

        avg_score = np.mean(scores[-100:])

        print('episode: ', e,'score: ', score, 'best score %.2f' % best_score,
            ' average score %.1f' % avg_score, 'best avg score %.2f' % best_avg_score,
            'epsilon %.2f' % agent.epsilon, 'steps', global_step)

        if score > best_score:
            best_score = score

        if avg_score > best_avg_score:
            if not load_checkpoint:
                agent.save_models()
            best_avg_score = avg_score

        eps_history.append(agent.epsilon)