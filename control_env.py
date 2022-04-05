import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image
from gym import spaces
import networkx as nx
import pdb
from utils import create_graph

PhotoImage = ImageTk.PhotoImage
UNIT = 100
HEIGHT = 10
WIDTH = 10
NUM_NODES = HEIGHT * WIDTH - 1
MAX_SOC = 24
MAX_STEPS = 240
CHARGING_REWARD = 0
PICKUP_REWARD = 1
DROP_REWARD = 5
DISCHARGE_RATE = 0.4
NOBATTERY_PENALTY = MAX_STEPS - (MAX_SOC / DISCHARGE_RATE)
CHARGING_RATE = 2.0

# np.random.seed(1)

LOCATIONS = [4, 5, 12, 20, 23]

class Env(tk.Tk):
    def __init__(self, normal_render = False):
        super().__init__()
        self.action_space = ['u','d','l','r','stay','serve','charge']
        self.action_size = len(self.action_space)
        self.observation_space = spaces.Box(
            low= np.array([0.]* (1 + 4)), 
            high =np.concatenate((np.array([np.float32(MAX_SOC)]), np.array([np.float32(NUM_NODES)]*4))), 
            dtype= np.float32)

        self.title('Taxi')
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, WIDTH * UNIT))
        self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        self.normal_render = normal_render
        
        self.counter = 0
        self.rewards = []
        self.goal = []
        self.ev_soc = MAX_SOC
        self.passenger = False
        self.on_duty = False
        self.no_pickup_counter = 0
        self.no_drop_counter = 0
        self.user_counter = 0
        self.waiting_time = 0
        self.charging_counter = 0
        self.info = dict()
        self.info['waiting_time'] = []

        self.car_location = self.canvas.coords(self.car)
        self.cs_location = [WIDTH-1, HEIGHT-1]
        self.set_reward(self.cs_location, CHARGING_REWARD)
        self.user_location = self.random_location()
        self.set_reward(self.user_location, PICKUP_REWARD)
        self.user_dest_location = self.random_location()
        self.set_reward(self.user_dest_location, DROP_REWARD)
        self.graph = create_graph(WIDTH, HEIGHT)

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white',
                            height=HEIGHT * UNIT,
                            width=WIDTH * UNIT)

        road_size = int(UNIT/10)
        for c in range(int(UNIT/2), (WIDTH * UNIT), UNIT):
            x0, y0, x1, y1 = c, 0, c, (WIDTH * UNIT)
            canvas.create_rectangle(x0-road_size, y0, x1+road_size, y1, fill='#add8e6', width=0)
            canvas.create_line(x0, y0, x1, y1, dash = (int(UNIT/10), int(UNIT/25)))
        for r in range(int(UNIT/2), (HEIGHT * UNIT), UNIT):
            x0, y0, x1, y1 = 0, r, (HEIGHT * UNIT), r
            canvas.create_rectangle(x0, y0-road_size, x1, y1+road_size, fill='#add8e6', width=0)
            canvas.create_line(x0, y0, x1, y1, dash = (int(UNIT/10), int(UNIT/25)))

        self.rewards = []
        self.goal = []
        x, y = UNIT/2, UNIT/2

        self.car = canvas.create_image(x, y, image=self.shapes[0])

        self.taken_locations = [[x,y]]

        canvas.pack()

        return canvas

    def random_location(self):
        location = [np.random.randint(WIDTH-1), np.random.randint(HEIGHT-1)]
        while [(UNIT * x) + UNIT / 2 for x in location] in self.taken_locations:
            location = [np.random.randint(WIDTH-1), np.random.randint(HEIGHT-1)]
        return location

    def load_images(self):
        img_unit = int(UNIT*0.2)
        car_image = Image.open("./img/car.png").convert('RGBA').resize((img_unit, img_unit))
        station_image = Image.open("./img/green-energy.png").convert('RGBA').resize((img_unit, img_unit))
        # User image taken from https://uxwing.com/person-profile-image-icon/
        user_image = Image.open("./img/person.png").convert('RGBA').resize((img_unit, img_unit))
        # Dest image taken from https://uxwing.com/area-icon/
        user_dest_image = Image.open("./img/area.png").convert('RGBA').resize((img_unit,img_unit))
        car = PhotoImage(car_image)
        station = PhotoImage(station_image)
        user = PhotoImage(user_image)
        dest = PhotoImage(user_dest_image)
        #TODO add image for destination

        return car, station, user, dest

    #TODO
    def reset_rewards(self):

        for reward in self.rewards:
            self.canvas.delete(reward['figure'])

        self.rewards.clear()
        self.goal.clear()

        self.no_pickup_counter = 0
        self.no_drop_counter = 0
        self.waiting_time = 0

        self.taken_locations = [self.coords_to_state(self.canvas.coords(self.car))]

        self.set_reward(self.cs_location, CHARGING_REWARD)
        self.user_location = self.random_location()
        self.set_reward(self.user_location, PICKUP_REWARD)
        self.user_dest_location = self.random_location()
        self.set_reward(self.user_dest_location, DROP_REWARD)

        self.passenger = False

    #TODO complete rewards
    def set_reward(self, state, reward):
        state = [int(state[0]), int(state[1])]
        x = int(state[0])
        y = int(state[1])
        temp = {}
        if reward == CHARGING_REWARD:
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image((UNIT * x) + UNIT / 2,
                                                       (UNIT * y) + UNIT / 2,
                                                       image=self.shapes[1])
            self.taken_locations.append([(UNIT * x) + UNIT / 2,(UNIT * y) + UNIT / 2])

        elif reward == PICKUP_REWARD:
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image((UNIT * x) + UNIT / 2,
                                                      (UNIT * y) + UNIT / 2,
                                                      image=self.shapes[2])
            self.taken_locations.append([(UNIT * x) + UNIT / 2,(UNIT * y) + UNIT / 2])

        elif reward == DROP_REWARD:
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image((UNIT * x) + UNIT / 2,
                                                      (UNIT * y) + UNIT / 2,
                                                      image=self.shapes[3])
            self.taken_locations.append([(UNIT * x) + UNIT / 2,(UNIT * y) + UNIT / 2])

            self.goal.append(temp['figure'])

        temp['coords'] = self.canvas.coords(temp['figure'])
        temp['state'] = state
        self.rewards.append(temp)

    #TODO
    def check_if_reward(self, state):
        check_list = dict()
        check_list['if_done'] = False
        check_list['if_goal'] = False
        rewards = 0

        found = False
        for reward in self.rewards:
            if found:
                break
            if reward['state'] == state:
                if reward['reward'] == PICKUP_REWARD and self.on_duty:
                    # rewards += reward['reward']
                    self.passenger = True
                    self.canvas.itemconfigure(reward['figure'], state='hidden')
                    self.info['waiting_time'].append(60*self.waiting_time/10)
                if reward['reward'] == DROP_REWARD and self.passenger:
                    rewards += reward['reward']
                    check_list['if_goal'] = True
                    self.passenger = False
                    self.on_duty = False
                if reward['reward'] == CHARGING_REWARD and not self.passenger:
                    for _ in range(int(np.ceil((MAX_SOC - self.ev_soc)/CHARGING_RATE))):
                        if self.counter == MAX_STEPS:
                            found = True
                            break
                        self.counter += 1
                        rewards -= PICKUP_REWARD
                        self.no_pickup_counter += 1
                        self.waiting_time += 1
                        self.ev_soc = min(self.ev_soc + CHARGING_RATE, MAX_SOC)

        # if self.passenger and self.no_drop_counter >= 8:
        #     rewards -= PICKUP_REWARD * 2
        
        if not self.passenger:
            rewards -= PICKUP_REWARD
            self.no_pickup_counter += 1
        elif self.passenger:
            self.no_drop_counter += 1

        if not self.passenger and not self.on_duty:
            self.waiting_time += 1

        if self.ev_soc <= 0:
            check_list['if_done'] = True
            rewards -= (MAX_STEPS - self.counter) * 1

        if self.counter == MAX_STEPS:
            check_list['if_done'] = True

        if check_list['if_goal']:
            self.reset_rewards()

        check_list['rewards'] = rewards

        return check_list

    def coords_to_state(self, coords):
        x = int((coords[0] - UNIT / 2) / UNIT)
        y = int((coords[1] - UNIT / 2) / UNIT)
        return [x,y]

    def state_to_node(self, coords):
        x, y = coords
        node = int((y * WIDTH) + x)
        return node

    def node_to_coords(self, node):
        x = node % HEIGHT
        y = node // WIDTH
        coords = ((UNIT * x) + UNIT / 2, (UNIT * y) + UNIT / 2)
        return coords

    #TODO
    def get_state(self):
        states = np.array(np.zeros(self.observation_space.shape[0], dtype=np.float32))
        states[0] = self.ev_soc
        states[1] = self.state_to_node(self.coords_to_state(self.canvas.coords(self.car)))
        if self.passenger:
            self.user_location = self.coords_to_state(self.canvas.coords(self.car))
        states[2] = self.state_to_node(self.user_location)
        states[3] = self.state_to_node(self.user_dest_location)
        states[4] = self.state_to_node(self.cs_location)
        return states

    def reset(self):
        self.update()
        if self.normal_render:
            time.sleep(0.5)
        x, y = self.canvas.coords(self.car)
        self.canvas.move(self.car, UNIT / 2 - x, UNIT / 2 - y)
        self.reset_rewards()
        self.ev_soc = MAX_SOC
        self.counter = 0
        self.user_counter = 0
        self.charging_counter = 0
        self.info = dict()
        self.info['waiting_time'] = []
        return self.get_state()

    #TODO
    def step(self, action):
        if action <=4:
            self.counter += 1
            self.render()
            next_coords = self.move(self.car, action)
            check = self.check_if_reward(self.coords_to_state(next_coords))
        else:
            check = self.move_path(self.car, action)
        
        done = check['if_done']

        if done: 
            self.info['served_users'] = self.user_counter
            self.info['count'] = self.counter
            self.info['to_charge'] = self.charging_counter
        
        reward = check['rewards']

        # pdb.set_trace()

        self.canvas.tag_raise(self.car)

        s_ = self.get_state()

        return s_, reward, done, self.info

    def move_path(self, target, action):
        s = self.canvas.coords(target)

        car_node = self.state_to_node(self.coords_to_state(s))

        if action == 5: 
            passenger_node = self.state_to_node(self.user_location)
            dest_node = self.state_to_node(self.user_dest_location)
            route_car2pass = nx.dijkstra_path(self.graph, car_node, passenger_node)
            route_car2dest = nx.dijkstra_path(self.graph, passenger_node, dest_node)
            final_route = np.concatenate([route_car2pass[1:], route_car2dest[1:]])
            self.user_counter += 1
            self.on_duty = True
        if action == 6:
            self.charging_counter += 1
            cs_node = self.state_to_node(self.cs_location)
            cs_route = nx.dijkstra_path(self.graph, car_node, cs_node)
            if len(cs_route) == 1:
                final_route = cs_route
            else:
                final_route = cs_route[1:]

        # print(final_route)
        for i in final_route:
            coords = self.node_to_coords(i)
            aux_s = self.canvas.coords(target)
            self.canvas.move(target, coords[0]-aux_s[0], coords[1]-aux_s[1])
            self.render()
            self.counter += 1
            self.ev_soc -= DISCHARGE_RATE
            check = self.check_if_reward(self.coords_to_state(self.canvas.coords(target)))
            if self.ev_soc <= 0 or self.counter == MAX_STEPS:
                break

        return check

    def move(self, target, action):
        s = self.canvas.coords(target)

        base_action = np.array([0, 0])

        if action == 0:  # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
                self.ev_soc -= DISCHARGE_RATE
        elif action == 1:  # down
            if s[1] < (HEIGHT - 1) * UNIT:
                base_action[1] += UNIT
                self.ev_soc -= DISCHARGE_RATE
        elif action == 2:  # right
            if s[0] < (WIDTH - 1) * UNIT:
                base_action[0] += UNIT
                self.ev_soc -= DISCHARGE_RATE
        elif action == 3:  # left
            if s[0] > UNIT:
                base_action[0] -= UNIT
                self.ev_soc -= DISCHARGE_RATE

        self.canvas.move(target, base_action[0], base_action[1])

        s_ = self.canvas.coords(target)

        return s_

    def render(self):
        if self.normal_render:
            time.sleep(0.07)
        self.update()

from dqn_agent import DQNAgent
import matplotlib.pyplot as plt

EPISODES = 100
GAMMA = 0.9
EPSILON = 0.01
BATCH = 32
EPS_DEC = 5e-6
LR = 1e-3

if __name__ == "__main__":
    """Set normal_render = True to see the environment at normal speed"""
    env = Env(normal_render=False)
    agent = DQNAgent(GAMMA, EPSILON, env.observation_space.shape[0], env.action_size, BATCH, eps_dec=EPS_DEC, 
        name = f'lr{LR}_b{BATCH}_dec{EPS_DEC}_g{GAMMA}', lr=LR, chkpt_dir= 'chkpt/test')

    print(f'Training parameters: ep: {EPISODES}, g: {GAMMA}, b: {BATCH}, dec: {EPS_DEC}, lr: {LR}')

    best_avg_score = best_score = -np.inf
    load_checkpoint = True

    if load_checkpoint:
        agent.load_models()

    global_step = 0
    scores, avg_scores = [], []
    eps_history, step = [], [] 
    waiting_time, avg_wt = [], []
    hours, avg_hours = [], []
    users, avg_users = [], [] 
    to_charge, avg_to_charge = [], []

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

        scores.append(score)
        step.append(global_step)
        hours.append(info['count']/10)
        waiting_time.append(np.mean(info['waiting_time']))
        users.append(info['served_users'])
        to_charge.append(info['to_charge'])

        avg_score = np.mean(scores[-100:])
        avg_hour = np.mean(hours[-100:])
        avg_waiting_time = np.mean(waiting_time[-100:])
        avg_user = np.mean(users[-100:])
        avg_2charge = np.mean(to_charge[-100:])
        
        avg_wt.append(avg_waiting_time)
        avg_scores.append(avg_score)
        avg_hours.append(avg_hour) 
        avg_users.append(avg_user)
        avg_to_charge.append(avg_2charge)

        print('episode: ', e,'score: ', score, 'best score: %.2f' % best_score,
            ' avg score: %.1f' % avg_score, 'best avg score: %.2f' % best_avg_score,
            'epsilon: %.2f' % agent.epsilon, 'steps:', global_step, 'ep duration:', info['count'])

        if score > best_score:
            best_score = score

        if avg_score > best_avg_score:
            if not load_checkpoint:
                agent.save_models()
            best_avg_score = avg_score

        eps_history.append(agent.epsilon)

    if load_checkpoint:
        slice = 1
        tag='eval'
    else:
        slice = 100
        tag='training'

        plt.figure()
        plt.plot(eps_history[::slice])
        plt.xlabel('Episodes')
        plt.ylabel('Epsilon')
        plt.savefig(f'chkpt/test/lr{LR}_b{BATCH}_dec{EPS_DEC}_g{GAMMA}_epsilon_{tag}.png')

    plt.figure()
    plt.plot(hours[::slice])
    plt.xlabel('Episodes')
    plt.ylabel('Working Time (hour)')
    plt.ylim(0, 24)
    plt.savefig(f'chkpt/test/lr{LR}_b{BATCH}_dec{EPS_DEC}_g{GAMMA}_workt_{tag}.png')

    plt.figure()
    plt.plot(avg_hours[::slice])
    plt.xlabel('Episodes')
    plt.ylabel('Avg. Working Time (hour)')
    plt.ylim(0, 24)
    plt.savefig(f'chkpt/test/lr{LR}_b{BATCH}_dec{EPS_DEC}_g{GAMMA}_avg_workt_{tag}.png')

    plt.figure()
    plt.plot(scores[::slice])
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.savefig(f'chkpt/test/lr{LR}_b{BATCH}_dec{EPS_DEC}_g{GAMMA}_scores_{tag}.png')
    
    plt.figure()
    plt.plot(avg_scores[::slice])
    plt.xlabel('Episodes')
    plt.ylabel('Avg. Rewards')
    plt.savefig(f'chkpt/test/lr{LR}_b{BATCH}_dec{EPS_DEC}_g{GAMMA}_avg_scores_{tag}.png')

    plt.figure()
    plt.plot(users[::slice])
    plt.xlabel('Episodes')
    plt.ylabel('Served Users')
    plt.ylim(0, 50)
    plt.savefig(f'chkpt/test/lr{LR}_b{BATCH}_dec{EPS_DEC}_g{GAMMA}_users_{tag}.png')

    plt.figure()
    plt.plot(avg_users[::slice])
    plt.xlabel('Episodes')
    plt.ylabel('Avg. Served Users')
    plt.ylim(0, 50)
    plt.savefig(f'chkpt/test/lr{LR}_b{BATCH}_dec{EPS_DEC}_g{GAMMA}_avg_users_{tag}.png')

    plt.figure()
    plt.plot(waiting_time[::slice])
    plt.xlabel('Episodes')
    plt.ylabel('Waiting Time (minutes)')
    plt.ylim(0, 70)
    plt.savefig(f'chkpt/test/lr{LR}_b{BATCH}_dec{EPS_DEC}_g{GAMMA}_waitt_{tag}.png')

    plt.figure()
    plt.plot(avg_wt[::slice])
    plt.xlabel('Episodes')
    plt.ylabel('Avg. Waiting Time (minutes)')
    plt.ylim(0, 70)
    plt.savefig(f'chkpt/test/lr{LR}_b{BATCH}_dec{EPS_DEC}_g{GAMMA}_avg_waitt_{tag}.png')

    plt.figure()
    plt.plot(to_charge[::slice])
    plt.xlabel('Episodes')
    plt.ylabel('Number of Charging Times')
    plt.ylim(0,20)
    plt.savefig(f'chkpt/test/lr{LR}_b{BATCH}_dec{EPS_DEC}_g{GAMMA}_2charge_{tag}.png')

    plt.figure()
    plt.plot(avg_to_charge[::slice])
    plt.xlabel('Episodes')
    plt.ylabel('Avg. Number of Charging Times')
    plt.ylim(0,20)
    plt.savefig(f'chkpt/test/lr{LR}_b{BATCH}_dec{EPS_DEC}_g{GAMMA}_avg_2charge_{tag}.png')
    
    