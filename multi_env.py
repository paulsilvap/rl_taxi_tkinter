from random import random
import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image
from gym import spaces
import random
import networkx as nx
from utils import create_graph

PhotoImage = ImageTk.PhotoImage
UNIT = 50
HEIGHT = 10
WIDTH = 10
NUM_NODES = HEIGHT * WIDTH - 1
MAX_SOC = 24
MAX_STEPS = 240
TIME_STEP_DURATION = int(60 / (MAX_STEPS / 24))
CHARGING_REWARD = 0
PICKUP_REWARD = 1
DROP_REWARD = 5
DISCHARGE_RATE = 0.4
NOBATTERY_PENALTY = MAX_STEPS - (MAX_SOC / DISCHARGE_RATE)
CHARGING_RATE = 2.0
ALPHA = HEIGHT + WIDTH - 2
BETA = 10
NUMBER_OF_ROADS = (HEIGHT * WIDTH * 2) - HEIGHT - WIDTH
NUMBER_OF_PASSENGERS = 5
NUMBER_OF_CS = 4
NUMBER_OF_EV = 1

class Env(tk.Tk):
    def __init__(self, normal_render = False):
        super().__init__()
        self.action_space = spaces.Discrete(14)
        self.action_size = 14
        self.observation_space = spaces.Box(
            low= np.array([0.]* (1 + NUMBER_OF_ROADS+(NUMBER_OF_EV * 2)+(NUMBER_OF_PASSENGERS*2)+NUMBER_OF_CS)), 
            high =np.concatenate((np.array([np.float32(MAX_SOC)]), 
                np.array([np.float32(NUM_NODES)]*(NUMBER_OF_ROADS+(NUMBER_OF_EV * 2)+(NUMBER_OF_PASSENGERS*2)+NUMBER_OF_CS)))), 
            dtype= np.float32)

        self.title('Taxi')
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, WIDTH * UNIT))
        self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        self.normal_render = normal_render
        
        self.counter = 0
        self.aux_counter = 0
        self.rewards = []
        self.rewards_info = {}
        self.ev_soc = MAX_SOC
        self.passenger = False
        self.on_duty = False
        self.stand_by = True
        self.user_counter = 0
        self.waiting_time = 0
        self.wt_rectifier = 0
        self.cs_waiting_time = 0
        self.charging_counter = 0
        self.info = {}
        self.info['waiting_time'] = []
        self.info['cs'] = {}
        self.image_tags ={}
        self.image_tags['cs'] = []
        self.image_tags['source'] = []
        self.image_tags['destination'] = [] 

        self.car_location = self.canvas.coords(self.car)
        self.cs_location = [WIDTH-1, HEIGHT-1]

        self.set_cs()
        self.set_users()
        self.graph = create_graph(WIDTH, HEIGHT, weight_limit=TIME_STEP_DURATION+1)

    def set_cs(self):
        self.cs_info = {}

        for i in range(NUMBER_OF_CS):
            self.cs_info[i] = {}
        
        self.cs_info[0]['position'] = [WIDTH-1, HEIGHT-1]
        self.cs_info[1]['position'] = [WIDTH-1, 0]
        self.cs_info[2]['position'] = [0, HEIGHT-1]
        self.cs_info[3]['position'] = [int(WIDTH/2)-1, int(HEIGHT/2)-1]

        for k,v in self.cs_info.items():
            self.cs_info[k]['waiting_time'] = (random.randrange(0,80,TIME_STEP_DURATION))
            self.info['cs'][k] = 0
            self.set_reward(v['position'], CHARGING_REWARD)

    def update_cs(self):
        for k,v in self.cs_info.items():
            cs_state = v['waiting_time']
            if cs_state > 5:
                self.cs_info[k]['waiting_time'] = cs_state - TIME_STEP_DURATION
            elif cs_state <= 0:
                self.cs_info[k]['waiting_time'] = random.randrange(0,80,TIME_STEP_DURATION)
            
    def set_users(self):
        self.user_info = {}

        for i in range(NUMBER_OF_PASSENGERS):
            self.user_info[i] = {}
            self.user_info[i]['source'] = self.random_location()
            self.set_reward(self.user_info[i]['source'], PICKUP_REWARD)
            self.user_info[i]['destination'] = self.random_location()
            self.set_reward(self.user_info[i]['destination'], DROP_REWARD)
            self.user_info[i]['waiting_time'] = 0

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

        x, y = UNIT/2, UNIT/2

        self.car = canvas.create_image(x, y, image=self.shapes[0])

        self.taken_locations = [self.coords_to_state([x,y])]

        canvas.pack()

        return canvas

    def random_location(self):
        location = [np.random.randint(WIDTH-1), np.random.randint(HEIGHT-1)]
        while location in self.taken_locations:
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

        return car, station, user, dest

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
            self.image_tags['cs'].append(temp['figure'])

        elif reward == PICKUP_REWARD:
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image((UNIT * x) + UNIT / 2,
                                                      (UNIT * y) + UNIT / 2,
                                                      image=self.shapes[2])
            self.image_tags['source'].append(temp['figure'])

        elif reward == DROP_REWARD:
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image((UNIT * x) + UNIT / 2,
                                                      (UNIT * y) + UNIT / 2,
                                                      image=self.shapes[3])
            self.image_tags['destination'].append(temp['figure'])

        self.taken_locations.append(state)
        temp['state'] = state
        self.rewards.append(temp)

    def update_reward(self, state, reward, id):
        x = int(state[0])
        y = int(state[1])
        temp = {}
        if reward == CHARGING_REWARD:
            temp['reward'] = reward
            temp['figure'] = self.image_tags['cs'][id]

        elif reward == PICKUP_REWARD:
            temp['reward'] = reward
            temp['figure'] = self.image_tags['source'][id]

        elif reward == DROP_REWARD:
            temp['reward'] = reward
            temp['figure'] = self.image_tags['destination'][id]           

        self.taken_locations.append(state)
        self.canvas.moveto(temp['figure'],(UNIT * x) + UNIT / 2, (UNIT * y) + UNIT / 2)
        temp['state'] = state
        self.rewards.append(temp)

    def check_if_reward(self, state):
        check_list = dict()
        check_list['if_done'] = False
        check_list['if_goal'] = False
        rewards = 0

        if not self.passenger:
            self.waiting_time = self.counter if self.aux_counter == 0 else self.counter - self.aux_counter

        found = False
        for reward in self.rewards:
            if found:
                break
            if reward['state'] == state:
                if (reward['reward'] == PICKUP_REWARD and 
                self.on_duty and 
                self.user_info[self.picked_user]['source'] == state):
                    self.passenger = True
                    self.canvas.itemconfigure(reward['figure'], state='hidden')
                    self.waiting_time -= self.wt_rectifier
                    self.info['waiting_time'].append(60*self.waiting_time/10)
                if (reward['reward'] == DROP_REWARD and 
                self.passenger and 
                self.user_info[self.picked_user]['destination'] == state):
                    self.user_counter += 1
                    rewards += ((ALPHA * self.user_counter* 3/4) if self.pre_pickup == 0 else 
                                                                    (ALPHA * self.user_counter* 3/4) / self.pre_pickup)
                    check_list['if_goal'] = True
                    self.passenger = False
                    self.on_duty = False
                if (reward['reward'] == CHARGING_REWARD and 
                (not self.passenger) and
                (not self.stand_by) and 
                self.cs_info[self.picked_cs]['position'] == state):
                    self.cs_waiting_time = self.cs_info[self.picked_cs]['waiting_time']
                    for _ in range(int(self.cs_waiting_time/TIME_STEP_DURATION)):
                        if self.counter == MAX_STEPS:
                            found = True
                            break
                        self.check_done()
                    if not found:
                        self.charging_counter += 1
                        self.info['cs'][self.picked_cs] += 1
                        charging_time = int(np.ceil((MAX_SOC - self.ev_soc)/CHARGING_RATE)) 
                        self.cs_info[self.picked_cs]['waiting_time'] = charging_time * TIME_STEP_DURATION
                        for _ in range(charging_time):
                            if self.counter == MAX_STEPS:
                                found = True
                                break
                            self.check_done()
                            self.ev_soc = min(self.ev_soc + CHARGING_RATE, MAX_SOC)
                        self.stand_by = True

        if self.ev_soc < 0.4 or self.counter == MAX_STEPS:
            check_list['if_done'] = True
            rewards -= (MAX_STEPS - self.counter) * 2

        if check_list['if_goal']:
            self.canvas.itemconfigure(self.image_tags['source'][self.picked_user], state='normal')
            self.reset_rewards()
        elif self.waiting_time >= BETA and not self.on_duty:
            rewards -= self.waiting_time * 2
            self.info['waiting_time'].append(60*self.waiting_time/10)
            self.reset_rewards()

        check_list['rewards'] = rewards

        return check_list

    def check_done(self):
        self.counter += 1
        self.waiting_time = self.counter if self.aux_counter == 0 else self.counter - self.aux_counter
        self.update_cs()
        self.update_graph()

    def reset_rewards(self):
        '''
        Reset parameters once the passenger has been taken to destination
        '''
        self.rewards.clear()

        self.waiting_time = 0
        self.aux_counter = self.counter
        self.wt_rectifier = 0
        self.passenger = False
        self.on_duty = False

        self.taken_locations = [self.coords_to_state(self.canvas.coords(self.car))]

        for k,v in self.cs_info.items():
            self.update_reward(v['position'], CHARGING_REWARD, k)
        for k,v in self.user_info.items():
            self.user_info[k]['source'] = self.random_location()
            self.update_reward(self.user_info[k]['source'], PICKUP_REWARD, k)
            self.user_info[k]['destination'] = self.random_location()
            self.update_reward(self.user_info[k]['destination'], DROP_REWARD, k)
            self.user_info[k]['waiting_time'] = 0

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
        states = []
        states.append(self.ev_soc if self.ev_soc > 0.1 else 0)
        states.append(self.state_to_node(self.coords_to_state(self.canvas.coords(self.car))))
        for v in self.user_info.values():
            states.append(self.state_to_node(v['source']))
            states.append(self.state_to_node(v['destination']))
        states.append(self.waiting_time)
        for v in self.cs_info.values():
            states.append(v['waiting_time'])
        for edge in self.graph.edges():
            states.append(self.graph.get_edge_data(edge[0],edge[1])['state'])
        return np.array(states)

    def update_graph(self):
        if self.counter % 10 == 0:
            for edge in self.graph.edges():
                self.graph[edge[0]][edge[1]]['state'] = np.random.randint(1, TIME_STEP_DURATION + 1) 

    #TODO
    def step(self, action):
        if action <=4:
            self.update_cs()
            self.counter += 1
            self.update_graph()
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

        self.canvas.tag_raise(self.car)

        s_ = self.get_state()

        return s_, reward, done, self.info

    def move_path(self, target, action):
        s = self.canvas.coords(target)

        car_node = self.state_to_node(self.coords_to_state(s))

        if action >= 5 and action <= 9:
            self.picked_user = action%5
            pass_node = self.state_to_node(self.user_info[self.picked_user]['source'])
            dest_node = self.state_to_node(self.user_info[self.picked_user]['destination'])
            l_car_to_pass, r_car_to_pass = nx.single_source_dijkstra(self.graph, car_node, pass_node, weight='state')
            self.pre_pickup = len(r_car_to_pass) -1
            l_car_to_dest, r_car_to_dest = nx.single_source_dijkstra(self.graph, pass_node, dest_node, weight='state')
            self.wt_rectifier = round(l_car_to_dest/ TIME_STEP_DURATION)
            if len(r_car_to_pass) > 1:
                r_car_to_pass = r_car_to_pass[1:]
            final_route = np.concatenate([r_car_to_pass, r_car_to_dest[1:]])
            final_length = round(l_car_to_pass/TIME_STEP_DURATION) + self.wt_rectifier
            self.on_duty = True
        elif action >= 10 and action <= 13:
            self.stand_by = False
            self.picked_cs = action%10
            cs_node = self.state_to_node(self.cs_info[self.picked_cs]['position'])
            l_car_to_cs, r_car_to_cs = nx.single_source_dijkstra(self.graph, car_node, cs_node, weight='state')
            if len(r_car_to_cs) == 1:
                final_route = r_car_to_cs
            else:
                final_route = r_car_to_cs[1:]
            final_length = round(l_car_to_cs / TIME_STEP_DURATION)

        if self.counter + final_length >= 240:
            self.wt_rectifier = 0
            for i in range(final_length):
                self.counter += 1
                coords = self.node_to_coords(final_route[i])
                check = self.move_aux(coords, target)
                if self.ev_soc < 0.4 or self.counter == MAX_STEPS:
                    break
        else:
            self.counter += final_length
            for i in final_route:
                coords = self.node_to_coords(i)
                check = self.move_aux(coords, target)
                if self.ev_soc < 0.4 or self.counter == MAX_STEPS:
                    break

        return check

    def move_aux(self, coords, target):
        aux_s = self.canvas.coords(target)
        self.canvas.move(target, coords[0]-aux_s[0], coords[1]-aux_s[1])
        self.render()
        self.update_graph()
        self.update_cs()
        self.ev_soc -= DISCHARGE_RATE
        return self.check_if_reward(self.coords_to_state(self.canvas.coords(target)))

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
            time.sleep(0.5)
        self.update()

    def reset(self):
        if self.normal_render:
            time.sleep(0.5)
        self.update()

        x, y = self.canvas.coords(self.car)
        self.canvas.move(self.car, UNIT / 2 - x, UNIT / 2 - y)
        
        self.ev_soc = MAX_SOC
        self.counter = 0
        self.user_counter = 0
        self.charging_counter = 0
        self.info.clear()
        self.info['waiting_time'] = []
        self.info['cs'] = {}
        self.stand_by = True

        for k in self.cs_info.keys():
            self.cs_info[k]['waiting_time'] = (random.randrange(0,80,TIME_STEP_DURATION))
            self.info['cs'][k] = 0
        self.reset_rewards()
        return self.get_state()


from dqn_agent import DQNAgent
import matplotlib.pyplot as plt
import numpy as np
from multi_env import Env

EPISODES = 100000
GAMMA = 0.9
EPSILON = 1.0
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
    load_checkpoint = False

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
        writer.add_scalar('Epsilon', agent.epsilon, e)

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
