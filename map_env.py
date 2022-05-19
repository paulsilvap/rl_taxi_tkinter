import random
import pdb
import time
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import tkinter as tk

from utils import create_map, label_speed, graph_from_gdfs
from gym import spaces
from random import randint
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

DAYS = 2
gdf_edges, gdf_nodes, G, gdf_cs, s_df, result = create_map(DAYS)

NUMBER_OF_PASSENGERS = 10
MAX_NUMBER_OF_CONNECTIONS = int((max(sorted((d for _, d in G.degree()), reverse= True)))/2)
NUMBER_OF_CS = len(gdf_cs)
NUMBER_OF_ROADS = G.number_of_edges()
NUMBER_OF_EV = 1
MAX_SOC = 48
DISCHARGE_RATE = 0.4
CHARGING_RATE = 2.0
MAX_STEPS = 288
TIME_STEP_DURATION = int(60 / (MAX_STEPS / 24))
ALPHA = 18 # Check later
BETA = 10
OUT_OFF_BATTERY_PENALTY = MAX_STEPS - (MAX_SOC / DISCHARGE_RATE)

class Env(tk.Tk):
    def __init__(self, normal_render = False, show = False):
        super().__init__()
        ''' Define the action size and the observation space'''
        self.action_size = NUMBER_OF_CS + MAX_NUMBER_OF_CONNECTIONS + NUMBER_OF_PASSENGERS + 1
        self.observation_space =  spaces.Box(
            low = np.array([0.] * (NUMBER_OF_ROADS + (NUMBER_OF_EV * 2) + (NUMBER_OF_PASSENGERS * 3) + NUMBER_OF_CS)),
            high = np.array([6.] * (NUMBER_OF_ROADS + (NUMBER_OF_EV * 2) + (NUMBER_OF_PASSENGERS * 3) + NUMBER_OF_CS), dtype=np.float32),
            dtype = np.float32
        )

        self.graph = G
        self.intersections = gdf_nodes
        self.cs = gdf_cs
        self.cs_nodes = self.cs['NODE_ID'].values
        self.roads = gdf_edges
        self.aux_road_result = result.drop(columns=result.geometry.name)
        self.mask_101 = (self.aux_road_result['ROAD_RANK'] == '101').values
        self.mask_103 = ((self.aux_road_result['ROAD_RANK'] == '103') | (self.aux_road_result['ROAD_RANK'] == '106')).values
        self.mask_107 = (self.aux_road_result['ROAD_RANK'] == '107').values
        self.color_mapper = {'green': (0.0, 0.8, 0.0, 1.0), 'yellow': (0.8, 0.8, 0.0, 1.0), 'red': (0.8, 0.0, 0.0, 1.0)}
        self.day = 1
        self.max_days = DAYS
        self.traffic = s_df[self.day].groupby('ts')
        self.traffic_time = iter(list(self.traffic.groups.keys()))
        self.daily_graph = {}
        self.daily_roads = {}
        for d in range(1, self.max_days+1):
            self.daily_graph[d] = {}
            self.daily_roads[d] = {}

        self.title('Jeonju')
        self.create_plot()
        self.canvas = self._build_canvas()
        self.normal_render = normal_render
        self.show = show

        self.counter = 0
        self.aux_counter = 0
        self.traffic_counter = 0
        self.rewards = []
        self.passenger = False
        self.user_counter = 0
        self.waiting_time = 0
        self.wt_rectifier = 0
        self.cs_waiting_time = 0
        self.charging_counter = 0
        self.info = {}
        self.info['waiting_time'] = []
        self.info['cs'] = {}

        self.set_cs()
        self.set_users()
        self.canvas.get_tk_widget().pack(side='top',fill='both', expand=True)

    def color_roads(self):
        return [self.color_mapper[v] for v in self.roads['TC'].values]

    def create_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(10,7))
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        self.ax = self.roads['geometry'].plot(ax=self.ax, color=self.color_roads(), lw=1, alpha=None, zorder=-1)
        self.roads = self.roads.drop(columns=self.roads.geometry.name)
        # self.ax.scatter(self.intersections['geometry'].x, self.intersections['geometry'].y, s = 2, zorder=-1, color='k')
        self.intersections = self.intersections.drop(columns=self.intersections.geometry.name)
        # self.ax.set(xlim = (127.05693, 127.19335), ylim = (35.81962, 35.88782))
        self.ax.set(xlim = (127.09, 127.15), ylim = (35.83, 35.86))
        self.fig.tight_layout()
        self.ax.scatter(self.cs['geometry'].x, self.cs['geometry'].y, color='xkcd:brown')

    def set_cs(self):
        self.cs_info = self.cs.T.to_dict()

        for k in self.cs_info.keys():
            self.cs_info[k]['waiting_time'] = (random.randrange(0, 80, TIME_STEP_DURATION))
            self.info['cs'][k] = 0

    def update_cs(self):
        for k,v in self.cs_info.items():
            cs_state = v['waiting_time']
            if cs_state > 5:
                self.cs_info[k]['waiting_time'] = cs_state - TIME_STEP_DURATION
            elif cs_state <= 0:
                self.cs_info[k]['waiting_time'] = random.randrange(0,80,TIME_STEP_DURATION)

    def random_location(self):
        picked_node = random.choice(self.intersections.index.values)
        while (picked_node in self.cs_nodes) or (picked_node in self.taken_nodes):
            picked_node = random.choice(self.intersections.index.values)
        loc = [self.intersections.at[picked_node, 'x'], self.intersections.at[picked_node, 'y']]
        self.taken_nodes.append(picked_node)
        return loc, picked_node

    def _build_canvas(self):
        # self.attributes('-fullscreen', True)
        canvas = FigureCanvasTkAgg(self.fig, self)
        
        self.ev_info = {}
        self.taken_nodes = []
        for i in range(NUMBER_OF_EV):
            self.ev_info[i] = {}
            location, node = self.random_location()
            self.ev_info[i]['location'] = location
            self.ev_info[i]['NODE_ID'] = node
            self.ev_info[i]['SOC'] = MAX_SOC
            self.ev_info[i]['on_duty'] = False
            self.ev_info[i]['stand_by'] = True
            self.ev_info[i]['scatter'] = self.ax.scatter(self.ev_info[i]['location'][0], self.ev_info[i]['location'][1], color='b', marker='s')
        
        return canvas

    def set_user_info(self, i):
        source, source_node = self.random_location()
        self.user_info[i]['source'] = source
        self.user_info[i]['NODE_ID_source'] = source_node
        self.user_info[i]['scatter_source'] = self.ax.scatter(self.user_info[i]['source'][0], self.user_info[i]['source'][1], color='m', marker='H')
        destination, dest_node = self.random_location()
        self.user_info[i]['destination'] = destination
        self.user_info[i]['NODE_ID_destination'] = dest_node
        self.user_info[i]['scatter_destination'] = self.ax.scatter(self.user_info[i]['destination'][0], self.user_info[i]['destination'][1], color='m', marker='X')
        self.user_info[i]['waiting_time'] = 0
            # self.info['waiting_time'][i] = 0

    def set_users(self):
        self.user_info = {}
        
        for i in range(NUMBER_OF_PASSENGERS):
            self.user_info[i] = {}
            self.set_user_info(i)

    # @profile
    def update_roads(self, groups, date):
        if self.traffic_counter in self.daily_roads[self.day].keys():
            self.roads = self.daily_roads[self.day][self.traffic_counter]
        else:
            s = groups.get_group(date).set_index('LINK_ID')['speed']
            self.aux_road_result['speed'] = self.aux_road_result['LINK_ID'].map(s).fillna(self.aux_road_result['speed'])
            self.roads = label_speed(self.aux_road_result, self.mask_101, self.mask_103, self.mask_107)
            self.daily_roads[self.day][self.traffic_counter] = self.roads
    
    # @profile
    def update_graph(self):
        if self.counter >= self.traffic_counter + 1:
            time = next(self.traffic_time)
            self.update_roads(self.traffic, time)
            if self.traffic_counter in self.daily_graph[self.day].keys():
                self.graph = self.daily_graph[self.day][self.traffic_counter]
            else:
                roads = self.roads.iloc[:, [2,5]]
                self.graph = graph_from_gdfs(roads)
                self.daily_graph[self.day][self.traffic_counter] = self.graph
            if self.show:
                self.ax.collections[0].set_color(self.color_roads())
            self.traffic_counter += 1

    def discharge_ev(self, index):
        if self.ev_info[index] != 0.4:
            self.ev_info[index]['SOC'] = (int(self.ev_info[index]['SOC']*10) - int((DISCHARGE_RATE)*10))/10
        else:
            self.ev_info[index]['SOC'] = 0

    def move(self, index, target, action):
        s = target['NODE_ID']
        sample = self.graph.neighbors(s)
        s_ = next((x for i,x in enumerate(sample) if i==action), None)
        if s_ is not None:
            self.ev_info[index]['NODE_ID'] = s_
            self.ev_info[index]['location'] = [self.intersections.at[s_, 'x'], self.intersections.at[s_, 'y']]
            self.ev_info[index]['scatter'].set_offsets(
                np.c_[self.ev_info[index]['location'][0], self.ev_info[index]['location'][1]])
            self.discharge_ev(index)
        else:
            s_ = s
        return s_

    def get_state(self):
        states = []
        states.append(self.ev_info[0]['SOC'])
        states.append(int(self.ev_info[0]['NODE_ID']))
        for v in self.user_info.values():
            states.append(int(v['NODE_ID_source']))
            states.append(int(v['NODE_ID_destination']))
            states.append(v['waiting_time'])
        for v in self.cs_info.values():
            states.append(v['waiting_time'])
        for edge in self.graph.edges():
            for v in self.graph.get_edge_data(edge[0], edge[1]).values():
                states.append(v['state'])    
        return np.array(states)

    # @profile
    def check_done(self):
        self.counter += 1 
        for k in self.user_info.keys():
            self.user_info[k]['waiting_time'] = self.counter if self.aux_counter == 0 else self.counter - self.aux_counter
        self.update_cs()
        self.update_graph()

    def reset_rewards(self, index):
        self.rewards.clear()

        for k in self.user_info.keys():
            self.user_info[k]['waiting_time'] = 0
        self.aux_counter = self.counter
        self.wt_rectifier = 0
        self.passenger = False
        self.ev_info[index]['on_duty'] = False

        self.taken_nodes = [self.ev_info[index]['NODE_ID']]

        for k in self.user_info.keys():
            source, source_node = self.random_location()
            self.user_info[k]['source'] = source
            self.user_info[k]['NODE_ID_source'] = source_node
            self.user_info[k]['scatter_source'].set_offsets(
                np.c_[self.user_info[k]['source'][0], self.user_info[k]['source'][1]])
            destination, dest_node = self.random_location()
            self.user_info[k]['destination'] = destination
            self.user_info[k]['NODE_ID_destination'] = dest_node
            self.user_info[k]['scatter_destination'].set_offsets(
                np.c_[self.user_info[k]['destination'][0], self.user_info[k]['destination'][1]])
            self.user_info[k]['waiting_time'] = 0

    # @profile
    def check_if_reward(self, index, next_node):
        check_list = {}
        check_list['if_done'] = False
        check_list['if_goal'] = False
        rewards = 0

        found = False
        if not self.passenger:
            for k in self.user_info.keys():
                self.user_info[k]['waiting_time'] = self.counter if self.aux_counter == 0 else self.counter - self.aux_counter

        # pdb.set_trace()

        if self.ev_info[index]['on_duty'] and self.user_info[self.picked_user]['NODE_ID_source'] == next_node:
            self.passenger = True
            self.user_info[self.picked_user]['waiting_time'] -= self.wt_rectifier
            self.info['waiting_time'].append(60*self.user_info[self.picked_user]['waiting_time']/10)
        if self.passenger and self.user_info[self.picked_user]['NODE_ID_destination'] == next_node:
            self.user_counter += 1
            rewards += ((ALPHA * self.user_counter* 3/4) if self.pre_pickup == 0 else 
                            (ALPHA * self.user_counter* 3/4) / self.pre_pickup)
            check_list['if_goal'] = True
            self.passenger = False
            self.ev_info[index]['on_duty'] = False
        if ((not self.passenger) and (not self.ev_info[index]['stand_by']) and 
            self.cs_info[self.picked_cs]['NODE_ID'] == next_node):
            self.cs_waiting_time = self.cs_info[self.picked_cs]['waiting_time']
            for _ in range(int(self.cs_waiting_time/TIME_STEP_DURATION)):
                if self.counter == MAX_STEPS:
                    found = True
                    break
                self.check_done()
            if not found:
                self.charging_counter += 1
                self.info['cs'][self.picked_cs] += 1
                charging_time = int(np.ceil((MAX_SOC - self.ev_info[index]['SOC'])/CHARGING_RATE))
                self.cs_info[self.picked_cs]['waiting_time'] = charging_time * TIME_STEP_DURATION
                for _ in range(charging_time):
                    if self.counter == MAX_STEPS:
                        found = True
                        break
                    self.check_done()
                    self.ev_info[index]['SOC'] = min(self.ev_info[index]['SOC'] + CHARGING_RATE, MAX_SOC)
                self.ev_info[index]['stand_by'] = True

        # pdb.set_trace()

        if self.ev_info[index]['SOC'] < 0.4 or self.counter == MAX_STEPS:
            check_list['if_done'] = True
            rewards -= (MAX_STEPS - self.counter) * 2

        if check_list['if_goal']:
            self.reset_rewards(index)
        elif not self.ev_info[index]['on_duty'] and self.user_info[0]['waiting_time'] >= BETA:
            rewards -= self.user_info[0]['waiting_time'] * 2
            self.info['waiting_time'].append(60*self.user_info[0]['waiting_time']/10)
            self.reset_rewards(index)

        check_list['rewards'] = rewards

        # pdb.set_trace()
        return check_list

    # @profile
    def move_aux(self, index, next_node):
        self.update_cs()
        self.update_graph()
        if self.show:
            self.render()
        self.ev_info[index]['NODE_ID'] = next_node
        # node = self.intersections.loc[next_node]
        # self.ev_info[index]['location'] = [node['geometry'].x, node['geometry'].y]
        # self.ev_info[index]['location'] = [node['x'], node['y']]
        self.ev_info[index]['location'] = [self.intersections.at[next_node, 'x'], self.intersections.at[next_node, 'y']]
        self.ev_info[index]['scatter'].set_offsets(
            np.c_[self.ev_info[index]['location'][0], self.ev_info[index]['location'][1]])
        self.discharge_ev(index)
        # pdb.set_trace()
        return self.check_if_reward(index, next_node)

    # @profile
    def move_path(self, index, target, action):
        car_node = target['NODE_ID']
        
        pass_action = MAX_NUMBER_OF_CONNECTIONS + 1
        cs_action = pass_action + NUMBER_OF_PASSENGERS

        # print(f'{action=}, {self.counter=}')
        if action >= pass_action and action <= (cs_action - 1):
            self.picked_user = action % pass_action
            pass_node = self.user_info[self.picked_user]['NODE_ID_source']
            dest_node = self.user_info[self.picked_user]['NODE_ID_destination']
            l_car_to_pass, r_car_to_pass = nx.single_source_dijkstra(self.graph, car_node, pass_node, weight='state')
            self.pre_pickup = len(r_car_to_pass) - 1
            l_car_to_dest, r_car_to_dest = nx.single_source_dijkstra(self.graph, pass_node, dest_node, weight='state')
            self.wt_rectifier = round(l_car_to_dest/TIME_STEP_DURATION)
            if len(r_car_to_pass) > 1:
                r_car_to_pass = r_car_to_pass[1:]
            final_route = np.concatenate([r_car_to_pass, r_car_to_dest[1:]])
            final_length = round(l_car_to_pass/TIME_STEP_DURATION) + self.wt_rectifier
            self.ev_info[index]['on_duty'] = True
        elif action >= cs_action and action <= (cs_action + NUMBER_OF_CS - 1):
            self.ev_info[index]['stand_by'] = False
            self.picked_cs = action % cs_action
            cs_node = self.cs_info[self.picked_cs]['NODE_ID']
            l_car_to_cs, r_car_to_cs = nx.single_source_dijkstra(self.graph, car_node, cs_node, weight='state')
            if len(r_car_to_cs) == 1:
                final_route = r_car_to_cs
            else:
                final_route = r_car_to_cs[1:]
            final_length = round(l_car_to_cs / TIME_STEP_DURATION)

        # pdb.set_trace()

        if self.counter + final_length >= MAX_STEPS:
            self.wt_rectifier = 0
            for i in range(final_length):
                self.counter += 1
                check = self.move_aux(index, final_route[i])
                if target['SOC'] < 0.4 or self.counter == MAX_STEPS:
                    break
        else:
            self.counter += final_length
            for i in final_route:
                check = self.move_aux(index, i)
                if target['SOC'] < 0.4 or self.counter == MAX_STEPS:
                    break

        return check

    # @profile
    def step(self, action):
        # print(self.ev_info)
        # pdb.set_trace()
        if action <= MAX_NUMBER_OF_CONNECTIONS:
            self.update_cs()
            self.counter += 1
            self.update_graph()
            if self.show:
                self.render()
            for k,v in self.ev_info.items():
                next_node = self.move(k, v, action)
                check = self.check_if_reward(k, next_node)
        else:
            for k,v in self.ev_info.items():
                check = self.move_path(k, v, action)

        done = check['if_done']

        if done:
            self.info['served_users'] = self.user_counter
            self.info['count'] = self.counter
            self.info['to_charge'] = self.charging_counter

        # print(self.ev_info)
        reward = check['rewards']

        s_ = self.get_state()

        # pdb.set_trace()

        return s_, reward, done, self.info

    def reset(self):
        if self.show:
            self.render()

        if self.day % self.max_days != 0:
            self.day += 1
        else:
            self.day = 1

        self.traffic = s_df[self.day].groupby('ts')
        self.traffic_time = iter(list(self.traffic.groups.keys()))

        self.ev_info[0]['SOC'] = MAX_SOC
        self.counter = 0
        self.traffic_counter = 0
        self.user_counter = 0
        self.charging_counter = 0
        self.info.clear()
        self.info['waiting_time'] = []
        self.info['cs'] = {}
        self.ev_info[0]['stand_by'] = True

        for k in self.cs_info.keys():
            self.cs_info[k]['waiting_time'] = (random.randrange(0,80,TIME_STEP_DURATION))
            self.info['cs'][k] = 0
        self.reset_rewards(0)
        return self.get_state()

    def render(self):
        if self.normal_render:
            time.sleep(1)
            pass
        self.update()
        self.canvas.draw()