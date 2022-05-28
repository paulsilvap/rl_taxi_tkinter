import random
import pdb
import time
import gym
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import tkinter as tk
import geopandas as gpd

from shapely.geometry import box
from utils import create_map, label_speed, graph_from_gdfs
from gym import spaces
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

np.random.seed(100)

class Env(tk.Tk):
    def __init__(
        self, 
        normal_render = False, 
        show = False,
        days = 2,
        hours_per_day = 24,
        minutes_per_hour = 60, 
        n_passengers = 10, 
        n_ev = 1,
        min_threshold = 10,     # lower bound for waiting time uniform distribution at charging station 
        min_soc = 0,
        max_soc = 58,           # IONIQ 5
        charge_time = 60,       # FAST CHARGER: 50 kW DC station from 10% to 80% in minutes
        discharge_rate = 0.2,
        time_step_duration = 1,
        alpha = 15,
        beta = 20,              # minutes that passenger will wait for e-taxi
        soc_threshold = 0.15,   # threshold for charge allocator
        occurrence_rate = 0.5
        ):
        super().__init__()

        gdf_edges, gdf_nodes, G, gdf_cs, s_df, result = create_map(days)

        self.n_passenger = n_passengers
        self.n_ev = n_ev
        self.min_threshold = min_threshold
        self.min_soc = min_soc
        self.max_soc = max_soc 
        self.charge_time = charge_time
        self.duration = time_step_duration
        self.min = minutes_per_hour
        self.max_steps = (hours_per_day * minutes_per_hour) / self.duration
        self.charge_rate = self.max_soc / self.charge_time
        self.discharge_rate = discharge_rate
        self.alpha = alpha
        self.beta = beta
        self.soc_threshold = soc_threshold
        self.occurrence_rate = occurrence_rate

        self.graph = G
        self.intersections = gdf_nodes
        self.cs = gdf_cs
        self.roads = gdf_edges
        self.aux_road_result = result.drop(columns=result.geometry.name)
        self.speed = s_df

        self.divide_map()

        self.max_neighbours = int((max(sorted((d for _, d in G.degree()), reverse= True)))/2)
        self.n_cs = len(self.cs)
        self.n_roads = self.graph.number_of_edges()
        self.charge_rate = self.max_soc / self.charge_time

        ''' Define the action size and the observation space'''
        self.action_size = self.max_neighbours
        self.observation_space_size = (self.n_roads + (self.n_ev * 2) + (self.n_passenger * 3) + self.n_cs)
        self.observation_space =  spaces.Box(
            low = np.array([-1.] * self.observation_space_size, dtype = np.float32),
            high = np.array([4000000000] * self.observation_space_size, dtype=np.float32),
            dtype = np.float32
        )

        self.cs_nodes = self.cs['NODE_ID'].values
        self.intersection_nodes = self.intersections.index.values
        self.mask_101 = (self.aux_road_result['ROAD_RANK'] == '101').values
        self.mask_103 = ((self.aux_road_result['ROAD_RANK'] == '103') | (self.aux_road_result['ROAD_RANK'] == '106')).values
        self.mask_107 = (self.aux_road_result['ROAD_RANK'] == '107').values
        self.color_mapper = {'green': (0.0, 0.8, 0.0, 1.0), 'yellow': (0.8, 0.8, 0.0, 1.0), 'red': (0.8, 0.0, 0.0, 1.0)}
        self.day = 1
        self.max_days = days
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
        self.user_counter = 0
        self.waiting_time = 0
        self.cs_waiting_time = 0
        self.charging_counter = 0
        self.info = {}
        self.info['waiting_time'] = []
        self.info['ev_wt'] = []
        self.info['ev_ct'] = []
        self.info['cs'] = {}

        self.set_cs()
        self.set_users()
        self.canvas.get_tk_widget().pack(side='top',fill='both', expand=True)

    def color_roads(self):
        return [self.color_mapper[v] for v in self.roads['TC'].values]

    def divide_map(self):
        minx, miny, maxx, maxy = self.roads.total_bounds
        self.city_boundaries = [minx, miny, maxx, maxy]
        self.boundary_1 = [minx, miny, minx + (maxx-minx)/2, miny + (maxy-miny)/2]
        self.boundary_2 = [minx + (maxx-minx)/2, miny + (maxy-miny)/2, maxx, maxy]
        self.boundary_3 = [minx, miny + (maxy-miny)/2, minx + (maxx-minx)/2, maxy]
        self.boundary_4 = [minx + (maxx-minx)/2, miny, maxx, miny + (maxy-miny)/2]
        self.city_box = {
            0: box(self.boundary_1[0], self.boundary_1[1], self.boundary_1[2], self.boundary_1[3]),
            1: box(self.boundary_2[0], self.boundary_2[1], self.boundary_2[2], self.boundary_2[3]),
            2: box(self.boundary_3[0], self.boundary_3[1], self.boundary_3[2], self.boundary_3[3]),
            3: box(self.boundary_4[0], self.boundary_4[1], self.boundary_4[2], self.boundary_4[3])
        }
        self.city_section = {}
        for i in self.city_box.keys():
            self.city_section[i] = gpd.clip(self.intersections, mask=self.city_box[i]).index.values

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
            self.cs_info[k]['waiting_time'] = (random.randrange(self.min_threshold, self.charge_time, self.duration))
            self.cs_info[k]['ev_id'] = []
            self.info['cs'][k] = 0

    def update_cs(self):
        for k,v in self.cs_info.items():
            cs_state = v['waiting_time']
            if cs_state > 0:
                self.cs_info[k]['waiting_time'] = cs_state - self.duration
            elif cs_state <= 0:
                self.cs_info[k]['waiting_time'] = random.randrange(self.min_threshold, self.charge_time, self.duration)

    def random_location(self, target):
        nodes = self.intersection_nodes
        if target == 'source':
            if self.counter > self.min * 5 and self.counter < self.min * 11:
                nodes = [*self.city_section[0], *self.city_section[2]]
            elif self.counter > self.min * 16 and self.counter < self.min * 22:
                nodes = [*self.city_section[1], *self.city_section[3]]
            if self.counter > self.min * 5 and self.counter < self.min * 11:
                nodes = [*self.city_section[1], *self.city_section[3]]
            elif self.counter > self.min * 16 and self.counter < self.min * 22:
                nodes = [*self.city_section[0], *self.city_section[2]]
        
        picked_node = random.choice(nodes)
        while (picked_node in self.cs_nodes) or (picked_node in self.taken_nodes):
            picked_node = random.choice(nodes)
        loc = [self.intersections.at[picked_node, 'x'], self.intersections.at[picked_node, 'y']]
        self.taken_nodes.append(picked_node)
            
        return loc, picked_node

    def _build_canvas(self):
        # self.attributes('-fullscreen', True)
        canvas = FigureCanvasTkAgg(self.fig, self)
        
        self.ev_info = {}
        self.taken_nodes = []
        for i in range(self.n_ev):
            self.ev_info[i] = {}
            location, node = self.random_location('ev')
            self.ev_info[i]['location'] = location
            self.ev_info[i]['NODE_ID'] = node
            self.ev_info[i]['SOC'] = self.max_soc
            self.ev_info[i]['status'] = 'idle'
            self.ev_info[i]['scatter'] = self.ax.scatter(self.ev_info[i]['location'][0], self.ev_info[i]['location'][1], color='b', marker='s')
            self.ev_info[i]['passenger'] = -1
            self.ev_info[i]['route'] = None
            self.ev_info[i]['timer'] = 0
            # self.ev_info[i]['charging_time'] = 0
            self.ev_info[i]['charging'] = False
            self.ev_info[i]['serving'] = False
        
        return canvas

    def set_user_info(self, i):
        if np.random.random() < self.occurrence_rate:
            self.user_info[i]['status'] = 'standing'
            self.user_info[i]['waiting_time'] = 0
            source, source_node = self.random_location('source')
            destination, dest_node = self.random_location('destination')
        else:
            self.user_info[i]['status'] = 'no_call'
            self.user_info[i]['waiting_time'] = -1
            source, source_node = [np.nan, np.nan], -1
            destination, dest_node = [np.nan, np.nan], -1
        
        self.user_info[i]['source'] = source
        self.user_info[i]['NODE_ID_source'] = source_node
        self.user_info[i]['scatter_source'] = self.ax.scatter(source[0], source[1], color='m', marker='H')
        self.user_info[i]['destination'] = destination
        self.user_info[i]['NODE_ID_destination'] = dest_node
        self.user_info[i]['scatter_destination'] = self.ax.scatter(destination[0], destination[1], color='m', marker='X')
            
    def update_user_info(self, k):
        if np.random.random() < self.occurrence_rate:
            self.user_info[k]['status'] = 'standing'
            self.user_info[k]['waiting_time'] = 0
            source, source_node = self.random_location('source')
            destination, dest_node = self.random_location('destination')
        else:
            self.user_info[k]['status'] = 'no_call'
            self.user_info[k]['waiting_time'] = -1
            source, source_node = [np.nan, np.nan], -1
            destination, dest_node = [np.nan, np.nan], -1

        self.user_info[k]['source'] = source
        self.user_info[k]['NODE_ID_source'] = source_node
        self.user_info[k]['scatter_source'].set_offsets(np.c_[source[0], source[1]])
        self.user_info[k]['destination'] = destination
        self.user_info[k]['NODE_ID_destination'] = dest_node
        self.user_info[k]['scatter_destination'].set_offsets(
            np.c_[destination[0], destination[1]])
            
    def set_users(self):
        self.user_info = {}
        
        for i in range(self.n_passenger):
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
            if self.traffic_counter % 5 == 0:
                try:
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
                except StopIteration:
                    print(f'{self.counter=}, {self.traffic_counter=}\n')
                    print(f'{self.info=}\n')
                    print(f'{self.ev_info=}\n')
                    print(f'{self.user_info=}\n')
                    print(f'{self.cs_info=}\n')
            self.traffic_counter += 1

    def discharge_ev(self, index):
        self.ev_info[index]['SOC'] = (int(self.ev_info[index]['SOC']*10) - int((self.discharge_rate)*10))/10

    # @profile
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

        if (self.ev_info[index]['SOC'] < self.max_soc * self.soc_threshold) and (self.counter < self.max_steps):
            check = self.charge_allocator(index, s_)
        elif (self.counter < self.max_steps):
            check = self.passenger_allocator(index, s_)
        else:
            check = self.check_if_reward(index, s_)

        return check

    def charge_allocator(self, index, target):
        cs_loc = []
        aux_cs = {}
        for k, v in self.cs_info.items():
            l_car_to_cs, r_car_to_cs = nx.single_source_dijkstra(self.graph, target, v['NODE_ID'], weight='state')
            aux_cs[k] = [l_car_to_cs, r_car_to_cs]
            cs_loc.append(l_car_to_cs + v['waiting_time'])

        self.ev_info[index]['status'] = 'charging'
        self.picked_cs = cs_loc.index(min(cs_loc))
        l_car_to_cs, r_car_to_cs = aux_cs[self.picked_cs]
        if len(r_car_to_cs) > 1:
            final_route = r_car_to_cs[1:]
        else:
            final_route = []
        if len(final_route) == 0:
            self.ev_info[index]['timer'] = 0
        else:
            self.ev_info[index]['timer'] = self.graph.adj[int(target)][int(final_route[0])][0]['state']
            self.ev_info[index]['route'] = iter(final_route)
            
        check = self.check_if_reward(index, target)
        
        return check

    # @profile
    def serve(self, index, target):
        if self.ev_info[index]['timer'] == 1:
            node = next(self.ev_info[index]['route'])
            route = list(self.ev_info[index]['route'])
            self.ev_info[index]['route'] = iter(route)
            check = self.move_aux(index, node)
            if len(route) > 0: 
                self.ev_info[index]['timer'] = self.graph.adj[int(node)][int(route[0])][0]['state']
        else:
            self.ev_info[index]['timer'] -= 1
            node = target['NODE_ID']
            check = self.check_if_reward(index, node)

        return check

    # @profile
    def passenger_allocator(self, index, target):
        aux_pass = {}
        for k, v in self.user_info.items():
            if v['status'] == 'standing':
                aux_pass[k] = nx.dijkstra_path_length(self.graph, target, v['NODE_ID_source'], weight='state')

        if len(aux_pass) > 0:
            picked_user = min(aux_pass, key=aux_pass.get)
            pass_node = self.user_info[picked_user]['NODE_ID_source']
            dest_node = self.user_info[picked_user]['NODE_ID_destination']
            l_car_to_pass = aux_pass[picked_user]
            r_car_to_pass = nx.dijkstra_path(self.graph, target, pass_node, weight='state')
            pre_pickup = len(r_car_to_pass) - 1
            l_car_to_dest, r_car_to_dest = nx.single_source_dijkstra(self.graph, pass_node, dest_node, weight='state')
            if len(r_car_to_pass) > 1:
                r_car_to_pass = r_car_to_pass[1:]
            else:
                r_car_to_pass = [] 
            final_route = [*r_car_to_pass, *r_car_to_dest[1:]]
            # final_length = l_car_to_dest + l_car_to_pass

            if len(final_route) * self.discharge_rate < self.ev_info[index]['SOC']:
                self.user_info[picked_user]['status'] = 'served'
                self.pre_pickup = pre_pickup
                self.picked_user = picked_user
                self.ev_info[index]['status'] = 'serving'
                self.ev_info[index]['passenger'] = picked_user
                self.ev_info[index]['timer'] = self.graph.adj[int(target)][int(final_route[0])][0]['state']
                self.ev_info[index]['route'] = iter(final_route)
            
        check = self.check_if_reward(index, target)
        
        return check

    def get_state(self):
        ev = [[v['SOC'],float(v['NODE_ID'])] for v in self.ev_info.values()]
        ev = [d for data in ev for d in data]
        user = [[float(v['NODE_ID_source']), float(v['NODE_ID_destination']), v['waiting_time']] for v in self.user_info.values()]
        user = [d for data in user for d in data]
        cs = [v['waiting_time'] for v in self.cs_info.values()]
        traffic = [data['state'] for _, _, data in self.graph.edges(data=True)]
        return np.array([*ev, *user, *cs, *traffic], dtype=np.float32)

    # @profile
    def reset_rewards(self, index):

        for k in self.user_info.keys():
            self.user_info[k]['waiting_time'] = 0
        self.aux_counter = self.counter
        if self.ev_info[index]['status'] != 'no_battery':
            self.ev_info[index]['serving'] = False
            self.ev_info[index]['status'] = 'idle'
            self.ev_info[index]['passenger'] = -1

        self.taken_nodes = [self.ev_info[index]['NODE_ID']]

        for k in self.user_info.keys():
            self.update_user_info(k)

    def charge(self, index, target):
        if self.cs_info[self.picked_cs]['NODE_ID'] == target['NODE_ID']:
            check = self.check_if_reward(index, target['NODE_ID'])
        else:
            check = self.serve(index, target)

        return check

    # @profile
    def check_if_reward(self, index, next_node):
        check_list = {}
        check_list['if_done'] = False
        check_list['if_goal'] = False
        rewards = 0

        if not self.ev_info[index]['serving']:
            for k, v in self.user_info.items():
                if v['status'] != 'no_call':
                    self.user_info[k]['waiting_time'] = self.counter if self.aux_counter == 0 else self.counter - self.aux_counter

        if self.ev_info[index]['passenger'] != -1:
            picked_user = self.ev_info[index]['passenger']

            if self.ev_info[index]['status'] == 'serving' and self.user_info[picked_user]['NODE_ID_source'] == next_node:
                
                self.ev_info[index]['serving'] = True
                self.info['waiting_time'].append(self.user_info[picked_user]['waiting_time']*self.duration)
            if (
                self.ev_info[index]['serving'] and
                self.user_info[picked_user]['NODE_ID_destination'] == next_node):

                self.user_info[picked_user]['status'] = 'standing'
                self.user_counter += 1
                rewards += self.alpha if self.pre_pickup == 0 else (self.alpha / self.pre_pickup)
                check_list['if_goal'] = True
                self.ev_info[index]['serving'] = False
                self.ev_info[index]['status'] = 'idle'

        if (
            (not self.ev_info[index]['serving']) and
            (self.ev_info[index]['status']) == 'charging' and 
            self.cs_info[self.picked_cs]['NODE_ID'] == next_node):

            if self.charging_counter == len(self.info['ev_wt']) and not self.ev_info[index]['charging']:
                self.info['ev_wt'].append(self.cs_info[self.picked_cs]['waiting_time'])

            if self.cs_info[self.picked_cs]['waiting_time'] == 0:
                self.ev_info[index]['charging'] = True
                charging_time = int(np.ceil((self.max_soc - self.ev_info[index]['SOC'])/self.charge_rate))
                self.info['ev_ct'].append(charging_time)
                self.cs_info[self.picked_cs]['waiting_time'] = charging_time * self.duration

            if self.ev_info[index]['charging']:
                self.ev_info[index]['SOC'] = min(self.ev_info[index]['SOC'] + self.charge_rate, self.max_soc)
                    
            if self.ev_info[index]['SOC'] == self.max_soc:
                self.info['cs'][self.picked_cs] += 1
                self.charging_counter += 1
                self.ev_info[index]['charging'] = False
                self.ev_info[index]['status'] = 'idle'

        if self.ev_info[index]['SOC'] == self.min_soc and self.ev_info[index]['status'] != 'no_battery':
            self.ev_info[index]['status'] = 'no_battery'
            self.info['active'] = self.counter
            rewards -= (self.max_steps - self.counter) * 2

        if self.counter == self.max_steps:
            check_list['if_done'] = True
        
        if check_list['if_goal']:
            self.reset_rewards(index)
        elif (
            (self.ev_info[index]['status'] == 'idle' or self.ev_info[index]['status'] == 'no_battery') and 
            any(v['waiting_time'] >= self.beta for v in self.user_info.values())):
            for k, v in self.user_info.items():
                if v['waiting_time'] >= 0:
                    idx = k
                    break
            rewards -= self.user_info[idx]['waiting_time']
            self.info['waiting_time'].append(self.user_info[idx]['waiting_time'] * self.duration)
            # print('no_patience')
            # if np.random.random() < self.occurrence_rate:
                # print('after_no_patience')
            self.reset_rewards(index)
        elif (all(v['status'] == 'no_call' for v in self.user_info.values())):
            # print('no_call')
            if np.random.random() < self.occurrence_rate:
                # print('after_no_call')
                self.reset_rewards(index)

        check_list['rewards'] = rewards

        return check_list

    # @profile
    def move_aux(self, index, next_node):
        self.update_cs()
        self.update_graph()
        if self.show:
            self.render()
        self.ev_info[index]['NODE_ID'] = next_node
        self.ev_info[index]['location'] = [self.intersections.at[next_node, 'x'], self.intersections.at[next_node, 'y']]
        self.ev_info[index]['scatter'].set_offsets(
            np.c_[self.ev_info[index]['location'][0], self.ev_info[index]['location'][1]])
        self.discharge_ev(index)
        return self.check_if_reward(index, next_node)

    # @profile
    def step(self, action):
        self.update_cs()
        self.counter += 1
        self.update_graph()
        if self.show:
            self.render()
        for k,v in self.ev_info.items():
            if v['status'] == 'idle':
                check = self.move(k, v, action)
            elif v['status'] == 'serving':
                check = self.serve(k, v)
            elif v['status'] == 'charging':
                check = self.charge(k, v)
            else:
                check = self.check_if_reward(k, v['NODE_ID'])

        done = check['if_done']

        if done:
            self.info['served_users'] = self.user_counter
            self.info['count'] = self.counter
            self.info['to_charge'] = self.charging_counter
            self.info['total_calls'] = len(self.info['waiting_time'])
            self.info['response_rate'] = self.user_counter/len(self.info['waiting_time']) if len(self.info['waiting_time']) > 0 else 0
            if 'active' not in self.info.keys():
                self.info['active'] = self.counter

        reward = check['rewards']

        s_ = self.get_state()

        return s_, reward, done, self.info

    def reset(self):
        if self.show:
            self.render()

        if self.day % self.max_days != 0:
            self.day += 1
        else:
            self.day = 1

        self.traffic = self.speed[self.day].groupby('ts')
        self.traffic_time = iter(list(self.traffic.groups.keys()))

        self.counter = 0
        self.traffic_counter = 0
        self.user_counter = 0
        self.charging_counter = 0
        self.info.clear()
        self.info['waiting_time'] = []
        self.info['ev_wt'] = []
        self.info['cs'] = {}
        self.info['ev_ct'] = []
        for k in self.ev_info.keys():
            self.ev_info[k]['SOC'] = self.max_soc
            self.ev_info[k]['timer'] = 0
            self.ev_info[k]['route'] = None
            self.ev_info[k]['charging'] = False
            self.ev_info[k]['status'] = 'idle'
            self.reset_rewards(k)

        for k in self.cs_info.keys():
            self.cs_info[k]['waiting_time'] = (random.randrange(self.min_threshold, self.charge_time, self.duration))
            self.info['cs'][k] = 0

        return self.get_state()

    def render(self):
        if self.normal_render:
            time.sleep(0.1)
        self.update()
        self.canvas.draw()

class GymEnv(gym.Env):
    def __init__(
            self,
            normal_render = False, 
            show = False,
            days = 2,
            hours_per_day = 24,
            minutes_per_hour = 60, 
            n_passengers = 10, 
            n_ev = 1,
            min_threshold = 10,    
            min_soc = 0,
            max_soc = 58,           
            charge_time = 60,       
            discharge_rate = 0.2,
            time_step_duration = 1,
            alpha = 15,
            beta = 20,              
            soc_threshold = 0.15,
            occurrence_rate = 0.5  
        ):
        super().__init__()
        self.game = Env(
            normal_render = normal_render, 
            show = show,
            days = days,
            hours_per_day = hours_per_day,
            minutes_per_hour = minutes_per_hour, 
            n_passengers = n_passengers, 
            n_ev = n_ev,
            min_threshold = min_threshold,    
            min_soc = min_soc,
            max_soc = max_soc,          
            charge_time = charge_time,      
            discharge_rate = discharge_rate,
            time_step_duration = time_step_duration,
            alpha = alpha,
            beta = beta,
            soc_threshold = soc_threshold,
            occurrence_rate= occurrence_rate
        )
        self.observation_space = self.game.observation_space
        self.action_space = spaces.Discrete(self.game.action_size)

    def step(self, action):
        obs, reward, done, info = self.game.step(action)
        return obs, reward, done, info

    def render(self):
        self.game.render()

    def reset(self):
        obs = self.game.reset()
        return obs
