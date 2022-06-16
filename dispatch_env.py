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
from utils import create_map, label_speed, graph_from_gdfs, truncated_normal_distribution
from gym import spaces
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

np.random.seed(100)
random.seed(100)

class Env(tk.Tk):
    # @profile
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
        occurrence_rate = 0.5,
        max_number_of_requests = 100
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
        self.max_n_of_requests = max_number_of_requests

        self.graph = G
        self.intersections = gdf_nodes
        self.cs = gdf_cs
        self.roads = gdf_edges
        self.aux_road_result = result.drop(columns=result.geometry.name)
        self.speed = s_df
        y1 = truncated_normal_distribution(0, 18*60, 9*60, 4*60, 1, int(self.max_n_of_requests/2))
        y2 = truncated_normal_distribution(14*60, 24*60, 16*60, 5*60, 1, int(self.max_n_of_requests/2))
        # self.pass_distribution = random.sample(range(int(self.max_steps)), self.max_n_of_requests)
        self.pass_distribution = np.sort([*y1, *y2])

        self.divide_map()

        self.node_neighbours = {n:int(d/2) for n, d in G.degree()}
        self.max_neighbours = int((max(sorted((d for _, d in G.degree()), reverse= True)))/2)
        self.n_cs = len(self.cs)
        self.n_roads = self.graph.number_of_edges()
        self.charge_rate = self.max_soc / self.charge_time

        ''' Define the action size and the observation space'''
        self.action_size = self.max_neighbours
        self.observation_space_size = (self.n_roads + (self.n_ev * 3) + (self.n_passenger * 4) + self.n_cs)
        self.observation_space =  spaces.Box(
            low = np.array([-1.] * self.observation_space_size, dtype = np.float32),
            high = np.array([4000000000] * self.observation_space_size, dtype=np.float32),
            dtype = np.float32
        )

        self.ev_status = {
            'idle': 0,
            'serving': 1,
            'charging': 2,
            'no_battery': 3, 
        }

        self.user_status = {
            'standing': 0,
            'served': 1,
            'no_call': 2,
        }

        self.cs_nodes = self.cs['NODE_ID'].values
        self.intersection_nodes = self.intersections.index.values
        self.mask_101 = (self.aux_road_result['ROAD_RANK'] == '101').values
        self.mask_103 = ((self.aux_road_result['ROAD_RANK'] == '103') | (self.aux_road_result['ROAD_RANK'] == '106')).values
        self.mask_107 = (self.aux_road_result['ROAD_RANK'] == '107').values
        self.color_mapper = {'green': (0.0, 0.8, 0.0, 1.0), 'yellow': (0.8, 0.8, 0.0, 1.0), 'red': (0.8, 0.0, 0.0, 1.0)}
        self.day = 1
        self.max_days = days
        self.traffic = s_df[self.day].groupby('ts')
        self.traffic_time = iter(self.traffic.groups.keys())
        self.daily_graph = {}
        self.daily_roads = {}
        self.daily_traffic_state = {}
        self.daily_passenger_calls = {}
        self.daily_cs_waiting_time = {}
        for d in range(1, self.max_days+1):
            self.daily_graph[d] = {}
            self.daily_roads[d] = {}
            self.daily_traffic_state[d] = {}
            self.daily_passenger_calls[d] = {}
            self.daily_cs_waiting_time[d] = {}
            for i in range(self.n_passenger):
                self.daily_passenger_calls[d][i] = {}
            for i in range(self.n_cs):
                self.daily_cs_waiting_time[d][i] = {}

        self.info = {}
        self.pass_set = set()
        
        self.base_variables()

        self.title('Jeonju')
        self.create_plot()
        self.canvas = self._build_canvas()
        self.normal_render = normal_render
        self.show = show

        self.set_cs()
        self.set_users()
        self.canvas.get_tk_widget().pack(side='top',fill='both', expand=True)

    # @profile
    def color_roads(self):
        return [self.color_mapper[v] for v in self.roads['TC'].values]

    # @profile
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

    def base_variables(self):
        self.counter = 0

        self.user_counter = 0
        self.user_dropped_counter = 0
        self.no_pass_counter = 0
        self.total_request_counter = 0
        
        self.traffic_counter = 0
        self.charging_counter = 0

        self.info['waiting_time'] = []
        self.info['ev_wt'] = []
        self.info['ev_ct'] = []
        self.info['driving_to_cs'] = []
        self.info['driving_to_pass'] = []
        self.info['real_driving_to_pass'] = []
        self.info['ev_idle_time'] = 0
        self.info['ev_serving_time'] = 0
        self.info['ev_charging_time'] = 0
        self.info['ev_no_battery_time'] = 0
        self.info['ev_total_distance'] = 0
        # self.info['real_driving_to_cs'] = []
        self.info['cs'] = {}
        self.info['active'] = {}
        # self.info['ev_total_trip'] = {}
        self.info['ev_where_at_user'] = {}

    def base_variables_ev(self, i):
        self.ev_info[i]['SOC'] = self.max_soc
        self.ev_info[i]['status'] = 'idle'
        self.ev_info[i]['passenger'] = -1
        self.ev_info[i]['cs'] = -1
        self.ev_info[i]['driving_to_pass'] = -1
        self.ev_info[i]['driving_to_cs'] = -1
        self.ev_info[i]['route'] = None
        self.ev_info[i]['single_move'] = -1
        self.ev_info[i]['timer'] = 0
        self.ev_info[i]['idle_timer'] = 0
        self.ev_info[i]['serving_timer'] = 0
        self.ev_info[i]['charging_timer'] = 0
        self.ev_info[i]['no_battery_timer'] = 0
        self.ev_info[i]['charging'] = False
        self.ev_info[i]['serving'] = False
        self.ev_info[i]['waiting'] = False
        self.ev_info[i]['where_at_user'] = []
        self.ev_info[i]['distance'] = 0

    # @profile
    def create_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(10,7))
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        self.ax = self.roads['geometry'].plot(ax=self.ax, color=self.color_roads(), lw=1, alpha=None, zorder=-1)
        self.roads = self.roads.drop(columns=self.roads.geometry.name)
        # self.ax.scatter(self.intersections['geometry'].x, self.intersections['geometry'].y, s = 2, zorder=-1, color='k')
        self.intersections = self.intersections.drop(columns=self.intersections.geometry.name)
        # self.ax.set(xlim = (self.city_boundaries[0], self.city_boundaries[2]), ylim = (self.city_boundaries[1], self.city_boundaries[3]))
        self.fig.tight_layout()
        self.ax.scatter(self.cs['geometry'].x, self.cs['geometry'].y, color='xkcd:brown')

    # @profile
    def set_cs(self):
        self.cs_info = self.cs.T.to_dict()

        for k in self.cs_info.keys():
            self.cs_info[k]['waiting_time'] = random.randrange(self.min_threshold, self.charge_time, self.duration)
            self.cs_info[k]['ev_id'] = []
            self.info['cs'][k] = 0

    # @profile
    def update_cs(self):
        for k,v in self.cs_info.items():
            cs_state = v['waiting_time']
            if cs_state > 0:
                self.cs_info[k]['waiting_time'] = cs_state - self.duration
            elif cs_state <= 0:
                if self.counter in self.daily_cs_waiting_time[self.day][k].keys():
                    self.cs_info[k]['waiting_time'] = self.daily_cs_waiting_time[self.day][k][self.counter]
                else:    
                    self.cs_info[k]['waiting_time'] = random.randrange(self.min_threshold, self.charge_time, self.duration)
                    self.daily_cs_waiting_time[self.day][k][self.counter] = self.cs_info[k]['waiting_time']

    # @profile
    def random_location(self, target):
        nodes = self.intersection_nodes
        if target == 'source':
            if self.counter > self.min * 5 and self.counter < self.min * 11:
                nodes = [*self.city_section[0], *self.city_section[2]]
            elif self.counter > self.min * 16 and self.counter < self.min * 22:
                nodes = [*self.city_section[1], *self.city_section[3]]
        elif target == 'destination':
            if self.counter > self.min * 5 and self.counter < self.min * 11:
                nodes = [*self.city_section[1], *self.city_section[3]]
            elif self.counter > self.min * 16 and self.counter < self.min * 22:
                nodes = [*self.city_section[0], *self.city_section[2]]
        
        picked_node = random.choice(nodes)
        while (picked_node in self.cs_nodes) or (picked_node in self.taken_nodes):
            picked_node = random.choice(nodes)
        self.taken_nodes.add(picked_node)
            
        return picked_node
    
    # @profile
    def _build_canvas(self):
        # self.attributes('-fullscreen', True)
        canvas = FigureCanvasTkAgg(self.fig, self)
        
        self.ev_info = {}
        self.taken_nodes = set()
        for i in range(self.n_ev):
            self.ev_info[i] = {}
            node = self.random_location('ev')
            location = [self.intersections.at[node, 'x'], self.intersections.at[node, 'y']]
            self.ev_info[i]['NODE_ID'] = node
            self.ev_info[i]['scatter'] = self.ax.scatter(location[0], location[1], color='b', marker='s')
            self.base_variables_ev(i)
        
        return canvas

    # @profile
    def set_user_info(self, i):
        if np.random.random() < self.occurrence_rate:
            self.user_info[i]['status'] = 'standing'
            self.user_info[i]['waiting_time'] = 0
            self.user_info[i]['counter'] = 0
            source_node = self.random_location('source')
            dest_node = self.random_location('destination')
            source = [self.intersections.at[source_node, 'x'],self.intersections.at[source_node, 'y']]
            destination = [self.intersections.at[dest_node, 'x'],self.intersections.at[dest_node, 'y']]
        else:
            self.user_info[i]['status'] = 'no_call'
            self.user_info[i]['waiting_time'] = -1
            self.user_info[i]['counter'] = -1
            source, source_node = [np.nan, np.nan], -1
            destination, dest_node = [np.nan, np.nan], -1
        
        
        self.user_info[i]['NODE_ID_source'] = source_node
        self.user_info[i]['scatter_source'] = self.ax.scatter(source[0], source[1], color='m', marker='H')
        self.user_info[i]['NODE_ID_destination'] = dest_node
        self.user_info[i]['scatter_destination'] = self.ax.scatter(destination[0], destination[1], color='m', marker='X')
        self.user_info[i]['picked_up'] = False
    
    # @profile
    def update_user_info(self, i):
        if i not in self.pass_set:
            if self.counter in self.pass_distribution and np.random.random() < self.occurrence_rate: 
                self.taken_nodes.discard(self.user_info[i]['NODE_ID_source'])
                self.taken_nodes.discard(self.user_info[i]['NODE_ID_destination'])
                self.total_request_counter += 1
                self.user_info[i]['status'] = 'standing'
                self.user_info[i]['waiting_time'] = 0
                self.user_info[i]['counter'] = self.counter
                if self.counter in self.daily_passenger_calls[self.day][i].keys():
                    source_node, dest_node = self.daily_passenger_calls[self.day][i][self.counter]
                    self.taken_nodes.add(source_node)
                    self.taken_nodes.add(dest_node)
                else:
                    source_node = self.random_location('source')
                    dest_node = self.random_location('destination')
                    self.daily_passenger_calls[self.day][i][self.counter] = [source_node, dest_node]
                if self.show:
                    source = [self.intersections.at[source_node, 'x'],self.intersections.at[source_node, 'y']]
                    destination = [self.intersections.at[dest_node, 'x'],self.intersections.at[dest_node, 'y']]                
            else:
                self.user_info[i]['status'] = 'no_call'
                self.user_info[i]['waiting_time'] = -1
                self.user_info[i]['counter'] = -1
                dest_node = source_node = -1
                if self.show:
                    destination = source = [np.nan, np.nan]

            self.user_info[i]['NODE_ID_source'] = source_node
            self.user_info[i]['NODE_ID_destination'] = dest_node
            if self.show:
                self.user_info[i]['scatter_source'].set_offsets(np.c_[source[0], source[1]])
                self.user_info[i]['scatter_destination'].set_offsets(np.c_[destination[0], destination[1]])
            self.user_info[i]['picked_up'] = False

    # @profile
    def set_users(self):
        self.user_info = {}
        
        for i in range(self.n_passenger):
            self.user_info[i] = {}
            self.set_user_info(i)

    # @profile
    def update_users(self):
        self.taken_nodes.clear()
        for v in self.ev_info.values():
            self.taken_nodes.add(v['NODE_ID'])

        for k in self.user_info.keys():
            self.update_user_info(k)

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

    # @profile
    def discharge_ev(self, index, current_node, next_node):
        length = self.graph.adj[current_node][next_node][0]['LENGTH']
        km = length / 500
        self.ev_info[index]['distance'] = self.ev_info[index]['distance'] + km
        consumption = km * self.discharge_rate
        self.ev_info[index]['SOC'] = self.ev_info[index]['SOC'] - round(consumption, 2)

    # @profile
    def move(self, index, target, action):
        s = target['NODE_ID']
        sample = self.graph.neighbors(s)
        sample = list(sample)
        if self.ev_info[index]['single_move'] == -1:
            self.ev_info[index]['single_move'] = sample[action]
            self.ev_info[index]['timer'] = self.graph.adj[int(s)][sample[action]][0]['state']

        if self.ev_info[index]['timer'] > 1:
            self.ev_info[index]['timer'] -= 1
            check = self.check_if_reward(index, target['NODE_ID'])
        elif self.ev_info[index]['timer'] == 1:
            s_ = self.ev_info[index]['single_move']
            self.ev_info[index]['single_move'] = -1
            self.ev_info[index]['NODE_ID'] = s_
            if self.show:
                loc = [self.intersections.at[s_, 'x'], self.intersections.at[s_, 'y']]
                self.ev_info[index]['scatter'].set_offsets(np.c_[loc[0], loc[1]])
                self.discharge_ev(index, s, s_)

            if (self.ev_info[index]['SOC'] < self.max_soc * self.soc_threshold) and (self.counter < self.max_steps):
                check = self.charge_allocator(index, s_)
            elif (self.counter < self.max_steps):
                check = self.passenger_allocator(index, s_)
            else:
                check = self.check_if_reward(index, s_)

        return check

    # @profile
    def charge_allocator(self, index, target):
        aux_cs = {}
        for k, v in self.cs_info.items():
            l_car_to_cs = nx.dijkstra_path_length(self.graph, target, v['NODE_ID'], weight='state')
            aux_cs[k] = l_car_to_cs + v['waiting_time']

        picked_cs = min(aux_cs, key=aux_cs.get)
        self.ev_info[index]['status'] = 'charging'
        self.ev_info[index]['cs'] = picked_cs
        l_car_to_cs = aux_cs[picked_cs]
        self.ev_info[index]['driving_to_cs'] = l_car_to_cs
        self.info['driving_to_cs'].append(l_car_to_cs)
        r_car_to_cs = nx.dijkstra_path(self.graph, target, self.cs_info[picked_cs]['NODE_ID'], weight='state')
        if len(r_car_to_cs) > 1:
            final_route = r_car_to_cs[1:]
        else:
            final_route = []
        if len(final_route) == 0:
            self.ev_info[index]['timer'] = 0
        else:
            self.ev_info[index]['timer'] = self.graph.adj[int(target)][int(final_route[0])][0]['state']
            self.ev_info[index]['route'] = iter(final_route)
            
        return self.check_if_reward(index, target)

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
            l_car_to_dest, r_car_to_dest = nx.single_source_dijkstra(self.graph, pass_node, dest_node, weight='state')
            if len(r_car_to_pass) > 1:
                r_car_to_pass = r_car_to_pass[1:]
            else:
                r_car_to_pass = [] 
            final_route = [*r_car_to_pass, *r_car_to_dest[1:]]
            # final_length = l_car_to_dest + l_car_to_pass

            if len(final_route) * self.discharge_rate < self.ev_info[index]['SOC']:
                self.user_info[picked_user]['status'] = 'served'
                self.ev_info[index]['driving_to_pass'] = l_car_to_pass
                # self.info['driving_to_pass'].append(l_car_to_pass)
                self.ev_info[index]['status'] = 'serving'
                self.ev_info[index]['passenger'] = picked_user
                self.ev_info[index]['where_at_user'].append(target)
                self.pass_set.add(picked_user)
                self.ev_info[index]['timer'] = self.graph.adj[int(target)][int(final_route[0])][0]['state']
                self.ev_info[index]['route'] = iter(final_route)
            
        return self.check_if_reward(index, target)

    def status(self, obj, status):
        if obj == 'ev':
            _status = self.ev_status[status]
        elif obj == 'user':
            _status = self.user_status[status]
        return _status 

    # @profile
    def get_state(self):
        ev = [[v['SOC'],float(v['NODE_ID']), self.status('ev',v['status'])] for v in self.ev_info.values()]
        ev = [d for data in ev for d in data]
        user = [[float(v['NODE_ID_source']), float(v['NODE_ID_destination']), 
        v['waiting_time'], self.status('user',v['status'])] for v in self.user_info.values()]
        user = [d for data in user for d in data]
        cs = [v['waiting_time'] for v in self.cs_info.values()]
        if self.counter in self.daily_traffic_state[self.day].keys():
            traffic = self.daily_traffic_state[self.day][self.counter]
        else:
            traffic = [data['state'] for _, _, data in self.graph.edges(data=True)]
            self.daily_traffic_state[self.day][self.counter] = traffic
        return np.array([*ev, *user, *cs, *traffic], dtype=np.float32)

    # @profile
    def charge(self, index, target):
        if self.cs_info[self.ev_info[index]['cs']]['NODE_ID'] == target['NODE_ID']:
            check = self.check_if_reward(index, target['NODE_ID'])
        else:
            check = self.serve(index, target)

        return check

    # @profile
    def check_if_reward(self, index, next_node):
        check_list = {}
        check_list['if_done'] = False
        rewards = 0

        # if self.ev_info[index]['NODE_ID'] == next_node and self.ev_info[index]['status'] == 'idle':
        #     rewards -= 0.15

        if self.ev_info[index]['passenger'] != -1:
            picked_user = self.ev_info[index]['passenger']

            if (not self.user_info[picked_user]['picked_up'] and 
                self.user_info[picked_user]['waiting_time'] >= self.beta):
                self.info['waiting_time'].append(self.user_info[picked_user]['waiting_time'] * self.duration)
                rewards -= self.user_info[picked_user]['waiting_time'] * 2
                self.ev_info[index]['driving_to_pass'] = -1
                self.ev_info[index]['serving'] = False
                self.ev_info[index]['status'] = 'idle'
                self.ev_info[index]['passenger'] = -1
                self.pass_set.remove(picked_user)
                self.user_dropped_counter += 1
                self.update_user_info(picked_user)

            if (self.ev_info[index]['status'] == 'serving' and 
                self.user_info[picked_user]['NODE_ID_source'] == next_node and
                not self.ev_info[index]['serving']):
                
                self.ev_info[index]['serving'] = True
                self.user_info[picked_user]['picked_up'] = True
                self.info['waiting_time'].append(self.user_info[picked_user]['waiting_time']*self.duration)
            
            if (self.ev_info[index]['serving'] and self.user_info[picked_user]['NODE_ID_destination'] == next_node):

                self.user_counter += 1
                driving_to_pass = self.ev_info[index]['driving_to_pass']
                self.info['driving_to_pass'].append(driving_to_pass)
                self.info['real_driving_to_pass'].append(self.user_info[picked_user]['waiting_time']*self.duration)
                rewards += self.alpha if driving_to_pass == 0 else (self.alpha / driving_to_pass)
                self.ev_info[index]['driving_to_pass'] = -1
                self.ev_info[index]['serving'] = False
                self.ev_info[index]['status'] = 'idle'
                self.ev_info[index]['passenger'] = -1
                self.pass_set.remove(picked_user)
                self.update_user_info(picked_user)
                
        if self.ev_info[index]['cs'] != -1:
            picked_cs = self.ev_info[index]['cs']

            if (self.ev_info[index]['status'] == 'charging' and self.cs_info[picked_cs]['NODE_ID'] == next_node):
                
                # if self.ev_info[index]['driving_to_cs'] != -1:
                #     driving_to_cs = self.ev_info[index]['driving_to_cs']
                #     rewards += self.alpha if driving_to_cs == 0 else (self.alpha / driving_to_cs)
                #     self.ev_info[index]['driving_to_cs'] = -1

                if (not self.ev_info[index]['waiting'] and not self.ev_info[index]['charging']):
                    self.info['ev_wt'].append(self.cs_info[picked_cs]['waiting_time'])
                    self.ev_info[index]['waiting'] = True

                if self.cs_info[picked_cs]['waiting_time'] == 0:
                    self.ev_info[index]['charging'] = True
                    self.ev_info[index]['waiting'] = False
                    charging_time = int(np.ceil(((self.max_soc)- self.ev_info[index]['SOC'])/self.charge_rate))
                    self.info['ev_ct'].append(charging_time)
                    self.cs_info[picked_cs]['waiting_time'] = charging_time * self.duration

                if self.ev_info[index]['charging']:
                    self.ev_info[index]['SOC'] = min(self.ev_info[index]['SOC'] + self.charge_rate, self.max_soc)
                        
                if self.ev_info[index]['SOC'] == self.max_soc:
                    self.info['cs'][picked_cs] += 1
                    self.charging_counter += 1
                    self.ev_info[index]['charging'] = False
                    self.ev_info[index]['status'] = 'idle'
                    self.ev_info[index]['cs'] = -1

        if self.ev_info[index]['SOC'] == self.min_soc and self.ev_info[index]['status'] != 'no_battery':
            self.ev_info[index]['status'] = 'no_battery' 
            self.info['active'][index] = self.counter
            rewards -= (self.max_steps - self.counter)

        if self.counter == self.max_steps:
            check_list['if_done'] = True
            if self.ev_info[index]['charging']:
                self.info['cs'][self.ev_info[index]['cs']] += 1
                self.charging_counter += 1
            if index not in self.info['active'].keys():
                self.info['active'][index] = self.counter
            # self.info['ev_total_trip'][index] = self.ev_info[index]['total_trip']
            self.info['ev_where_at_user'][index] = self.ev_info[index]['where_at_user']

        if all(v['status'] == 'standing' for v in self.user_info.values()):
            if all(v['waiting_time'] >= self.beta for v in self.user_info.values()):
                # rewards -= self.beta*self.n_passenger
                rewards -= self.beta * 2
                for v in self.user_info.values():
                    self.info['waiting_time'].append(v['waiting_time'] * self.duration)
                self.update_users()

        ''' Check if any 'standing' passenger has been waiting for more than beta.
        If this is the case, replace those passenger requests.'''
        for k, v in self.user_info.items():
            if k not in self.pass_set:
                if v['waiting_time'] >= self.beta:
                    self.info['waiting_time'].append(v['waiting_time'] * self.duration)
                    self.update_user_info(k)

        check_list['rewards'] = rewards

        return check_list

    # @profile
    def move_aux(self, index, next_node):
        current_node = self.ev_info[index]['NODE_ID']
        self.ev_info[index]['NODE_ID'] = next_node
        if self.show:
            loc = [self.intersections.at[next_node, 'x'], self.intersections.at[next_node, 'y']]
            self.ev_info[index]['scatter'].set_offsets(np.c_[loc[0], loc[1]])
        self.discharge_ev(index, current_node, next_node)
        return self.check_if_reward(index, next_node)

    # @profile
    def step(self, action):
        self.update_cs()
        self.counter += 1
        self.update_graph()
        if self.show:
            self.render()

        if (all(v['status'] == 'no_call' for v in self.user_info.values())):
            self.no_pass_counter += 1
            self.update_users()
        
        for k, v in self.user_info.items():
            if not v['picked_up'] and v['status'] != 'no_call':
                self.user_info[k]['waiting_time'] = self.counter if v['counter'] == 0 else self.counter - v['counter']

        check = {}

        if self.n_ev == 1:
            action = [action]
        
        for i, action in enumerate(action):
            '''Action masking to avoid invalid actions'''
            action = action % (self.node_neighbours[self.ev_info[i]['NODE_ID']])
            if self.ev_info[i]['status'] == 'idle':
                self.ev_info[i]['idle_timer'] += 1
                check[i] = self.move(i, self.ev_info[i], action)
            elif self.ev_info[i]['status'] == 'serving':
                self.ev_info[i]['serving_timer'] += 1
                check[i] = self.serve(i, self.ev_info[i])
            elif self.ev_info[i]['status'] == 'charging':
                self.ev_info[i]['charging_timer'] += 1
                check[i] = self.charge(i, self.ev_info[i])
            else:
                self.ev_info[i]['no_battery_timer'] += 1
                check[i] = self.check_if_reward(i, self.ev_info[i]['NODE_ID'])

        done = all(v['if_done'] for v in check.values())

        if done:
            '''General Environment Information'''
            self.info['no_pass'] = self.no_pass_counter/self.counter
            self.info['count'] = self.counter

            '''Results related to E-taxi'''
            self.info['to_charge'] = self.charging_counter
            self.info['avg_driving_to_pass'] = np.mean(self.info['driving_to_pass']) if len(self.info['driving_to_pass']) > 0 else np.nan
            self.info['real_avg_driving_to_pass'] = np.mean(self.info['real_driving_to_pass']) if len(self.info['real_driving_to_pass']) > 0 else np.nan
            self.info['avg_driving_to_cs'] = np.mean(self.info['driving_to_cs']) if self.charging_counter > 0 else np.nan
            # self.info['real_avg_driving_to_cs'] = np.mean(self.info['real_driving_to_cs']) if len(self.info['real_driving_to_cs']) > 0 else np.nan
            self.info['avg_ev_ct'] = np.mean(self.info['ev_ct']) if self.charging_counter > 0 else np.nan
            self.info['avg_ev_wt'] = np.mean(self.info['ev_wt']) if self.charging_counter > 0 else np.nan
            for k, v in self.ev_info.items():
                self.info['ev_idle_time'] += (v['idle_timer']/self.info['active'][k] if self.info['active'][k] > 0 else np.nan)
                self.info['ev_serving_time'] += (v['serving_timer']/self.info['active'][k] if self.info['active'][k] > 0 else np.nan)
                self.info['ev_charging_time'] += (v['charging_timer']/self.info['active'][k] if self.info['active'][k] > 0 else np.nan)
                self.info['ev_no_battery_time'] += (v['no_battery_timer']/self.info['active'][k] if self.info['active'][k] > 0 else np.nan)
                self.info['ev_total_distance'] += v['distance']
            self.info['avg_idle_time'] = (self.info['ev_idle_time']/self.n_ev)
            self.info['avg_serving_time'] = (self.info['ev_serving_time']/self.n_ev)
            self.info['avg_charging_time'] = (self.info['ev_charging_time']/self.n_ev)
            self.info['avg_no_battery_time'] = (self.info['ev_no_battery_time']/self.n_ev)
            
            '''Results related to passenger'''
            self.info['served_users'] = self.user_counter
            self.info['non_served_users'] = self.user_dropped_counter
            self.info['total_requests'] = self.total_request_counter
            self.info['total_calls'] = len(self.info['waiting_time'])
            self.info['request_answered'] = self.user_dropped_counter + self.user_counter 
            self.info['answered_rate'] = self.info['request_answered']/self.total_request_counter if self.total_request_counter > 0 else 0
            self.info['successful_rate'] = self.user_counter / self.info['request_answered'] if self.total_request_counter > 0 else 0
            self.info['unsuccessful_rate'] = self.user_dropped_counter / self.info['request_answered'] if self.total_request_counter > 0 else 0

        reward = sum([v['rewards'] for v in check.values()]) 

        s_ = self.get_state()

        return s_, reward, done, self.info

    # @profile
    def reset(self):
        if self.show:
            self.render()

        if self.day % self.max_days != 0:
            self.day += 1
        else:
            self.day = 1

        self.traffic = self.speed[self.day].groupby('ts')
        self.traffic_time = iter(self.traffic.groups.keys())

        self.info.clear()
        self.pass_set.clear()
        self.taken_nodes.clear()
        
        self.base_variables()
        
        for k in self.ev_info.keys():
            self.base_variables_ev(k)
            self.taken_nodes.add(self.ev_info[k]['NODE_ID'])

        for k in self.cs_info.keys():
            if self.counter in self.daily_cs_waiting_time[self.day][k].keys():
                self.cs_info[k]['waiting_time'] = self.daily_cs_waiting_time[self.day][k][self.counter]
            else:    
                self.cs_info[k]['waiting_time'] = random.randrange(self.min_threshold, self.charge_time, self.duration)
                self.daily_cs_waiting_time[self.day][k][self.counter] = self.cs_info[k]['waiting_time']
            self.info['cs'][k] = 0

        for k in self.user_info.keys():
            self.update_user_info(k)

        return self.get_state()

    def render(self):
        if self.normal_render:
            time.sleep(0.09)
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
            occurrence_rate = 0.5,
            max_number_of_requests = 100
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
            occurrence_rate= occurrence_rate,
            max_number_of_requests = max_number_of_requests
        )
        self.observation_space = self.game.observation_space
        self.action_space = spaces.Discrete(self.game.action_size)
        self.total_reward = 0

    def step(self, action):
        obs, reward, done, info = self.game.step(action)
        self.total_reward += reward
        return obs, reward, done, info

    def render(self):
        self.game.render()

    def reset(self):
        self.total_reward = 0
        obs = self.game.reset()
        return obs
