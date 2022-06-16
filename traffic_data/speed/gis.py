import matplotlib.pyplot as plt
import geopandas as gpd
import json
import networkx as nx
import osmnx as ox
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import time
import pandas as pd
from shapely.geometry import Point, box, Polygon, LineString
from shapely.ops import nearest_points
from shapely import wkt

with open('cs_data.json', 'r') as f:
    cs = json.load(f)

cs_list = np.array(cs['response']['body']['items']['item'])

def create_edges(edge_list, groups, date):
    result = pd.merge(edge_list, groups.get_group(date).loc[:, 'LINK_ID':], how='left', on='LINK_ID')
    result['speed'].fillna(result['MAX_SPD'], inplace=True)

    return label_speed(result), result

def update_edges(result, groups, date):
    s = groups.get_group(date).loc[:, 'LINK_ID':].set_index('LINK_ID')['speed']
    result['speed'] = result['LINK_ID'].map(s).fillna(result['speed'])

    return label_speed(result), result

def label_speed(result):
    gdf_edges = result.set_index(['F_NODE','T_NODE','LINK_ID'])

    gdf_edges.loc[(gdf_edges['ROAD_RANK'] == '101') & (gdf_edges['speed'] > 80), 'TC'] = 'green'
    gdf_edges.loc[(gdf_edges['ROAD_RANK'] == '101') & ((gdf_edges['speed'] >= 40) & (gdf_edges['speed'] <= 80)),'TC'] = 'yellow'
    gdf_edges.loc[(gdf_edges['ROAD_RANK'] == '101') & (gdf_edges['speed'] < 40), 'TC'] = 'red'
    gdf_edges.loc[(gdf_edges['ROAD_RANK'] == '103') & (gdf_edges['speed'] > 50), 'TC'] = 'green'
    gdf_edges.loc[(gdf_edges['ROAD_RANK'] == '103') & ((gdf_edges['speed'] >= 30) & (gdf_edges['speed'] <= 50)),'TC'] = 'yellow'
    gdf_edges.loc[(gdf_edges['ROAD_RANK'] == '103') & (gdf_edges['speed'] < 30), 'TC'] = 'red'
    gdf_edges.loc[(gdf_edges['ROAD_RANK'] == '106') & (gdf_edges['speed'] > 50), 'TC'] = 'green'
    gdf_edges.loc[(gdf_edges['ROAD_RANK'] == '106') & ((gdf_edges['speed'] >= 30) & (gdf_edges['speed'] <= 50)),'TC'] = 'yellow'
    gdf_edges.loc[(gdf_edges['ROAD_RANK'] == '106') & (gdf_edges['speed'] < 30), 'TC'] = 'red'
    gdf_edges.loc[(gdf_edges['ROAD_RANK'] == '107') & (gdf_edges['speed'] > 25), 'TC'] = 'green' 
    gdf_edges.loc[(gdf_edges['ROAD_RANK'] == '107') & ((gdf_edges['speed'] >= 15) & (gdf_edges['speed'] <= 25)),'TC'] = 'yellow'
    gdf_edges.loc[(gdf_edges['ROAD_RANK'] == '107') & (gdf_edges['speed'] < 15), 'TC'] = 'red'

    gdf_edges['state'] = (gdf_edges['LENGTH'] * 6) / (gdf_edges['speed'] * 100)
    gdf_edges['state'] = gdf_edges['state'].round(0).astype('Int8') + 1

    return gdf_edges

# bbox=(208700, 358000, 215300, 362650)
bbox=(205000, 358000, 217500, 365500)
limits = [127.05693, 35.81962, 127.19335, 35.88782]
limits = [127.09, 35.83, 127.15, 35.86]
new_bbox = box(limits[0], limits[1], limits[2], limits[3])
gdf_box = gpd.GeoDataFrame({'geometry': gpd.GeoSeries([new_bbox])}).set_crs('EPSG:4326')

node = gpd.read_file(
                    # '../traffic_data/[2022-03-28]NODELINKDATA/MOCT_NODE.shp',
                    '../traffic_data/[2021-11-15]NODELINKDATA/MOCT_NODE.shp',
                    bbox=bbox,
                    ignore_fields=['REMARK', 'TURN_P', 'NODE_TYPE','NODE_NAME'])

node = node.to_crs("EPSG:4326")
node = gpd.clip(node, mask=new_bbox)
node = node[node['NODE_ID'] != '3060202801']
node = node[node['NODE_ID'] != '3060202701']
# node = node.overlay(gdf_box, how='intersection')
# node['NODE_ID'] = pd.to_numeric(node['NODE_ID'], downcast='unsigned')

link = gpd.read_file(
                    # '../traffic_data/[2022-03-28]NODELINKDATA/MOCT_LINK.shp',
                    '../traffic_data/[2021-11-15]NODELINKDATA/MOCT_LINK.shp',  
                    bbox=bbox,
                    ignore_fields=['REMARK', 'REST_VEH', 'REST_W', 'REST_H', 'ROAD_USE', 
                                    'CONNECT','ROAD_TYPE', 'ROAD_NO', 'MULTI_LINK',
                                    'ROAD_NAME','LANES'])
link = link.to_crs("EPSG:4326")
link = gpd.clip(link, mask=new_bbox)
# link = link.overlay(gdf_box, how='intersection')

# link['LINK_ID'] = pd.to_numeric(link['LINK_ID'], downcast='unsigned')
# link['F_NODE'] = pd.to_numeric(link['F_NODE'], downcast='unsigned')
# link['T_NODE'] = pd.to_numeric(link['T_NODE'], downcast='unsigned')
# link['ROAD_NAME'] = link['ROAD_NAME'].astype("category")
link['ROAD_RANK'] = link['ROAD_RANK'].astype("category")
link['MAX_SPD'] = link['MAX_SPD'].astype('Int32')
link['LENGTH'] = pd.to_numeric(link['LENGTH'], downcast='float')

link = link[link['F_NODE'].isin(node['NODE_ID'])]
link = link[link['T_NODE'].isin(node['NODE_ID'])]
node = node[node['NODE_ID'].isin(link['F_NODE'])]
node = node[node['NODE_ID'].isin(link['T_NODE'])]
# print(node['NODE_ID'])
# print(node[node['NODE_ID'].isin(link['F_NODE'])])

# '''Get a list with the id of each link, where link is a specific section of the road'''
# for l in list(link['LINK_ID']):
    # print(l)

# s_df = pd.read_csv('../traffic_data/jeonju.csv')
s_df = pd.read_csv('traffic_data/211201.csv')

# s_df.columns = ['date', 'time', 'LINK_ID', 'UNK' ,'speed']
# s_df = s_df.loc[:, ['time', 'LINK_ID', 'speed']]

# s_df['time'] = s_df['time'].astype("category")
# s_df['LINK_ID'] = pd.to_numeric(s_df['LINK_ID'], downcast='unsigned')
# s_df['speed'] = s_df['speed'].astype('Int8')

# groups = s_df.groupby('time')

s_df.columns = ['ts','LINK_ID','speed']

''' Reduce memory footprint of data'''
s_df2 = s_df.copy()
s_df2['ts'] = s_df2['ts'].astype("category")
# s_df2['LINK_ID'] = pd.to_numeric(s_df2['LINK_ID'], downcast='unsigned')
s_df2['LINK_ID'] = s_df2['LINK_ID'].astype(str)
s_df2['speed'] = s_df2['speed'].astype('Int32')

groups = s_df2.groupby('ts')

date_list = list(groups.groups.keys())

gdf_edges, result = create_edges(link, groups, date_list[0])

gdf_nodes = node.set_index('NODE_ID')
# gdf_nodes = gdf_nodes.drop(['3060202801', '3060202701'])
gdf_nodes['x'] = gdf_nodes['geometry'].x
gdf_nodes['y'] = gdf_nodes['geometry'].y

# gdf_edges = gdf_edges.drop(['3060202801','3060202701'], level=1)
assert gdf_nodes.index.is_unique and gdf_edges.index.is_unique

G = ox.graph_from_gdfs(gdf_nodes, gdf_edges)
edges_types = gdf_edges['TC'].value_counts()
color_list = [(0.0, 0.8, 0.0, 1.0), (0.8, 0.8, 0.0, 1.0), (0.8, 0.0, 0.0, 1.0)]
color_mapper = pd.Series(color_list, index=edges_types.index).to_dict()
ec = [color_mapper[v] for v in gdf_edges['TC']]

fig, ax = plt.subplots(figsize=(10,7), facecolor='k')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
# g_osm = ox.graph_from_bbox(35.88782,35.81962,127.19335,127.05693, network_type="drive")
# _, ax = ox.plot_graph(g_osm, ax=ax, show=False, close=False, node_size=0)
ax = gdf_edges['geometry'].plot(ax=ax, color=ec, lw=1, alpha=None, zorder=-1)
ax.scatter(gdf_nodes['geometry'].x, gdf_nodes['geometry'].y, s = 2, zorder=-1, color='k')

# geo = ox.graph_to_gdfs(G, nodes=False)['ROAD_NAME']
# count = 0
# for k,v in geo.items():
#     # if v == '권삼득로':
#     if v == '당산로':
#         print(k,v)
#         count+=1
# print(count)

# route = nx.dijkstra_path(G,'3050034000','3060030100')
# route_1 = nx.dijkstra_path(G,'3050007100','3050039300')
# route = nx.dijkstra_path(G,3050034000,3060030100)
# route_1 = nx.dijkstra_path(G,3050007100,3050039300)

ax.set(xlim = (limits[0], limits[2]), ylim = (limits[1], limits[3]))
fig.tight_layout()

cs = {
    'statId': [],
    'geometry': []
}
for elem in cs_list:
    if (
        elem['busiId'] == 'ME' and 
    ((float(elem['lng']) <= limits[2] and float(elem['lng']) >= limits[0])) and 
    (float(elem['lat']) <= limits[3] and float(elem['lat']) >= limits[1])):
        cs['statId'].append(elem['statId'])
        cs['geometry'].append(Point(float(elem['lng']), float(elem['lat'])))
gdf_cs = gpd.GeoDataFrame(cs, crs="EPSG:4326")
# ax.scatter(gdf_cs['geometry'].x,gdf_cs['geometry'].y, color='m') # 'xkcd:very light blue'

pts = node.geometry.unary_union
def near(point, pts=pts):
    nearest = node.geometry == nearest_points(point, pts)[1]
    return np.asarray(node[nearest])[0]
gdf_cs[['NODE_ID','geometry']] = gdf_cs.apply(lambda row: near(row['geometry']), axis=1, result_type='expand')
gdf_cs['position'] = gdf_cs.apply(lambda row: [row['geometry'].x,row['geometry'].y], axis=1)
# print(gdf_cs)
ax.scatter(gdf_cs['geometry'].x,gdf_cs['geometry'].y, color='xkcd:brown') # 'xkcd:very light blue'
# plt.show()

# root = tk.Tk()
# # root.attributes('-fullscreen', True)
# chart = FigureCanvasTkAgg(fig, root)
# chart.get_tk_widget().pack(side='top', fill='both', expand=True)

# def render():
#     # time.sleep(0.4)
#     root.update()
#     chart.draw()

# x = gdf_nodes['x'][route[0]]
# y = gdf_nodes['y'][route[0]]
# x_1 = gdf_nodes['x'][route_1[0]]
# y_1 = gdf_nodes['y'][route_1[0]]
# route_coordinates = ax.scatter(x,y, color= 'b')
# route_coordinates_1 = ax.scatter(x_1,y_1, color = 'b')
# j = 0
# render()
# for i in date_list[1:]:
#     gdf_edges, result = update_edges(result, groups, i)
#     edges_types = gdf_edges['TC'].value_counts()
#     color_mapper = pd.Series(color_list, index=edges_types.index).to_dict()
#     ec = [color_mapper[v] for v in gdf_edges['TC']]
#     ax.collections[0].set_color(ec)
#     j+=1
#     try:
#         route_coordinates.set_offsets(np.c_[gdf_nodes['x'][route[j]],gdf_nodes['y'][route[j]]])
#         route_coordinates_1.set_offsets(np.c_[gdf_nodes['x'][route_1[j]],gdf_nodes['y'][route_1[j]]])   
#     except:
#         pass
#     render()

import random

from gym import spaces
from random import randint
import pdb

# degree_sequence = sorted((d for _, d in G.degree()), reverse= True)
# dmax = max(degree_sequence)
# print(min(degree_sequence))
# print(gdf_nodes.loc['3060202801'])

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
    def __init__(self, normal_render = False):
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
        self.roads = gdf_edges
        self.aux_road_result = result
        self.roads_color = [(0.0, 0.8, 0.0, 1.0), (0.8, 0.8, 0.0, 1.0), (0.8, 0.0, 0.0, 1.0)]
        self.traffic = s_df2.groupby('ts')
        self.traffic_time = iter(list(self.traffic.groups.keys()))

        self.title('Jeonju')
        self.create_plot()
        self.canvas = self._build_canvas()
        self.normal_render = normal_render

        self.counter = 0
        self.aux_counter = 0
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

    def color_roads(self):
        edges_types = self.roads['TC'].value_counts()
        color_mapper = pd.Series(self.roads_color, index=edges_types.index).to_dict()
        return [color_mapper[v] for v in self.roads['TC']]

    def create_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(15,12), facecolor='k')
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        self.ax = self.roads['geometry'].plot(ax=self.ax, color=self.color_roads(), lw=1, alpha=None, zorder=-1)
        self.ax.scatter(self.intersections['geometry'].x, self.intersections['geometry'].y, s = 2, zorder=-1, color='k')
        self.ax.set(xlim = (127.05693, 127.19335), ylim = (35.81962, 35.88782))
        self.fig.tight_layout()
        self.ax.scatter(self.cs['geometry'].x, self.cs['geometry'].y, color='xkcd:brown')
        # self.ax.scatter(self.intersections.loc['3060202801'].x,self.intersections.loc['3060202801'].y, color='r')
        # print(self.graph.adj['3060202801'])

    def set_cs(self):
        self.cs_info = self.cs.T.to_dict()

        for k, v in self.cs_info.items():
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
        picked_node = self.intersections.sample()
        while self.cs.isin([picked_node.index.values[0]]).any().any() or (picked_node.index.values[0] in self.taken_nodes):
            picked_node = self.intersections.sample()
        loc = [picked_node.iloc[0]['x'], picked_node.iloc[0]['y']]
        self.taken_nodes.append(picked_node.index.values[0])
        return loc, picked_node.index.values[0]

    def _build_canvas(self):
        self.attributes('-fullscreen', True)
        canvas = FigureCanvasTkAgg(self.fig, self)
        canvas.get_tk_widget().pack(side='top',fill='both', expand=True)
        
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

    def update_roads(self, groups, date):
        s = groups.get_group(date).loc[:, 'LINK_ID':].set_index('LINK_ID')['speed']
        self.aux_road_result['speed'] = self.aux_road_result['LINK_ID'].map(s).fillna(self.aux_road_result['speed'])
        self.roads = label_speed(self.aux_road_result)
    
    def update_graph(self):
        self.update_roads(self.traffic, next(self.traffic_time))
        self.ax.collections[0].set_color(self.color_roads())
        self.graph = ox.graph_from_gdfs(self.intersections, self.roads)

    def discharge_ev(self, index):
        if self.ev_info[index] != 0.4:
            self.ev_info[index]['SOC'] = (int(self.ev_info[index]['SOC']*10) - int((DISCHARGE_RATE)*10))/10
        else:
            self.ev_info[index]['SOC'] = 0

    # def charge_ev(self, index):


    def move(self, index, target, action):
        s = target['NODE_ID']
        sample = self.graph.neighbors(s)
        s_ = next((x for i,x in enumerate(sample) if i==action), None)
        if s_ is not None:
            self.ev_info[index]['NODE_ID'] = s_
            node = self.intersections.loc[s_]
            self.ev_info[index]['location'] = [node['geometry'].x, node['geometry'].y]
            self.ev_info[index]['scatter'].set_offsets(
                np.c_[self.ev_info[index]['location'][0], self.ev_info[index]['location'][1]])
            self.discharge_ev(index)
        else:
            s_ = s
        return s_

    def get_state(self):
        states = []
        states.append(self.ev_info[0]['SOC'])
        states.append(self.ev_info[0]['NODE_ID'])
        for v in self.user_info.values():
            states.append(v['NODE_ID_source'])
            states.append(v['NODE_ID_destination'])
            states.append(v['waiting_time'])
        for v in self.cs_info.values():
            states.append(v['waiting_time'])
        for edge in self.graph.edges():
            for v in self.graph.get_edge_data(edge[0], edge[1]).values():
                states.append(v['state'])    
        return np.array(states)

    def check_done(self):
        self.counter += 1 
        for k,v in self.user_info.items():
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

    def check_if_reward(self, index, next_node):
        check_list = {}
        check_list['if_done'] = False
        check_list['if_goal'] = False
        rewards = 0

        found = False
        if not self.passenger:
            for k,v in self.user_info.items():
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
                charging_time = int(np.ceil((MAX_SOC - self.ev_info['SOC'])/CHARGING_RATE))
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
            self.reset_rewards(index)

        check_list['rewards'] = rewards

        # pdb.set_trace()
        return check_list

    def move_aux(self, index, next_node):
        self.update_cs()
        self.update_graph()
        self.render()
        self.ev_info[index]['NODE_ID'] = next_node
        node = self.intersections.loc[next_node]
        self.ev_info[index]['location'] = [node['geometry'].x, node['geometry'].y]
        self.ev_info[index]['scatter'].set_offsets(
            np.c_[self.ev_info[index]['location'][0], self.ev_info[index]['location'][1]])
        self.discharge_ev(index)
        # pdb.set_trace()
        return self.check_if_reward(index, next_node)

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
            self.stand_by = False
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

    def step(self, action):
        # print(self.ev_info)
        # pdb.set_trace()
        if action <= MAX_NUMBER_OF_CONNECTIONS:
            self.update_cs()
            self.counter += 1
            self.update_graph()
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
        self.render()

        self.traffic_time = iter(list(self.traffic.groups.keys()))

        self.ev_info[0]['SOC'] = MAX_SOC
        self.counter = 0
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
            # time.sleep(2)
            pass
        self.update()
        self.canvas.draw()

# from dqn_agent import DQNAgent

EPISODES = 100
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
    # agent = DQNAgent(GAMMA, 
    #                 EPSILON, 
    #                 env.observation_space.shape[0], 
    #                 env.action_size, 
    #                 BATCH, 
    #                 eps_dec=EPS_DEC, 
    #                 name = f'lr{LR}_b{BATCH}_dec{EPS_DEC}_g{GAMMA}', 
    #                 lr=LR, 
    #                 chkpt_dir= dir)

    print(f'Training parameters: {EPISODES=}, {GAMMA=}, {BATCH=}, {EPS_DEC=}, {LR=}')

    best_avg_score = best_score = -np.inf
    load_checkpoint = False

    # if load_checkpoint:
    #     agent.load_models()

    global_step = 0
    scores = []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()

        while not done:
            global_step += 1

            # action = agent.choose_action(state)
            action = randint(0, env.action_size - 1)
            next_state, reward, done, info = env.step(action)
            
            score += reward

            # if not load_checkpoint:
            #     agent.store_transition(state, action, reward, next_state, done)
            #     agent.train()

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
        # writer.add_scalar('Epsilon', agent.epsilon, e)

        scores.append(score)

        avg_score = np.mean(scores[-100:])

        print('episode: ', e,'score: ', score, 'best score: %.2f' % best_score,
            ' avg score: %.1f' % avg_score, 'best avg score: %.2f' % best_avg_score,
            # 'epsilon: %.2f' % agent.epsilon, 
            'steps:', global_step, 'ep duration:', info['count'])

        if score > best_score:
            best_score = score

        if avg_score > best_avg_score:
            # if not load_checkpoint:
            #     agent.save_models()
            best_avg_score = avg_score