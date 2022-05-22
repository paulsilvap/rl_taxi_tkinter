import numpy as np
import networkx as nx
import json
import geopandas as gpd
import pandas as pd
import osmnx as ox
from shapely.geometry import Point, box
from shapely.ops import nearest_points

def create_graph(row, column, weight_limit=2):

    matrix = np.reshape(np.array([x for x in range(column*row)]),(column,row))

    graph = nx.Graph()

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if j+1 < len(matrix):
                graph.add_edge(matrix[i][j], matrix[i][j+1], state = np.random.randint(1,weight_limit))
            if i+1 < len(matrix):
                graph.add_edge(matrix[i][j], matrix[i+1][j], state = np.random.randint(1,weight_limit))
            if j-1 >= 0:
                graph.add_edge(matrix[i][j], matrix[i][j-1], state = np.random.randint(1,weight_limit))
            if i-1 >= 0:
                graph.add_edge(matrix[i][j], matrix[i-1][j], state = np.random.randint(1,weight_limit))
    
    return graph

def create_edges(edge_list, groups, date):
    result = pd.merge(edge_list, groups.get_group(date).loc[:, 'LINK_ID':], how='left', on='LINK_ID')
    result['speed'].fillna(result['MAX_SPD'], inplace=True)

    mask_101 = result['ROAD_RANK'].values == '101'
    mask_103 = (result['ROAD_RANK'].values == '103') | (result['ROAD_RANK'].values == '106')
    mask_107 = result['ROAD_RANK'].values == '107'

    return label_speed(result, mask_101, mask_103, mask_107), result

# def update_edges(result, groups, date):
#     s = groups.get_group(date).loc[:, 'LINK_ID':].set_index('LINK_ID')['speed']
#     result['speed'] = result['LINK_ID'].map(s).fillna(result['speed'])

#     return label_speed(result), result

# @profile
def label_speed(gdf_edges, mask_101, mask_103, mask_107):
    aux = gdf_edges['speed'].values
    length = gdf_edges['LENGTH'].values
    green_mask = (mask_101 & (aux > 80)) | (mask_103 & (aux > 50)) | (mask_107 & (aux > 25))
    yellow_mask = ((mask_101 & ((aux >= 40) & (aux <= 80))) | (mask_103 & ((aux >= 30) & (aux <= 50))) | (mask_107 & ((aux >= 15) & (aux <= 25))))
    red_mask = (mask_101 & (aux < 40)) | (mask_103 & (aux < 30)) | (mask_107 & (aux < 15))
    gdf_edges.loc[green_mask, 'TC'] = 'green'
    gdf_edges.loc[yellow_mask, 'TC'] = 'yellow'
    gdf_edges.loc[red_mask, 'TC'] = 'red'

    aux_state = (length * 6) / (aux * 100)
    gdf_edges['state'] = aux_state.round(0) + 1

    return gdf_edges.set_index(['F_NODE','T_NODE','LINK_ID'])

# @profile
def graph_from_gdfs(gdf_edges):
    G = nx.MultiDiGraph()

    attr_names = gdf_edges.columns.to_list()
    for (u, v, _), attr_vals in zip(gdf_edges.index, gdf_edges.values):
        G.add_edge(u, v, key=0, **{name: val for name, val in zip(attr_names, attr_vals)})

    return G

def create_map(days=1):
    with open('./traffic_data/cs/cs_data.json', 'r') as f:
        cs = json.load(f)

    cs_list = np.array(cs['response']['body']['items']['item'])

    # bbox=(208700, 358000, 215300, 362650)
    bbox=(205000, 358000, 217500, 365500)
    # limits = [127.05693, 35.81962, 127.19335, 35.88782]
    limits = [127.09, 35.83, 127.15, 35.86]
    new_bbox = box(limits[0], limits[1], limits[2], limits[3])
    # gdf_box = gpd.GeoDataFrame({'geometry': gpd.GeoSeries([new_bbox])}).set_crs('EPSG:4326')

    node = gpd.read_file(
                        './traffic_data/roads/[2021-11-15]NODELINKDATA/MOCT_NODE.shp',
                        bbox=bbox,
                        ignore_fields=['REMARK', 'TURN_P', 'NODE_TYPE','NODE_NAME'])

    node = node.to_crs("EPSG:4326")
    node = gpd.clip(node, mask=new_bbox)
    node = node[node['NODE_ID'].values != '3060202801']
    node = node[node['NODE_ID'].values != '3060202701']
    #TODO: find invalid nodes automatically
    node['NODE_ID'] = pd.to_numeric(node['NODE_ID'], downcast='unsigned')

    link = gpd.read_file(
                        './traffic_data/roads/[2021-11-15]NODELINKDATA/MOCT_LINK.shp',  
                        bbox=bbox,
                        ignore_fields=['REMARK', 'REST_VEH', 'REST_W', 'REST_H', 'ROAD_USE', 
                                        'CONNECT','ROAD_TYPE', 'ROAD_NO', 'MULTI_LINK',
                                        'ROAD_NAME','LANES'])
    link = link.to_crs("EPSG:4326")
    link = gpd.clip(link, mask=new_bbox)

    link['LINK_ID'] = pd.to_numeric(link['LINK_ID'], downcast='unsigned')
    link['F_NODE'] = pd.to_numeric(link['F_NODE'], downcast='unsigned')
    link['T_NODE'] = pd.to_numeric(link['T_NODE'], downcast='unsigned')
    # link['ROAD_NAME'] = link['ROAD_NAME'].astype("category")
    link['ROAD_RANK'] = link['ROAD_RANK'].astype("category")
    link['MAX_SPD'] = link['MAX_SPD'].astype('Int32')
    link['LENGTH'] = pd.to_numeric(link['LENGTH'], downcast='float')

    link = link[link['F_NODE'].isin(node['NODE_ID'])]
    link = link[link['T_NODE'].isin(node['NODE_ID'])]
    node = node[node['NODE_ID'].isin(link['F_NODE'])]
    node = node[node['NODE_ID'].isin(link['T_NODE'])]

    # '''Get a list with the id of each link, where link is a specific section of the road'''
    # for l in list(link['LINK_ID']):
        # print(l)

    daily_traffic = {}

    for d in range(1, days+1):
        if d < 10:
            traffic_df = pd.read_csv(f'./traffic_data/speed/21120{d}.csv')
        else:
            traffic_df = pd.read_csv(f'./traffic_data/speed/2112{d}.csv')

        traffic_df.columns = ['ts', 'LINK_ID', 'speed']

        traffic_df['ts'] = traffic_df['ts'].astype('category')
        traffic_df['LINK_ID'] = pd.to_numeric(traffic_df['LINK_ID'], downcast='unsigned')
        traffic_df['speed'] = traffic_df['speed'].astype('Int32')

        # print(len(list(traffic_df.groupby('ts').groups.keys())))
        daily_traffic[d] = traffic_df

    groups = daily_traffic[1].groupby('ts')

    date_list = list(groups.groups.keys())

    gdf_edges, result = create_edges(link, groups, date_list[0])

    gdf_nodes = node.set_index('NODE_ID')
    gdf_nodes['x'] = gdf_nodes['geometry'].x
    gdf_nodes['y'] = gdf_nodes['geometry'].y

    assert gdf_nodes.index.is_unique and gdf_edges.index.is_unique

    # G = ox.graph_from_gdfs(gdf_nodes, gdf_edges)
    G = graph_from_gdfs(gdf_edges)

    # g_osm = ox.graph_from_bbox(35.88782,35.81962,127.19335,127.05693, network_type="drive")
    # _, ax = ox.plot_graph(g_osm, ax=ax, show=False, close=False, node_size=0)

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

    pts = node.geometry.unary_union
    def near(point, pts=pts):
        nearest = node.geometry == nearest_points(point, pts)[1]
        return np.asarray(node[nearest])[0]
    gdf_cs[['NODE_ID','geometry']] = gdf_cs.apply(lambda row: near(row['geometry']), axis=1, result_type='expand')
    gdf_cs['position'] = gdf_cs.apply(lambda row: [row['geometry'].x,row['geometry'].y], axis=1)

    return gdf_edges, gdf_nodes, G, gdf_cs, daily_traffic, result
