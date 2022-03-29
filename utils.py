import numpy as np
import networkx as nx

def create_graph(row, column):

    matrix = np.reshape(np.array([x for x in range(column*row)]),(column,row))

    graph = nx.Graph()

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if j+1 < len(matrix):
                graph.add_edge(matrix[i][j], matrix[i][j+1])
            if i+1 < len(matrix):
                graph.add_edge(matrix[i][j], matrix[i+1][j])
            if j-1 >= 0:
                graph.add_edge(matrix[i][j], matrix[i][j-1])
            if i-1 >= 0:
                graph.add_edge(matrix[i][j], matrix[i-1][j])
    
    return graph

