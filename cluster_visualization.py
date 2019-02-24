from apyori import apriori
import pandas as pd
import numpy as np
import re
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
# import csv
# from matplotlib import pyplot as plt
import holoviews as hv
import networkx as nx
from holoviews import opts
import holoviews.plotting.bokeh
# from langdetect import detect as ldetect
# import tqdm
import pickle


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def apriori_cluster(baskets_file='data/ynet/articles_orgs_baskets_190222.pkl'):
    baskets = load_pickle(baskets_file)
    print(type(baskets))
    baskets = [list(baskets[b].difference(set(['ynet']))) for b in baskets]
    result = apriori(baskets, min_support=0.002, min_confidence=0.0, min_lift=0.002,
                     max_length=None)
    for i, r in enumerate(result):
        print(i, r.items, r.support)

def visualize(idx2org, org2idx, org2pairs, distance, affinity, distance_threshold):
    graph_nodes = []
    for i in range(len(idx2org)):
        for j in range(i):
            d = distance[i, j]
            if d <= distance_threshold:
                graph_nodes.append((idx2org[i], idx2org[j], d,))
    hv.extension('bokeh')
    graph = hv.Graph(graph_nodes, vdims='weight')
    graph = hv.element.graphs.layout_nodes(graph, layout=nx.layout.fruchterman_reingold_layout,
                                           kwargs={'weight': 'weight'})
    # labels = hv.Labels(graph.nodes, ['x', 'y'], 'club')
    # (graph * labels.opts(text_font_size='8pt', text_color='white', bgcolor='gray'))
    graph.opts(node_size=2, edge_line_width=hv.dim('weight'))
    hv.save(graph, "out.html")

if __name__ == '__main__':
    apriori_cluster()
