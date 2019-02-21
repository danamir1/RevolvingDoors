import pandas as pd
import numpy as np
import re
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from ExtractEntities import load_pickle
import csv
from matplotlib import pyplot as plt
import holoviews as hv
import networkx as nx
from holoviews import opts
import holoviews.plotting.bokeh
from langdetect import detect as ldetect
import tqdm
def build_index_dictionary(organizations):
    return {i: org for i, org in enumerate(organizations)}, {org: i for i, org in enumerate(organizations)}

def build_org2pairs(orgs, pairs):
    org2pairs = {org: [] for org in orgs}
    for pair in pairs:
        for org in pair:
            org2pairs[org].append(pair)
    return org2pairs


def build_distance_matrix(index2org,org2index, org2pairs, pairs_count,
                          org_counts):
    # computes a distance matrix for all pairs of organizations with jaccard similarity

    n = len(index2org)
    distance_matrix = np.ones((n, n))
    for i in range(len(index2org)):
        org = index2org[i]
        for org_pair in org2pairs[index2org[i]]:
            org_pair = [x for x in org_pair if x != org][0]
            j = org2index[org_pair]
            # 1 - Jaccard(org1, org2)
            distance = 1 - ((pairs_count[frozenset([org ,org_pair])] * 2) / (org_counts[org] + org_counts[
                                                                             org_pair]))

            distance_matrix[i,j] = distance ** 10 # high polynomial to make non 1 values closer to 0
            distance_matrix[j,i] = distance ** 10
    return distance_matrix


def match_entities(filtered_orgs, entities_dataset):
    # interactive matching of entities to the entities dataset. TODO: use merge sets and maya for better matching
    new_orgs = {'org_name':[],'entity_name':[],'id':[]}
    unmatched_orgs = []
    for org in tqdm.tqdm(filtered_orgs):
        candidates = entities_dataset[entities_dataset['name'].str.contains(org,regex=False)]
        if ldetect(org) != 'he':
            en_candidates = entities_dataset[entities_dataset['eng_name'].str.contains(org,regex=False)]
            candidates = pd.concat([candidates, en_candidates]).drop_duplicates().reset_index(drop=True)
        if candidates.shape[0] > 1:
            vals = list(candidates['name'].values)
            vals_dict = {i:v for i,v in enumerate(vals)}
            for i,v in enumerate(vals):
                print('%d: %s' % (i,v))
            choice = -1
            while True:
                try:
                    choice = int(input("choose the best fit name for organization:  %s" % org))
                    val = vals_dict[choice]
                    break
                except:
                    print("please provide valid index")
            if choice == -1:
                unmatched_orgs.append(org)
            else:
                val = vals_dict[choice]
                id = candidates[candidates['name'] == val]['id'].values[0]
        elif candidates.shape[0] == 1:
            val = candidates['name'].values[0]
            id = candidates['id'].values[0]
        else:
            print("No match found for organization: %s" % org)
            unmatched_orgs.append(org)
            continue
        new_orgs['org_name'].append(org)
        new_orgs['entity_name'].append(val)
        new_orgs['id'].append(id)
        pd.DataFrame(new_orgs).to_csv('filtered_by_entity_db.csv', encoding='utf8')
        pd.DataFrame({'org_name': unmatched_orgs}).to_csv("unmatched_orgs.csv",encoding='utf8')
    pd.DataFrame(new_orgs).to_csv('filtered_by_entity_db.csv', encoding='utf8')
    pd.DataFrame({'org_name': unmatched_orgs}).to_csv("unmatched_orgs.csv", encoding='utf8')
    return new_orgs



if __name__ == "__main__":

    # dictionary of {entity_name: count} - number of occurences in unique articles
    org_counts = load_pickle("organizations_counts_0209_merged.pkl")
    # dictionary pf {frozenset pair of names: count} - number of co-occurences in unique articles
    pairs_counts = load_pickle("organizations_pairs_counts_0209_merged.pkl")

    # create some convenience mappings for distance matrices
    idx2org, org2idx = build_index_dictionary(org_counts)
    org2pairs = build_org2pairs(org_counts, pairs_counts)

    distance_matrix = build_distance_matrix(idx2org, org2idx, org2pairs, pairs_counts, org_counts)
    affinity_matrix = (distance_matrix < 0.9).astype(np.float32)
    N = len(org_counts)

    #spectral clustering
    clustering = SpectralClustering(affinity='precomputed')
    clusters = clustering.fit_predict(affinity_matrix)

    # t-sne for 2-D embeddings
    tsne= TSNE(n_components=2,
               perplexity=10,
               metric='precomputed')
    tsne_res = tsne.fit_transform(distance_matrix)

    # ------ t-sne + clustering plotting -----
    plt.scatter(x=tsne_res[:, 0], y=tsne_res[:, 1], c=clusters)
    for i in np.random.choice(np.arange(distance_matrix.shape[0]), 100):
        org = idx2org[i]
        diff = 5
        x, y = tsne_res[i, 0], tsne_res[i, 1]
        plt.plot(x, y, x + diff, y + diff, marker='o')
        plt.text(x=x + diff, y=y + diff, s=org[::-1])
    plt.show()

    # ---- HoloViews graph creation ----
    graph_nodes = []
    for i in range(N):
        for j in range(i):
            org_i = idx2org[i]
            org_j= idx2org[j]
            if distance_matrix[i,j] == 1:
                continue
            else:
                # distance =  (1 - distance_matrix[i,j]) * ((org_counts[org_i] + org_counts[org_j]) / 2)
                distance = 2 * pairs_counts[frozenset([org_j, org_i])] / (org_counts[org_i] + org_counts[
                    org_j])

                graph_nodes.append((idx2org[i], idx2org[j], distance,))
    hv.extension('bokeh')
    graph = hv.Graph(graph_nodes, vdims='weight')
    graph =hv.element.graphs.layout_nodes(graph, layout=nx.layout.fruchterman_reingold_layout,
                                          kwargs={'weight': 'weight'})
    graph.opts(node_size=5, edge_line_width=hv.dim('weight'))
    hv.save(graph,"out.html")


