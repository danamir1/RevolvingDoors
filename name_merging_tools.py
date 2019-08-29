import networkx as nx
import math
from functools import reduce
import tqdm
from networkx.linalg.algebraicconnectivity import algebraic_connectivity
from matplotlib import pyplot as plt
from utils import load_pickle, save_pickle
from itertools import combinations
from fuzzywuzzy import fuzz
import pandas as pd

def find_bottlenecks_in_pairs_graph(pairs):
    """

    :param pairs:
    :return:
    """
    nodes = set(reduce(lambda x, y: x + y, pairs))
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from([tuple(p) for p in pairs])
    G = G.to_undirected()
    bottlenecks = []
    for subgraph in tqdm.tqdm(nx.connected_component_subgraphs(G)):
        if subgraph.size() > 3:
            betweenness_scores = nx.edge_betweenness_centrality(subgraph)
            max_edge = max(betweenness_scores.keys(), key=lambda x: betweenness_scores[x])
            # print(max_edge, betweenness_scores[max_edge])
            if 'רשות השרות הלאומי אזרחי' in subgraph.nodes:  # weird bug causes this specific calculation to
                #  get stuck
                continue
            connectivity = algebraic_connectivity(subgraph, normalized=True)
            # print(subgraph.nodes)
            bottlenecks.append((max_edge, betweenness_scores[max_edge], connectivity, subgraph))
    bottlenecks = sorted(bottlenecks, key=lambda x: x[2], reverse=False)
    print(len(bottlenecks))

    for i, b in enumerate(bottlenecks[:100]):
        plt.clf()
        plt.axis('off')
        print("\n\n\n-------\n\n\n")
        print(i, ")")
        print("\n".join([str(x) for x in b[:-1]]))
        print("\n".join([str(x) for x in b[-1].edges]))

        layout = nx.spring_layout(b[-1], k=2 / math.sqrt(b[-1].order()))
        nx.draw_networkx(b[-1], pos=layout, with_labels=True,
                         labels={n: n[::-1] for n in b[-1].nodes}, node_size=4, font_size=8, alpha=0.7,
                         node_color='black', edge_color='gray')
        plt.savefig("%d.pdf" % i)
        plt.clf()
        plt.axis('off')
        nx.draw_networkx(b[-1], pos=layout, with_labels=False, node_size=4, alpha=0.7, node_color='black',
                         edge_color='gray')
        plt.savefig("%d_without_names.png" % i)


def merge_entities_sets(merge_pairs):
    """
    given pairs of terms that should be merged, generates a dictionary which maps each term to the
    set of all terms it is merged with.
    """
    orgs2merge_sets = {}
    for org1, org2 in merge_pairs:
        if (org1 in orgs2merge_sets) and (org2 in orgs2merge_sets):
            # both orgs are in sets thus the sets should be merged and all 'pointers' should be corrected.
            orgs2merge_sets[org1].update(orgs2merge_sets[org2])
            new_set = orgs2merge_sets[org1]
            for org in orgs2merge_sets[org2]:
                orgs2merge_sets[org] = new_set
        elif org1 in orgs2merge_sets:
            orgs2merge_sets[org1].add(org2)
            orgs2merge_sets[org2] = orgs2merge_sets[org1]
        elif org2 in orgs2merge_sets:
            orgs2merge_sets[org2].add(org1)
            orgs2merge_sets[org1] = orgs2merge_sets[org2]
        else:
            new_set = {org1, org2}
            orgs2merge_sets[org1] = new_set
            orgs2merge_sets[org2] = new_set
    return orgs2merge_sets

def find_best_textual_representative():
    all_pairs = load_pickle("all_pairs_with_fixes.pkl")
    org2mergeset = merge_entities_sets(all_pairs)
    unique_merge_sets = []
    for m_set in org2mergeset.values():
        if m_set not in unique_merge_sets:
            unique_merge_sets.append(m_set)
    set2term = {}
    for merge_set in unique_merge_sets:
        g = nx.Graph()
        g.add_nodes_from(list(merge_set))

        edges = [(v1, v2, {'weight': fuzz.ratio(v1, v2)}) for v1, v2 in combinations(merge_set, 2)]
        g.add_edges_from(edges)
        g = g.to_undirected()

        max_degree_node = max(g.degree(weight='weight'), key=lambda x: x[1])
        set2term[frozenset(merge_set)] = max_degree_node[0]
    save_pickle(set2term, 'all_data_set2term_final.pkl')
    save_pickle(org2mergeset, 'all_data_org2mergeset_final.pkl')

def unite_orgs_and_map_to_terms():
    org2set = load_pickle('all_data_org2mergeset_final.pkl')
    set2term = load_pickle('all_data_set2term_final.pkl')
    org2baskets = load_pickle('orgs_to_articles_190225.pkl')
    guests = pd.read_csv('filtered_guests_final.csv', encoding='utf8')
    ynet_orgs = list(org2baskets.keys())
    committees_orgs = guests.company.unique().tolist()
    all_orgs = set(ynet_orgs + committees_orgs)
    org2term = {"organization": [], "term": []}
    for org in all_orgs:
        org2term["organization"].append(org)
        org2term["term"].append(org)
        if org in org2set:
            for syn_org in org2set[org]:
                if syn_org != org:
                    org2term["organization"].append(syn_org)
                    org2term["term"].append(org)

    df = pd.DataFrame(org2term)
    df.to_csv('map_org_to_term.csv', encoding='utf8', index=False)