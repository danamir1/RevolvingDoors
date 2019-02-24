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
from langdetect import detect as ldetect
import tqdm
import pickle
import json

ENTITIES_DATABASE = 'data/entities/entities_database.csv'
MAYA_MATCH = 'data/ynet/maya_name_matches.csv'
TYPE_MAP = 'types_mapping.json'


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_json(path):
    with open(path, encoding='utf-8') as f:
        # f = f.read().encode('utf-8')
        return json.load(f)


def get_type_mapping():
    return {
        "חברה פרטית": "company",
        "חברה ציבורית": "company",
        "חברה פרטית מחוייבת במאזן": "company",
        "חברת חו\"ל": "company",
                             "שותפות מוגבלת": "company",
        "NaN": "other",
        "government_office": "government",
        "municipality": "government",
        "religion_service": "government",
        "foreign_company": "company",
        "law_mandated_organization": "government",
        "provident_fund": "company",
        "university": "company",
        "health_service": "company",
        "professional_association": "company",
        "house_committee": "company",
        "foreign_representative": "other",
        "municipal_parties": "other",
        "conurbation": "government",
        "local_planning_committee": "government",
        "local_community_committee": "government",
        "religious_court_sacred_property": "government",
        "drainage_authority": "government",
        "municipal_precinct": "other",
        "west_bank_corporation": "company",
        "company": "company",
        "association": "company",
        "cooperative": "company",
        "ottoman-association": "company",
        "private": "company"
    }



def apriori_cluster(baskets_file='data/ynet/articles_orgs_baskets_190222.pkl'):
    baskets = load_pickle(baskets_file)
    print(type(baskets))
    baskets = [list(baskets[b].difference(set(['ynet']))) for b in baskets]
    result = apriori(baskets, min_support=0.002, min_confidence=0.0, min_lift=0.002,
                     max_length=None)
    for i, r in enumerate(result):
        print(i, r.items)  # , r.support)


def edit_distance(s1, s2):
    # from https://stackoverflow.com/questions/2460177/edit-distance-in-python
    m = len(s1) + 1
    n = len(s2) + 1

    tbl = {}
    for i in range(m): tbl[i, 0] = i
    for j in range(n): tbl[0, j] = j
    for i in range(1, m):
        for j in range(1, n):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            tbl[i, j] = min(tbl[i, j - 1] + 1, tbl[i - 1, j] + 1, tbl[i - 1, j - 1] + cost)

    return tbl[i, j]


def match_entities(orgs):
    # interactive matching of entities to the entities dataset. TODO: use merge sets and maya for better matching
    new_orgs = {'org_name': [], 'entity_name': [], 'id': [], 'type': [], 'sub_type': [],
                'candidates': []}
    maya = pd.read_csv(MAYA_MATCH)
    entities_dataset = pd.read_csv(ENTITIES_DATABASE)
    # print(entities_dataset.columns.values)
    type_map = get_type_mapping()
    for i, org in enumerate(orgs):
        print('-->', org)
        # candidates = entities_dataset[entities_dataset['name'].str.contains(org, regex=False)]
        candidates = entities_dataset[org in entities_dataset.name]
        if ldetect(org) != 'he':
            en_candidates = entities_dataset[
                entities_dataset['eng_name'].str.contains(org, regex=False)]
            candidates = pd.concat([candidates, en_candidates]).drop_duplicates().reset_index(
                drop=True)
        if candidates.size == 0:
            # no candidate
            new_orgs['org_name'].append(org)
            new_orgs['entity_name'].append(org)
            new_orgs['id'].append(-1)
            new_orgs['sub_type'].append('other')
            new_orgs['type'].append('other')
            new_orgs['candidates'].append(0)
        else:
            # look for the one with the smallest edit distance
            min_dis = float('inf')
            best_candidate = None
            print('-->', org)
            for index, row in candidates.iterrows():
                print(row['name'])
                if ldetect(org) != 'he':
                    distance = edit_distance(org, row['name_eng'])
                else:
                    distance = edit_distance(org, row['name'])
                if distance < min_dis:
                    min_dis = distance
                    best_candidate = row
            new_orgs['org_name'].append(org)
            new_orgs['entity_name'].append(best_candidate['name'])
            new_orgs['id'].append(best_candidate['id'])
            new_orgs['candidates'].append(candidates.size)
            new_orgs['sub_type'].append(best_candidate['type'])
            new_orgs['type'].append(type_map[best_candidate['type']])
        if i > 20:
            break

    # pd.DataFrame(new_orgs).to_csv('data/ynet/filtered_by_entity_db.csv', encoding='utf8')
    # pd.DataFrame({'org_name': unmatched_orgs}).to_csv("data/ynet/unmatched_orgs.csv",encoding='utf8')
    pd.DataFrame(new_orgs).to_csv('data/ynet/matched.csv', encoding='utf8')
    # pd.DataFrame({'org_name': unmatched_orgs}).to_csv("data/ynet/unmatched_orgs.csv", encoding='utf8')
    return new_orgs


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
    # apriori_cluster()
    # todo try: http://holoviews.org/gallery/demos/bokeh/lesmis_example.html
    # todo search: https://bokeh.pydata.org/en/latest/docs/user_guide/interaction/widgets.html
    orgs = list(load_pickle('data/ynet/orgs_to_articles_190222.pkl').keys())
    match_entities(orgs)
