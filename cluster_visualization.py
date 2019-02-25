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
from fuzzywuzzy import fuzz

ENTITIES_DATABASE_ORIG = 'data/entities/entities_database.csv'
ENTITIES_DATABASE = 'data/entities/entities_database_clean.csv'
MAYA_MATCH = 'data/ynet/maya_name_matches.csv'
TYPE_MAP = 'types_mapping.json'
MERGE = 'data/ynet/all_data_org2mergeset.pkl'


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
    result = apriori(baskets, min_support=0.0002, min_confidence=0.0, min_lift=0.0002,
                     max_length=2)

    sort = sorted(filter(lambda r: len(r.items) > 1, result), key=lambda x: x.ordered_statistics[
        0].lift,
                  reverse=True)
    for i, r in enumerate(sort):
        # print(r)
        print(i, r.items, r.ordered_statistics[0].lift)  # , r.support)


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
    # todo maya matched orig name
    # todo entities main office
    # todo sum num candidates by set size of mergesets
    # todo choose candidate by smallest candidate group
    # todo load newest data
    # interactive matching of entities to the entities dataset. TODO: use merge sets and maya for better matching
    new_orgs = {'org_name': [], 'entity_name': [], 'id': [], 'type': [], 'sub_type': [],
                'candidates': []}
    maya = pd.read_csv(MAYA_MATCH)
    merge = load_pickle(MERGE)
    entities_dataset = pd.read_csv(ENTITIES_DATABASE)
    # print(entities_dataset.columns.values)
    type_map = get_type_mapping()
    for i, org_name in enumerate(orgs):
        # print('-->', org)
        org_set = merge[org_name] if org_name in merge else set([org_name])
        candidates_set = []
        for org in org_set:
            matched_maya = False
            # first use maya's matches
            maya_match = maya[maya['name'] == org]
            if maya_match.size > 0:
                new_org = maya_match['official_name'].iloc[0]
                # print('MAYA', org, '->', new_org)
                org = new_org
                matched_maya = True

            # candidates = entities_dataset[entities_dataset['name'].str.contains(org, regex=False)]
            candidates = entities_dataset[entities_dataset['name'].str.contains('\\b' + org + '\\b',
                                                                                regex=True) == True]
            if ldetect(org) != 'he':
                candidates = entities_dataset[entities_dataset['name_eng'].str.contains(
                    '\\b({}|{}|{})\\b'.format(org.upper(), org, org.lower()), regex=True) == True]
                # candidates = pd.concat([candidates, en_candidates]).drop_duplicates().reset_index(
                #     drop=True)
                print('EN----------', org)
                print(candidates)
            if candidates.size == 0:
                # no candidate
                candidates_set.append((None, 0))

                candidate = {'org_name':org_name, 'entity_name':org_name,'id':-1}
                if matched_maya:
                    candidate['sub_type'] = 'company'
                    candidate['type'] = 'company'
                else:
                    candidate['sub_type'] = 'other'
                    candidate['type'] = 'other'
            else:
                # look for the one with the smallest edit distance
                cell_name = 'name' if ldetect(org) == 'he' else 'name_eng'
                rows = map(lambda r: r[1], candidates.iterrows())
                best_candidate = min(rows, key=lambda row: fuzz.token_sort_ratio(
                    org, str(row[cell_name])))
                # print('chosen:', row['name'])
                candidate = {'org_name': org_name, 'entity_name': best_candidate['name'],
                             'id':best_candidate['id'],'sub_type':best_candidate['type'],
                             'type':type_map[best_candidate['type']]}
            candidates_set.append((candidate, candidates.size))
        total_num_candidates = sum([c[1] for c in candidates_set])
        candidate_match = max(candidates_set, key=lambda c:c[1] if c[1]>0 else float('inf'))[0]
        # new_orgs = {'org_name': [], 'entity_name': [], 'id': [], 'type': [], 'sub_type': [],
        #         'candidates': []}
        for k in new_orgs:
            if k != 'candidates':
                new_orgs[k].append(candidate_match[k])
        new_orgs['candidates'].append(total_num_candidates)
        if (i + 1) % 50 == 0:

            print('done', i)
            break

    # pd.DataFrame(new_orgs).to_csv('data/ynet/filtered_by_entity_db.csv', encoding='utf8')
    # pd.DataFrame({'org_name': unmatched_orgs}).to_csv("data/ynet/unmatched_orgs.csv",encoding='utf8')
    pd.DataFrame(new_orgs).to_csv('data/ynet/ynet_matched.csv', encoding='utf8')
    pd.DataFrame(new_orgs).to_excel('data/ynet/ynet_matched.xlsx', encoding='utf8')
    # pd.DataFrame({'org_name': unmatched_orgs}).to_csv("data/ynet/unmatched_orgs.csv", encoding='utf8')
    return new_orgs

def clean_main_office():
    # data['result'] = data['result'].map(lambda x: x.lstrip('+-').rstrip('aAbBcC'))
    entities_dataset = pd.read_csv(ENTITIES_DATABASE_ORIG)
    entities_dataset['name'] = entities_dataset['name'].map(lambda x: str(x).replace('/המשרד הראשי',
                                                                                  ''))
    entities_dataset.to_csv(ENTITIES_DATABASE)
    entities_dataset.to_excel(ENTITIES_DATABASE.split('.')[0]+'.xlsx')

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
    # clean_main_office()
