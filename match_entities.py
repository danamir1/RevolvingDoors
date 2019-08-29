import pandas as pd
import numpy as np
import re
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering, SpectralClustering

from langdetect import detect as ldetect
import tqdm
import pickle
import json

from multiprocessing.pool import ThreadPool
from fuzzywuzzy import fuzz



ENTITIES_DATABASE = 'entities_database_with_universities.csv'
MAYA_MATCH = 'maya_name_matches.csv'
TYPE_MAP = 'types_mapping.json'
MERGE = 'all_data_org2mergeset_final.pkl'

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

def get_candidate(original_org, org, entities_dataset, maya, type_map):
    matched_maya = False
    # first use maya's matches -- names that we already have their mapping to official name
    maya_match = maya[maya['name'] == org]
    if maya_match.size > 0:
        new_org = maya_match['official_name'].iloc[0]
        org = new_org
        matched_maya = True

    try:
        candidates = entities_dataset[
            entities_dataset['name'].str.contains('\\b' + org + '\\b', regex=True) == True]
        if ldetect(org) != 'he':
            candidates = entities_dataset[entities_dataset['name_eng'].str.contains(
                '\\b({}|{}|{})\\b'.format(org.upper(), org, org.lower()), regex=True) == True]

    except:
        print('except', org)
        candidates = pd.DataFrame(columns=list(entities_dataset.columns.values))
    if candidates.size == 0:
        # no candidate
        candidate = {'org_name': original_org, 'entity_name': original_org, 'id': -1}
        if matched_maya:
            candidate['sub_type'] = 'company'
            candidate['type'] = 'company'
            candidate['score'] = float('inf')
        else:
            candidate['sub_type'] = 'other'
            candidate['type'] = 'other'
            candidate['score'] = -float('inf')

    else:
        # look for the one with the smallest edit distance
        cell_name = 'name' if ldetect(org) == 'he' else 'name_eng'
        rows = list(map(lambda r: r[1], candidates.iterrows()))
        fuzz_scores = [fuzz.token_sort_ratio(org, str(row[cell_name])) for row in rows]
        max_index, max_score = max(enumerate(fuzz_scores), key=lambda x: x[1])
        best_candidate = rows[max_index]

        candidate = {'org_name': original_org, 'entity_name': best_candidate['name'],
                     'id': best_candidate['id'], 'sub_type': best_candidate['type'],
                     'type': type_map[best_candidate['type']],
                     # 'score': candidates.shape[0]}
                     'score': max_score}
    return candidate

def choose_candidate_from_merge_set(original_org, org_set,entities_dataset, maya, type_map):
    candidates_set = []
    for org in org_set:
        candidate = get_candidate(original_org, org, entities_dataset, maya, type_map)
        candidates_set.append(candidate)
    return max(candidates_set, key=lambda x: x['score'])

def match_entities_multithread(orgs):
    #  matching of entities to the entities dataset based on fuzzy matching and merge sets.
    new_orgs = {'org_name': [], 'entity_name': [], 'id': [], 'type': [], 'sub_type': [],
                'candidates': []}
    maya = pd.read_csv(MAYA_MATCH)
    merge = load_pickle(MERGE)
    entities_dataset = pd.read_csv(ENTITIES_DATABASE)

    type_map = get_type_mapping()
    pool = ThreadPool(10)
    for i, org_name in enumerate(tqdm.tqdm(orgs)):
        org_set = merge[org_name] if org_name in merge else {org_name}

        if len(org_set) > 1:
            # use thread pool for parallel matching for terms in merge set
            threads = []
            candidates = []
            for org in org_set:
                async_result = pool.apply_async(get_candidate,(org_name, org, entities_dataset, maya,
                                                               type_map))
                threads.append(async_result)
            for thread in threads:
                candidates.append(thread.get())

            result = max(candidates, key=lambda x: x['score'])
        else:
            result = choose_candidate_from_merge_set(org_name, org_set, entities_dataset, maya, type_map)

        for k in new_orgs:
            if k != 'candidates':
                new_orgs[k].append(result[k])
        new_orgs['candidates'].append(0)
    pool.close()
    pd.DataFrame(new_orgs).to_csv('ynet_matched_final.csv', encoding='utf8')
    pd.DataFrame(new_orgs).to_excel('ynet_matched_final.xlsx', encoding='utf8')

    return new_orgs


if __name__ == '__main__':

    orgs = list(load_pickle('orgs_to_articles_190225.pkl').keys())
    matches = match_entities_multithread(orgs=orgs)

