try:
    from polyglot.text import Text
except:
    Text = None
from collections import Counter, defaultdict
from itertools import combinations
from utils import load_pickle, save_pickle, read_list_from_csv

import tqdm
import pandas as pd
import math
import csv
import wikipedia
import networkx as nx
from networkx.linalg.algebraicconnectivity import algebraic_connectivity
from functools import reduce
from matplotlib import pyplot as plt
from name_merging_tools import find_bottlenecks_in_pairs_graph, merge_entities_sets, find_best_textual_representative

try:
    from selenium import webdriver
    from selenium.webdriver.common.keys import Keys
except:
    webdriver = None
    Keys = None

import time
import re
from fuzzywuzzy import fuzz

ORG_TAG = 'I-ORG'


def string_from_entity(entity):
    return ' '.join(entity._collection)


def articles_generator(path):
    """
    generate iteratively pairs of (url, article_text)) from a given path with articles
    """
    with open(path, "r",
              encoding="utf8") as f:
        text = f.read()
    return (t.split(URL_SPLIT) for t in text.split(ARTICLE_SPLIT)[:-1])


def save_articles_to_csv(path):
    """
    utility for changing article data format
    """
    with open(path.replace(".txt", ".csv"), "w", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(['url', 'text'])
        for url, text in articles_generator(path):
            writer.writerow([url, text])


def analyze_discovery_percentage(articles_path="articles_text_heb_20042019.txt",
                                 org2counts_path="orgainizations_counts_0209.pkl",
                                 out_path='discovery_ratio_0209.csv'):
    """
    Calculate the ratio between the number of times a term was identified as ORG in NER extraction and the
    total number of occurrences
    :param articles_path: path to file with articles text
    :param org2counts_path: path to dictionary of organizations and their counts in Ynet data
    :param out_path: path to save result into
    """

    with open(articles_path, "r",
              encoding="utf8") as f:
        text = f.read()
    counter = load_pickle(org2counts_path)
    terms_counts = [(term, count) for term, count in counter.items()]
    terms_counts_total = [{"term": term, "positive_count": count, "total_count": text.count(term)} for
                          term, count in tqdm.tqdm(terms_counts)]
    df = pd.DataFrame(terms_counts_total)
    df["discovery_ratio"] = df["positive_count"] / df["total_count"]
    df.to_csv(out_path, encoding="utf8")



def filter_valid_terms(min_count=10, min_discovery_ratio=0.1, in_path="data/discovery_ratio_0209.csv",
                       out_path="data/filtered_organizations_0209.csv"):
    """
    filter terms from data based on their discovery ration and minimal number of appearances in corpus.
    Assumes that names with low discovery ratio are usually mistakes and names with low frequency will not
    allow proper analysis.
    """
    data = pd.read_csv(in_path, encoding='utf8')
    data = data[data['positive_count'] > min_count]
    data = data[data['discovery_ratio'] > min_discovery_ratio]
    data['term'].to_csv(out_path, encoding='utf8')


def collect_wiki_categories_for_entities(entities_to_search, out_path='categories_counts.csv'):
    """
    Collect counts for wikipedia categories related to the first result for each entity in
    entities_to_search. Those categories will then be used to find best result for top-5 wiki search
    results for entities.
    """
    wikipedia.set_lang('he')
    categories_counter = Counter()
    for entity in tqdm.tqdm(entities_to_search):

        first_res = wikipedia.search(entity, results=1)
        try:
            page = wikipedia.page(first_res)
        except Exception as e:
            continue
        for category in page.categories:
            categories_counter[category] += 1
    counts = pd.DataFrame({'category': list(categories_counter.keys()), 'count':
        list(categories_counter.values())})
    counts.to_csv(out_path, encoding='utf8')


def find_names_from_maya(entities_to_search, out_path='maya_name_matches_committees.csv'):
    """
    Use maya search engine to find mappings from our entities to company official name
    """
    names_data = {'name': [], 'official_name': []}
    base_url = 'https://maya.tase.co.il/'
    maya_title_suffix = ' - פרטי חברה - מאיה - מערכת אינטרנט להודעות | הבורסה לניירות ערך'
    driver = webdriver.Chrome()
    for entity in tqdm.tqdm(entities_to_search):
        driver.get(base_url)
        elem = driver.find_element_by_id('searchDesktop')
        elem.clear()
        elem.send_keys(entity)
        time.sleep(1)
        try:
            elem = driver.find_element_by_class_name('resultRow')
            ref = elem.get_attribute('href')
            driver.get(ref)
            title = driver.title[:-len(maya_title_suffix)]
            if entity in title.split(' '):
                print(title)
                names_data['name'].append(entity)
                names_data['official_name'].append(title)
                time.sleep(1)
        except Exception as e:
            print(e)
            continue
    data = pd.DataFrame(names_data)
    data.to_csv(out_path, encoding='utf8')




ARTICLE_SPLIT = "****END_ARTICLE*****"
URL_SPLIT = "****END_URL*****"
RUN_NAME = "heb_20042019"


def manually_filter_terms(terms, out_path='filtered_organizations_0209_manual_elimination.csv'):
    result = []
    for term in terms:
        while True:
            print(term)
            answer = input('is it a valid organization?')
            if answer == 'y':
                result.append(term)
                break
            elif answer == 'n':
                break
    pd.DataFrame({'term': result}).to_csv(out_path,
                                          encoding='utf8')


def find_merge_candidates_by_fuzzy_matching(terms, out_path='organizations_fuzzy_matching.csv'):
    """
    uses fuzzy matching to find candidates for merging
    :param terms: iterable of terms to match
    """
    def _score(term1, term2):
        return fuzz.token_sort_ratio(term1, term2)

    def _get_kth_elem(list_of_tuples, i):
        return [x[i] for x in list_of_tuples]

    matches = [(t1, t2, _score(t1, t2)) for t1, t2 in combinations(terms, 2)]
    df = pd.DataFrame({'term_1': _get_kth_elem(matches, 0),
                       'term_2': _get_kth_elem(matches, 1),
                       'score': _get_kth_elem(matches, 2)})
    df = df.sort_values('score', ascending=False)[:1000]
    df.to_csv(out_path, encoding='utf8')


def find_merge_candidates_by_wikipedia_search(terms, categories, out_path="wikipedia_merge_candidates.csv"):
    """
    find pairs of words which are mapped in wikipedia search to the same organization. uses categories
    indicative for organization pages to find best result. choosing first in top-5 matched to categories if
    none of them is matched, uses first result.
    :param terms: iterable of terms to merge
    :param categories: iterable of categories from wiki to match against.
    """
    wikipedia.set_lang('he')
    categories = set(categories)
    term2page = {}
    for term in tqdm.tqdm(terms):
        try:
            results = wikipedia.search(term, results=5)
            pages = [wikipedia.page(x) for x in results]
            cat_matches = [len(categories.intersection(p.categories)) > 0 for p in pages]
            if not any(cat_matches): # choose 1st as none of them matched
                term2page[term] = pages[0].title
            else:
                term2page[term] = pages[cat_matches.index(True)].title
        except:
            print("couldn't match wiki page for term %s" % term)
            continue

    merge_candidates = {'term1': [], 'term2': []}
    for term1, term2 in combinations(term2page.keys(), 2):
        if term2page[term1] == term2page[term2]:
            merge_candidates['term1'].append(term1)
            merge_candidates['term2'].append(term2)
    df = pd.DataFrame(merge_candidates)
    df.to_csv(out_path, encoding='utf8')



def map_set_to_term(orgs2merge_sets, org_counts):
    """
    Deprecated - mapping each merge set to term with highest count in ynet data
    """
    unique_merge_sets = []
    for m_set in orgs2merge_sets.values():
        if m_set not in unique_merge_sets:
            unique_merge_sets.append(m_set)
    unique_merge_sets = [frozenset(s) for s in unique_merge_sets]
    for merge_set in unique_merge_sets:
        for org in merge_set:
            orgs2merge_sets[org] = merge_set
    set2term = {}
    for set in unique_merge_sets:
        set2term[set] = max(set, key=lambda org: org_counts[org])
        print("for set: {} chose representative: {}".format(set, set2term[set]))
    return set2term



def update_counts_by_merge_set(orgs2merge_sets, set2term, org_counts, org_pairs_counts,
                               org_counts_path='organizations_counts_0209_merged.pkl',
                               pairs_counts_path='organizations_pairs_counts_0209_merged.pkl'):

    new_org_counts = Counter()
    new_pairs_counts = Counter()
    for org, count in org_counts.items():
        if org not in orgs2merge_sets:
            new_org_counts[org] += count
        else:
            new_org_counts[set2term[orgs2merge_sets[org]]] += count
    save_pickle(orgs2merge_sets, 'orgs2mergesets.pkl')
    save_pickle(set2term, 'mergesets2name.pkl')
    for pair, count in org_pairs_counts.items():
        pair_list = list(pair)
        for i in range(len(pair_list)):
            org = pair_list[i]
            if org in orgs2merge_sets:
                pair_list[i] = set2term[orgs2merge_sets[org]]
        pair = frozenset(pair_list)
        if len(pair) == 1:
            continue
        new_pairs_counts[pair] += count

    save_pickle(new_org_counts, org_counts_path)
    save_pickle(new_pairs_counts, pairs_counts_path)


def filter_counters_by_org_list(org_counts, org_pairs_counts, filtered_list):
    filtered_set = set(filtered_list)
    org_counts = {k: v for k, v in org_counts.items() if k in filtered_set}
    org_pairs_counts = {k: v for k, v in org_pairs_counts.items() if all([(x in filtered_set) for x in k])}
    return org_counts, org_pairs_counts


def load_merge_pairs():
    # load all merge pairs for ynet
    def extract_pairs(path):
        pairs = pd.read_csv(path, encoding='utf8')
        pairs = list(zip(pairs['term1'].values, pairs['term2'].values))
        return pairs

    wiki_pairs = extract_pairs("wikipedia_merge_candidates_filtered.csv")
    fuzzy_matching_pairs = extract_pairs("organizations_fuzzy_matching_after_elimination.csv")
    fuzzy_partial_pairs = extract_pairs("organizations_fuzzy_partial_matching_after_elimination.csv")
    return wiki_pairs + fuzzy_matching_pairs + fuzzy_partial_pairs


def merge_and_filter_pairs():
    ynet_pairs = load_merge_pairs()
    pairs1 = load_pickle('pairs_1_fratio_greater_than_89.pkl')
    pairs2 = load_pickle('pairs_2_iou_and_intersection.pkl')
    pairs3 = load_pickle('pairs_3_iou_and_fration_70.pkl')
    pairs4 = load_pickle('pairs_4_fratio_89_manual.pkl')
    guests_pairs = pairs1 + pairs2 + pairs3 + pairs4
    to_remove_pairs = pd.read_csv('pairs_to_remove.txt', header=None, names=['term1', 'term2'])
    to_remove_pairs = list(set(zip(to_remove_pairs.term1.tolist(), to_remove_pairs.term2.tolist())))
    to_remove_pairs = [(x[0].strip().strip("'"), x[1].strip().strip("'")) for x in to_remove_pairs]

    all_pairs = ynet_pairs + guests_pairs
    for pair in to_remove_pairs:
        if pair in all_pairs:
            all_pairs.remove(pair)
        elif (pair[1], pair[0]) in all_pairs:
            all_pairs.remove((pair[1], pair[0]))
        else:
            print("couldn't find pair: ", pair)
    save_pickle(all_pairs, "all_pairs_with_fixes_final.pkl")


def convert_pickle_to_csv(pkl_path='articles_orgs_baskets_190225.pkl', csv_path="articles_orgs_baskets_190225.csv"):
    article2orgs = load_pickle(pkl_path)
    data_rows = {"organization": [], "article": []}
    for article, orgs in article2orgs.items():
        for org in orgs:
            data_rows['organization'].append(org)
            data_rows['article'].append(article.strip())
    df = pd.DataFrame(data_rows)
    df.to_csv(csv_path, encoding='utf8', index=False)

def extract_ner_from_articles(articles_path='articles_text_{}.txt".format(RUN_NAME)', filtered_terms=None,
                              org2mergesets=None, mergesets2name=None):
    orgs_counter = Counter()
    orgs_pairs_counter = Counter()
    articles_orgs_baskets = {}
    orgs2articles = defaultdict(lambda: set())
    with open(articles_path, "r",
              encoding="utf8") as f:
        text = f.read()
    for article in tqdm.tqdm(text.split(ARTICLE_SPLIT)[:-1]):
        if URL_SPLIT not in article:
            continue
        url, article = article.split(URL_SPLIT)
        doc = Text(article)
        doc.hint_language_code = 'he'
        try:
            entities = doc.entities
        except UnboundLocalError as e:  # Some bug in polyglot..
            print("failed to extract NER for article in url: %s with text: \n %s" % (url, article))
            continue
        org_entities = set([string_from_entity(ent) for ent in entities if ent.tag == ORG_TAG])
        if filtered_terms is not None:
            org_entities = filter(lambda x: x in filtered_terms, org_entities)
            org_entities = map(lambda x: x if x not in orgs2mergesets else mergesets2name[frozenset(
                                                                                          orgs2mergesets[x])],
                           org_entities)
        org_entities = set(list(org_entities))
        articles_orgs_baskets[url] = org_entities
        orgs_counter.update(org_entities)
        orgs_pairs_counter.update(set([frozenset(c) for c in combinations(org_entities, 2)]))
        for org in org_entities:
            orgs2articles[org].add(url)

    # edit paths if neccessary!
    save_pickle(orgs_counter, "orgainizations_counts_190225_after_merges.pkl")
    save_pickle(orgs_pairs_counter, "orgainizations_pairs_counts_190225_after_merges.pkl")
    save_pickle(articles_orgs_baskets, "articles_orgs_baskets_190225.pkl")
    save_pickle(dict(orgs2articles), "orgs_to_articles_190225.pkl")

if __name__ == "__main__":
    # uncomment relevant call:

    # extract_ner_from_articles()


    # guests = pd.read_csv('guests_final_no_intersection_merges.csv', encoding='utf8')
    # orgs = guests.company.unique().tolist()
    # find_names_from_maya(orgs)

    # unite_orgs_and_map_to_terms()


    # analyze_discovery_percentage()

    # filter_valid_terms()

    # categories = list(pd.read_csv('categories_counts_filtered.csv', encoding='utf8')['category'].values)
    # find_merge_candidates_by_wikipedia_search(filtered_terms, categories)

    # merge_entities_sets(merge_pairs=merge_pairs, org_counts=org_counter, org_pairs_counts=pairs_counter)
    # filtered_terms = list(
    #     pd.read_csv('filtered_organizations_0209_manual_elimination.csv', encoding='utf8')['term'].values)
    # org_counter, pairs_counter = filter_counters_by_org_list(org_counter, pairs_counter, filtered_terms)
    # org_counter = load_pickle('orgainizations_counts_0209.pkl')
    # pairs_counter = load_pickle('orgainizations_pairs_counts_0209.pkl')
    # collect_wiki_categories_for_entities(filtered_terms)
    # manually_filter_terms(filtered_terms)
    # find_names_from_maya(filtered_terms)
    # pairs = load_merge_pairs()
    # pairs1 = load_pickle('pairs_1_fratio_greater_than_89.pkl')
    # pairs2 = load_pickle('pairs_2_iou_and_intersection.pkl')
    # pairs3 = load_pickle('pairs_3_iou_and_fration_70.pkl')
    # pairs = pairs1 + pairs2 + pairs3 + pairs
    # find_bottlenecks_in_pairs_graph(pairs)
    # convert_pickle_to_csv()
    # merge_and_filter_pairs()
    # find_best_textual_representative()
    orgs2mergesets = load_pickle("all_data_org2mergeset.pkl")
    mergesets2name = load_pickle("all_data_set2term.pkl")
    filtered_terms = set(read_list_from_csv('filtered_organizations_0209_manual_elimination.csv', 'term'))

