# import spacy
# nlp = spacy.load('en_core_web_sm')
# from polyglot.text import Text
from collections import Counter
from itertools import combinations
import pickle
import tqdm
import pandas as pd
# pd = None
import csv
import wikipedia
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import re
from fuzzywuzzy import fuzz
ORG_TAG = 'I-ORG'

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def string_from_entity(entity):
    return ' '.join(entity._collection)

def articles_generator(path):
    with open(path, "r",
              encoding="utf8") as f:
        text = f.read()
    return (t.split(URL_SPLIT) for t in text.split(ARTICLE_SPLIT)[:-1])

def save_to_csv(path):
    with open(path.replace(".txt", ".csv"), "w", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(['url', 'text'])
        for url, text in articles_generator(path):
            writer.writerow([url, text])


def analyze_discovery_percentage():
    with open("/Users/danamir/CS/Needle in the haystack/FinalProject/articles_text_heb_20042019.txt", "r",
              encoding="utf8") as f:
        text = f.read()
    counter = load_pickle("/Users/danamir/CS/Needle in the "
                          "haystack/FinalProject/orgainizations_counts_0209.pkl")
    terms_counts = [(term, count) for term, count in counter.items()]
    terms_counts_total = [{"term": term, "positive_count": count, "total_count": text.count(term)} for
                          term, count in tqdm.tqdm(terms_counts)]
    df = pd.DataFrame(terms_counts_total)
    df["discovery_ratio"] = df["positive_count"] / df["total_count"]
    df.to_csv("discovery_ratio_0209.csv", encoding="utf8")
    print("here")

def filter_valid_terms(min_count=10, min_discovery_ratio=0.1, in_path="discovery_ratio_0209.csv",
                       out_path="filtered_organizations_0209.csv"):
    data = pd.read_csv(in_path, encoding='utf8')
    data = data[data['positive_count'] > min_count]
    data = data[data['discovery_ratio'] > min_discovery_ratio]
    data['term'].to_csv(out_path, encoding='utf8')


def collect_wiki_categories_for_entities(entities_to_search):
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
    counts.to_csv('categories_counts.csv',encoding='utf8')

def find_names_from_maya(entities_to_search):
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
    data.to_csv('maya_name_matches.csv', encoding='utf8')
        # elem.send_keys(Keys.RETURN)



# nlp = spacy.load('en')
ARTICLE_SPLIT = "****END_ARTICLE*****"
URL_SPLIT = "****END_URL*****"
RUN_NAME = "heb_20042019"


def manually_filter_terms(filtered_terms):
    result = []
    for term in filtered_terms:
        while True:
            print(term)
            answer = input('is it a valid organization?')
            if answer == 'y':
                result.append(term)
                break
            elif answer == 'n':
                break
    pd.DataFrame({'term': result}).to_csv('filtered_organizations_0209_manual_elimination.csv',
                                          encoding='utf8')


def find_merge_candidates_by_fuzzy_matching(terms):
    def _score(term1, term2):
        return fuzz.token_sort_ratio(term1, term2)
    def _get_kth_elem(list_of_tuples, i):
        return [x[i] for x in list_of_tuples]
    matches = [(t1, t2, _score(t1,t2)) for t1, t2 in combinations(terms, 2)]
    df = pd.DataFrame({'term_1': _get_kth_elem(matches, 0),
                       'term_2': _get_kth_elem(matches, 1),
                       'score': _get_kth_elem(matches, 2)})
    df = df.sort_values('score', ascending=False)[:1000]
    df.to_csv('organizations_fuzzy_matching.csv', encoding='utf8')


def find_merge_candidates_by_wikipedia_search(terms, categories):
    wikipedia.set_lang('he')
    categories = set(categories)
    term2page = {}
    for term in tqdm.tqdm(terms):
        try:
            results = wikipedia.search(term, results=5)
            pages = [wikipedia.page(x) for x in results]
            cat_matches = [len(categories.intersection(p.categories)) > 0 for p in pages]
            if not any(cat_matches):
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
    df.to_csv("wikipedia_merge_candidates.csv", encoding='utf8')

def merge_entities_sets(merge_pairs, org_counts, org_pairs_counts):
    orgs2merge_sets = {}
    for org1, org2 in merge_pairs:
        if (org1 in orgs2merge_sets) and (org2 in orgs2merge_sets):
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
        set2term[set] = max(set, key= lambda org: org_counts[org])
        print("for set: {} chose representative: {}".format(set, set2term[set]))
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

    save_pickle(new_org_counts, 'organizations_counts_0209_merged.pkl')
    save_pickle(new_pairs_counts, 'organizations_pairs_counts_0209_merged.pkl')


def filter_counters_by_org_list(org_counts, org_pairs_counts, filtered_list):
    filtered_set = set(filtered_list)
    org_counts = {k:v for k,v in org_counts.items() if k in filtered_set}
    org_pairs_counts = {k: v for k, v in org_pairs_counts.items() if all([(x in filtered_set) for x in k])}
    return org_counts, org_pairs_counts


def load_merge_pairs():
    def extract_pairs(path):
        pairs = pd.read_csv(path, encoding='utf8')
        pairs = list(zip(pairs['term1'].values, pairs['term2'].values))
        return pairs
    wiki_pairs = extract_pairs("wikipedia_merge_candidates_filtered.csv")
    fuzzy_matching_pairs = extract_pairs("organizations_fuzzy_matching_after_elimination.csv")
    fuzzy_partial_pairs = extract_pairs("organizations_fuzzy_partial_matching_after_elimination.csv")
    return wiki_pairs + fuzzy_matching_pairs + fuzzy_partial_pairs


if __name__ == "__main__":
    # orgs_counter = load_pickle('orgainizations_counts.pkl')
    # save_to_csv("articles_text_heb_20162019.txt")
    # analyze_discovery_percentage()
    # filter_valid_terms()
    filtered_terms = list(pd.read_csv('filtered_organizations_0209_manual_elimination.csv',encoding='utf8')['term'].values)
    # categories = list(pd.read_csv('categories_counts_filtered.csv', encoding='utf8')['category'].values)
    # find_merge_candidates_by_wikipedia_search(filtered_terms, categories)
    org_counter = load_pickle('orgainizations_counts_0209.pkl')
    pairs_counter = load_pickle('orgainizations_pairs_counts_0209.pkl')
    org_counter, pairs_counter = filter_counters_by_org_list(org_counter, pairs_counter, filtered_terms)
    merge_pairs = load_merge_pairs()


    merge_entities_sets(merge_pairs=merge_pairs, org_counts=org_counter, org_pairs_counts=pairs_counter)

    # collect_wiki_categories_for_entities(filtered_terms)
    # manually_filter_terms(filtered_terms)
    # find_names_from_maya(filtered_terms)
    # orgs_counter = Counter()
    # orgs_pairs_counter = Counter()
    # with open("articles_text_{}.txt".format(RUN_NAME), "r",
    #           encoding="utf8") as f:
    #     text = f.read()
    # for article in tqdm.tqdm(text.split(ARTICLE_SPLIT)[:-1]):
    #     if URL_SPLIT not in article:
    #         continue
    #     url, article = article.split(URL_SPLIT)
    #     doc = Text(article)
    #     doc.hint_language_code = 'he'
    #     try:
    #         entities = doc.entities
    #     except UnboundLocalError as e: # Some bug in polyglot..
    #         print("failed to extract NER for article in url: %s with text: \n %s" % (url, article))
    #         continue
    #     org_entities = set([string_from_entity(ent) for ent in entities if ent.tag == ORG_TAG])
    #     orgs_counter.update(org_entities)
    #     orgs_pairs_counter.update(set([frozenset(c) for c in combinations(org_entities,2)]))
    # save_pickle(orgs_counter, "orgainizations_counts_0209.pkl")
    # save_pickle(orgs_pairs_counter, "orgainizations_pairs_counts_0209.pkl")
