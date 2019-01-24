# import spacy
# nlp = spacy.load('en_core_web_sm')
from polyglot.text import Text
from collections import Counter
from itertools import combinations
import pickle
import tqdm

ORG_TAG = 'I-ORG'

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def string_from_entity(entity):
    return ' '.join(entity._collection)

# nlp = spacy.load('en')
ARTICLE_SPLIT = "****END_ARTICLE*****"
URL_SPLIT = "****END_URL*****"
RUN_NAME = "heb_20162019"
if __name__ == "__main__":
    orgs_counter = Counter()
    orgs_pairs_counter = Counter()
    with open("articles_text_{}.txt".format(RUN_NAME), "r",
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
        except UnboundLocalError as e: # Some bug in polyglot..
            print("failed to extract NER for article in url: %s with text: \n %s" % (url, article))
            continue
        org_entities = set([string_from_entity(ent) for ent in entities if ent.tag == ORG_TAG])
        orgs_counter.update(org_entities)
        orgs_pairs_counter.update(combinations(org_entities,2))
    save_pickle(orgs_counter, "orgainizations_counts.pkl")
    save_pickle(orgs_pairs_counter, "orgainizations_pairs_counts.pkl")