import pandas as pd
from apyori import apriori
from collections import namedtuple
from functools import partial
import numpy as np
from scipy.sparse import dok_matrix
from ExtractEntities import load_pickle
from matplotlib import pyplot as plt
from sklearn.cluster import SpectralClustering, SpectralCoclustering
from itertools import product
import networkx as nx
from functools import reduce
import holoviews as hv
from langdetect import detect as ldetect
from bokeh.models import GraphRenderer
from bokeh.models.graphs import NodesAndLinkedEdges
import seaborn as sns
MATCHES_PATH = 'all_org_to_entity_matches.csv'
ENTITY = 'entity'
BASKET = 'basket'
YNET_KEYS = {'entity': 'organization', 'basket': 'article'}
COMMITTEES_KEYS = {'entity': 'company', 'basket': 'meeting_id'}
pair = namedtuple('Pair',['items','score'])
SECTOR2NUM = {
    'government': 0,
    'company': 1,
    'other': 2
}

KEYS_DICT = {
    'ynet': YNET_KEYS,
    'committees': COMMITTEES_KEYS
}

def edit_plot_height_and_width(html_file, height, width):
    import bs4
    with open(html_file, 'r') as inf:
        txt = inf.read()
        soup = bs4.BeautifulSoup(txt)
        js = soup.find_all(attrs={"type": "text/javascript"})[-1]
        # js = soup.find_all(text = re.compile("plot_height"))
        fixed_text = js.string.replace('\"plot_height\":300', '\"plot_height\":' + str(height))
        fixed_text = fixed_text.replace('\"plot_width\":300', '\"plot_width\":' + str(width))
        js.string.replace_with(fixed_text)
    with open(html_file, 'w') as f:
        f.write(str(soup))

def datarows_to_sets(df, key_column='organization', value_column='article'):
    return df.groupby(key_column)[value_column].unique().apply(lambda x: set(x.tolist())).to_dict()

def build_index_dictionary(organizations):
    return {i: org for i, org in enumerate(organizations)}, {org: i for i, org in enumerate(organizations)}


def normalize_scores(pairs, apply_root=False):
    max_val = max(pairs, key=lambda x: x.score).score
    def norm(s):
        s = s / max_val
        if apply_root:
            s = s ** 0.5
        return s
    return [pair(p.items, norm(p.score)) for p in pairs]


def analyze_graph(df, data_kind='ynet', similarity='pmi', min_support=0.0002, save_to='out.html'):
    keys = KEYS_DICT[data_kind]
    orgs2baskets = datarows_to_sets(df, key_column=keys[ENTITY], value_column=keys[BASKET])
    baskets2orgs = datarows_to_sets(df, key_column=keys[BASKET], value_column=keys[ENTITY])
    pairs = extract_pair_by_min_support(baskets2orgs, min_support=min_support, similarity=similarity,
                                        to_print=True, org2basket=orgs2baskets)
    pairs = normalize_scores(pairs, apply_root=similarity=='jaccard')
    pairs = [p for p in pairs if p.score > 0]
    matches = pd.read_csv(MATCHES_PATH, encoding='utf8', index_col='org_name').to_dict(orient='index')
    def create_desc_dict(org):
        keys = ['type','entity_name']
        return {k: matches[org][k] for k in keys}
    org2desc = {org: create_desc_dict(org) for org in orgs2baskets.keys()}
    idx2org, org2idx = build_index_dictionary(orgs2baskets.keys())
    create_network_graph(org2idx, orgs2baskets, org2desc, pairs, save_to=save_to, color_by='cluster')
    return pairs

def get_pmi(result):
    return np.log(result.ordered_statistics[0].lift)

def get_weighted_pmi(result, total_count, subtract):
    supp = result.support * total_count
    lift = result.ordered_statistics[0].lift
    return np.log(lift * ((supp - subtract) / supp) + (subtract / supp))



def calculate_jaccard(result, org2basket):
    item1, item2 = result.items
    intersection = org2basket[item1].intersection(org2basket[item2])
    union = org2basket[item1].union(org2basket[item2])
    return len(intersection) / len(union)

def extract_pair_by_min_support(baskets_dict, min_support=0.0002, filter_set=None, similarity='pmi',
                                to_print=False, **kwargs):
    if filter_set is None:
        filter_set = set()

    min_count = len(baskets_dict) * min_support
    print("min support equivalent to {} baskets".format(min_count))
    score_to_func = {
        'pmi': get_pmi,
        'weighted_pmi': partial(get_weighted_pmi, total_count=len(baskets_dict), subtract=min_count // 2),
        'jaccard': partial(calculate_jaccard, org2basket=kwargs['org2basket'])
    }
    baskets_list = [list(baskets_dict[b].difference(filter_set)) for b in baskets_dict]
    result = apriori(baskets_list, min_support=min_support, min_confidence=0.0, max_length=2)
    result = filter(lambda r: len(r.items) > 1, result)
    final_results = []
    for i, r in enumerate(result):
        r = pair(items=r.items, score=score_to_func[similarity](r))
        final_results.append(r)
    final_results = sorted(final_results, key=lambda x: x.score, reverse=True)
    if to_print:
        print("\n".join([str(x) for x in final_results[:1000]]))
    return final_results

def spectral_bi_cluster(org2idx: dict, org2desc: dict, pairs, affinity_matrix):
    def valid_pair(p):
        o1, o2 = p.items
        types = org2desc[o1]['type'], org2desc[o2]['type']
        return ('government' in types) and ('company' in types)
    pairs = [p for p in pairs if valid_pair(p)]
    nodes = set(list(reduce(lambda x,y: list(x) + list(y), [p.items for p in pairs])))
    gov_orgs = [org for org in org2desc.keys() if (org2desc[org]['type'] == 'government') and org in nodes]
    com_orgs = [org for org in org2desc.keys() if org2desc[org]['type'] == 'company' and org in nodes]
    _, gov_orgs2idx = build_index_dictionary(gov_orgs)
    _, com_orgs2idx = build_index_dictionary(com_orgs)
    print(len(com_orgs), len(gov_orgs))
    bipartite_matrix = dok_matrix((len(com_orgs), len(gov_orgs)))
    for com, gov in product(com_orgs, gov_orgs):
        bipartite_matrix[com_orgs2idx[com], gov_orgs2idx[gov]] = affinity_matrix[org2idx[com], org2idx[gov]]
    bipartite_matrix = bipartite_matrix.tocsc()
    print(bipartite_matrix.min())
    num_clusters = 19
    co_clustering = SpectralCoclustering(num_clusters)
    co_clustering.fit(bipartite_matrix)
    clustering = np.full((len(org2idx),), fill_value= num_clusters,dtype=np.int32)
    for org, idx in org2idx.items():
        org_type = org2desc[org]['type']
        if (org not in gov_orgs2idx) and (org not in com_orgs2idx):
            continue
        new_idx = gov_orgs2idx[org] if org_type == 'government' else com_orgs2idx[org]
        clust = co_clustering.column_labels_ if org_type == 'government' else co_clustering.row_labels_
        clustering[idx] = clust[new_idx]
    return clustering


def create_network_graph(org2idx, org2baskets, org2desc ,pairs, save_to='out.html',
                         color_by='cluster',size_by='log_count'):


    N = len(org2idx)
    affinity_matrix = dok_matrix((N, N), dtype=np.float32)
    for edge in pairs:
        org1, org2 = edge.items
        affinity_matrix[org2idx[org1], org2idx[org2]] = edge.score
        affinity_matrix[org2idx[org2], org2idx[org1]] = affinity_matrix[org2idx[org1], org2idx[org2]]

    # spectral clustering
    # clustering = SpectralClustering(affinity='precomputed', n_clusters=20)
    # clusters = clustering.fit_predict(affinity_matrix.tocsc())
    clusters = spectral_bi_cluster(org2idx, org2desc, pairs, affinity_matrix)
    graph = nx.Graph()
    nodes = reduce(lambda x, y: list(x) + list(y), [x.items for x in pairs])
    nodes = list(set(nodes))
    nodes = [(x, {"sector": org2desc[x]['type'] , "sector_num": SECTOR2NUM[org2desc[x]['type']] , 'cluster':
    clusters[
        org2idx[x]], 'log_count': np.log(len(org2baskets[x])) * 2}) for x in nodes]
    graph.add_nodes_from(nodes)
    graph_edges = []
    for edge in pairs:
        p1, p2 = edge.items
        val = edge.score
        types = [org2desc[p1]['type'], org2desc[p2]['type']]
        if 'government' in types and 'company' in types:
            types = 1
        else:
            types=0.2
        graph_edges.append((p1, p2, {'weight': val, 'types': types}))
    graph.add_edges_from(graph_edges)

    hv.extension('bokeh')
    graph = hv.Graph.from_networkx(graph, nx.layout.fruchterman_reingold_layout, weight='weight')

    graph.opts(node_cmap='Category20', node_size=size_by, edge_line_width='weight',
               node_color=color_by, node_line_width=0, edge_color='types', edge_cmap='Greys')
    hv.save(graph,save_to)
    edit_plot_height_and_width(save_to, 800, 1200)

def get_matches(df: pd.DataFrame, data_type):
    matches = pd.read_csv(MATCHES_PATH, encoding='utf8')
    companies = set(df[KEYS_DICT[data_type][ENTITY]].unique().tolist())
    matches = matches[matches.org_name.apply(lambda n: n in companies)]
    matches.sub_type[matches.sub_type != matches.sub_type] = 'other'
    matches.sub_type = matches.sub_type.apply(lambda x: x if ldetect(x) != 'he' else x[::-1])
    return  matches

def types_histogram():
    data_type = 'committees'
    df = pd.read_csv(paths[data_type], encoding='utf8')

    matches_com = get_matches(df, data_type)
    data_type = 'ynet'
    df = pd.read_csv(paths[data_type], encoding='utf8')
    matches_gov = get_matches(df, data_type)
    matches_com['dataset'] = 'committees'
    matches_gov['dataset'] = 'ynet'
    matches = pd.concat([matches_com, matches_gov])
    matches = matches[matches.sub_type != 'other']
    ax = sns.countplot(x='sub_type', hue='dataset', data=matches)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig('sub_types_hist.pdf')

if __name__ == '__main__':
    data_type = 'committees'
    paths = {
        'committees': 'filtered_guests_final.csv',
        'ynet': 'articles_orgs_baskets_190225.csv'
    }
    df = pd.read_csv(paths[data_type], encoding='utf8')
    similarity = 'pmi'
    pairs = analyze_graph(df, data_type, similarity, 0.0002, '{}_{}_bi_spectral.html'.format(data_type, similarity))
    # scores = [p.score for p in pairs]
    # scores = np.array(scores)
    # scores /= scores.max()
    # scores = scores ** 0.5
    # plt.hist(scores, bins=50)
    # plt.figure()
    # pairs = analyze_graph(df, 'ynet', 'pmi', 0.0002)
    # scores = [p.score for p in pairs]
    # scores = np.array(scores)
    # scores /= scores.max()
    # scores = scores[scores >= 0]
    # plt.hist(scores, bins=50)
    # plt.show()
