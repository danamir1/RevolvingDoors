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
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
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
paths = {
        'committees': 'filtered_guests_final.csv',
        'ynet': 'articles_orgs_baskets_190225.csv'
    }

# -------- APML clustering code ----------------

def kmeans_pp_init(X, k, metric):
    """
    The initialization function of kmeans++, returning k centroids.
    :param X: The data matrix.
    :param k: The number of clusters.
    :param metric: a metric function like specified in the kmeans documentation.
    :return: kxD matrix with rows containing the centroids.
    """
    N = X.shape[0]
    current_centroids = X[[np.random.choice(np.arange(0,N))],:]
    while current_centroids.shape[0] < k:
        current_distances = metric(X,current_centroids)
        current_minimal_distance = np.min(current_distances, axis=1)
        sampling_prob = np.power(current_minimal_distance, 2)
        sampling_prob = sampling_prob / np.sum(sampling_prob)
        new_centroid = X[[np.random.choice(np.arange(0,N),p=sampling_prob)], :]
        current_centroids = np.concatenate((current_centroids, new_centroid))
    return current_centroids

def euclid(X, Y):
    """
    return the pair-wise euclidean distance between two data matrices.
    :param X: NxD matrix.
    :param Y: MxD matrix.
    :return: NxM euclidean distance matrix.
    """
    N, d = X.shape
    M, d2 = Y.shape
    assert d2 == d
    distances = cdist(X,Y, metric='euclidean')
    return distances


def euclidean_centroid(X):
    """
    return the center of mass of data points of X.
    :param X: a sub-matrix of the NxD data matrix that defines a cluster.
    :return: the centroid of the cluster.
    """

    return np.mean(X, axis=0)

def kmeans(X, k, iterations=10, metric=euclid, center=euclidean_centroid, init=kmeans_pp_init):
    """
    The K-Means function, clustering the data X into k clusters.
    :param X: A NxD data matrix.
    :param k: The number of desired clusters.
    :param iterations: The number of iterations.
    :param metric: A function that accepts two data matrices and returns their
            pair-wise distance. For a NxD and KxD matrices for instance, return
            a NxK distance matrix.
    :param center: A function that accepts a sub-matrix of X where the rows are
            points in a cluster, and returns the cluster centroid.
    :param init: A function that accepts a data matrix and k, and returns k initial centroids.
    :param stat: A function for calculating the statistics we want to extract about
                the result (for K selection, for example).
    :return: a tuple of (clustering, centroids, statistics)
    clustering - A N-dimensional vector with indices from 0 to k-1, defining the clusters.
    centroids - The kxD centroid matrix.
    """

    current_centroids = init(X=X,k=k, metric=metric)
    best_centroid_per_point = None
    for i in range(iterations):
        # find best cluster for each point
        distances_to_centroids = metric(X, current_centroids)
        best_centroid_per_point = np.argmin(distances_to_centroids, axis=1)
        new_centroids = []
        for j in range(k):
            new_j_centroid = center(X[best_centroid_per_point == j,:])
            new_centroids.append(new_j_centroid)
        current_centroids = np.array(new_centroids)

    stats = {}
    stats["final_distances"] = np.min(metric(X, current_centroids), axis=1)
    stats["total_cost"] = np.sum(stats["final_distances"])
    return best_centroid_per_point, current_centroids, stats

def spectral(Adj, k):
    """
    Cluster the data into k clusters using the spectral clustering algorithm.
    :param X: A NxD data matrix.
    :param k: The number of desired clusters.
    :param similarity_param: m for mnn, sigma for the Gaussian kernel.
    :param similarity: The similarity transformation of the data.
    :return: clustering, as in the kmeans implementation.
    """
    N = Adj.shape[0]
    adj_sum = np.sum(Adj, axis=0)
    # avoid division by zero
    adj_sum_nz = adj_sum.copy()
    adj_sum_nz[adj_sum == 0] = 1
    inv_sqrt_D = 1 / (adj_sum_nz ** 0.5)
    inv_sqrt_D[adj_sum == 0] = 1
    laplacian = np.eye(N) - (Adj * inv_sqrt_D.reshape((1,N)) * inv_sqrt_D.reshape((N,1))) # replace matrix
    # multiplication by diagonal matrix with multiplication with vector with brodacasting.
    laplacian = (0.5 * laplacian) + (0.5 * laplacian.T)
    w,v = eigh(laplacian)
    spectral_signature = v[:,:k]
    spect_norms = np.linalg.norm(spectral_signature, axis=1, keepdims=True)
    spect_norms[spect_norms == 0] = 1 # avoid zero division
    spectral_signature = spectral_signature / spect_norms
    clusters, centers, stats =  kmeans(spectral_signature,k)
    return clusters

# -------- APML clustering code ----------------

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

def get_pairs(df, data_kind, min_support=0.0002, similarity='jaccard'):
    keys = KEYS_DICT[data_kind]
    orgs2baskets = datarows_to_sets(df, key_column=keys[ENTITY], value_column=keys[BASKET])
    baskets2orgs = datarows_to_sets(df, key_column=keys[BASKET], value_column=keys[ENTITY])
    pairs = extract_pair_by_min_support(baskets2orgs, min_support=min_support, similarity=similarity,
                                        to_print=True, org2basket=orgs2baskets)
    pairs = normalize_scores(pairs, apply_root='root' in similarity)
    return pairs

def analyze_graph(df, data_kind='ynet', similarity='pmi', min_support=0.0002,
                  save_to='out.html', **clustering_kwargs):
    """
    construction and analysis of interests graph
    :param df: either single dataframe of item, set pairs from ynet or committees or list of them
    :param data_kind: 'ynet' , 'committees' or list of both
    :param similarity: name of similarity metric
    :param min_support: minimal count ratio of item appearances in sets
    :param save_to: html path to save interactive graphs to
    :param clustering_kwargs: arguments for the clustering method
    """
    if type(data_kind) in [list, tuple]: # handle edges from multiple data sources
        orgs2baskets = {}
        items2pairs = {}
        pairs_2maxscores = {}
        for kind, d, supp in zip(data_kind, df, min_support):
            keys = KEYS_DICT[kind]
            _orgs2baskets = datarows_to_sets(d, key_column=keys[ENTITY], value_column=keys[BASKET])
            _baskets2orgs = datarows_to_sets(d, key_column=keys[BASKET], value_column=keys[ENTITY])
            _pairs = extract_pair_by_min_support(_baskets2orgs, min_support=supp,
                                                 similarity=similarity,
                                                to_print=True, org2basket=_orgs2baskets)
            _pairs = normalize_scores(_pairs, apply_root='root' in similarity)
            _pairs = [p for p in _pairs if p.score > 0]
            for p in _pairs:
                if p.items in pairs_2maxscores and p.score < pairs_2maxscores[p.items]:
                    continue
                else:
                    pairs_2maxscores[p.items] = p.score
                    items2pairs[p.items] = p
            for org, basket in _orgs2baskets.items():
                if org in orgs2baskets:
                    orgs2baskets[org].update(basket)
                else:
                    orgs2baskets[org] = _orgs2baskets[org]

        pairs = list(items2pairs.values())
    else: # standard version
        keys = KEYS_DICT[data_kind]
        orgs2baskets = datarows_to_sets(df, key_column=keys[ENTITY], value_column=keys[BASKET])
        baskets2orgs = datarows_to_sets(df, key_column=keys[BASKET], value_column=keys[ENTITY])
        pairs = extract_pair_by_min_support(baskets2orgs, min_support=min_support, similarity=similarity,
                                            to_print=True, org2basket=orgs2baskets)
        pairs = normalize_scores(pairs, apply_root='root' in similarity)
        pairs = [p for p in pairs if p.score > 0]

    matches = pd.read_csv(MATCHES_PATH, encoding='utf8', index_col='org_name').to_dict(orient='index')
    def create_desc_dict(org):
        keys = ['type','entity_name']
        return {k: matches[org][k] for k in keys}
    org2desc = {org: create_desc_dict(org) for org in orgs2baskets.keys()}
    idx2org, org2idx = build_index_dictionary(orgs2baskets.keys())
    create_network_graph(org2idx, orgs2baskets, org2desc, pairs, save_to=save_to, color_by='cluster',
                         **clustering_kwargs)
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
        'jaccard': partial(calculate_jaccard, org2basket=kwargs['org2basket']),
        'root_jaccard': partial(calculate_jaccard, org2basket=kwargs['org2basket'])
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
                         color_by='cluster',size_by='log_count', clustering_type='spectral', n_clusters=20):


    N = len(org2idx)
    affinity_matrix = dok_matrix((N, N), dtype=np.float32)
    for edge in pairs:
        org1, org2 = edge.items
        affinity_matrix[org2idx[org1], org2idx[org2]] = edge.score
        affinity_matrix[org2idx[org2], org2idx[org1]] = affinity_matrix[org2idx[org1], org2idx[org2]]

    # spectral clustering
    if clustering_type == 'spectral':
        clustering = SpectralClustering(affinity='precomputed', n_clusters=n_clusters)
        clusters = clustering.fit_predict(affinity_matrix.tocsc())
    elif clustering_type == 'bi_spectral':
        clusters = spectral_bi_cluster(org2idx, org2desc, pairs, affinity_matrix)
    elif clustering_type == 'our_spectral':
        affinity_matrix = affinity_matrix.toarray()
        clusters = spectral(affinity_matrix, n_clusters) # using APML HW version which seem to be better
        # than sklearn (maybe they don't normalize laplacian from both sides)
    else:
        raise ValueError('choose valid clustering type!')

    # visualize
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
            types=0.4 # hide intra-sector edges
        graph_edges.append((p1, p2, {'weight': val, 'types': types}))
    graph.add_edges_from(graph_edges)

    hv.extension('bokeh')
    graph = hv.Graph.from_networkx(graph, nx.layout.fruchterman_reingold_layout, weight='weight')

    graph.opts(node_cmap='Category20', node_size=size_by, edge_line_width='weight',
               node_color=color_by, node_line_width=0, edge_color='types', edge_cmap='Greys')


    # plot each cluster separately
    clusters_plots = []
    for cluster_value in np.unique(clusters):
        cluster_nodes = [n for n in nodes if n[1]['cluster'] == cluster_value]
        cluster_edges = [e for e in graph_edges if (clusters[org2idx[e[0]]] == cluster_value) and (
            clusters[org2idx[e[1]]] == cluster_value)]
        if len(cluster_edges) == 0:
            continue
        cluster_graph = nx.Graph()
        cluster_graph.add_nodes_from(cluster_nodes)
        cluster_graph.add_edges_from(cluster_edges)
        c_graph = hv.Graph.from_networkx(cluster_graph, nx.layout.fruchterman_reingold_layout, weight='weight')
        c_labels = hv.Labels(c_graph.nodes, ['x', 'y'], 'index')
        c_labels.opts(text_font_size='8pt')
        c_graph.opts(node_cmap='Category20', node_size=size_by, edge_line_width='weight',
               node_color='cluster', node_line_width=0, edge_cmap='Greys')
        c_graph = c_graph * c_labels
        clusters_plots.append(c_graph)
    layout = hv.Layout(clusters_plots).cols(4)
    layout = (graph + layout).cols(1)
    hv.save(layout, save_to)
    edit_plot_height_and_width(save_to, 800, 1200)
    result = {'entity': [], 'cluster': []}
    for idx, c in enumerate(clusters):
        result['entity'].append(org2idx[idx])
        result['cluster'].append(int(c))
    df = pd.DataFrame(result)
    df.to_csv(save_to.replace('.html', '.csv'))


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


def scores_histogram():
    fig, axes = plt.subplots(1,3)
    data_type = 'ynet'
    similarities = ['jaccard', 'root_jaccard', 'pmi']
    scores = [get_pairs(pd.read_csv(paths[data_type], encoding='utf8'), data_type, 0.0002, s) for s in \
            similarities]
    scores = [[p.score for p in pairs] for pairs in scores]
    for sim, a, sco in zip(similarities, axes, scores):
        a.set_title(sim)
        sns.distplot(sco, ax=a)
    plt.savefig('scores_hist.pdf')

def compare_scores():
    data_type = 'ynet'
    df = pd.read_csv(paths[data_type], encoding='utf8')
    pairs_jaccard = get_pairs(df, data_type, 0.0002, 'root_jaccard')
    pairs_jaccard = sorted(pairs_jaccard, key=lambda x: x.score, reverse=True)
    pairs2jaccardrank ={p.items: i for i,p in enumerate(pairs_jaccard)}
    pairs_pmi = get_pairs(df, data_type, 0.0002, 'pmi')
    pairs_pmi = sorted(pairs_pmi, key=lambda x: x.score, reverse=True)
    jaccard_set = set([p.items for p in pairs_jaccard[:50]])
    pairs2pmirank = {p.items: i for i, p in enumerate(pairs_pmi)}
    pmi_set = set([p.items for p in pairs_pmi[:50]])
    diff_scores = []
    for pair in pairs2pmirank:
        diff_scores.append((pair, pairs2pmirank[pair] - pairs2jaccardrank[pair]))
    diff_scores = sorted(diff_scores, key=lambda x: x[1], reverse=True)

    print("\n\n\n ---------------- \n\n\n")

    print("higher jaccard:  ")
    print("\n".join([str(x) for x in diff_scores[:20]]))

    print("higher pmi:  ")
    print("\n".join([str(x) for x in diff_scores[-20:]]))
    print('pmi but not jaccard')
    print(pmi_set.difference(jaccard_set))
    print('jaccard but not pmi')
    print(jaccard_set.difference(pmi_set))


def main():
    data_types = ['ynet', 'committees']
    dfs = [pd.read_csv(paths[dtype]) for dtype in data_types]

    clustering_type = 'our_spectral'
    n_clusters = 20
    similarity = 'pmi'
    analyze_graph(dfs,
                  data_types,
                  similarity,
                  [0.0002, 0.0008],
                  '{}_{}_{}.html'.format("-".join(data_types), similarity, clustering_type),
                  clustering_type=clustering_type,
                  num_clusters=n_clusters)



if __name__ == '__main__':
    main()