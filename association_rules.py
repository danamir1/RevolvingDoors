from analyze_interests import *
from collections import defaultdict

def get_org2supp(orgs2baskets, num_baskets):
    """
    Calculates organization support
    :param orgs2baskets: organization to baskets dictionary
    :param num_baskets: number of baskets in total
    :return: organization to support dictionary
    """
    org2supp = dict()
    for org in orgs2baskets:
        org2supp[org] = len(orgs2baskets[org]) / float(num_baskets)
    return org2supp


def get_government_company(match):
    """
    Returns a set of government organization names and another set of companies names
    :param match: a matching data frame
    """
    government = match[match['type'] == 'government']
    government_names = set(government['entity_name']).union(set(government['org_name']))
    company = match[match['type'] == 'company']
    company_names = set(company['entity_name']).union(set(company['org_name']))
    return government_names, company_names


def associate_pairs(df, data_kind, match, min_support=0.0002, filter_set=None):
    '''Extracts a list of association rules in the form of company -> government'''
    if filter_set is None:
        filter_set = set()

    keys = KEYS_DICT[data_kind]
    orgs2baskets = datarows_to_sets(df, key_column=keys[ENTITY], value_column=keys[BASKET])
    baskets2orgs = datarows_to_sets(df, key_column=keys[BASKET], value_column=keys[ENTITY])
    baskets_list = [list(baskets2orgs[b].difference(filter_set)) for b in baskets2orgs]
    org2supp = get_org2supp(orgs2baskets, len(baskets_list))
    government_names, company_names = get_government_company(match)

    # get association rules
    apriori_result = apriori(baskets_list, min_support=min_support, min_confidence=0.0,
                             max_length=2)
    # get pairs only
    result = filter(lambda r: len(r.items) > 1, apriori_result)
    # get ordered statistics
    result = map(lambda res: res.ordered_statistics, result)

    # filter only government / company:
    def gov_comp_pair(ordered_stat):
        (org1,) = ordered_stat.items_base
        (org2,) = ordered_stat.items_add
        if (org1 in government_names and org2 in company_names) or (org2 in government_names and
                                                                            org1 in company_names):
            return True
        return False

    result = filter(lambda res: gov_comp_pair(res[0]), result)

    # get ordered stats of company -> governement
    def first_company(ordered_stat):
        (org,) = ordered_stat.items_base
        return org in company_names

    result = map(lambda res: res[0] if first_company(res[0]) else res[1], result)

    # create data of rules, confidence, interest
    rules = defaultdict(lambda: [])

    def item(some_set):
        # gets single item from a set
        (some_item,) = some_set
        return some_item

    for s in result:
        rule_interest = s.confidence - org2supp[item(s.items_add)]
        if rule_interest >= 0.25:
            rules['antecedent'].append(item(s.items_base))
            rules['consequent'].append(item(s.items_add))
            rules['confidence'].append(s.confidence)
            rules['lift'].append(s.lift)
            rules['interest'].append(rule_interest)
            rules['antecedent_support'].append(org2supp[item(s.items_base)])
            rules['consequent_support'].append(org2supp[item(s.items_add)])

    for r in rules:
        print(r, len(rules[r]))

    pd.DataFrame(rules).to_csv('data/' + data_kind + '_association_rules.csv', encoding='utf8')
    pd.DataFrame(rules).to_excel('data/' + data_kind + '_association_rules.xlsx', encoding='utf8')


def top_interest(df, data_type, top_num=3):
    """
    Extract the top x mapping to each government entity, based on association rules dataFrame.
    """
    interest = defaultdict(lambda: [])
    government_names = df['consequent'].unique()
    for gov_index, government_name in enumerate(government_names):
        gov = df[df['consequent'] == government_name]
        gov = gov.sort_values(by=['interest'], ascending=False)
        interest['government'].append(government_name)
        counter = 0

        for index, rule in gov.iterrows():
            if counter >= top_num:
                break
            interest['top ' + str(counter + 1)].append(rule['antecedent'])
            counter += 1
        while counter < top_num:
            interest['top ' + str(counter + 1)].append('')
            counter += 1
    pd.DataFrame(interest).to_csv('data/' + data_type + '_top_association_rules.csv',
                                  encoding='utf8')
    pd.DataFrame(interest).to_excel('data/' + data_type + '_top_association_rules.xlsx',
                                    encoding='utf8')


def random_sample(rules, sample_size=30):
    """
    Samples association rules for manual check of accuracy
    """
    sample = rules.sample(n=sample_size)
    sample.to_csv('data/' + data_type + '_sample2_association_rules.csv', encoding='utf8')
    sample.to_excel('data/' + data_type + '_sample2_association_rules.xlsx', encoding='utf8')

def sample_from_top(rules, top=500, sample_size=30):
    """
    Samples association rules from the one with best interest.
    """
    top_rules = rules.sort_values(by=['interest'], ascending=False).head(top)
    sample = top_rules.sample(n=sample_size)
    sample.to_csv('data/' + data_type + '_sample_top_association_rules.csv', encoding='utf8')
    sample.to_excel('data/' + data_type + '_sample_top_association_rules.xlsx', encoding='utf8')

if __name__ == '__main__':
    data_type = 'committees' #'ynet'  #
    paths = {
        'committees': 'filtered_guests_final.csv',
        'ynet': 'articles_orgs_baskets_190225.csv'
    }
    df = pd.read_csv(paths[data_type], encoding='utf8')
    match = pd.read_csv('all_org_to_entity_matches.csv', encoding='utf8')
    associate_pairs(df, data_type, match)
    df = pd.read_csv('data/' + data_type + '_association_rules.csv', encoding='utf8')
    top_interest(df, data_type)
    random_sample(df)
