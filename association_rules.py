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
    government = match[match['type']=='government']
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
    apriori_result = apriori(baskets_list, min_support=min_support, min_confidence=0.0, max_length=2)
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
        (some_item,)=some_set
        return some_item
    for s in result:
        # print(s)
        rules['antecedent'].append(item(s.items_base))
        rules['consequent'].append(item(s.items_add))
        rules['confidence'].append(s.confidence)
        rules['lift'].append(s.lift)
        rules['interest'].append(s.confidence-org2supp[item(s.items_add)])
        rules['antecedent_support'].append(org2supp[item(s.items_base)])
        rules['consequent_support'].append(org2supp[item(s.items_add)])
    # rules['antecedent'] = list(map(lambda s: item(s.items_base), result))
    # rules['consequent'] = list(map(lambda s: item(s.items_add), result))
    # rules['confidence'] = list(map(lambda s: s.confidence, result))
    # rules['lift'] = list(map(lambda s: s.lift, result))
    # rules['interest'] = list(map(lambda s: s.confidence-org2supp[item(s.items_add)], result))
    # rules['antecedent_support'] = list(map(lambda s: org2supp[item(s.items_base)], result))
    # rules['consequent_support'] = list(map(lambda s: org2supp[item(s.items_add)], result))

    for r in rules:
        print(r, len(rules[r]))

    pd.DataFrame(rules).to_csv('data/association_rules.csv', encoding='utf8')
    pd.DataFrame(rules).to_excel('data/association_rules.xlsx', encoding='utf8')


if __name__ == '__main__':
    data_type = 'committees'
    paths = {
        'committees': 'filtered_guests_final.csv',
        'ynet': 'articles_orgs_baskets_190225.csv'
    }
    df = pd.read_csv(paths[data_type], encoding='utf8')
    match = pd.read_csv('all_org_to_entity_matches.csv', encoding='utf8')
    associate_pairs(df, data_type, match)