import pandas as pd
from fuzzywuzzy import fuzz
from ExtractEntities import map_set_to_term, merge_entities_sets, save_pickle, load_pickle
from collections import namedtuple
from langdetect import detect as ldetect
import glob
import os
import tqdm
LOCAL_PATH_TO_DATA = "C:\\Users\\Ella\\PycharmProjects\\needle_final_project\\RevolvingDoors\\data\\"
MAYA_PATH = '/Users/danamir/theDoors/A.Companies_BoardMembers/data/extractions/feb24_18/full'
ORG2TERM_PATH = 'map_org_to_term.csv'


def add_fuzzy_matching_to_maya(maya, org2term):
    org2term = {k:v for k,v in zip(org2term.organization.values, org2term.term.values)}
    prior_companies = maya.prior_company.unique().tolist()
    prior_to_match = {'prior_company': [], 'best_org_match': [], 'best_org_match_score': [],
                      'best_term_match': []}
    for c in tqdm.tqdm(prior_companies):
        best_match = max([(org, fuzz.ratio(org, c)) for org in org2term.keys()], key=lambda x: x[1])
        prior_to_match['prior_company'].append(c)
        prior_to_match['best_org_match_score'].append(best_match[1])
        prior_to_match['best_org_match'].append(best_match[0])
        prior_to_match['best_term_match'].append(org2term[best_match[0]])
    df = pd.DataFrame(prior_to_match)
    # maya['best_org_match'] = maya.prior_company.apply(lambda c: max([(org, fuzz.ratio(org, c)) for org in
    #                                                                   org2term.keys()], key=lambda x: x[1]))
    # maya['best_org_match_score'] = maya['best_org_match'].apply(lambda x: x[1])
    # maya['best_org_match'] = maya['best_org_match'].apply(lambda x: x[0])
    # maya['best_term_match'] = maya['best_org_match'].apply(lambda x: org2term[x])
    return df


def unite_maya(maya_dir=MAYA_PATH):
    paths = glob.glob(os.path.join(maya_dir, "*.csv"))
    dfs = [pd.read_csv(path, encoding='utf8') for path in paths]
    return pd.concat(dfs)

def find_potential_revolving_doors(path2mayas_data, path2org2term):
    maya = pd.read_csv(path2mayas_data, converters={'PRIOR_JOBS': lambda x: _parse_prior_jobs_tuples(x)})
    # matches = pd.read_csv(path2matches)
    # set2term = load_pickle(path2set2term)
    org2term = pd.read_csv(path2org2term)
    cols = maya.keys()
    revolving_doors = []
    maya = lateral_explode(maya, 'PRIOR_JOBS')
    maya = maya[maya.PRIOR_JOBS == maya.PRIOR_JOBS]
    maya['prior_role'] = maya['PRIOR_JOBS'].apply(lambda pj: pj.role)
    maya['prior_company'] = maya['PRIOR_JOBS'].apply(lambda pj: pj.company)
    maya['prior_duration'] = maya['PRIOR_JOBS'].apply(lambda pj: pj.duration)
    matches = add_fuzzy_matching_to_maya(maya, org2term)
    #
    # for idx, row in maya:
    #     new_job, old_job = row["COMPANY_NAME"], row["PRIOR_JOBS"][1]
    #     if is_suspicious(old_job, matches, set2term):
    #         revolving_doors.append(row)
    return matches


def _parse_prior_jobs_tuples(string):
    jobs = string.split(",")
    prior_job = namedtuple("PriorJob", ["role", "company", "duration"])
    jobs_tuples = [prior_job(*j.strip("(").strip(")").split("##")) for j in jobs if len(j.strip("(").strip(
        ")").split("##")) == 3]
    return jobs_tuples


def lateral_explode(dataframe, fieldname):
    """
    from https://stackoverflow.com/a/47244708/6873867
    """
    temp_fieldname = fieldname + '_made_tuple_'
    dataframe[temp_fieldname] = dataframe[fieldname].apply(tuple)
    list_of_dataframes = []
    for values in dataframe[temp_fieldname].unique().tolist():
        list_of_dataframes.append(pd.DataFrame({
            temp_fieldname: [values] * len(values),
            fieldname: list(values),
        }))
    dataframe = dataframe[list(set(dataframe.columns) - set([fieldname]))] \
        .merge(pd.concat(list_of_dataframes), how='left', on=temp_fieldname)
    del dataframe[temp_fieldname]

    return dataframe


def filter_maya_by_data_matches(maya_exploded: pd.DataFrame, org_matches: pd.DataFrame, maya_matches:
pd.DataFrame):
    merged_maya = maya_exploded.merge(maya_matches, on="prior_company")
    merged_maya = merged_maya[merged_maya.best_org_match_score >= 85]
    org_matches['prior_company'] = org_matches['org_name']
    merged_maya = merged_maya.merge(org_matches[['prior_company','type']], on="prior_company")
    merged_maya = merged_maya[merged_maya.type != 'company']
    return merged_maya

def find_occupational_shifts_with_interests():
    # occupational shifts from government to private sector with potential interest
    revolving_doors = pd.read_csv("data\\revolving_doors.csv", encoding="utf8")

    # interests map by association rules of the private sector in the government sector
    committees_association_rules = pd.read_csv("data\\committees_association_rules.csv", encoding="utf8")
    ynet_association_rules = pd.read_csv("data\\ynet_association_rules.csv", encoding="utf8")

    all_rules = pd.concat([committees_association_rules, ynet_association_rules])

    # all analyzed data of organizations and their ids and types
    org_to_entity = pd.read_csv("data\\all_org_to_entity_matches.csv", encoding="utf8")

    # add id field to association rules table
    all_rules_cols = all_rules.keys().tolist()
    # merge preparation
    all_rules_cols[all_rules_cols.index('antecedent')] = 'org_name'
    all_rules_cols[all_rules_cols.index("consequent")] = "best_term_match"
    all_rules.columns = all_rules_cols
    all_rules_with_id = all_rules.merge(org_to_entity[["id", "org_name"]], on="org_name")

    # new_cols = [n.lower() for n in revolving_doors.keys().tolist()]
    # new_cols[new_cols.index("company_rasham_num")] = "id"
    # new_cols[new_cols.index("type")] = "prior_type"
    # revolving_doors.columns = new_cols
    # revolving_doors_matches = revolving_doors[revolving_doors.id.apply(lambda x: x in org_to_entity.id.values)]

    revolving_doors_matches = pd.read_csv("data\\revolving_doors_matches.csv", encoding="utf8")
    rd_iterests = revolving_doors_matches.merge(all_rules_with_id, on= ["id", "best_term_match"])
    rd_iterests.to_csv("data\\revolving_doors_iterests.csv", encoding="utf8", index=False)


if __name__ == "__main__":
    # path2mayas_data = LOCAL_PATH_TO_DATA + "maya\\full\\2018.csv"
    # pairs = find_potential_revolving_doors(path2mayas_data)
    # edit_plot_height_and_width("out.html", 500, 1000)
    # all_maya = unite_maya()
    all_maya_path = 'all_maya.csv'
    # all_maya.to_csv(all_maya_path, encoding='utf8', index=False)
    # matches = find_potential_revolving_doors(all_maya_path, path2org2term=ORG2TERM_PATH)
    # matches.to_csv('maya_prior_company_matches.csv', encoding='utf8', index=False)
    maya = pd.read_csv(all_maya_path, converters={'PRIOR_JOBS': lambda x: _parse_prior_jobs_tuples(x)})
    maya = lateral_explode(maya, 'PRIOR_JOBS')
    maya = maya[maya.PRIOR_JOBS == maya.PRIOR_JOBS]
    maya['prior_role'] = maya['PRIOR_JOBS'].apply(lambda pj: pj.role)
    maya['prior_company'] = maya['PRIOR_JOBS'].apply(lambda pj: pj.company)
    maya['prior_duration'] = maya['PRIOR_JOBS'].apply(lambda pj: pj.duration)
    maya_matches = pd.read_csv('maya_prior_company_matches.csv', encoding='utf8')
    entity_matches = pd.read_csv('all_org_to_entity_matches.csv', encoding='utf8')
    merged = filter_maya_by_data_matches(maya, entity_matches, maya_matches)
    merged.to_csv('revolving_doors.csv', encoding='utf8', index=False)
    # find_occupational_shifts_with_interests()