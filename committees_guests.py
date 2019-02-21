# IMPORTS
import os
import pandas as pd
import re
from tqdm import tqdm
from datetime import date
from difflib import SequenceMatcher
from itertools import combinations
from collections import defaultdict
import numpy as np
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from RevolvingDoors.ExtractEntities import map_set_to_term, merge_entities_sets, save_pickle

# GLOBALS
meetings_info = pd.DataFrame()
first_names = set()
last_names = set()
companies_info = pd.DataFrame()
LOCAL_PATH_TO_DATA = "C:\\Users\\Ella\\PycharmProjects\\needle_final_project\\data\\"
WORDS_TO_CLEAN = set()
count_companies_per_name = set()


def _load_globals(path2meetings_info, path2first_names, path2last_names, path2companies_info, path2titles,
                  path2companies_oracle, path2cleaning_list):
    global meetings_info, first_names, last_names, companies_info, titles, companies_oracle, WORDS_TO_CLEAN
    meetings_info = pd.read_csv(path2meetings_info)
    with open(path2first_names, 'r', encoding='utf-8') as fns:
        first_names = set(fns.read().split('\n'))
    with open(path2last_names, 'r', encoding='utf-8') as lns:
        last_names = set(lns.read().split('\n'))
    with open(path2cleaning_list, 'r', encoding='utf-8') as lns:
        WORDS_TO_CLEAN = set(lns.read().split('\n'))


def extract_guests_from_meetings(protocols_path, meetings_info_file):
    rows = {str(y): [] for y in range(2004, 2019)}
    cols = ['full_name', 'job_title', 'company', 'raw_company', 'meeting_id', 'meeting_title', 'meeting_date',
            'reference_link']
    all_rows = []
    for meeting_id, meeting_file_path in tqdm(_get_files_path(protocols_path)):
        committee_id, meeting_date, meeting_title, session_content, reference, year = _get_meeting_info(meetings_info,
                                                                                                        meeting_id)
        if committee_id is None:
            continue
        meeting_title = meeting_title if type(meeting_title) is str else session_content
        guests = _get_meeting_guests(meeting_file_path)
        # TODO extract knesset members in committee

        for full_name, job_title, cleaned_work_place, work_place in guests:
            guest_tuple = (
                full_name, job_title, cleaned_work_place, work_place, meeting_id, meeting_title, meeting_date,
                reference)
            rows[year].append(guest_tuple)
            all_rows.append(guest_tuple)

    for year in rows:
        df = pd.DataFrame(rows[year], columns=cols)
        save_guests_path = 'data\\guests\\{}'.format(date.today().__str__())
        os.makedirs(save_guests_path, exist_ok=True)
        df.to_csv(save_guests_path + '\\{}.csv'.format(year), sep=",", line_terminator='\n', encoding='utf-8')


# INNER TOOLZ

def _get_files_path(path2parentfolder):
    for folder in os.listdir(path2parentfolder):
        for protocol in os.listdir(os.path.join(path2parentfolder, folder)):
            if protocol.endswith('.csv'):
                committee_id = protocol[:-4]
                yield committee_id, os.path.join(path2parentfolder, folder, protocol)


def _get_meeting_info(meetings_info, meeting_id):
    meeting_info = meetings_info.loc[meetings_info['id'] == int(meeting_id)].values
    if meeting_info.shape[0] == 0:
        return None, None, None
    meeting_info = meeting_info[0]
    year = meeting_info[24][-4:]
    committee_id, meeting_date, meeting_title, session_content, reference = meeting_info[1], meeting_info[24], \
                                                                            meeting_info[3], meeting_info[4], \
                                                                            meeting_info[31],
    if "," in str(meeting_title):
        meeting_title = meeting_title.replace(",", "")
    return committee_id, meeting_date, meeting_title, session_content, reference, year


def _is_guest_pattern(gr):
    # skip committee staff by ignoring rows with :
    return (re.search('\s-\s|\s–\s', gr) or len(gr.split(",")) > 2) and len(gr) and ":" not in gr


def _get_meeting_guests(meeting_file_path):
    df = pd.read_csv(meeting_file_path)
    # print(df)
    if 'header' not in df.columns:
        return []
    # get the row with index == מוזמנים
    relevant_row = df.loc[df['header'].isin(['מוזמנים', ':מוזמנים', 'מוזמנים:'])].values
    if relevant_row.shape[0] == 0:
        # No such rows where found
        return []
    text_block = relevant_row[0][1]
    if type(text_block) == float:
        return []
    guests_rows = [gr for gr in text_block.split('\n') if _is_guest_pattern(gr)]
    guests = [g for g in _parse_rows(guests_rows)]
    return guests


def _concatenate_guests():
    all_dfs = []
    for year in range(2004, 2019):
        # last_update = date.today().__str__()
        last_update = '2019-02-18'
        save_guests_path = 'data\\guests\\{}'.format(last_update)
        df = pd.read_csv(save_guests_path + '\\{}.csv'.format(year))
        all_dfs.append(df)
    all_rows = pd.concat(all_dfs)
    df = all_rows[all_rows.full_name.str.contains("\w\s\w", regex=True)]
    df.to_csv("guests_0.csv", sep=",", line_terminator='\n', encoding='utf-8')
    return df


def _collect_companies_pairs_and_scores(iteration_number):
    df = pd.read_csv("guests_{}.csv".format(iteration_number))
    d = {"c1": [], "c2": [], "ratio": [], "fratio": [], "pfratio": [], "ptfratio": [], "iou": [], "intersection": [],
         "union": []}

    if iteration_number != 0:
        df = update_indicative_names(df)
        gb = df[df.is_indicative_name].groupby("company")
    else:
        gb = df.groupby("company")

    for id, pair in tqdm(enumerate(_get_candidates_for_similarity(df))):
        d["c1"].append(pair[0])
        d["c2"].append(pair[1])
        ratio, fratio, pfratio, ptfratio = _get_similarity_score(pair)
        d["ratio"].append(ratio)
        d["fratio"].append(fratio)
        d["pfratio"].append(pfratio)
        d["ptfratio"].append(ptfratio)
        iou, intersection, union = _get_jaccard_score(gb, pair)
        d["iou"].append(iou)
        d["intersection"].append(intersection)
        d["union"].append(union)
    candidates_scores = pd.DataFrame(d)
    candidates_scores.to_csv("candidates_scores_{}.csv".format(iteration_number), sep=",", line_terminator='\n',
                             encoding='utf-8')


def _merge_companies_names(condition, condition_name, iteration_number):
    candidates_scores = pd.read_csv("candidates_scores_{}.csv".format(iteration_number - 1))
    all_guests = pd.read_csv("guests_{}.csv".format(iteration_number - 1))
    c1 = candidates_scores[condition(candidates_scores)]["c1"].tolist()
    c2 = candidates_scores[condition(candidates_scores)]["c2"].tolist()
    pairs = list(zip(c1, c2))
    orgs2merge_sets = merge_entities_sets(pairs)
    companies_count = all_guests.company.value_counts()
    set2term = map_set_to_term(orgs2merge_sets, companies_count)
    save_pickle(orgs2merge_sets, "guests_orgs2merge_sets_{}_{}.pkl".format(iteration_number, condition_name))
    save_pickle(set2term, "guests_set2term_{}_{}.pkl".format(iteration_number, condition_name))
    for org, merge_set in orgs2merge_sets.items():
        all_guests[all_guests.company == org].company = set2term[merge_set]
    all_guests.to_csv("guests_{}.csv".format(iteration_number), sep=",", line_terminator='\n', encoding='utf-8')
    return all_guests


def _get_candidates_for_similarity(df):
    seen = set()
    gb = df.groupby("full_name")
    names = df.full_name.unique().tolist()
    for name in names:
        name_companies = [c for c in gb.get_group(name).company.unique().tolist() if c is not np.nan]
        for comb in combinations(name_companies, 2):
            frozen_comb = frozenset(comb)
            if frozen_comb in seen:
                continue
            else:
                seen.add(frozen_comb)
                yield comb


def update_indicative_names(df: pd.DataFrame, threshold=5):
    count_companies_per_name = df.groupby("full_name").company.nunique().reset_index(name='count_companies_per_name')
    joined_df = df.join(count_companies_per_name)
    joined_df["is_indicative_name"] = joined_df[joined_df.count_companies_per_name >= threshold]
    return joined_df


def _get_similarity_score(pair):
    s = SequenceMatcher(lambda x: x == " ", pair[0], pair[1])
    return s.ratio(), fuzz.ratio(pair[0], pair[1]), fuzz.partial_ratio(pair[0], pair[1]), fuzz.partial_token_sort_ratio(
        pair[0], pair[1])


def _get_jaccard_score(gb, pair):
    company_fullnames = set(gb.get_group(pair[0]).full_name.unique().tolist())
    subcompany_fullnames = set(gb.get_group(pair[1]).full_name.unique().tolist())
    intersection_size = len(company_fullnames.intersection(subcompany_fullnames))
    union_size = len(company_fullnames.union(subcompany_fullnames))
    score = intersection_size / union_size
    return score, intersection_size, union_size

def _clean_work_place(raw_work_place):
    for word in WORDS_TO_CLEAN:
        if raw_work_place.startswith(word):
            return _clean_work_place(raw_work_place[len(word):])
    return raw_work_place


def _parse_rows(guests_rows):
    for row in guests_rows:
        row_parts = re.split(r',|\s-\s|\s–\s', row)
        hyphen_loc = row.find("-")
        comma_loc = row.find(",")
        if comma_loc > -1 and hyphen_loc > -1 and comma_loc < hyphen_loc:
            raw_full_name = row_parts.pop(-1)
            row_parts.insert(0, raw_full_name)
        if ":" in row_parts[0]:
            continue
        full_name, valid, job_title = _parse_full_name(row_parts[0], original_row=row, parts=row_parts)
        if not valid or len(row_parts) == 1:
            continue
        # TODO verify if workplace in companies file
        work_place = row_parts[-1].strip()
        cleaned_work_place = _clean_work_place(work_place)
        # work_place = _guess_company_name(row_parts[-1].strip())
        if len(row_parts) > 2:
            job_title = row_parts[1].strip().strip("–").strip()
        yield full_name, job_title, cleaned_work_place, work_place


def _parse_full_name(full_name_raw, original_row="", parts=[]):
    parts = re.split(r'\s+', full_name_raw)
    valid = False
    full_name = None
    full_job_title = None
    job_title = []
    first_name = []
    last_name = []
    for p in parts:
        if p in titles and not len(first_name):
            valid = True
        elif p in first_names and not len(first_name):
            first_name = [p]
            valid = True
        elif p in last_names or '-' in p:
            last_name.append(p)
        else:
            job_title.append(p)
    if valid:
        full_name = ' '.join(first_name + last_name)
        full_job_title = ' '.join(job_title).strip()

    return full_name, valid, full_job_title


if __name__ == '__main__':
    path2meetings_info = LOCAL_PATH_TO_DATA + "committee-meetings.csv"
    path2firstnames = LOCAL_PATH_TO_DATA + "\\lexicons\\first_names.txt"  # with _ it's the newer version
    path2lastnames = LOCAL_PATH_TO_DATA + "\\lexicons\\last_names.txt"
    path2titles = LOCAL_PATH_TO_DATA + "\\lexicons\\titles.txt"
    path2companies_oracle = LOCAL_PATH_TO_DATA + "\\lexicons\\companies_oracle.csv"
    path2companies = LOCAL_PATH_TO_DATA + "companies.csv"
    path2cleaning_list = LOCAL_PATH_TO_DATA + "WORDS_TO_CLEAN.txt"
    _load_globals(path2meetings_info, path2firstnames, path2lastnames, path2companies, path2titles,
                  path2companies_oracle, path2cleaning_list)

    path2protocols = LOCAL_PATH_TO_DATA + "committees_csvs"
    # extract_guests_from_meetings(path2protocols, path2meetings_info)
    _collect_companies_pairs_and_scores(iteration_number=)
    _merge_companies_names(iteration_number=1, condition=)
