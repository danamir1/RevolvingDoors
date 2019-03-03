import pandas as pd
import fuzzywuzzy as fw
from RevolvingDoors.ExtractEntities import map_set_to_term, merge_entities_sets, save_pickle, load_pickle
from collections import namedtuple, combinations
from langdetect import detect as ldetect

LOCAL_PATH_TO_DATA = "C:\\Users\\Ella\\PycharmProjects\\needle_final_project\\RevolvingDoors\\data\\"


def is_suspicious(old_job, data, set2term, suspicious_types=["government"]):
    row_old_job = recognize(old_job, data, set2term)
    if not row_old_job:
        return False
    elif row_old_job["type"] in suspicious_types:
        return True


def fuzzy_filter(maya, org2term):
    maya[fw.ratio(maya.old_job, org2term) > sim_threshold]



#
# def recognize(old_job, data, set2term, sim_threshold=85):
#     name_col = "name" if ldetect(old_job) == "he" else "name_eng"
#     name2term
#     maya = maya[maya[name_col] in ]
#     # for idx, row in data.iterrows():
#     #     if row[name_col].str.contains(old_job):
#     #         return row
#     # for merge_set, term in set2term.items():
#     #     for synonym in merge_set:
#     #         if fw.ratio(old_job, synonym) > sim_threshold:
#     #             return data[data.org_name == term]
#     #             # TODO add entities with no set2term


def find_potential_revolving_doors(path2mayas_data, path2matches, path2set2term):
    maya = pd.read_csv(path2mayas_data, converters={'PRIOR_JOBS': lambda x: _parse_prior_jobs_tuples(x)})
    matches = pd.read_csv(path2matches)
    set2term = load_pickle(path2set2term)
    # org2term = pd.read() # TODO dan
    cols = maya.keys()
    revolving_doors = []
    lateral_explode(maya)
    # TODO split named tuples to cols


    for idx, row in maya:
        new_job, old_job = row["COMPANY_NAME"], row["PRIOR_JOBS"][1]
        if is_suspicious(old_job, matches, set2term):
            revolving_doors.append(row)
    return pd.DataFrame(revolving_doors, columns=cols)


def _parse_prior_jobs_tuples(string):
    jobs = string.split(",")
    prior_job = namedtuple("PriorJob", ["role", "company", "duration"])
    jobs_tuples = [prior_job(j.strip("(").strip(")").split("##")) for j in jobs]
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


def edit_plot_height_and_width(html_file, height, width):
    import bs4
    import re
    with open(html_file) as inf:
        txt = inf.read()
        soup = bs4.BeautifulSoup(txt)
        js = soup.find_all(attrs={"type": "text/javascript"})[-1]
        # js = soup.find_all(text = re.compile("plot_height"))
        fixed_text = js.text.replace('\"plot_height\":300', '\"plot_height\":' + str(height))
        js.replace_with(fixed_text)
        # new_tag = soup.new_tag(name=js.name, attrs={"type": "text/javascript"})
        # new_tag.string = "example.net"
        # a_tag.i.replace_with(new_tag)


if __name__ == "__main__":
    # path2mayas_data = LOCAL_PATH_TO_DATA + "maya\\full\\2018.csv"
    # pairs = find_potential_revolving_doors(path2mayas_data)
    edit_plot_height_and_width("out.html", 500, 1000)