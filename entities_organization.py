'''
Creating one dataset for all companies (private sector, goverment, assoiations)
'''

from openpyxl import load_workbook, Workbook
import pandas as pd
import os
import sys

companies = ['data/details.csv', 'data/entities.csv',
             'data/special-entities.csv','data/company-details.csv',
             'data/cooperatives.csv', 'data/association-registry.csv']
goverment_csv = ''
associations_csv = ''


# def extract_name(str):


def goverment_hierarchy():
    enteties = pd.read_csv('data/entities_1.csv')
    goverment_enteties = enteties[enteties.kind == 'government_office']
    hierarchy = pd.DataFrame(columns=['name', 'parent', 'id', 'kind', 'fingerprint'])
    counter = 1
    parents = set()
    for index, row in goverment_enteties.iterrows():
        # print(row)
        if '/' in row['name']:
            name = row['name'].split('/')
            if '-' in name[0]:
                name = name[0].split('-') + name[1:]
            parent = name[0].strip()
            if parent.startswith('ה') and parent[1:] in parents:
                parent = parent[1:]
            elif 'ה' + parent in parents:
                parent = 'ה' + parent
            parents.add(parent)
            child = '/'.join(name[1:]).strip()
            hierarchy.loc[counter] = [child, parent, row['id'], row['kind'], row['fingerprint']]
            counter += 1
            # print(child, '--', parent)
    print(parents)
    hierarchy.to_excel('goverment_hierarchy.xlsx')
    hierarchy.to_csv('goverment_hierarchy.csv')

# TRANSLATOR = {'חברה פרטית':'private','חברה ציבורית':'public',
#               'חברה פרטית מחוייבת במאזן':'private balance due',
#               'חברת חו"ל':'non israeli',
#               'שותפות מוגבלת':''}
# private = ['חברה פרטית','חברה פרטית מחוייבת במאזן','חברת חו"ל']
# private = []
def create_entities_database():
    output_dir = 'entities_database'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    # datasets = [pd.read_csv(csv_file) for csv_file in companies]
    ids = set()
    counter = 0
    database = pd.DataFrame(columns=['id', 'type', 'name', 'name_eng'])
    for db_name in companies:
        print('==>', db_name)
        db = pd.read_csv(db_name)
        columns = db.columns.values
        id_cell = 'id'
        if db_name == 'data/association-registry.csv':
            id_cell = 'Association_Number'
        if 'name' in columns:
            name_cell = 'name'
        elif 'company_name' in columns:
            name_cell = 'company_name'
        else:
            name_cell = 'Association_Name'

        if 'company_name_eng' in columns:
            name_eng_cell = 'company_name_eng'
        elif 'name_en' in columns:
            name_eng_cell = 'name_en'
        else:
            name_eng_cell = ''

        if 'kind' in columns:
            kind_cell = 'kind'
        elif 'company_government' in columns:
            kind_cell = 'company_type'

        else:
            kind_cell = ''

        for index, row in db.iterrows():
            # print(row['company_type'])
            id = int(row[id_cell])
            if id in ids:
                # print('===================================')
                # print(row)
                # print(database[database.id == id])
                continue
            ids.add(id)
            if name_eng_cell:
                name_eng = row[name_eng_cell]
            else:
                name_eng = ''
            # kind
            if kind_cell:
                kind = row[kind_cell]
            elif db_name == 'data/cooperatives.csv':
                kind = 'cooperative'
            elif db_name == 'data/association-registry.csv':
                kind = 'association'
            else:
                if row['company_is_municipal'] == 'TRUE':
                    kind = 'municipal'
                elif row['company_is_government'] == 'TRUE':
                    kind = 'government'
                # elif row['company_is_mafera']:
                #     kind = 'mafera'
                else:
                    kind = 'private'
            # print(id, kind, name,name_eng ,'', figerprint)
            database.loc[counter] = [id, kind, row[name_cell], name_eng]
            counter += 1
            if counter % 10000 ==0:
                print('done', counter)
                # break
        database.to_excel(output_dir+'/enttities_database_' + str(counter) + '.xlsx')
        database.to_csv(output_dir+'/enttities_database_' + str(counter) + '.csv')
    database.to_excel(output_dir+'/enttities_database.xlsx')
    database.to_csv(output_dir+'/enttities_database.csv')
    print('total of', counter, 'entities')

def csv2xlsx(file_name):
    df = pd.read_csv(file_name+'.csv')
    df.to_excel(file_name+'.xlsx')
if __name__ == '__main__':
    # goverment_hierarchy()
    create_entities_database()
    # csv2xlsx('details')