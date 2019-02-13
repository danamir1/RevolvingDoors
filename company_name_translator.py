'''
Translating companies names using Wikipedia
'''

import wikipedia
# import urllib2
import urllib.request as urllib2
from bs4 import BeautifulSoup
import wikipedia
import wikipediaapi


# def langBS(link):
#     # get languages
#     soup = BeautifulSoup(urllib2.urlopen(link))
#     links = [(el.get('lang'), el.get('title')) for el in soup.select('li.interlanguage-link > a')]
#
#     for language, title in links:
#         page_title = title.split(u' – ')[0]
#         wikipedia.set_lang(language)
#         page = wikipedia.page(page_title)
#         print(language, page.title)
#         # print page.summary
#         # print "-----"

def get_wiki_title(name):
    # ministry or deprtment
    if name.lower().startswith('ministry'):
        name = name + ' (Israel)'
    if name.lower().startswith('the '):
        name = name[4:]
    try:
        page = wikipedia.page(name)
        return page.title
    except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError) as e:
        print("Couldn't extract:", name, "because of:\n", e)
        return None
    # wikipedia.set_lang('heb')
    # page.lang_links('en')
    return page.title


def en2he(title):
    wiki_wiki = wikipediaapi.Wikipedia('en')
    page_py = wiki_wiki.page(title)
    langlinks = page_py.langlinks
    if 'he' in langlinks:
        return langlinks['he'].title
    return None
    # for k in sorted(langlinks.keys()):
    #     v = langlinks[k]
    #     print("%s: %s - %s: %s" % (k, v.language, v.title, v.fullurl))


if __name__ == '__main__':
    names = ['']
    name = 'Port Authority'
    name = 'Ahava'
    title = get_wiki_title(name)
    print(title)
    # print(langBS('https://en.wikipedia.org/wiki/Ministry_of_Health_(Israel)'))
    # wiki_langs('Ministry of Health (Israel)')
    hebrew_name = en2he(title)
    if hebrew_name == None:
        hebrew_name = en2he('Israel ' + title.title())
    print(hebrew_name)
