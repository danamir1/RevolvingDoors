import requests
from bs4 import BeautifulSoup
from bs4.element import Comment

import json
import os
import sys
import time
import tqdm
import re
from functools import wraps

YNET_EN_ROOT = "http://www.ynetnews.com/home/0,7340,L-3085,00.html"
YNET_HEB_ROOT = "https://www.ynet.co.il/home/0,7340,L-6,00.html"
DAY_IN_SECS = 60 * 60 * 24
DEFAULT_SINCE = time.strptime('2000','%Y')
ARCHIVED_SNAPSHOT_KEY = "archived_snapshots"
WAY_BACK_CDX_QUERY = "http://web.archive.org/cdx/search/cdx?url={}&output=json&collapse=timestamp:8&from={" \
                 "}&to={}"
WAY_BACK_URL_FORMAT = "http://web.archive.org/web/{}/{}"


def get_url_from_wb_cdx_response(response):
    timestamp = response[1]
    original = response[2]
    url = WAY_BACK_URL_FORMAT.format(timestamp, original)
    return url


def add_delay(delay_secs):
    # decorator to allow scraping without IP blocking
    def add_delay_fixed_time(func):
        @wraps(func)
        def func_with_delay(*args, **kwargs):
            gen = func(*args, **kwargs)
            for item in gen:
                time.sleep(delay_secs)
                yield item
        return func_with_delay
    return add_delay_fixed_time

def string_from_time(time_struct=None):
    if time_struct is None:
        time_struct = time.localtime()
    return time.strftime('%Y%m%d', time_struct)


def generate_times_strings(frequency=1, since=None, till=None):
    current_time = till if till is not None else time.localtime()
    current_sec_time = time.mktime(current_time)
    end_sec_time = time.mktime(since) if since is not None else time.mktime(DEFAULT_SINCE)
    while current_sec_time > end_sec_time:
        yield string_from_time(time.localtime(current_sec_time))
        current_sec_time -= frequency * DAY_IN_SECS


def retrieve_daily_snapshots(url, since=None, till=None):
    if since is None:
        since = "2010"
    if till is None:
        till = time.strftime("%Y%m%d", time.localtime())
    query = WAY_BACK_CDX_QUERY.format(url,since, till)
    print(query)
    resp = requests.get(query)
    js = resp.json()
    return [get_url_from_wb_cdx_response(r) for r in js[1:]]

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def text_from_soup(soup):
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    return u" ".join(t.strip() for t in visible_texts)

class YnetScraper(object):
    def __init__(self, urls, article_urls=None):
        self.urls = urls
        if article_urls is not None:
            self.articles_urls = article_urls
        else:
            self.articles_urls = self._extract_urls()

    def extract_url_from_link(self, link):
        m = re.search(r"http.*html", link["href"])
        if m is not None:
            return m.string[m.start(): m.end()]
        else:
            return None

    def _extract_urls(self):
        # ARTICLE_CLASSES = ["smallheader", "mta_title_href"]
        urls = set()
        for url in tqdm.tqdm(self.urls):
            time.sleep(0.5)
            try:
                content = requests.get(url).content
            except Exception as e:
                print("failed to extract articles links from snapshot: ", url)
                print(e, file=sys.stderr)
                continue
            bs = BeautifulSoup(content, "lxml")
            cur_url_links = bs.find_all(href=re.compile("article"))
            if len(cur_url_links) > 0:
                break
            cur_urls = [self.extract_url_from_link(link) for link in cur_url_links]
            cur_urls = [url for url in cur_urls if url is not None]
            urls.update(set(cur_urls))
        return urls

    def get_urls(self):
        return self.articles_urls

    @add_delay(0.5)
    def get_text(self):
        failures = []
        for url in tqdm.tqdm(self.articles_urls):
            resp = requests.get(url)
            if resp is None:
                print("unable to obtain article: ", url)
                failures.append(url)
                continue
            bs = BeautifulSoup(resp.content, "lxml")
            try:
                text_element = bs.find("article").find(class_="text14")
                text = text_from_soup(text_element)
            except Exception as e:
                print(e)
                print("unable to scrape text from url: ", url)
                continue

            yield url, text

if __name__ == "__main__":
    RUN_NAME = "heb_20162019"
    snapshots = retrieve_daily_snapshots(YNET_HEB_ROOT, since="20160101")
    with open("snapshots_{}.txt".format(RUN_NAME), "w") as f:
        f.write("\n".join(snapshots))
    scraper = YnetScraper(snapshots)
    with open("articles_{}.txt".format(RUN_NAME), "w") as f:
        f.write("\n".join(scraper.get_urls()))
    for url, text in scraper.get_text():
        with open("articles_text_{}.txt".format(RUN_NAME), "ab") as f:
            f.write(url.encode("utf8"))
            f.write("\n****END_URL*****\n".encode("utf8"))
            f.write(text.encode("utf8"))
            f.write("\n****END_ARTICLE*****\n".encode("utf8"))


