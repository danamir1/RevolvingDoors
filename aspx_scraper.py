import scrapy
from scrapy.crawler import CrawlerProcess
class SpidyQuotesViewStateSpider(scrapy.Spider):
    name = 'spidyquotes-viewstate'
    start_urls = ['https://main.knesset.gov.il/About/Lobbyist/Pages/LobbyistDetails.aspx']
    download_delay = 1.5

    def parse(self, response):
        # for author in response.css('select#author > option ::attr(value)').extract():
        yield scrapy.FormRequest(
            'https://main.knesset.gov.il/About/Lobbyist/Pages/LobbyistDetails.aspx',
            formdata={
                'l_id': '921',
                '__VIEWSTATE': response.css('input#__VIEWSTATE::attr(value)').extract_first()
            },
            callback=self.parse_results
        )

    def parse_tags(self, response):
        # for tag in response.css('select#tag > option ::attr(value)').extract():
        yield scrapy.FormRequest(
            'https://main.knesset.gov.il/About/Lobbyist/Pages/LobbyistDetails.aspx',
            formdata={
                'l_id': response.css(
                    'select#l_id > option[selected] ::attr(value)'
                ).extract_first(),
                'l_id': '921',
                '__VIEWSTATE': response.css('input#__VIEWSTATE::attr(value)').extract_first()
            },
            callback=self.parse_results,
        )

    def parse_results(self, response):
        for quote in response.css("div.LobbyistTopTitles"):
            yield {
                'quote': quote.css('span.LobbyistKnessetNum ::text').extract_first(),
                # 'author': quote.css('span.author ::text').extract_first(),
                # 'tag': quote.css('span.tag ::text').extract_first(),
            }

process = CrawlerProcess({
    'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)',
    'DOWNLOAD_DELAY': 2, 'AUTOTHROTTLE_ENABLED': True, 'AUTOTHROTTLE_START_DELAY': 3,
    'AUTOTHROTTLE_MAX_DELAY': 60, 'AUTOTHROTTLE_TARGET_CONCURRENCY': 2.0,
    'COOKIES_ENABLED': False
})

process.crawl(SpidyQuotesViewStateSpider)
process.start()