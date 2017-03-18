import scrapy
from scrapy.crawler import CrawlerProcess
from html.parser import HTMLParser
import re
import numpy as np
import json


class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return re.sub(r'(^\s)|(\s$)', '', re.sub(r'\s+', ' ',s.get_data()))

class MySpider(scrapy.Spider):
    name = "habrgeek"
    # ATTENTION! Not object safety
    # TODO: rework 
    habr_done = []
    gt_done = []
    total_need = 10

    def urls(self, domain, start_id, finish_id):
        return ('https://{}.ru/post/{}/'.format(domain, id) for id in np.random.choice(range(start_id, finish_id), finish_id-start_id-1, replace=False))

    def habr_urls(self):
        domain = "habrahabr"
        start_id = 100000
        finish_id = 319000
        return self.urls(domain, start_id, finish_id)
        
    def gt_urls(self):
        domain = "geektimes"
        start_id = 240200
        finish_id = 287006
        return self.urls(domain, start_id, finish_id)

    def start_requests(self):
        # first download habr
        for url in self.habr_urls():
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        result = self.habr_done
        if 'geektimes' in response.url:
            result = self.gt_done
        result.append({"url": response.url, 'title': strip_tags(response.css("h1").extract()[0]), 
                   "text": strip_tags(response.css("div.content.html_format").extract()[0])})
        print(len(result))
        if len(result) >= self.total_need and result == self.habr_done:
            # second download geektimes
            for url in self.gt_urls():
                yield scrapy.Request(url=url, callback=self.parse)

        elif len(result) >= self.total_need and result == self.gt_done:
            raise scrapy.exceptions.CloseSpider(reason=str(self.total_need) + ' pages already downloaded')
            

def download(count=1000, habr="habrahabr.json", gt="geektimes.json"):
    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'
    })

    MySpider.total_need = count
    process.crawl(MySpider)
    process.start()

    print("Save JSON files...")
    with open(habr, 'w', encoding='utf8') as json_file:
        data = json.dumps(MySpider.habr_done, ensure_ascii=False)
        json_file.write(data)

    with open(gt, 'w', encoding='utf8') as json_file:
        data = json.dumps(MySpider.gt_done, ensure_ascii=False)
        json_file.write(data)


if __name__ == "__main__":
    download()