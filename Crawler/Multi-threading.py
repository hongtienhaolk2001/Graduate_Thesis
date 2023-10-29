from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import pandas as pd
import threading
from time import sleep
from queue import Queue


class MultiThreading:
    def __init__(self, threads):
        self.url_list = []
        self.df = pd.DataFrame({'category': [], 'title': [], 'brief': [], 'date': [], 'content': [], 'sources': []})
        self.threads = threads
        self.browsers = []
        self.options = Options()
        self.options.headless = False
        self.options.add_argument("--window-size=1920,1080")
        self.service = Service(executable_path='./chromedriver.exe')

    def open_MultiBrowsers(self):
        try:
            self.browsers = [webdriver.Chrome(service=self.service, options=self.options) for _ in range(self.threads)]
            print(f"open {self.threads} browser successfully")
        except:
            print("open multi-browser is fail")

    def get_URLs(self, browser, page):
        try:
            browser.get(f"https://agro.gov.vn/vn/p{page}_2Ca-phe.html")
            # browser.get(f"https://agro.gov.vn/vn/p{i}_9Lua-gao.html")
            # browser.get(f"https://agro.gov.vn/vn/p{i}_10Cao-su.html"")
            sleep(self.threads)
            elements = browser.find_elements(By.CSS_SELECTOR, ".news [href]")
            links = [element.get_attribute('href') for element in elements]
            browser.close()
            print(f"crawl page {page} successfully")
            return [*set(links[0:10])]
        except:
            print(f"get Error at page {page}")

    def get_News(self, browser, url, category=2):
        try:
            browser.get(url)
            sleep(self.threads)
            title = browser.find_element(By.ID, "ctl00_maincontent_N_TIEUDE")
            date = browser.find_element(By.ID, "ctl00_maincontent_N_NGAYTHANG")
            brief = browser.find_element(By.ID, "ctl00_maincontent_N_TRICHDAN")
            content = browser.find_element(By.ID, "ctl00_maincontent_N_NOIDUNG")
            new_row = [category, title.text, brief.text, date.text, content.text, url]
            browser.close()
            print(f"crawl url: '{url}' successfully")
            return new_row

        except:
            new_row = {'category': category, 'title': "", 'brief': "", 'date': "",
                       'content': "", 'sources': url}
            browser.close()
            print(f"get Error at url: '{url}'")
            return new_row

    def crawl_Urls(self, page):
        queue = Queue(self.threads)
        for i in range(self.threads):
            browser = self.browsers[i]
            t = threading.Thread(target=lambda q, b, p: q.put(self.get_URLs(b, p)),
                                 args=(queue, browser, page + i,))
            t.start()
        try:
            sleep(10)
            for _ in range(self.threads):
                self.url_list += queue.get()
            print(f"total of urls: {len(self.url_list)}")
        except:
            print("get error when saving url")

    def crawl_News(self, url_index):
        queue = Queue(self.threads)
        for i in range(self.threads):
            browser = self.browsers[i]
            t = threading.Thread(target=lambda q, b, u: q.put(self.get_News(b, self.url_list[u])),
                                 args=(queue, browser, url_index + i,))
            t.start()
        try:
            sleep(10)
            for j in range(self.threads):
                self.df.loc[len(self.df.index)] = queue.get()
            print(f"total of crawled news: {len(self.df)}")
        except:
            print("get error when saving news")


if __name__ == '__main__':
    multi_threading = MultiThreading(threads=6)
    max_page = 264
    # Get url
    for page_i in range(1, max_page, multi_threading.threads):
        multi_threading.open_MultiBrowsers()
        multi_threading.crawl_Urls(page=page_i)
    # Crawl news
    current_url = 0
    for i in range(current_url, len(multi_threading.url_list), multi_threading.threads):
        multi_threading.open_MultiBrowsers()
        multi_threading.crawl_News(url_index=current_url)
    # Save
    multi_threading.df.to_csv(f'../data/crawled_data/coffee.csv')
    print("save successfully")
