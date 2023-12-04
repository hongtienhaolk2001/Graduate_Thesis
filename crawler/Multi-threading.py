from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from time import sleep
from queue import Queue
import pandas as pd
import threading
import GetData

#https://googlechromelabs.github.io/chrome-for-testing/#stable
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

    def crawl_Urls(self, page):
        queue = Queue(self.threads)
        for thread_i in range(self.threads):
            browser = self.browsers[thread_i]
            t = threading.Thread(target=lambda th, q, b, p: q.put(GetData.get_URLs(th, b, p)),
                                 args=(self.threads, queue, browser, page + thread_i,))
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
        for thread_i in range(self.threads):
            browser = self.browsers[thread_i]
            t = threading.Thread(target=lambda th, q, b, u: q.put(GetData.get_News(th, b, self.url_list[u])),
                                 args=(self.threads, queue, browser, url_index + thread_i,))
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

    try:  # Get url
        for page_i in range(1, max_page, multi_threading.threads):
            multi_threading.open_MultiBrowsers()
            multi_threading.crawl_Urls(page=page_i)
        # Crawl news
        current_url = 0
        for i in range(current_url, len(multi_threading.url_list), multi_threading.threads):
            multi_threading.open_MultiBrowsers()
            multi_threading.crawl_News(url_index=current_url)
    except:
        pass
    # Save
    multi_threading.df.to_csv(f'./coffee.csv')
    print("save successfully")
