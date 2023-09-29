from selenium import webdriver
from time import sleep
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import json
import pandas as pd


class NewsCrawler:
    def __init__(self) -> None:
        self.options = Options()
        self.options.headless = False
        self.options.add_argument("--window-size=1920,1080")

    def to_csv(self):
        pass

    def to_jsonl(self):
        pass
