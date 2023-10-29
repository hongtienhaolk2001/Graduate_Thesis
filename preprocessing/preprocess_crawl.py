import pandas as pd
import re
import glob
import os


class Preprocess:
    def __init__(self, path):
        self.df = pd.DataFrame

    def merge_Data(self, root_path):
        all_files = glob.glob(os.path.join(root_path, "*.csv"))
        self.df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

    def crawled_Data(self, text):
        html_pattern = re.compile('<.*?>')
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        text = html_pattern.sub(r' ', text)
        text = url_pattern.sub(r' ', text)
        text = text.replace("\n", ". ")
        text = text.replace("\r.", "")
        text = text.replace("\r", "")
        text = text.replace(";", "")
        return text

    def normalize_annotatation(self, text):
        khach_san = "\bkhach san ?|\bksan ?|\bks ?"
        return re.sub("\bnv ?", "nhân viên",
                      re.sub(khach_san, "khách sạn", text))

    def clean_text(self, number=True, special_char=True, punct=True):
        remove_number = lambda t: re.sub(" \d+", " ", t)
        remove_special_char = lambda t: re.compile("�+").sub(r'', t)
        remove_punctuation = lambda t: re.compile(r"[!#$%&()*+;<=>?@[\]^_`{|}~]").sub(r"", t)
        lower_case = lambda t: t.lower()


if __name__ == '__main__':
    preprocess = Preprocess()
    preprocess.merge_Data(root_path='../data/labeled_data/')
    preprocess.crawled_Data()
