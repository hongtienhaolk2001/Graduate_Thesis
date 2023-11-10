import numpy as np
import re


def normalize_annotation(text):
    return re.sub(r"\btp hcm?|\btphcm?|\bhcm?", "thành phố hồ chí minh",
           re.sub(r"\bđbscl?", "đồng bằng sông cửu long",
           re.sub(r"\bvn?", "việt nam",
           re.sub(r"\beu?", "châu âu",
           re.sub(r"\bxk?", "xuất khẩu",
           re.sub(r"\bdn?", "doanh nghiệp",
           re.sub(r"\bnnnt?", "nông nghiệp nông thôn",
           re.sub(r"\bts?", "tiến sĩ",
           re.sub(r"\bthạc sĩ?", "nông nghiệp nông thôn",
           re.sub(r"\bnnptnt|\bptnnnt?", "phát triển nông nghiệp nông thôn", text))))))))))


def remove_number(text):
    return re.sub(" \d+", " ", text)


def remove_punctuation(text):
    punctuation = re.compile(r"[!#$%&()*+;<=>?@[\]^_`{|}~]")
    return punctuation.sub(r"", text)


def remove_special_char(text):
    special_character = re.compile("�+|+")
    return special_character.sub(r'', text)


def remove_punctuation(text):
    punctuation = re.compile(r"[!#$%&()*+;<=>?@[\]^_`{|}~]")
    return punctuation.sub(r"", text)


def pipeline(text):
    """
    lowercase -> remove punctuation -> remove special char -> remove number -> annotation
    """
    return {"News": normalize_annotation(
                    remove_number(
                    remove_special_char(
                    remove_punctuation(
                    text["News"].lower()))))}


class Preprocess():
    def __init__(self, tokenizer, rdrsegmenter):
        self.tokenizer = tokenizer
        self.rdrsegmenter = rdrsegmenter
        self.feature = ['category', 'price', 'gov', 'market', 'intrinsic', 'extrinsic']

    def segment(self, example):
        return {"Segment": " ".join([" ".join(sen) for sen in self.rdrsegmenter.tokenize(example["News"])])}

    def tokenize(self, example):
        return self.tokenizer(example["Segment"], truncation=True)

    def label(self, example):
        return {'labels_regressor': np.array([example[i] for i in self.feature]),
                'labels_classifier': np.array([int(example[i] != 0) for i in self.feature])}

    def run(self, dataset):
        dataset = dataset.map(pipeline)
        dataset = dataset.map(self.segment)
        dataset = dataset.map(self.tokenize, batched=True)
        dataset = dataset.map(self.label)
        columns = ['Unnamed: 0', 'News', 'category', 'price', 'gov', 'market', 'intrinsic', 'extrinsic', 'Segment']
        dataset = dataset.remove_columns(columns)
        dataset.set_format("torch")
        return dataset


# if __name__ == '__main__':
#     df = concat_files(root_path='../data/labeled_data/', file_format='xlsx')

#     # preprocess.crawled_Data()
