import numpy as np


class Preprocess():
    def __init__(self, tokenizer, rdrsegmenter):
        self.tokenizer = tokenizer
        self.rdrsegmenter = rdrsegmenter
        self.feature = ['giai_tri', 'luu_tru', 'nha_hang', 'an_uong', 'di_chuyen', 'mua_sam']

    def segment(self, example):
        return {"Segment": " ".join([" ".join(sen) for sen in self.rdrsegmenter.tokenize(example["Review"])])}

    def tokenize(self, example):
        return self.tokenizer(example["Segment"], truncation=True)

    def label(self, example):
        return {'labels_regressor': np.array([example[i] for i in self.feature]),
                'labels_classifier': np.array([int(example[i] != 0) for i in self.feature])}

    def run(self, dataset):
        dataset = dataset.map(clean_text)
        dataset = dataset.map(self.segment)
        dataset = dataset.map(self.tokenize, batched=True)
        dataset = dataset.map(self.label)
        columns = ['Unnamed: 0', 'Review', 'giai_tri', 'luu_tru',
                   'nha_hang', 'an_uong', 'di_chuyen', 'mua_sam', 'Segment']
        dataset = dataset.remove_columns(columns)
        dataset.set_format("torch")
        return dataset


if __name__ == "__main__":
    pass
