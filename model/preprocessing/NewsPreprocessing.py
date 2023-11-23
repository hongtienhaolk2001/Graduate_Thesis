import numpy as np
import re
import json


def normalize_annotation(text):
    annotation_dict = {
        "nông nghiệp nông thôn việt nam": r"\b(nn ptnt)\b|\b(nnptnt)\b|\b(ptnnnt)\b|\b(pt nnnt)\b|\b(nnnt)\b|\b(nnvn)\b",
        "phát triển nông thôn": r"\b(ptnt)\b",
        "hecta": r"\b(héc ta)\b|\b(hécta)\b|\b(ha)\b|\b(hectare)\b",
        "đồng bằng sông cửu long": r"\b(đbscl)\b",
        "đồng bằng sông hồng": r"\b(đbsh)\b",
        "thành phố hồ chí minh": r"\b(tp hcm)\b|\b(tphcm)\b|\b(hcm)\b",
        "lúa mỳ": r"\b(lúa mì)\b",
        "hợp tác xã": r"\b(htx)\b",
        "bảo vệ thực vật": r"\b(bvtv)\b",
        "iraq": r"\b(irac)\b|\b(irắc)\b",
        "pakistan": r"\b(pakixtan)\b",
        "myanmar": r"\b(mianma)\b",
        "indonesia": r"\b(indonexia)\b|\b(inđônêxia)\b|\b(in đô nê xi a)\b",
        "bangladesh": r"\b(băngla đét)\b",
        "philippines": r"\b(philippin)\b|\b(philippine)\b",
        "Kazakhstan": r"\b(cadắcxtan)\b",
        "malaysia": r"\b(malayxia)\b|\b(ma lai xi a)\b|\b(malaixia)\b|\b(malaisia)\b",
        "đông xuân": r"\b(đx)\b",
        "thành phố": r"\b(tp)\b",
        "doanh nghiệp": r"\b(dn)\b",
        "nhập khẩu": r"\b(nk)\b",
        "xuất khẩu": r"\b(xk)\b",
        "châu âu": r"\b(eu)\b",
        "việt nam": r"\b(vn)\b",
        "phóng viên": r"\b(pv)\b",
    }
    for replace in annotation_dict:
        text = re.sub(annotation_dict[replace], replace, text)
    return text


def remove_irrelevant(text):
    with open("remove_list.txt", "r", encoding='utf8') as f:
        remove_list = f.read().split("\n")
    for i in remove_list:
        text = re.sub(i, "", text)
    inside_brackets = re.compile(r"\(.* ?\)")
    return inside_brackets.sub(" ", text)


def space_check(text):
    space = re.compile(r"\s{2,}")
    return space.sub(r" ", text).strip()


def remove_number(text):
    return re.sub(r"\d+", " ", text)


def remove_special_char(text):
    special_character = re.compile(r"�+|+|½+")
    return special_character.sub(r"", text)


def remove_punctuation(text):
    punctuation = re.compile(r"([^\w\s]|_)")
    return punctuation.sub(r" ", text)


def pipeline(text):
    """
    lowercase -> irrelevant -> punctuation -> special char -> number -> annotation
    """
    return {"News": space_check(
        normalize_annotation(
            remove_punctuation(
                remove_special_char(
                    remove_number(
                        remove_irrelevant(
                            text["News"].lower()))))))}


class Preprocess:
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


if __name__ == '__main__':
    # df = concat_files(root_path='../data/labeled_data/', file_format='xlsx')
    # print(remove_irrelevant("tổng giám đốc cp đồng bằng ths tiến sĩ thac si"))
    t = "BẢN TIN THỊ TRƯỜNG VÀ THƯƠNG MẠI LÚA GẠO VIỆT NAM ½ (TUẦN 23 - 29/03/2009) Theo chỉ đạo của Bộ Công thương và Bộ NN&PTNT, cho đến hết tháng 6/2009, Việt Nam sẽ xuất khẩu khoảng 3,4 triệu tấn gạo trong tổng số 3,7 triệu tấn theo hợp đồng đã ký."
    print(pipeline(t))
    # preprocess.crawled_Data()
