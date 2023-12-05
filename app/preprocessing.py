import re


def normalize_annotation(text):
    annotation_dict = {
        "nông nghiệp nông thôn việt nam": r"\b(nn ptnt)\b|\b(nnptnt)\b|\b(ptnnnt)\b|\b(pt nnnt)\b|\b(nnnt)\b|\b(nnvn)\b",
        "phát triển nông thôn": r"\b(ptnt)\b",
        "hecta": r"\b(héc ta)\b|\b(hécta)\b|\b(ha)\b|\b(hectare)\b",
        "đồng bằng sông cửu long": r"\b(đbscl)\b",
        "đồng bằng sông hồng": r"\b(đbsh)\b",
        "thành phố hồ chí minh": r"\b(tp hcm)\b|\b(tphcm)\b|\b(hcm)\b",
        "hiệp hội lương thực việt nam": r"\b(HHLTVN)\b",
        "doanh nghiệp xuất khẩu": r"\b(dnxk)\b",
        "ngân hàng nhà nước": r"\b(nhnn)\b",
        "ngân hàng thương mại": r"\b(nhtm)\b",
        "giao thông vận tải": r"\b(gtvt)\b",
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
        "usd": r"\b(đô la)\b",
        "việt nam": r"\b(vn)\b",
        "phóng viên": r"\b(pv)\b",
    }
    for replace in annotation_dict:
        text = re.sub(annotation_dict[replace], replace, text)
    return text


def remove_irrelevant(text):
    remove_list = ['tổng giám đốc', 'tgđ', 'giám đốc', 'chủ tịch', 'cp',
                   'thạc sĩ', 'ths', 'tiến sĩ', 'ts', 'gs', 'pgs', 'tnhh', 'hđqt']
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
    lowercase -> irrelevant -> punctuation -> special char -> number -> annotation -> space check
    """
    return space_check(
        normalize_annotation(
            remove_punctuation(
                remove_special_char(
                    remove_number(
                        remove_irrelevant(
                            text.lower()))))))


class Preprocess:
    def __init__(self, tokenizer, rdrsegmenter):
        self.tokenizer = tokenizer
        self.rdrsegmenter = rdrsegmenter

    def tokenize(self, sentence):
        sentence = pipeline(sentence)
        segment = " ".join([" ".join(sent) for sent in self.rdrsegmenter.tokenize(sentence)])

        inputs = self.tokenizer(segment, truncation=True, return_tensors='pt')
        return inputs


def output(encode_result):
    result = {
        "news": "sentence",
        "results": {}
    }
    factors = {"category": ["Không xác định", "Nông sản từ lúa, gạo", "Nông sản cà phê", "Nông sản cao su"],
               "price": ["Không xác định", "Giảm", "Ổn định", "Tăng"],
               "market": ["Không xác định", "Nguồn cung lớn hơn nhu cầu", "Nguồn cung và cầu ổn định",
                          "Nhu cầu lớn hơn Nguồn cung"],
               "polices": ["Không xác định",
                           "Chính sách", "Hiệp định", "Khác"],
               "internal": ["Không xác định", "Liên quan đến sản lượng nông sản",
                            "Liên quan đến chất lượng nông sản.", "Chi phí sản suất liên quan"],
               "external": ["Không xác định", "Dịch bệnh", "Thiên tai", "Khủng hoảng"]
               }
    for i, r in enumerate(factors):
        result["results"][r] = factors[r][int(encode_result[i])]
    return result
