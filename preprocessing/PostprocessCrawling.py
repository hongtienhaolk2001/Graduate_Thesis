import re

from FilesProcessing import concat_files


def preprocess_crawled_Data(text):
    """
    - remove url (ex: https://xyz.com, www.xyz.com, etc.)
    - remove newline (\n) by "."
    - remove ";"
    """
    url_pattern = re.compile(r'https?://\S+|www\.\S+')  # find url
    text = url_pattern.sub(r' ', text)  # replace url by space " "
    text = text.replace("\n", ". ")  # replace newline by space ". "
    text = text.replace("\r.", "")  # remove Carriage Return
    text = text.replace("\r", "")  #
    text = text.replace("..", ".")  # replace ".." by space "."
    return text


def add_Label(dataframe):
    dataframe['price'] = [0] * len(dataframe)
    dataframe['gov'] = [0] * len(dataframe)
    dataframe['market'] = [0] * len(dataframe)
    dataframe['intrinsic'] = [0] * len(dataframe)
    dataframe['extrinsic'] = [0] * len(dataframe)
    return dataframe


def pipeline(dataframe):
    dataframe['content'] = dataframe['content'].apply(preprocess_crawled_Data)
    dataframe['brief'] = dataframe['brief'].apply(preprocess_crawled_Data)
    dataframe['title'] = dataframe['title'].apply(preprocess_crawled_Data)
    dataframe = add_Label(dataframe)
    print("preprocessing crawled data successfully")
    return dataframe


if __name__ == '__main__':
    df = concat_files(root_path='../data/raw_data/', file_format='csv')
    df = pipeline(df)
    df.to_csv(r'../data/crawled_data/crawled_rice_news.csv')
