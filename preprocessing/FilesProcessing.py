import os
import pandas as pd


def merge_and_drop_Col(dataframe):
    dataframe['News'] = dataframe['title'] + ' ' + dataframe['brief']
    dataframe = dataframe.drop(columns=['Unnamed: 0', 'title', 'date', 'brief', 'content', 'sources'])
    return dataframe


def concat_files(root_path, file_format='csv'):
    """
    Merge Excel file a format used to label news data.
    """
    files = os.listdir(root_path)
    df = []
    for f in files:
        print(f)
        _file = root_path + "/" + f
        if file_format == 'csv':
            df.append(pd.read_csv(_file))
        else:
            df.append(pd.read_excel(_file))
    df_full = pd.concat(df, ignore_index=True)
    df_full = df_full.drop(df_full.columns[[0]], axis=1)  # remove "#" column
    print("concentrating successfully")
    # try:
    #     df_full.to_csv(output_path)
    #     print("saving successfully")
    # except:
    #     print("get error when saving merged file")
    return df_full

# if __name__ == "__main__":
#     concat_files(root_path='../data/raw_data/',
#                  output_path="../data/_data/merge_data.csv")
