import os
import pandas as pd
import train


def merge_and_drop_Col(dataframe, cols):
    dataframe['News'] = dataframe['title'] + ' ' + dataframe['brief']
    dataframe = dataframe.drop(columns=cols)
    print("merge and drop columns successfully")
    return dataframe


def concat_files(root_path, file_format='csv'):
    """
    Merge Excel file a format used to label news data.
    """
    files = os.listdir(root_path)
    dataframe = []
    for f in files:
        print(f)
        _file = root_path + "/" + f
        if file_format == 'csv':
            dataframe.append(pd.read_csv(_file))
        else:
            dataframe.append(pd.read_excel(_file))
    df_full = pd.concat(dataframe, ignore_index=True)
    df_full = df_full.drop(df_full.columns[[0]], axis=1)  # remove "#" column
    print("concentrating successfully")
    return df_full


if __name__ == "__main__":
    df = concat_files(root_path=os.path.join(train.root_path, r"data/labeled_data"), file_format='xlsx')
    print(df.columns)
    drop_cols = ['title', 'date', 'brief', 'content', 'sources']
    df = merge_and_drop_Col(df, drop_cols)
    df.to_csv(os.path.join(train.root_path, r"data/original_data/original_data.csv"), index=False)
    print(f"save at {os.path.join(train.root_path, r'data/original_data/original_data.csv')} ")
