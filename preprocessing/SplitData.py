import numpy as np
import pandas as pd


def split_train_test(dataframe, test_size, seed=19133022):
    shuffled = np.random.default_rng(seed=seed).permutation(len(dataframe))
    num_test = int(test_size * len(dataframe))
    test_index = shuffled[:num_test]
    train_index = shuffled[num_test:]
    return dataframe.iloc[train_index], dataframe.iloc[test_index]


def check_ratio(full_dataset, train_set, test_set):
    calc_ratio = lambda dataframe: (dataframe.iloc[:, :-1] != 0).sum() / len(dataframe)
    variance = lambda ratio_1, ratio_2, diff: np.all(((ratio_1 - ratio_2).abs() < diff).values)
    ratio = calc_ratio(full_dataset)
    train_ratio = calc_ratio(train_set)
    test_ratio = calc_ratio(test_set)
    condition_1 = variance(ratio, test_ratio, 0.002)
    condition_2 = variance(ratio, train_ratio, 0.002)
    return condition_1 and condition_2


def main():
    df = pd.read_csv("../data/original_data/original_data.csv")
    # df = pd.read_csv("../datasets/data_original/Original-datasets.csv")
    train_set, test_set = split_train_test(dataframe=df, test_size=0.2)
    print("Samples train: ", len(train_set))
    print("Samples val: ", len(test_set))
    # Save to csv
    train_set.to_csv('../data/training_data/train_datasets.csv')
    test_set.to_csv('../data/training_data/test_datasets.csv')
    print("save test & train datasets successfully")


if __name__ == '__main__':
    main()
