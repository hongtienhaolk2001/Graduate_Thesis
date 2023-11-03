import numpy as np
import pandas as pd


def split_train_test(data, test_size):
    shuffled = np.random.permutation(len(data))
    num_test = int(test_size * len(data))
    test_index = shuffled[:num_test]
    train_index = shuffled[num_test:]
    return data.iloc[train_index], data.iloc[test_index]


def main():
    all_datasets = pd.read_csv("../data/original_data/original_data.csv")
    while True:
        train_set, test_set = split_train_test(data=all_datasets, test_size=0.2)
        ratio = (all_datasets.iloc[:, 1:] != 0).sum() / len(all_datasets)
        train_ratio = (train_set.iloc[:, 1:] != 0).sum() / len(train_set)
        val_ratio = (test_set.iloc[:, 1:] != 0).sum() / len(test_set)
        if np.all(((ratio - val_ratio).abs() < 0.002).values) and np.all(((ratio - train_ratio).abs() < 0.002).values):
            break
    print("Samples train: ", len(train_set))
    print("Samples val: ", len(test_set))
    # Save to csv
    train_set.to_csv('../data/training_data/train_datasets.csv')
    test_set.to_csv('../data/training_data/test_datasets.csv')
    print("save test & train datasets successfully")


if __name__ == '__main__':
    main()
