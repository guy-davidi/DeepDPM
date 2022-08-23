import numpy as np
import torch
from scipy.optimize import linear_sum_assignment as linear_assignment
import pandas as pd


def check_args(args, data_dim):
    # Make sure that the NIW prior's nu is defined correctly
    args.NIW_prior_nu = args.NIW_prior_nu or (data_dim + 2)
    if args.NIW_prior_nu < data_dim + 1:
        raise Exception(
            f"The chosen NIW nu hyperparameter need to be at least D+1 (D is the data dim). Set --NIW_prior_nu to at least {data_dim + 1}")

    # Ensure that there is no evaluation if no labels exist
    if not args.use_labels_for_eval:
        args.evaluate_every_n_epochs = 0


def best_cluster_fit(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_assignment(w.max() - w)
    best_fit = []
    for i in range(y_pred.size):
        for j in range(len(row_ind)):
            if row_ind[j] == y_pred[i]:
                best_fit.append(col_ind[j])
    return best_fit, row_ind, col_ind, w


def cluster_acc(y_true, y_pred):
    best_fit, row_ind, col_ind, w = best_cluster_fit(y_true, y_pred)
    return w[row_ind, col_ind].sum() * 1.0 / y_pred.size


def read_ts_dataset(args):
    root_dir_dataset = args.dir
    df_train = pd.read_csv(root_dir_dataset + '/' + args.dataset + '_TRAIN.tsv', sep='\t', header=None)

    df_test = pd.read_csv(root_dir_dataset + '/' + args.dataset + '_TEST.tsv', sep='\t', header=None)

    y_train = df_train.values[:, 0]
    y_test = df_test.values[:, 0]

    x_train = df_train.drop(columns=[0])
    x_test = df_test.drop(columns=[0])

    x_train.columns = range(x_train.shape[1])
    x_test.columns = range(x_test.shape[1])

    x_train = x_train.values
    x_test = x_test.values

    # znorm
    std_ = x_train.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_

    std_ = x_test.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_

    torch.save(x_train, root_dir_dataset + '/' + 'train_data.pt')
    torch.save(torch.ByteTensor(y_train), root_dir_dataset + '/' + 'train_labels.pt')
    torch.save(x_test, root_dir_dataset + '/' + 'test_data.pt')
    torch.save(torch.ByteTensor(y_test), root_dir_dataset + '/' + 'test_labels.pt')
