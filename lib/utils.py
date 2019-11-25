import logging
import os
import csv
import random
import pandas as pd
import pickle
import sys
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse import linalg
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime



def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger


def prepare_train_valid_test_2d(data, p):
    p_valid_size = 0.2
    train_size = int(data.shape[0] * (1 - p - p_valid_size))
    valid_size = int(data.shape[0] * p_valid_size)

    train_set = data[0:train_size]
    valid_set = data[train_size: train_size + valid_size]
    test_set = data[train_size + valid_size:]

    return train_set, valid_set, test_set


def create_data_lstm_ed(data, seq_len, r, input_dim, output_dim, horizon):
    K = data.shape[1]
    T = data.shape[0]
    bm = binary_matrix(r, T, K)
    _data = data.copy()
    _std = np.std(data)

    _data[bm == 0] = np.random.uniform(_data[bm == 0] - _std, _data[bm == 0] + _std)

    en_x = np.zeros(shape=((T - seq_len - horizon) * K, seq_len, input_dim))
    de_x = np.zeros(shape=((T - seq_len - horizon) * K, horizon, output_dim))
    de_y = np.zeros(shape=((T - seq_len - horizon) * K, horizon, output_dim))

    _idx = 0
    for k in range(K):
        for i in range(T - seq_len - horizon):
            en_x[_idx, :, 0] = _data[i:i + seq_len, k]
            de_y[_idx, :, 0] = data[i + seq_len:i + seq_len + horizon, k]

            _idx += 1
    return en_x, de_x, de_y


def load_dataset_lstm_ed(seq_len, horizon, input_dim, output_dim, dataset, r, p, **kwargs):
    raw_data = np.load(dataset)['data']
    # reshape in case raw_data is rank 1 array
    raw_data = np.reshape(raw_data, (raw_data.shape[0], kwargs['model'].get('num_nodes')))

    raw_data = raw_data/1000
    # import matplotlib.pyplot as plt
    # plt.plot(raw_data, label='raw_data')
    # plt.legend()
    # plt.show()

    print('|--- Splitting train-test set.')
    train_data2d, valid_data2d, test_data2d = prepare_train_valid_test_2d(data=raw_data, p=p)
    print('|--- Normalizing the train set.')
    data = {}
    scaler = MinMaxScaler(copy=True)
    scaler.fit(train_data2d)
    train_data2d_norm = scaler.transform(train_data2d)
    valid_data2d_norm = scaler.transform(valid_data2d)
    test_data2d_norm = scaler.transform(test_data2d)

    data['test_data_norm'] = test_data2d_norm.copy()

    encoder_input_train, decoder_input_train, decoder_target_train = create_data_lstm_ed(train_data2d_norm,
                                                                                         seq_len=seq_len,
                                                                                         r=r,
                                                                                         input_dim=input_dim,
                                                                                         output_dim=output_dim,
                                                                                         horizon=horizon)
    encoder_input_val, decoder_input_val, decoder_target_val = create_data_lstm_ed(valid_data2d_norm,
                                                                                   seq_len=seq_len, r=r,
                                                                                   input_dim=input_dim,
                                                                                   output_dim=output_dim,
                                                                                   horizon=horizon)
    encoder_input_eval, decoder_input_eval, decoder_target_eval = create_data_lstm_ed(test_data2d_norm,
                                                                                      seq_len=seq_len, r=r,
                                                                                      input_dim=input_dim,
                                                                                      output_dim=output_dim,
                                                                                      horizon=horizon)

    for cat in ["train", "val", "eval"]:
        e_x, d_x, d_y = locals()["encoder_input_" + cat], locals()[
            "decoder_input_" + cat], locals()["decoder_target_" + cat]
        print(cat, "e_x: ", e_x.shape, "d_x: ", d_x.shape, "d_y: ", d_y.shape)
        data["encoder_input_" + cat] = e_x
        data["decoder_input_" + cat] = d_x
        data["decoder_target_" + cat] = d_y
    data['scaler'] = scaler

    return data


def create_data_att_res_seq2seq(data, seq_len, r, input_dim, output_dim, horizon):
    K = data.shape[1]
    T = data.shape[0]
    bm = binary_matrix(r, T, K)
    _data = data.copy()
    _std = np.std(data)

    _data[bm == 0] = np.random.uniform(_data[bm == 0] - _std, _data[bm == 0] + _std)

    en_x = np.zeros(shape=((T - seq_len - horizon) * K, seq_len, input_dim))
    de_x = np.zeros(shape=((T - seq_len - horizon) * K, horizon, output_dim))
    de_y = np.zeros(shape=((T - seq_len - horizon) * K, horizon, output_dim))

    _idx = 0
    for k in range(K):
        for i in range(T - seq_len - horizon):
            en_x[_idx, :, 0] = _data[i:i + seq_len, k]
            de_y[_idx, :, 0] = data[i + seq_len:i + seq_len + horizon, k]

            _idx += 1
    return en_x, de_x, de_y


def load_dataset_att_res_seq2seq(seq_len, horizon, input_dim, output_dim, dataset, r, p, **kwargs):
    raw_data = np.load(dataset)['data']
    # reshape in case raw_data is rank 1 array
    raw_data = np.reshape(raw_data, (raw_data.shape[0], kwargs['model'].get('num_nodes')))

    raw_data = raw_data/1000
    # import matplotlib.pyplot as plt
    # plt.plot(raw_data, label='raw_data')
    # plt.legend()
    # plt.show()

    print('|--- Splitting train-test set.')
    train_data2d, valid_data2d, test_data2d = prepare_train_valid_test_2d(data=raw_data, p=p)
    print('|--- Normalizing the train set.')
    data = {}
    scaler = MinMaxScaler(copy=True)
    scaler.fit(train_data2d)
    train_data2d_norm = scaler.transform(train_data2d)
    valid_data2d_norm = scaler.transform(valid_data2d)
    test_data2d_norm = scaler.transform(test_data2d)

    data['test_data_norm'] = test_data2d_norm.copy()

    encoder_input_train, decoder_input_train, decoder_target_train = create_data_att_res_seq2seq(train_data2d_norm,
                                                                                         seq_len=seq_len,
                                                                                         r=r,
                                                                                         input_dim=input_dim,
                                                                                         output_dim=output_dim,
                                                                                         horizon=horizon)
    encoder_input_val, decoder_input_val, decoder_target_val = create_data_att_res_seq2seq(valid_data2d_norm,
                                                                                   seq_len=seq_len, r=r,
                                                                                   input_dim=input_dim,
                                                                                   output_dim=output_dim,
                                                                                   horizon=horizon)
    encoder_input_eval, decoder_input_eval, decoder_target_eval = create_data_att_res_seq2seq(test_data2d_norm,
                                                                                      seq_len=seq_len, r=r,
                                                                                      input_dim=input_dim,
                                                                                      output_dim=output_dim,
                                                                                      horizon=horizon)

    for cat in ["train", "val", "eval"]:
        e_x, d_x, d_y = locals()["encoder_input_" + cat], locals()[
            "decoder_input_" + cat], locals()["decoder_target_" + cat]
        print(cat, "e_x: ", e_x.shape, "d_x: ", d_x.shape, "d_y: ", d_y.shape)
        data["encoder_input_" + cat] = e_x
        data["decoder_input_" + cat] = d_x
        data["decoder_target_" + cat] = d_y
    data['scaler'] = scaler

    return data


def cal_error(test_arr, prediction_arr):
    with np.errstate(divide='ignore', invalid='ignore'):
        # cal mse
        error_mae = mean_absolute_error(test_arr, prediction_arr)
        print('MAE: %.3f' % error_mae)

        # cal rmse
        error_mse = mean_squared_error(test_arr, prediction_arr)
        error_rmse = np.sqrt(error_mse)
        print('RMSE: %.3f' % error_rmse)

        # cal mape
        y_true, y_pred = np.array(test_arr), np.array(prediction_arr)
        error_mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        print('MAPE: %.3f' % error_mape)
        error_list = [error_mae, error_rmse, error_mape]
        return error_list


def binary_matrix(r, row, col):
    tf = np.array([1, 0])
    bm = np.random.choice(tf, size=(row, col), p=[r, 1.0 - r])
    return bm


def save_metrics(error_list, log_dir, alg):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    error_list.insert(0, dt_string)
    with open(log_dir + alg + "_metrics.csv", 'a') as file:
        writer = csv.writer(file)
        writer.writerow(error_list)
