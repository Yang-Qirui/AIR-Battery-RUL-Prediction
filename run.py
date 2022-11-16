import copy
import json
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
from adaTransformer import Transformer
from data_loader import load, load_from_pickle
import gc
import time
import os
from tqdm import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
gc.collect()
torch.cuda.empty_cache()


def build_sequences(text, window_size):
    # text:list of capacity
    x, y = [], []
    for i in range(len(text) - window_size):
        sequence = text[i:i+window_size]
        target = text[i+1:i+1+window_size]

        x.append(sequence)
        y.append(target)

    return np.array(x), np.array(y)


# leave-one-out evaluation: one battery is sampled randomly; the remainder are used for training.
def get_train_test(data_dict, name, window_size=8):
    data_sequence = data_dict[name]['capacity']
    train_data, test_data = data_sequence[:window_size +
                                          1], data_sequence[window_size+1:]
    train_x, train_y = build_sequences(
        text=data_sequence, window_size=window_size)
    # for k, v in data_dict.items():
    #     if k != name or len(data_dict.keys()) == 1:
    #         data_x, data_y = build_sequences(
    #             text=v['capacity'], window_size=window_size)
    #         train_x, train_y = np.r_[train_x, data_x], np.r_[train_y, data_y]
    # print(train_x)
    # print(train_y)
    return train_x, train_y, list(train_data), list(test_data)


def relative_error(y_test, y_predict, threshold):
    true_re, pred_re = len(y_test), 0
    for i in range(len(y_test)-1):
        if y_test[i] <= threshold >= y_test[i+1]:
            true_re = i - 1
            break
    for i in range(len(y_predict)-1):
        if y_predict[i] <= threshold:
            pred_re = i - 1
            break
    return abs(true_re - pred_re)/true_re if abs(true_re - pred_re)/true_re <= 1 else 1


def evaluation(y_test, y_predict):
    mse = mean_squared_error(y_test, y_predict)
    rmse = sqrt(mean_squared_error(y_test, y_predict))
    return rmse


def setup_seed(seed):
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def train(Battery, Battery_list, epoch, dist_old, weight_mat, model, optimizer, feature_size=8, len_seq=1, num_layers=1,
          alpha=0.0):
    loss_list = []
    dist_mat = torch.zeros(num_layers, len_seq)
    for i in tqdm(range(len(Battery_list))):
        name = Battery_list[i]
        # print(f"Training {name} ...")
        window_size = feature_size
        train_x, train_y, train_data, test_data = get_train_test(
            Battery, name, window_size)
        train_size = len(train_x)
        # print('sample size: {}'.format(train_size))
        # print('rated capacity: {}'.format(train_data[0]))
        rated_capacity = train_data[0]
        criterion = nn.MSELoss()
        # for epoch in range(EPOCH):
        # (batch_size, seq_len, input_size)
        X = np.reshape(train_x/rated_capacity,
                       (-1, 1, feature_size)).astype(np.float32)
        # shape 为 (batch_size, 1)
        y = np.reshape(train_y[:, -1]/rated_capacity,
                       (-1, 1)).astype(np.float32)
        X, y = torch.from_numpy(X).to(
            device), torch.from_numpy(y).to(device)
        output, decode, list_encoding = model(X)
        loss_adapt, dist, weight_mat = model.adapt_encoding_weight(
            list_encoding, weight_mat)
        dist_mat = dist_mat.to(device)
        dist = dist.to(device)
        dist_mat = dist_mat + dist
        output = output.reshape(-1, 1)
        loss = criterion(output, y) + alpha * \
            criterion(decode, X.reshape(-1, feature_size)) \
            + 0.5 * loss_adapt
        optimizer.zero_grad()              # clear gradients for this training step
        loss.backward()                    # backpropagation, compute gradients
        optimizer.step()                   # apply gradients
        loss_list.append(loss.item())
    if epoch > 0:
        weight_mat = model.update_weight_Boosting(
            weight_mat, dist_old, dist_mat)
    loss = np.array(loss_list).mean(axis=0)
    # if (epoch + 1) % 10 == 0:
    #     test_x = train_data.copy()
    #     point_list = []
    #     while (len(test_x) - len(train_data)) < len(test_data):
    #         x = np.reshape(np.array(
    #             test_x[-feature_size:])/rated_capacity, (-1, 1, feature_size)).astype(np.float32)
    #         # print(x.shape)
    #         # (batch_size,feature_size=1,input_size)
    #         x = torch.from_numpy(x).to(device)
    #         # pred shape (batch_size=1, feature_size=1)
    #         pred, _, _ = model(x)
    #         next_point = pred.data.cpu().numpy()[
    #             0, 0] * rated_capacity
    #         # The test values are added to the original sequence to continue to predict the next point
    #         test_x.append(next_point)
    #         # Saves the predicted value of the last point in the output sequence
    #         point_list.append(next_point)
    #     # Save all the predicted values
    #     y_.append(point_list)
    #     loss_list.append(loss)
    #     rmse = evaluation(y_test=test_data, y_predict=y_[-1])
    #     re = relative_error(
    #         y_test=test_data, y_predict=y_[-1], threshold=rated_capacity*0.7)
    #     #print('epoch:{:<2d} | loss:{:<6.4f} | RMSE:{:<6.4f} | RE:{:<6.4f}'.format(epoch, loss, rmse, re))
    # if epoch % 100 == 0:
    #     print(f"Epoch {epoch}, Done ...")
    # if metric == 're':
    #     score = [re]
    # elif metric == 'rmse':
    #     score = [rmse]
    # else:
    #     score = [re, rmse]
    # if (loss < 0.0005) and (score_[0] < score[0]):
    #     break
    # score_ = score.copy()
    # score_list.append(score_)
    # result_list.append(y_[-1])
    return loss, weight_mat, dist_mat


def test(Battery, Battery_list, model, feature_size=8):
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()
    for i in tqdm(range(len(Battery_list))):
        name = Battery_list[i]
        _, _, train_data, test_data = get_train_test(
            Battery, name, window_size=feature_size)
        test_x = train_data.copy()
        rated_capacity = train_data[0]
        # print('test', len(test_data))
        while (len(test_x) - len(train_data)) < len(test_data):
            x = np.reshape(np.array(
                test_x[-feature_size:])/rated_capacity, (-1, 1, feature_size)).astype(np.float32)
            x = torch.from_numpy(x).to(device)
            with torch.no_grad():
                pred, _, _ = model(x)
            next_point = pred.data.cpu().numpy()[0, 0] * rated_capacity
            test_x.append(next_point)
        # print(len(test_x[len(train_data):]))

        loss = criterion(
            torch.from_numpy(np.array(test_x[len(train_data):])), torch.from_numpy(np.array(test_data)))
        total_loss += loss.item()
    loss = total_loss / len(Battery_list)
    return loss


def predict(Battery, Battery_list, model, feature_size=8, terminal_rate=0.8):
    model.eval()
    with torch.no_grad():
        result_list = []
        window_size = feature_size
        for i in range(len(Battery_list)):
            name = Battery_list[i]
            print(f"Testing {name} ...")
            train_data = list(
                Battery[name]['capacity'][:window_size + 1])
            rated_capacity = Battery[name]['capacity'][0]
            aa = train_data.copy()
            pred_list = Battery[name]['capacity'][:window_size + 1].tolist()
            while pred_list[-1] > rated_capacity * terminal_rate or len(pred_list) < len(Battery[name]['capacity']):
                X = np.reshape(np.array(
                    aa[-feature_size:])/rated_capacity, (-1, 1, feature_size)).astype(np.float32)
                X = torch.from_numpy(X).to(device)
                pred, _, _ = model(X)
                next_point = pred.data.cpu().numpy()[0, 0] * rated_capacity
                aa.append(next_point)
                pred_list.append(next_point)
                if np.sum(np.array(pred_list) ** 2) / len(pred_list) < 0.1:
                    break
            result_list.append(pred_list)
        return result_list


'''
Rated_Capacity = 1.1
window_size = 64
feature_size = window_size
dropout = 0.0
EPOCH = 500
nhead = 16
weight_decay = 0.0
noise_level = 0.0
alpha = 0.01
lr = 0.0005    # learning rate
hidden_dim = 32
num_layers = 1
is_load_weights = False
metric = 're'
re mean: 0.0536
Rated_Capacity = 1.1
window_size = 64
feature_size = window_size
dropout = 0.0
EPOCH = 500
nhead = 16
weight_decay = 0.0
noise_level = 0.0
alpha = 0.01
lr = 0.0005    # learning rate
hidden_dim = 32
num_layers = 1
is_load_weights = False
metric = 'rmse'
rmse mean: 0.0690
'''


def get_optimal_params():
    Rated_Capacity = 1.1
    window_size = 64
    feature_size = window_size
    dropout = 0.0
    EPOCH = 20
    nhead = 16
    is_load_weights = False
    weight_decay = 0.0
    noise_level = 0.0
    num_layers = 1
    metric = 're'

    Battery_list = ['CS2_35', 'CS2_36']
    Battery = load(Battery_list)

    states = {}
    for lr in [1e-4, 5e-4, 1e-3, 1e-2]:
        for hidden_dim in [16, 32, 64]:
            for alpha in [1e-4, 1e-3, 1e-2]:
                show_str = 'lr={}, num_layers={}, hidden_dim={}'.format(
                    lr, num_layers, hidden_dim)
                print(show_str)
                SCORE = []
                for seed in range(5):
                    print('seed:{}'.format(seed))
                    score_list, _ = train(Battery=Battery, Battery_list=Battery_list, lr=lr, feature_size=feature_size, hidden_dim=hidden_dim, num_layers=num_layers, nhead=nhead,
                                          weight_decay=weight_decay, EPOCH=EPOCH, seed=seed, dropout=dropout, alpha=alpha,
                                          noise_level=noise_level, metric=metric, is_load_weights=is_load_weights)
                    print(np.array(score_list))
                    print(
                        metric + ': {:<6.4f}'.format(np.mean(np.array(score_list))))
                    print(
                        '------------------------------------------------------------------')
                    for s in score_list:
                        SCORE.append(s)

                print(
                    metric + ' mean: {:<6.4f}'.format(np.mean(np.array(SCORE))))
                states[show_str] = np.mean(np.array(SCORE))
                print(
                    '===================================================================')

    min_key = min(states, key=states.get)
    print('optimal parameters: {}, result: {}'.format(
        min_key, states[min_key]))


def save_result(batteries, battery_names, predict_results, error_dict, s_time, window_size=100, terminal_rate=0.8, type='train'):
    for i in range(len(battery_names)):
        name = battery_names[i]
        battery = batteries[name]
        fig, ax = plt.subplots(1, figsize=(12, 8))

        ax.plot(battery['cycle'], battery['capacity'], 'b.', label=name)
        ax.plot([x for x in range(len(predict_results[i]))], predict_results[i])

        target_terminal_cycle = -1
        pred_terminal_cycle = -1
        for j, c in enumerate(battery['capacity']):
            if c <= battery['capacity'][0] * terminal_rate:
                target_terminal_cycle = j
                break

        if target_terminal_cycle < 0:
            target_terminal_cycle = max(battery['cycle'].keys())

        for j, c in enumerate(predict_results[i]):
            if c <= battery['capacity'][0] * terminal_rate:
                pred_terminal_cycle = j
                break

        if pred_terminal_cycle < 0:
            pred_terminal_cycle = max(battery['cycle'].keys())

        error_dict[name] = pred_terminal_cycle - target_terminal_cycle

        plt.plot([min(battery['cycle'].keys()), max(battery['cycle'].keys())], [battery['capacity'][0] * terminal_rate, battery['capacity'][0] * terminal_rate],
                 c='black', lw=1, ls='--')
        plt.axvline(window_size)
        ax.set(xlabel='Discharge cycles', ylabel='Capacity (Ah)')
        ax.set_title(f'{type} case {name}. Error {error_dict[name]} cycle.')
        plt.legend()
        plt.savefig(f'./result/{type}/{s_time}/{name}.png')
        plt.close()

    with open(f'./result/error/{type}/{s_time}.json', 'w') as f:
        json.dump(error_dict, f)


def main():
    window_size = 64
    feature_size = window_size
    dropout = 0.0
    EPOCH = 200
    nhead = 16
    weight_decay = 0.0
    noise_level = 0.0
    alpha = 0.01
    lr = 0.0005    # learning rate
    hidden_dim = 64
    num_layers = 2
    is_load_weights = False
    metric = 'rmse'
    train_size = 55
    terminal_rate = 0.8

    seed = 0
    SCORE = []

    s_time = int(round(time.time() * 1000))
    s_time = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(s_time / 1000))

    train_batteries, train_names, valid_batteries, valid_names, test_batteries, test_names = load_from_pickle(
        train_size=train_size)



    if not is_load_weights:
        print('seed:{}'.format(seed))
        model = Transformer(feature_size=feature_size, hidden_dim=hidden_dim, num_layers=num_layers, nhead=nhead, dropout=dropout,
                            noise_level=noise_level)
        model = model.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay)
        best_score = np.inf
        weight_mat, dist_mat = None, None

        for epoch in range(EPOCH):
            print('Epoch:', epoch)
            print('Training ...')
            loss, weight_mat, dist_mat = train(Battery=train_batteries, Battery_list=train_names, epoch=epoch, model=model, optimizer=optimizer,
                                               dist_old=dist_mat, weight_mat=weight_mat, feature_size=feature_size, num_layers=num_layers, alpha=alpha)
            print('Validating ...')
            val_loss = test(Battery=valid_batteries, Battery_list=valid_names,
                            model=model, feature_size=feature_size)
            test_loss = test(Battery=test_batteries, Battery_list=test_names,
                             model=model, feature_size=feature_size)
            print('Valid %.6f, Test %.6f' % (val_loss, test_loss))
            if val_loss < best_score:
                best_score = val_loss
                torch.save(model, "./result/model/adaTransformer-{}.pth".format(
                    s_time))
    else:
        '''choose a version of model'''
        model = torch.load('./result/model/adaTransformer.pth')

    os.makedirs(f"./result/test/{s_time}")
    os.makedirs(f"./result/train/{s_time}")

    train_error_dict = {}
    test_error_dict = {}

    predict_results = predict(train_batteries, train_names, model,
                              feature_size=feature_size, terminal_rate=terminal_rate)
    save_result(train_batteries, train_names, predict_results,
                train_error_dict, s_time, type='train')

    predict_results = predict(
        test_batteries, test_names, model, feature_size=feature_size, terminal_rate=terminal_rate)
    save_result(test_batteries, test_names, predict_results,
                test_error_dict, s_time, type='test')

    print('Done ....')


if __name__ == "__main__":
    main()
