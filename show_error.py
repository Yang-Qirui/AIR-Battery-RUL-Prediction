from cmath import sqrt
import json
import matplotlib.pyplot as plt
import numpy as np
import pickle

def mse(x):
    return np.sum(x ** 2 / len(x)).real


def rmse(x):
    return sqrt(mse(x)).real


def mae(x):
    return np.sum(abs(x) / len(x)).real


f_test = open("./result/error/test.json", 'r')
test_json = json.load(f_test)
f_train = open("./result/error/train.json", 'r')
train_json = json.load(f_train)

test_errs = test_json.values()
train_errs = train_json.values()

test_id = [x for x in range(len(test_errs))]
train_id = [x for x in range(len(train_errs))]

fig = plt.figure()
test_ax = fig.add_subplot(2, 2, 1)
train_ax = fig.add_subplot(2, 2, 2)

test_ax.scatter(test_id, test_errs)
test_ax.set(xlabel='case id', ylabel='error cycle')
test_ax.set_title(f'Test set errors')
input = np.array(list(test_errs))

txt = 'rmse : {:.3f} \n mae : {:.3f}'.format(
    rmse(input), mae(input), mse(input))
test_ax.text(0.5 * (min(test_id) + max(test_id)), 0.5 *
             (min(test_errs) + max(test_errs)), txt, ha='center', c='red')

train_ax.scatter(train_id, train_errs)
train_ax.set(xlabel='case id', ylabel='error cycle')
train_ax.set_title(f'Train set errors')

input = np.array(list(train_errs))
txt = 'rmse : {:.3f} \n mae : {:.3f}'.format(
    rmse(input), mae(input), mse(input))
train_ax.text(0.5 * (min(test_id) + max(train_id)), 0.5 *
             (min(train_errs) + max(train_errs)), txt, ha='center', c='red')

test_relative = []
train_relative = []
for key,value in test_json.items():
    f = open(f"./dataset/AIR/{key}.pkl",'rb')
    pkl = pickle.load(f)
    cycle_cnt = len(pkl[key]['data'].keys())
    test_relative.append(abs(value) / cycle_cnt)

for key,value in train_json.items():
    f = open(f"./dataset/AIR/{key}.pkl",'rb')
    pkl = pickle.load(f)
    cycle_cnt = len(pkl[key]['data'].keys())
    train_relative.append(abs(value) / cycle_cnt)

train_r_ax = fig.add_subplot(2,2,3)
test_r_ax = fig.add_subplot(2,2,4)

test_r_ax.scatter(test_id, test_relative)
test_r_ax.set(xlabel='case id', ylabel='relative error cycle')
test_r_ax.set_title(f'Test set relative errors')

train_r_ax.scatter(train_id, train_relative)
train_r_ax.set(xlabel='case id', ylabel='relative error cycle')
train_r_ax.set_title(f'Train set relative errors')

fig.subplots_adjust(wspace=0.3,hspace=0.5)

print(np.mean(test_relative))
print(np.mean(train_relative))
# fig.legend()
fig.savefig("./err.png")