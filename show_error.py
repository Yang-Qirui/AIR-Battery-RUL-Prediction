from cmath import sqrt
import json
import matplotlib.pyplot as plt
import numpy as np


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
print()
train_errs = train_json.values()
print(rmse(np.array(list(train_errs))))

test_id = [x for x in range(len(test_errs))]
train_id = [x for x in range(len(train_errs))]

fig = plt.figure()
test_ax = fig.add_subplot(1, 2, 1)
train_ax = fig.add_subplot(1, 2, 2)

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
# fig.legend()
fig.savefig("./err.png")
