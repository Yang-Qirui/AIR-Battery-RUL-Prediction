import glob
import pickle
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def drop_outlier(array, count, bins):
    index = []
    range_ = np.arange(1, count, bins)
    for i in range_[:-1]:
        array_lim = array[i:i+bins]
        sigma = np.std(array_lim)
        mean = np.mean(array_lim)
        th_max, th_min = mean + sigma*2, mean - sigma*2
        idx = np.where((array_lim < th_max) & (array_lim > th_min))
        idx = idx[0] + i
        index.extend(list(idx))
    return np.array(index)


def show_battery_data(Battery_list, Battery):
    fig, ax = plt.subplots(1, figsize=(12, 8))
    color_list = ['b:', 'g--', 'r-.', 'c.']
    for name, color in zip(Battery_list, color_list):
        df_result = Battery[name]
        ax.plot(df_result['cycle'], df_result['capacity'],
                color, label='Battery_'+name)
    ax.set(xlabel='Discharge cycles', ylabel='Capacity (Ah)',
           title='Capacity degradation at ambient temperature of 1°C')
    plt.legend()


def load(Battery_list=['CS2_35', 'CS2_36']):
    dir_path = 'dataset/CALCE/'
    Battery = {}
    for name in Battery_list:
        print('Load Dataset ' + name + ' ...')
        path = glob.glob(dir_path + name + '/*.xlsx')

        dates = []
        for p in path:
            df = pd.read_excel(p, sheet_name=1)
            print('Load ' + str(p) + ' ...')
            # print(df['Date_Time'][0])
            dates.append(df['Date_Time'][0])
        idx = np.argsort(dates)
        path_sorted = np.array(path)[idx]
        print(path_sorted)

        count = 0
        discharge_capacities = []
        health_indicator = []
        internal_resistance = []
        CCCT = []
        CVCT = []
        for p in path_sorted:
            df = pd.read_excel(p, sheet_name=1)
            print('Load ' + str(p) + ' ...')
            cycles = list(set(df['Cycle_Index']))
            for c in cycles:
                df_lim = df[df['Cycle_Index'] == c]
                # Charging
                df_c = df_lim[(df_lim['Step_Index'] == 2) |
                              (df_lim['Step_Index'] == 4)]
                c_v = df_c['Voltage(V)']
                c_c = df_c['Current(A)']
                c_t = df_c['Test_Time(s)']
                #CC or CV
                df_cc = df_lim[df_lim['Step_Index'] == 2]
                df_cv = df_lim[df_lim['Step_Index'] == 4]
                CCCT.append(np.max(df_cc['Test_Time(s)']) -
                            np.min(df_cc['Test_Time(s)']))
                CVCT.append(np.max(df_cv['Test_Time(s)']) -
                            np.min(df_cv['Test_Time(s)']))

                # Discharging
                df_d = df_lim[df_lim['Step_Index'] == 7]
                d_v = df_d['Voltage(V)']
                # print(d_v.max(),d_v.min())
                d_c = df_d['Current(A)']
                d_t = df_d['Test_Time(s)']
                d_im = df_d['Internal_Resistance(Ohm)']

                if (len(list(d_c)) != 0):
                    time_diff = np.diff(list(d_t))
                    d_c = np.array(list(d_c))[1:]
                    discharge_capacity = time_diff*d_c/3600  # Q = A*h
                    discharge_capacity = [np.sum(discharge_capacity[:n]) for n in range(
                        discharge_capacity.shape[0])]
                    discharge_capacities.append(-1*discharge_capacity[-1])

                    # print(-1*discharge_capacity[-1])
                    dec = np.abs(np.array(d_v) - 3.8)[1:]
                    start = np.array(discharge_capacity)[np.argmin(dec)]
                    dec = np.abs(np.array(d_v) - 3.4)[1:]
                    end = np.array(discharge_capacity)[np.argmin(dec)]
                    health_indicator.append(-1 * (end - start))

                    internal_resistance.append(np.mean(np.array(d_im)))
                    count += 1

        discharge_capacities = np.array(discharge_capacities)
        health_indicator = np.array(health_indicator)
        internal_resistance = np.array(internal_resistance)
        CCCT = np.array(CCCT)
        CVCT = np.array(CVCT)

        idx = drop_outlier(discharge_capacities, count, 40)
        df_result = pd.DataFrame({'cycle': np.linspace(1, idx.shape[0], idx.shape[0]),
                                  'capacity': discharge_capacities[idx],
                                  'SoH': health_indicator[idx],
                                  'resistance': internal_resistance[idx],
                                  'CCCT': CCCT[idx],
                                  'CVCT': CVCT[idx]})
        print(df_result)
        Battery[name] = df_result
    return Battery


def load_from_pickle(train_size=55,type='train'):
    dir_path = 'dataset/AIR'
    all_pkl = os.listdir(dir_path)
    print(f'Train size {train_size}, test size {len(all_pkl) - train_size}')
    random.shuffle(all_pkl) # random choose train set and test set
    if type == 'train':
        print('Load Train Dataset ...')
        train_pkl = all_pkl[:train_size]
        train_batteries, train_names = generate_dataset_from_pkl(train_pkl)
        return train_batteries, train_names
    else:
        print('Load Test Dataset ...')
        test_pkl = all_pkl[train_size:]
        test_batteries, test_names = generate_dataset_from_pkl(test_pkl)
        return test_batteries, test_names
    # print(len(train_pkl),len(test_pkl))


def generate_dataset_from_pkl(pkls):
    CHARGE_STAGES = [
        'Constant current-constant voltage charge', 'Constant current charge']
    DISCHARGE_STAGES = ['Constant current discharge_0', 'Constant current discharge_1',
                        'Constant current discharge_2', 'Constant current discharge_3']
    dir_path = 'dataset/AIR'
    Battery = {}
    for pkl_path in pkls:
        CCCT = []
        CVCT = []
        discharge_capacities = []
        health_indicator = []
        count = 0
        print("Loading " + pkl_path + " data ...")
        pkl_file = open("/".join([dir_path, pkl_path]), 'rb')
        pkl = pickle.load(pkl_file)[pkl_path.split(".")[0]]
        data = pkl['data']
        for cycle_id, cycle in data.items():
            df_charge = cycle[cycle['Status'].isin(CHARGE_STAGES)]
            charge_voltage = df_charge['Voltage (V)']
            charge_current = df_charge['Current (mA)'] / 1000  # mA to A
            charge_time = df_charge['Time (s)']

            df_charge_current = df_charge[df_charge['Status']
                                          == CHARGE_STAGES[0]]
            df_charge_voltage = df_charge[df_charge['Status']
                                          == CHARGE_STAGES[1]]

            CCCT.append(
                np.max(df_charge_current['Time (s)']) - np.min(df_charge_current['Time (s)']))
            CVCT.append(
                np.max(df_charge_voltage['Time (s)']) - np.min(df_charge_voltage['Time (s)']))

            df_discharge = cycle[cycle['Status'].isin(DISCHARGE_STAGES)]
            discharge_voltage = df_discharge['Voltage (V)']
            # print(discharge_voltage.max(),discharge_voltage.min())
            discharge_current = df_discharge['Current (mA)'] / 1000  # mA to A
            discharge_time = df_discharge['Time (s)']

            if (len(list(discharge_current)) != 0):
                time_diff = np.diff(list(discharge_time))
                discharge_current = np.array(list(discharge_current))[1:]
                discharge_capacity = time_diff * discharge_current / 3600
                discharge_capacity = [np.sum(discharge_capacity[:n]) for n in range(
                    discharge_capacity.shape[0])]

                # discharge_capacity = pkl['dq'][cycle_id]
                discharge_capacities.append(
                    pkl['dq'][cycle_id] / 1000)  # mAh to Ah

                dec = np.abs(np.array(discharge_voltage) - 3.1)[1:]
                start = np.array(discharge_capacity)[np.argmin(dec)]
                dec = np.abs(np.array(discharge_voltage) - 2.7)[1:]
                end = np.array(discharge_capacity)[np.argmin(dec)]
                health_indicator.append(-1 * (end - start))
                # print(-1 * (end - start))

                count += 1

        discharge_capacities = np.array(discharge_capacities)
        health_indicator = np.array(health_indicator)
        CCCT = np.array(CCCT)
        CVCT = np.array(CVCT)

        idx = drop_outlier(discharge_capacities, count, 40)
        df_result = pd.DataFrame({
            'cycle': np.linspace(1, idx.shape[0], idx.shape[0]),
            'capacity': discharge_capacities[idx],
            'SoH': health_indicator[idx],
            'CCCT': CCCT[idx],
            'CVCT': CVCT[idx]
        })
        # print(df_result)
        Battery[pkl_path.split(".")[0]] = df_result
    return Battery, [pkls[i].split(".")[0] for i in range(len(pkls))]


if __name__ == "__main__":
    train_batteries, train_names, test_batteries, test_names = load_from_pickle(
        train_size=55)
    fig = plt.figure()
    ax = plt.subplot(1,2,1)
    ax1 = plt.subplot(1,2,2)
    color_list = ['b:', 'g--', 'r-.', 'c:']
    for name, color in zip(train_names, color_list):
        battery = train_batteries[name]
        ax.plot(battery['cycle'], battery['capacity'],
                color, label='Battery_'+name)

    for name, color in zip(test_names, color_list):
        battery = test_batteries[name]
        ax1.plot(battery['cycle'], battery['capacity'],
                color, label='Battery_'+name)

    ax.set(xlabel='Discharge cycles', ylabel='Discharge Capacity (mAh)')
    ax1.set(xlabel='Discharge cycles', ylabel='Discharge Capacity (mAh)')

    plt.legend()
    plt.show()

    # load_from_pickle()
