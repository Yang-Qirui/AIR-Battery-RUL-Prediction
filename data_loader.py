import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
            dates.append(df['Date_Time'][0])
        idx = np.argsort(dates)
        path_sorted = np.array(path)[idx]

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
        Battery[name] = df_result
    return Battery
