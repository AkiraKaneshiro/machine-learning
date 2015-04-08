### Daniel Kronovet
### dbk2123@columbia.edu

'''
This is the data visualization module of the library.

The assumption is that the data is passed as a dictionary,
    with the keys representing various parameters of the model,
    and the values being pandas Series or numpy Arrays.
'''

import math

import matplotlib.pyplot as plt

COLORS = ['green', 'blue', 'red', 'orange', 'yellow', 'purple']

def plot_params_hist(data_dict):
    root_num_axes = int(math.ceil(math.sqrt(len(data_dict))))
    fig, axes = plt.subplots(root_num_axes, root_num_axes)
    keys = data_dict.keys()
    keys.sort()
    for i, param in enumerate(keys):
        plot = axes[i / root_num_axes][i % root_num_axes]
        data = data_dict[param]#.values
        color = COLORS[i % len(COLORS)]
        plot.hist(data, bins=50, label='param={}'.format(param), color=color)
        plot.legend(loc='best')
    plt.show()

def print_data_by_param(data_dict, label):
    print 50 * '#'
    print '{} as a function of param:'.format(label)
    print 'Param', '\tMean', '\t\tVar'
    for param in data_dict:
        data = data_dict[param]
        print param, '\t', data.mean(), '\t', data.std()**2
    print

def print_func_by_param(data_dict, f, label):
    print 50 * '#'
    print '{} and {} as a function of param:'.format(label, f.func_name)
    print 'Param', '\tMean', '\t\t\tVar', '\t\t{}'.format(f.func_name)
    for param in data_dict:
        data = data_dict[param]
        print param, '\t', data.mean(), '\t', data.std()**2, '\t', f(data)
    print

def compare_series(series_dict):
    for k in series_dict:
        plt.plot(series_dict[k], label=k)
    plt.legend(loc='best')
    plt.show()

def draw_sample(sample):
    x1 = [x[0] for x in sample]
    x2 = [x[1] for x in sample]
    plt.scatter(x1, x2)
