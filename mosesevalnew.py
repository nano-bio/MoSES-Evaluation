import csv
import json
from datetime import datetime
from time import sleep

xmin = -2
xmax = 2
xstep = 0.5
ymin = -2
ymax = 2
ystep = 0.5

abs_diff = 0.001

csv_file_names = [
    'data/2019-01-14_17-46-15.csv',
    'data/2019-01-14_14-39-06.csv',
    'data/2019-01-14_11-41-14.csv',
]

csv_file_names = [
    'data/2019-01-31_10-53-41.csv',
]


def float_ranger(x_min, x_max, x_step):
    max_points = 1000
    xrange = []
    x = x_min
    while x <= x_max + 0.000001:  # float rounding errors
        xrange.append(x)
        x += x_step
        if len(xrange) > max_points:
            break
    return xrange


positions = []
for x in float_ranger(xmin, xmax, xstep):
    for y in float_ranger(ymin, ymax, ystep):
        positions.append({
            'x': x,
            'y': y,
            'data': [],
            'gamma': [],
        })

for csv_file_name in csv_file_names:
    with open(csv_file_name) as f:
        reader = csv.reader(f)
        next(reader)

        for l in reader:
            if "None" in l: continue
            for pos in positions:
                if abs(float(l[1]) - pos['x']) < 0.001 and abs(float(l[2]) - pos['y']) < 0.001:
                    pos['data'].append([
                        datetime.strptime(l[0], '%Y-%m-%dT%H:%M:%S.%f'),
                        float(l[1]),
                        float(l[2]),
                        float(l[3]),
                        float(l[4])
                    ])
                    continue
_i = 0
for pos in positions:
    _i += 1
    for point in pos['data']:
        pos['gamma'].append([
            point[0],
            point[4] / (point[3] - point[4])
        ])
    import matplotlib.pyplot as plt
    import numpy

    A = numpy.array(pos['gamma'])
    plt.plot(A[:, 0], A[:, 1])
    plt.show()
    sleep(0.5)
