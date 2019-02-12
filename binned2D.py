#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import csv

with open('data/2019-01-31_14-18-06.csv') as f:
    next(f)
    darray = []
    for l in csv.reader(f):
        if 'None' in l: continue
        darray.append([
                float(l[1]),
                float(l[2]),
                float(l[3])*1e9,
                float(l[4])*1e9
            ])
        

points_per_axes = 61
xbins = np.linspace(-15,15,points_per_axes)
ybins = np.linspace(-15,15,31)

img_fc = []
img_target = []

for x in xbins:
    row_target = []
    row_fc = []
    for y in ybins:
        point_fc = []
        point_target = []
        
        csv_array_length = len(darray)-1
        while csv_array_length >=0:
            d = darray[csv_array_length]
            if (
                d[0] > x-0.25 and 
                d[0] < x+0.25 and 
                d[1] > y-0.5 and 
                d[1] < y+0.5
            ):
                point_fc.append(d[3])
                point_target.append(d[2])
                del darray[csv_array_length]
            csv_array_length -=1
        
        if point_target:
            row_target.append(float(np.mean(point_target)))
        else:
            row_target.append(0)
        if point_fc:
            row_fc.append(float(np.mean(point_fc)))
        else:
            row_fc.append(0)
    img_fc.append(row_fc)
    img_target.append(row_target)
print(darray)


plt.imshow(img_fc,  extent=[-15,15,-15,15])
plt.colorbar()
plt.show()

plt.imshow(img_target, interpolation='nearest', extent=[-15,15,-15,15])
plt.colorbar()
plt.show()
