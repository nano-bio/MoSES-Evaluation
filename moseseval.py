import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy import optimize

fn_electrons = 'data/electrons.txt'
fn_ions = 'data/ions.txt'

# number of measurement cycles
m_cycles = 25
# cutoffs
co_start = 250
co_end = 7850

# time to not use for fitting before and after a peak
peak_dist_before = 5
peak_dist_after = 2

# which measurement section to plot
fittoplot = 22

# lambda to remove nA unit from reading files
remove_nA = lambda s: float(s.decode("utf-8").replace(' nA', ''))

# import electrons
# check whether the file exists. if not, try to use it as a relative path.
if not os.path.exists(fn_electrons):
    fn_electrons = os.path.normcase(os.path.join(os.path.dirname(__file__), fn_electrons))

electrons = np.loadtxt(fn_electrons, skiprows=0, usecols=1, delimiter=',', converters={1: remove_nA},
                       dtype=np.float64)

# import ions
# check whether the file exists. if not, try to use it as a relative path.
if not os.path.exists(fn_ions):
    fn_electrons = os.path.normcase(os.path.join(os.path.dirname(__file__), fn_ions))

ions = np.loadtxt(fn_ions, skiprows=0, delimiter=',', converters={1: remove_nA}, dtype=np.float64)

# combine everything into one array: 0 = time, 1 = ions, 2 = electrons, 3 = gamma
data = np.zeros((ions.shape[0], 4), dtype=np.float64)
data[:, 0] = ions[:, 0]
data[:, 1] = ions[:, 1]
data[:, 2] = electrons[0:ions.shape[0]]
data[:, 3] = np.abs(data[:, 2]) / (np.abs(data[:, 1]) - np.abs(data[:, 2]))

# cutoff data in the beginning and end
data = data[np.where((data[:, 0] >= co_start) & (data[:, 0] <= co_end)), :][0]

# find maxima
maxima_ind = scipy.signal.argrelmax(data[:, 3], order=300)
maxima = np.zeros((len(maxima_ind[0]), 2), np.float64)
maxima[:, 1] = data[maxima_ind, 3]
maxima[:, 0] = data[maxima_ind, 0]

# use the <number of measurement cycles> highest peaks
maxima = maxima[np.argsort(maxima[:, 1])]
maxima = maxima[-m_cycles:, :]

# resort by time
maxima = maxima[np.argsort(maxima[:, 0])]

# define error function for fit
exp_decay_2 = lambda p, x: p[0] + p[2]*np.exp(-(x-p[1])/p[3]) + p[4]*np.exp(-(x-p[1])/p[5])
errfunc = lambda p, x, y: exp_decay_2(p, x) - y

# prepare empty fit parameter list
p = [None] * 6
# prepare figure for plotting
fig1 = plt.figure()
ax0 = fig1.add_subplot(2, 2, 1)
ax1 = fig1.add_subplot(2, 2, 2)
ax2 = fig1.add_subplot(2, 2, 3)
ax3 = fig1.add_subplot(2, 2, 4)

# prepare list of results
fit_results = np.zeros((maxima.shape[0] - 1, 6), dtype=np.float64)

# loop through all between-maxima-intervals
for i in range(0, maxima.shape[0] - 1):
    # select proper data interval
    fitdata = data[np.where((data[:, 0] >= maxima[i, 0] + peak_dist_after) & (data[:, 0] <= maxima[i + 1, 0] - peak_dist_before)), :][0]

    # starting values for fit
    p[0] = 0.08 # gamma
    p[1] = maxima[i, 0]# time offset
    p[2] = 0.05
    p[3] = 8
    p[4] = 0.01
    p[5] = 100

    # fit the data
    try:
        res = optimize.least_squares(errfunc, np.asarray(p, dtype=np.float64), args=(fitdata[:, 0], fitdata[:, 3]))
        p1 = res.x
        fit_results[i, :] = p1
    except TypeError:
        print('Could not fit section {}.'.format(i))
        p1 = [None] * 6

    if i == fittoplot - 1:
        fitplot, = ax0.plot(fitdata[:, 0], exp_decay_2(p1, fitdata[:, 0]), 'r-')

gammaplot, = ax0.plot(data[:, 0], data[:, 3])
maximaplot, = ax0.plot(maxima[:, 0], maxima[:, 1], 'x')
ax0.set_title('Gamma')
ax0.set_xlabel('Time (s)')
ax0.set_ylabel('Gamma')
ax0.legend([gammaplot, maximaplot, fitplot], ['Gamma', 'Peak Maxima', 'Fit to section {}'.format(fittoplot)])

electronplot, = ax1.plot(data[:, 0], data[:, 2])
ax1.set_title('Electron current')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Current (nA)')

ionplot, = ax2.plot(data[:, 0], data[:, 1])
ionelectronplot, = ax2.plot(data[:, 0], data[:, 1] + data[:, 2])
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Current (nA)')
ax2.legend([ionplot, ionelectronplot], ['Ion Current', 'Sum of Ion and Electron current'])

ax3.plot(range(0, maxima.shape[0] - 1), fit_results[:, 0], 'b+')
ax3.set_title('Gamma')
ax3.set_xlabel('Measurement Position')

plt.show()
