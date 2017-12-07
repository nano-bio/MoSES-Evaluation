import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy import optimize
import argparse

# using the argparse module to make use of command line options
parser = argparse.ArgumentParser(description="Evaluation script for MoSES measurements")

# add commandline options
parser.add_argument("--filename_ions",
                    "-f_ions",
                    help="Specify a filename (ions) to be evaluated. Defaults to ions.txt",
                    default='data\ions.txt')
parser.add_argument("--filename_electrons",
                    "-f_electrons",
                    help="Specify a filename (electrons) to be evaluated. Defaults to electrons.txt",
                    default='data\electrons.txt')
parser.add_argument("--fit_plot",
                    "-fp",
                    help="Specify which fit to plot. Defaults to 22",
					type=int,
                    default=22)

# parse it
args = parser.parse_args()

fn_electrons = args.filename_electrons
fn_ions = args.filename_ions

# number of measurement cycles
m_cycles = 25
# cutoffs
co_start = 250
co_end = 7850

# minimum optimality for fit. if larger than that, a "molecule-potential" (negative amplitude) is tried
min_optimality_req = 1e-6

# maximum time constant to be accepted for fit.
max_time_constant = 2000

# time to not use for fitting before and after a peak
peak_dist_before = 5
peak_dist_after = 2

# which measurement section to plot
fittoplot = args.fit_plot

# this determines, whether minima are used instead of maxima
minima_used = False

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
    fn_ions = os.path.normcase(os.path.join(os.path.dirname(__file__), fn_ions))

# export path for the fit results
export_file_results = fn_ions.split('_ions.')[0] + '_results.txt'
export_file_gamma = fn_ions.split('_ions.')[0] + '_gamma.txt'

ions = np.loadtxt(fn_ions, skiprows=0, delimiter=',', converters={1: remove_nA}, dtype=np.float64)

# combine everything into one array: 0 = time, 1 = ions, 2 = electrons, 3 = gamma
data = np.zeros((ions.shape[0], 4), dtype=np.float64)
data[:, 0] = ions[:, 0]
data[:, 1] = ions[:, 1]
try:
    data[:, 2] = electrons[0:ions.shape[0]]
except ValueError:
    # not enough electron measurement points -> let's do it the other way around
    # combine everything into one array: 0 = time, 1 = ions, 2 = electrons, 3 = gamma
    data = np.zeros((electrons.shape[0], 4), dtype=np.float64)
    data[:, 0] = ions[0:electrons.shape[0], 0]
    data[:, 1] = ions[0:electrons.shape[0], 1]
    data[:, 2] = electrons[:]

data[:, 3] = np.abs(data[:, 2]) / (np.abs(data[:, 1]) - np.abs(data[:, 2]))

# cutoff data in the beginning and end
data = data[np.where((data[:, 0] >= co_start) & (data[:, 0] <= co_end)), :][0]

# find extrema
extrema_ind = scipy.signal.argrelmax(data[:, 3], order=300)

if len(extrema_ind[0]) < m_cycles:
    print('Not enough extrema found. Trying minima.')
    extrema_ind = scipy.signal.argrelmin(data[:, 3], order=300)
    print(extrema_ind)
    minima_used = True

    if len(extrema_ind[0]) < m_cycles:
        print('Not enough minima either. Is m_cycles set correctly?')
        quit()

extrema = np.zeros((len(extrema_ind[0]), 2), np.float64)
extrema[:, 1] = data[extrema_ind, 3]
extrema[:, 0] = data[extrema_ind, 0]

# use the <number of measurement cycles> highest/lowest peaks
extrema = extrema[np.argsort(extrema[:, 1])]
if minima_used is not True:
    extrema = extrema[-m_cycles:, :]
else:
    extrema = extrema[:m_cycles, :]

# resort by time
extrema = extrema[np.argsort(extrema[:, 0])]

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
fit_results = np.zeros((m_cycles, 6), dtype=np.float64)
boundary_met = [None]*m_cycles # to note, whether a boundary was met by a fit parameter

# boundaries for fits
bounds_lower = [0.0, min(data[:, 0]), -1e6, 0, -1e6, 0]
bounds_upper = [0.3, max(data[:, 0]), 1e6, 900, 1e6, 900]

bounds_range = np.array(bounds_upper) - np.array(bounds_lower)

# loop through all between-extrema-intervals
for i in range(0, m_cycles):
    # select proper data interval
    try:
        fitdata = data[np.where((data[:, 0] >= extrema[i, 0] + peak_dist_after) & (data[:, 0] <= extrema[i + 1, 0] - peak_dist_before)), :][0]
    except IndexError:
        # last one
        fitdata = data[np.where(data[:, 0] >= extrema[i, 0] + peak_dist_after), :][0]

    # starting values for fit
    p[0] = 0.08 # gamma
    p[1] = extrema[i, 0]# time offset
    p[2] = 0.05
    p[3] = 8
    p[4] = 0.01
    p[5] = 100

    # fit the data
    try:
        res = optimize.least_squares(errfunc, np.asarray(p, dtype=np.float64), args=(fitdata[:, 0], fitdata[:, 3]), bounds=(bounds_lower, bounds_upper))
        p1 = res.x

        if res.optimality > min_optimality_req:
            print('Optimality larger than {} for cycle {}. Retrying with new parameters.'.format(min_optimality_req, i+1))
            # fit did not converge well. could be a molecule-potential-like case. try new starting parameters
            p[0] = fit_results[i-1, 0]  # gamma
            p[1] = extrema[i, 0]  # time offset
            p[2] = 0.05
            p[3] = 8
            p[4] = -0.01
            p[5] = 100

            res = optimize.least_squares(errfunc, np.asarray(p, dtype=np.float64), args=(fitdata[:, 0], fitdata[:, 3]), bounds=(bounds_lower, bounds_upper))
            p1 = res.x
            print('New optimality: {}'.format(res.optimality))

    except TypeError:
        print('Could not fit section {}.'.format(i+1))
        p1 = [None] * 6

    fit_results[i, :] = p1

    # try to detect, whether one of the parameters came close to a boundary.
    # bounds_range_percentage is 0 for value at bounds_lower, 1 for value at bounds_upper
    bounds_range_percentage=  np.divide(np.abs(bounds_lower + p1), bounds_range)
    # note if any of the values is close (1e-4) to a boundary
    boundary_met[i] = np.any(np.piecewise(bounds_range_percentage, [1 - bounds_range_percentage < 1e-4, bounds_range_percentage < 1e-4], [1, 1]))

    if i == fittoplot - 1:
        fitplot, = ax0.plot(fitdata[:, 0], exp_decay_2(p1, fitdata[:, 0]), 'r-')
        print('Fit results for {}'.format(fittoplot))
        print(fit_results[i, :])

gammaplot, = ax0.plot(data[:, 0], data[:, 3])
maximaplot, = ax0.plot(extrema[:, 0], extrema[:, 1], 'x')
ax0.set_title('$\gamma$ from raw data')
ax0.set_xlabel('time (s)')
ax0.set_ylabel('$\gamma$ (-)')
ax0.legend([gammaplot, maximaplot, fitplot], ['$\gamma$', 'peak extrema', 'fit to section {}'.format(fittoplot)])

electronplot, = ax1.plot(data[:, 0], data[:, 2])
ax1.set_title('electron current')
ax1.set_xlabel('time (s)')
ax1.set_ylabel('current (nA)')

ionplot, = ax2.plot(data[:, 0], data[:, 1])
ionelectronplot, = ax2.plot(data[:, 0], data[:, 1] + data[:, 2])
ax2.set_xlabel('time (s)')
ax2.set_ylabel('current (nA)')
ax2.legend([ionplot, ionelectronplot], ['ion current', 'sum of ion and electron current'])

for i in range(0, m_cycles):
    if boundary_met[i]:
        ax3.plot(i+1, fit_results[i, 0], 'r+')
    else:
        ax3.plot(i + 1, fit_results[i, 0], 'b+')

ax3.set_title('$\gamma$ evaluated')
ax3.set_ylabel('$\gamma$ (-)')
ax3.set_xlabel('measurement position')

fig1.tight_layout()
fig1.suptitle('data from: {}'.format(fn_electrons))

# export fit shit to a file
export_values = np.concatenate(
    (np.arange(1, m_cycles + 1).reshape((m_cycles, 1)), fit_results[:, [0, 3, 5]]), axis=1)
np.savetxt(export_file_results,
           export_values,
           fmt=('%d', '%10.4f', '%10.4f', '%10.4f'),
           delimiter='\t',
header='Measurement\tGamma\tTime constant 1\tTime constant 2')

#export gamma to a new file
np.savetxt(export_file_gamma,
           data[:, [0, 3]],
           fmt=('%10.1f', '%10.4f'),
           delimiter='\t',
header='Time (s)\tGamma')

plt.show()
