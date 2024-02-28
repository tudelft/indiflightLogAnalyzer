#!/usr/bin/env python3

from logTools import IndiflightLog, imuOffsetCorrection
from glob import glob
from os import path
import pandas as pd
import numpy as np

dataFolder = "/mnt/data/WorkData/BlackboxLogs/2024-02-27/ExperimentsForReal";

files = glob(path.join(dataFolder, "*.BFL"));

parameters = []
initialConditions = []

#%% go through logs and get G1 and G2 parameters
logs = []
pars = []
for runIdx, file in enumerate(files):
    log = IndiflightLog(file)
    logs.append(log)
    lastRow = log.data.iloc[-1] 

    parameterRow = {'run': runIdx}

    # G1/G2
    rowNames = ['x', 'y', 'z', 'p', 'q', 'r']

    for i, r in enumerate(rowNames):
        for col in range(4):
            parameterRow[f'G1_{r}_{col}'] = lastRow[f'fx_{r}_rls_x[{col}]']

        if i >= 3:
            for col in range(4,8):
                parameterRow[f'G2_{r}_{col-4}'] = lastRow[f'fx_{r}_rls_x[{col}]']

    # motors
    for motor in range(4):
        a, b, w0, tau = lastRow[[f'motor_{motor}_rls_x[{i}]' for i in range(4)]]
        a = a if a > 0 else 0xFFFF + a # integer overflow in the data... not nice
        wm = a+b
        lam = a / wm
        parameterRow[f'motor_{motor}_wm'] = wm
        parameterRow[f'motor_{motor}_k'] = lam
        parameterRow[f'motor_{motor}_w0'] = w0
        parameterRow[f'motor_{motor}_tau'] = tau

    # initial rotation rate, after 1400 ms
    initCondIdx = (log.data['timeMs'] - 1400).abs().idxmin()
    parameterRow['p0'] = log.data['gyroADC[0]'].loc[initCondIdx] # in rad/s
    parameterRow['q0'] = log.data['gyroADC[1]'].loc[initCondIdx]
    parameterRow['r0'] = log.data['gyroADC[2]'].loc[initCondIdx]

    # get minimum altitude from this point until the manual takeover
    takeoverIdx = log.flags[log.flags['disable'].apply(lambda x: 9 in x)].index[0]
    parameterRow['minH'] = -log.data['extPos[2]'].loc[initCondIdx:takeoverIdx].max()

    pars.append(parameterRow.copy())

#%% get mean and std as latex table

df = pd.DataFrame(pars)
df.set_index('run', inplace=True)

mean = df.mean()
std = df.std()

# from genGMc.py
G1_fromData = np.array([
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00],
       [-5.11538462e-07, -5.11538462e-07, -5.11538462e-07, -5.11538462e-07],
       [-2.34408005e-05, -2.34408005e-05,  2.34408005e-05, 2.34408005e-05],
       [-1.56027856e-05,  1.56027856e-05, -1.56027856e-05, 1.56027856e-05],
       [-3.00214289e-06,  3.00214289e-06,  3.00214289e-06, -3.00214289e-06]])

G2_fromData = np.array([
       [ 0.        , -0.        , -0.        ,  0.        ],
       [ 0.        , -0.        , -0.        ,  0.        ],
       [-0.00101221,  0.00101221,  0.00101221, -0.00101221]])

omegaMax = 4113.063728303113 # from prop bench test
k = 0.46 # from prop bench test
omega0 = 449.725817910626 # from prop bench test
tau = 0.02 # from prop bench test

G1df = pd.DataFrame([], index=rowNames)
G2df = pd.DataFrame([], index=rowNames[3:])
motordf = pd.DataFrame([], index=['$\omega_{max}$', '$k$', '$\omega_{k}$', '$\\tau$ [ms]'])
motordf['Benchtest'] = [omegaMax, k, omega0, tau]

for motor in range(4):
    G1df[f'True motor {motor}'] = G1_fromData[:, motor]
    G1df[f'Mean motor {motor}'] = mean.filter(regex=f'^G1_[xyzpqr]_{motor}').to_list()
    G1df[f'Std motor {motor}'] = std.filter(regex=f'^G1_[xyzpqr]_{motor}').to_list()

    G2df[f'True motor {motor}'] = G2_fromData[:, motor]
    G2df[f'Mean motor {motor}'] = mean.filter(regex=f'^G2_[pqr]_{motor}').to_list()
    G2df[f'Std motor {motor}'] = std.filter(regex=f'^G2_[pqr]_{motor}').to_list()

    motordf[f'Mean motor {motor}'] = mean.filter(regex=f'^motor_{motor}').to_list()
    motordf[f'Std motor {motor}'] = std.filter(regex=f'^motor_{motor}').to_list()


print((1e6                            *G1df).to_latex(float_format="%.3f"))
print((1e3                            *G2df).to_latex(float_format="%.3f"))
print((np.array([[1.,1.,1.,1000.]]).T *motordf).to_latex(float_format="%.3f"))

#%% Initial condition scatter

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
plt.close('all')

plt.rcParams.update({
    "text.usetex": True,
#    "font.family": "Helvetica",
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.grid": True,
    "axes.grid.which": 'both',
    "grid.linestyle": '--',
    "grid.alpha": 0.7,
    "axes.labelsize": 10,
    "axes.titlesize": 16,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.loc": 'upper right',
    "legend.fontsize": 9,
    "legend.columnspacing": 2.0,
    'figure.subplot.bottom': 0.17,
    'figure.subplot.left': 0.07,
    'figure.subplot.right': 0.96,
    'figure.subplot.top': 0.95,
    'figure.subplot.hspace': 0.3,
    'figure.subplot.wspace': 0.4,
    'figure.titlesize': 'large',
    'lines.linewidth': 1,
})

from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler('linestyle', ['-', '--', ':', '-.'])

p0, q0, r0 = df[['p0', 'q0', 'r0']].to_numpy(dtype=float).T * 180./np.pi
minH = df['minH']
fig = plt.figure(figsize=(6.7, 5))
ax = fig.add_subplot(111, projection='3d')
#ax.stem(p0, q0, r0, 2, 2, r0+800, color='black', shade=False)
_, stemlines,_ = ax.stem(p0, q0, r0, basefmt=' ', linefmt='grey', markerfmt=' ', bottom=-700)
stemlines.set_linestyle('--')
scatter = ax.scatter(p0, q0, r0, c=minH, cmap='viridis', alpha=1.)
colorbar = fig.colorbar(scatter, pad=0.1)
colorbar.set_label('Minimum Altitude during Recovery [m]')
ax.set_xlabel("Roll [deg/s]")
ax.set_ylabel("Pitch [deg/s]")
ax.set_zlabel("Yaw [deg/s]")
ax.set_zlim(bottom=-700)
ax.view_init(elev=17, azim=-137)
#plt.title("Initial rotation before excitation")

fig.savefig('InitialRotation.pdf', format='pdf')
#fig.show()


#%% Time plots excitation

log = IndiflightLog("/mnt/data/WorkData/BlackboxLogs/2024-02-27/Experiments500HzLoggingAndThrows/LOG00228.BFL")
timeMs = log.data['timeMs'] - 1435
boolarr = (timeMs > 0) & (timeMs < 457)
timeMs = timeMs[boolarr]
crop = log.data[boolarr]

f, axs = plt.subplots(1, 3, figsize=(9, 2.7), sharex='all')
for i in range(4):
    axs[0].plot(timeMs,
                crop[f'motor[{i}]'],
                label=f'Motor {i}')
    axs[0].set_xlabel("Time [ms]")
    axs[0].set_ylabel("Motor input $\delta$ [-]")

    axs[1].plot(timeMs,
                crop[f'omegaUnfiltered[{i}]'],
                label=f'Motor {i}')
    axs[1].set_xlabel("Time [ms]")
    axs[1].set_ylabel("Motor Rotation Rate $\Omega$ [rad/s]")


axs[2].plot(timeMs,
            180./np.pi * crop[[f'gyroADCafterRpm[{i}]' for i in range(3)]])
axs[2].set_xlabel("Time [ms]")
axs[2].set_ylabel("Body Rotation Rate $\omega$ [deg/s]")
axs[2].legend(["Roll", "Pitch", "Yaw"], loc='upper left')

axs[0].set_ylim(top=1.)
axs[1].set_ylim(top=4000.)
axs[0].legend( ncol=2 )
axs[1].legend( ncol=2 )

f.savefig('Excitation.pdf', format='pdf')

#%% regressors

plt.rcParams.update({
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.loc": 'upper right',
    "legend.fontsize": 9,
    "legend.columnspacing": 2.0,
    'figure.subplot.bottom': 0.18,
    'figure.subplot.left': 0.1,
    'figure.subplot.right': 0.95,
    'figure.subplot.top': 0.85,
    'figure.subplot.hspace': 0.3,
    'figure.subplot.wspace': 0.35,
    'figure.titlesize': 'large',
    'lines.linewidth': 1,
    "axes.formatter.limits": [-2, 3]
})

timeMs = log.data['timeMs'] - 1435
boolarr = (timeMs > 0) & (timeMs < 500)
timeMs = timeMs[boolarr]
crop = log.data[boolarr]

from logTools import Signal
order = 2
fc = 20 # Hz
r = np.array([-0.01, -0.01, 0.015])

omegaRaw = Signal(crop['timeS'], crop[[f'omegaUnfiltered[{i}]' for i in range(4)]])
gyroRaw = Signal(crop['timeS'], crop[[f'gyroADCafterRpm[{i}]' for i in range(3)]])
spfRaw = Signal(crop['timeS'], crop[[f'accUnfiltered[{i}]' for i in range(3)]])
spfRawCor = Signal(crop['timeS'], imuOffsetCorrection(spfRaw.y.copy(), gyroRaw.y, gyroRaw.dot().y, r))

omegaFilt = omegaRaw.filter('lowpass', order, fc)
gyroFilt = gyroRaw.filter('lowpass', order, fc)
spfFiltCor = spfRawCor.filter('lowpass', order, fc)

regSpf = 2. * omegaFilt.y * omegaFilt.diff().y
regRot = np.concatenate((regSpf, omegaFilt.dot().diff().y), axis=1)

for axi, ax in enumerate(['x', 'y', 'z']):
    frls, axs = plt.subplots(1, 2, figsize=(9, 3), sharex=True)

    theta = crop[[f'fx_{ax}_rls_x[{i}]' for i in range(4)]].to_numpy()
    reproduction = np.array([t.T @ r for t, r in zip(theta, regSpf)])

    axs[0].plot(timeMs, spfRawCor.diff().y[:, axi], alpha=0.5, lw=0.5, ls='--', label=f"Unfiltered")
    axs[0].plot(timeMs, spfFiltCor.diff().y[:, axi], lw=1.0, ls='-', label=f"synchro-filtered")
    axs[0].plot(timeMs, reproduction, lw=1.5, ls='-.', label=f"Online reproduction")
    axs[0].set_xlabel("Time [ms]")
    axs[0].set_ylabel("Specific Force Delta [N/kg]")
    axs[0].set_ylim(bottom=-7, top=7)
    axs[0].legend(loc="lower left")

    axs[1].plot(timeMs, theta)
    axs[1].set_ylim(bottom=-0.3e-5, top=0.3e-5)
    axs[1].set_xlabel("Time [ms]")
    axs[1].set_ylabel("Motor effectiveness estimates [$\\frac{N/kg}{(rad/s)^2}$]")
    axs[1].legend([f'Motor {i}' for i in range(4)])

    frls.suptitle(f"Online Estimation for {ax}-Axis Force Effectiveness")

    frls.savefig(f"Fx_estimation_{ax}.pdf", format='pdf')

axis_names = ['Roll', 'Pitch', 'Yaw']
for axi, ax in enumerate(['p', 'q', 'r']):
    frls, axs = plt.subplots(1, 3, figsize=(9, 3), sharex=True)

    theta = crop[[f'fx_{ax}_rls_x[{i}]' for i in range(8)]].to_numpy()
    reproduction = np.array([t.T @ r for t, r in zip(theta, regRot)])

    axs[0].plot(timeMs, gyroRaw.dot().diff().y[:, axi], alpha=0.5, lw=0.5, ls='--', label=f"Unfiltered")
    axs[0].plot(timeMs, gyroFilt.dot().diff().y[:, axi], lw=1.0, ls='-', label=f"synchro-filtered")
    axs[0].plot(timeMs, reproduction, lw=1.5, ls='-.', label=f"Online reproduction")
    axs[0].set_xlabel("Time [ms]")
    axs[0].set_ylabel("Rotation Acceleration Delta [$rad/s^2$]")
    axs[0].set_ylim(bottom=-30, top=30)
    axs[0].legend(loc="lower left")

    axs[1].plot(timeMs, theta[:, :4])
    axs[1].set_ylim(bottom=-2e-4, top=2e-4)
    axs[1].set_xlabel("Time [ms]")
    axs[1].set_ylabel("Effectiveness [$\\frac{Nm/(kg\cdot m^2)}{(rad/s)^2}$]")
    axs[1].legend([f'Motor {i}' for i in range(4)], loc='upper left', ncols=2)

    axs[2].plot(timeMs, theta[:, 4:])
    axs[2].set_ylim(bottom=-6e-3, top=6e-3)
    axs[2].set_xlabel("Time [ms]")
    axs[2].set_ylabel("Acceleration effectiveness [$\\frac{Nm/(kg\cdot m^2)}{(rad/s^2)}$]")
    axs[2].legend([f'Motor {i}' for i in range(4)], loc='upper right', ncols = 2)

    frls.suptitle(f"Online Estimation for {axis_names[axi]} Effectiveness")

    frls.savefig(f"Fx_estimation_{ax}.pdf", format='pdf')

# %% Plot trajectories

ft = plt.figure(figsize=(6.7, 5))
ax = ft.add_subplot(111, projection='3d')

throwFiles = ["/mnt/data/WorkData/BlackboxLogs/2024-02-27/Experiments500HzLoggingAndThrows/LOG00230.BFL",
    "/mnt/data/WorkData/BlackboxLogs/2024-02-27/Experiments500HzLoggingAndThrows/LOG00231.BFL",
    "/mnt/data/WorkData/BlackboxLogs/2024-02-27/Experiments500HzLoggingAndThrows/LOG00232.BFL",
    ]
times = [7629, 5377, 5063]
throwLogs = []
for file, time in zip(throwFiles, times):
    log = IndiflightLog(file)
    #startIdx = log.flags[log.flags['enable'].apply(lambda x: 0 in x)].index[0]
    startTime = time + 1700
    duration = 5000

    timeMs = log.data['timeMs'] - startTime
    boolarr = (timeMs > 0) & (timeMs < duration)
    timeMs = timeMs[boolarr]
    throwLogs.append(log.data[boolarr])

#startTime = 1000
#endTime = 5000

#for i, log in enumerate(logs[8:]):
    #if i == 2:
    #    continue
    #timeMs = log.data['timeMs'] - 1000
    #boolarr = (timeMs > 0) & (timeMs < 4000)
    #timeMs = timeMs[boolarr]
    #crop = log.data[boolarr]

for i, log in enumerate(throwLogs):
    x,y,z = log[[f'extPos[{i}]' for i in range(3)]].to_numpy().T
    ax.plot(y,x,-z, linestyle="-")
    ax.plot(y,x,0, linestyle="--")
    for k in range(200, len(x)-100, 100):
        ax.quiver(y[k], x[k], -z[k], y[k+100]-y[k], x[k+100]-x[k], -z[k+100]+z[k], lw=1.0, arrow_length_ratio=20, length=0.01)
        #q = ax.quiver(y[k], x[k], 0, y[k+100]-y[k], x[k+100]-x[k], 0)

ax.set_xlabel("East [m]")
ax.set_ylabel("North [m]")
ax.set_zlabel("Up [m]")
ax.view_init(elev=16, azim=-14)
ft.savefig('Trajectories.pdf', format='pdf')


