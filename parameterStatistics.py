
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

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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