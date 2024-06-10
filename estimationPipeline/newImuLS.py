#!/usr/bin/env python3
from argparse import ArgumentParser
import logging
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from logTools import IndiflightLog, RLS, Signal

if __name__=="__main__":
    logging.basicConfig(
        format='%(asctime)s -- %(name)s %(levelname)s: %(message)s',
        level=logging.INFO,
        )

    parser = ArgumentParser()
    parser.add_argument("datafile", help="single indiflight bfl logfile")
    parser.add_argument("--range","-r", required=False, nargs=2, action='append', type=float, metavar=('START', 'END'), help="time range to consider in ms since start of datafile")
    args = parser.parse_args()

    ranges = [tuple(r) for r in args.range] if args.range else []

    # fitting parameters
    fc = 50. # Hz. tau = 1/(2*pi*fc) if first order
    order = 2 # 1 --> simple first order. 2 and up --> butterworth

    gyroFilt = np.empty((0, 3))
    dgyroFilt = np.empty((0, 3))
    accFilt = np.empty((0, 3))
    logFull = None

    for r in ranges:
        if logFull is None:
            logFull = log = IndiflightLog(args.datafile, r)
            log.resetTime()
        else:
            log = IndiflightLog(args.datafile, r)
            log.resetTime()
            log.data['timeS'] += logFull.data['timeS'].iloc[-1] + 0.002
            log.data['timeMs'] += logFull.data['timeMs'].iloc[-1] + 2
            log.data['timeUs'] += logFull.data['timeUs'].iloc[-1] + 2000
            logFull.data = pd.concat( (logFull.data, log.data), ignore_index=True )

        gyro = Signal(log.data["timeS"], log.data[[f"gyroADCafterRpm[{i}]" for i in range(3)]] )
        acc  = Signal(log.data["timeS"], log.data[[f"accADCafterRpm[{i}]" for i in range(3)]] )

        gyroFilt = np.concatenate( (gyroFilt, gyro.filtfilt('lowpass', order, fc).y))
        dgyroFilt = np.concatenate( (dgyroFilt, gyro.filtfilt('lowpass', order, fc).dot().y))
        accFilt = np.concatenate( (accFilt, acc.filtfilt('lowpass', order, fc).y))

    logFull.resetTime()

    A = np.empty((gyroFilt.shape[0], 3, 3))
    for i, (w, dw, a) in enumerate(zip(gyroFilt, dgyroFilt, accFilt)):
        wx, wy, wz = w
        dwx, dwy, dwz = dw
        A[i] = -np.array([
            [-(wy*wy + wz*wz),    wx*wy - dwz   ,    wx*wz + dwy   ],
            [  wx*wy + dwz   ,  -(wx*wx + wz*wz),    wy*wz - dwx   ],
            [  wx*wz - dwy   ,    wy*wz + dwx   ,  -(wx*wx + wy*wy)],
            ])

    # stack regressors and observations
    Arows = A.reshape(-1, 3)
    yrows = accFilt.reshape(-1)

    # solve
    x, residuals, rank, s = np.linalg.lstsq(Arows, yrows, rcond=None)

    # statitics
    # https://learnche.org/pid/least-squares-modelling/multiple-linear-regression
    print()
    print(f"x [mm]: {(x*1e3).round(2)}")
    print(f"2norm(b - Ax) [m/s/s]: {residuals[0]:.3f}")
    N = len(logFull.data)
    SE = residuals[0] / np.sqrt(N - 3)
    print(f"SE [m/s/s]: {SE:.4f}")
    print(f"95% confidence ellipsoid semi-major axes [mm]: {2. * SE**2 / s * 1e3}")

    U, s, Vt = np.linalg.svd(Arows, full_matrices=False)
    scaling_factor = np.sqrt(N / (N - 3)) * np.sqrt(1 - 1 / (N * 0.95))
    semi_axes_lengths = scaling_factor * (1 / s)


    f, axs = plt.subplots(3, 1, sharex=True)
    timeMs = logFull.data['timeMs']

    axs[0].plot( timeMs, logFull.data[[f"gyroADCafterRpm[{i}]" for i in range(3)]])
    axs[0].plot( timeMs, gyroFilt )

    axs[1].plot( timeMs, dgyroFilt )

    axs[2].plot( timeMs, logFull.data[[f"accADCafterRpm[{i}]" for i in range(3)]])
    axs[2].plot( timeMs, accFilt )




