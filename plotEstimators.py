#!/usr/bin/env python3
from argparse import ArgumentParser, ArgumentError
from matplotlib import pyplot as plt
import logging
import sys

from logTools.indiflight_log_importer import IndiflightLog

if __name__=="__main__":
    # script to demonstrate how to use it
    parser = ArgumentParser()
    parser.add_argument("datafile", help="single indiflight bfl logfile")
    parser.add_argument("--range","-r", required=False, nargs=2, type=int, help="integer time range to consider in ms since start of datafile")
    parser.add_argument("-v", required=False, action='count', default=0, help="verbosity (can be given up to 3 times)")
    parser.add_argument("--no-cache", required=False, action='store_true', default=False, help="Do not load from or store to raw data cache")
    parser.add_argument("--clear-cache", required=False, action='store_true', default=False, help="Clear raw data cache")

    # clear cache, even if no other arguments are given
    if "--clear-cache" in sys.argv[1:]:
        IndiflightLog.clearCache()
        if len(sys.argv) == 2:
            exit(0)

    args = parser.parse_args()

    verbosity = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]
    logging.basicConfig(
        format='%(asctime)s -- %(name)s %(levelname)s: %(message)s',
        level=verbosity[min(args.v, 3)],
        )

    # import data
    log = IndiflightLog(args.datafile, args.range, not args.no_cache)
    tMs = log.data['timeMs']

    f, axs = plt.subplots(5, 3, sharex=True)
    axs[0, 0].plot(tMs, log.data[[f'fx_x_rls_x[{motor}]' for motor in range(4)]].to_numpy())
    axs[0, 0].set_ylabel("x")
    axs[1, 0].plot(tMs, log.data[[f'fx_y_rls_x[{motor}]' for motor in range(4)]].to_numpy())
    axs[1, 0].set_ylabel("y")
    axs[2, 0].plot(tMs, log.data[[f'fx_z_rls_x[{motor}]' for motor in range(4)]].to_numpy())
    axs[2, 0].set_ylabel("z")
    for i in range(3):
        axs[i, 0].set_ylim(-2e-6, 2e-6)
    axs[3, 0].plot(tMs, log.data[[f'fx_{axis}_rls_e_var' for axis in ['x', 'y', 'z']]].to_numpy())
    axs[4, 0].plot(tMs, log.data[[f'fx_{axis}_rls_lambda' for axis in ['x', 'y', 'z']]].to_numpy())

    axs[0, 1].plot(tMs, log.data[[f'fx_p_rls_x[{motor}]' for motor in range(4)]].to_numpy())
    axs[0, 1].set_ylabel("p")
    axs[1, 1].plot(tMs, log.data[[f'fx_q_rls_x[{motor}]' for motor in range(4)]].to_numpy())
    axs[1, 1].set_ylabel("q")
    axs[2, 1].plot(tMs, log.data[[f'fx_r_rls_x[{motor}]' for motor in range(4)]].to_numpy())
    axs[2, 1].set_ylabel("r")
    for i in range(2):
        axs[i, 1].set_ylim(-1e-4, 1e-4)
    axs[2, 1].set_ylim(-1e-5, 1e-5)
    axs[3, 1].plot(tMs, log.data[[f'fx_{axis}_rls_e_var' for axis in ['p', 'q', 'r']]].to_numpy())
    axs[4, 1].plot(tMs, log.data[[f'fx_{axis}_rls_lambda' for axis in ['p', 'q', 'r']]].to_numpy())

    axs[0, 2].plot(tMs, log.data[[f'fx_p_rls_x[{motor+4}]' for motor in range(4)]].to_numpy())
    axs[1, 2].plot(tMs, log.data[[f'fx_q_rls_x[{motor+4}]' for motor in range(4)]].to_numpy())
    axs[2, 2].plot(tMs, log.data[[f'fx_r_rls_x[{motor+4}]' for motor in range(4)]].to_numpy())
    for i in range(3):
        axs[i, 2].set_ylim(-1e-3, 1e-3)
    axs[3, 2].plot(tMs, log.data[[f'fx_{axis}_rls_e_var' for axis in ['p', 'q', 'r']]].to_numpy())
    axs[4, 2].plot(tMs, log.data[[f'fx_{axis}_rls_lambda' for axis in ['p', 'q', 'r']]].to_numpy())
    axs[0, 2].legend(["1","2","3","4"])

    for i in range(3):
        axs[4, i].set_ylim(0.9, 1.)

    f.show()

