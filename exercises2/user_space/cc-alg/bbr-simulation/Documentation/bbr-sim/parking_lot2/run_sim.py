import os
import shlex
import subprocess

import numpy as np

DEFAULT_PROBE_GAIN = 1.25
DEFAULT_DRAIN_GAIN = 0.9
DEFAULT_INIT_RATIO = 10

OUT_ROOT = "runs/auto"


def run(probe_gain, drain_gain, init_ratio, out_dir):
    out_file = f"probe_gain={probe_gain}:drain_gain={drain_gain}:init_ratio={init_ratio}.txt"
    out_path = os.path.join(out_dir, out_file)

    cmd = f"./parking_lot_sim --probe-gain {probe_gain} --drain-gain {drain_gain} --init-ratio {init_ratio}"
    cmd_list = shlex.split(cmd)

    with open(out_path, "w") as f:
        subprocess.Popen(cmd_list, stdout=f).wait()


def main():
    out_dir = os.path.join(OUT_ROOT, "sweep=probe_gain")
    os.makedirs(out_dir, exist_ok=True)
    for probe_gain in np.arange(1.1, 2.1, 0.1):
        probe_gain = round(probe_gain, 2)
        run(probe_gain, DEFAULT_DRAIN_GAIN, DEFAULT_INIT_RATIO, out_dir)

    out_dir = os.path.join(OUT_ROOT, "sweep=drain_gain")
    os.makedirs(out_dir, exist_ok=True)
    for drain_gain in np.arange(0.1, 1, 0.1):
        drain_gain = round(drain_gain, 2)
        run(DEFAULT_PROBE_GAIN, drain_gain, DEFAULT_INIT_RATIO, out_dir)

    out_dir = os.path.join(OUT_ROOT, "sweep=init_ratio")
    os.makedirs(out_dir, exist_ok=True)
    for init_ratio in np.arange(1, 11, 1):
        run(DEFAULT_PROBE_GAIN, DEFAULT_DRAIN_GAIN, init_ratio, out_dir)

    for drain_gain in [0.1, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1]:
        out_dir = os.path.join(OUT_ROOT, f"sweep=init_ratio:drain_gain={drain_gain}")
        os.makedirs(out_dir, exist_ok=True)
        for init_ratio in np.arange(1, 11, 1):
            run(DEFAULT_PROBE_GAIN, drain_gain, init_ratio, out_dir)

if __name__ == "__main__":
    main()