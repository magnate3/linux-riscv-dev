import ast
from collections import defaultdict
import os
import re
import pandas as pd
import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot the simulation results")
    parser.add_argument("-i", type=str, required=True, help="Input file or directory")
    return parser.parse_args()


def parse_experiment(fpath):
    assert os.path.exists(fpath)

    records = []
    with open(fpath, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("t="):
                regex = r'^t= ([0-9]*)\s*max_bw: (.*)'
                match = re.match(regex, line)
                if match:
                    record = {}
                    t = match.group(1)
                    record["t"] = int(t)
                    details = match.group(2)
                    regex = r'\s*(f\d*): (\d*\.?\d*)\s*'
                    matches = re.findall(regex, details)
                    for m in matches:
                        record[m[0]] = float(m[1])
                    records.append(record)
    df = pd.DataFrame(records)
    return df


def parse_params(fname: str, ext=""):
    fname = fname.replace(ext, "")
    params = fname.split(":")
    ret = {}
    for param in params:
        key, value = tuple(param.split("="))
        try:
            value = ast.literal_eval(value)
        except ValueError:
            pass
        ret[key] = value
    return ret


@dataclass
class Experiment:
    fpath: str
    df: pd.DataFrame

    def __init__(self, fpath: str):
        self.fpath = fpath
        self.expr_name = os.path.basename(fpath).replace(".txt", "")
        self.expr_dir = os.path.dirname(fpath)
        self.df = parse_experiment(fpath)
        self.params = parse_params(self.expr_name)


def plot_max_bw(exp: Experiment):
    fig, ax = plt.subplots()
    df = exp.df
    for f in df.columns:
        if f.startswith("f"):
            ax.plot(df["t"], df[f], label=f)
    ax.set_xlabel("Time (round trip number)")
    ax.set_ylabel("Bandwidth estimate (Mbps)")
    ax.grid(True)
    ax.legend()
    ax.set_ylim((0, None))
    ax.set_title(exp.expr_name)
    fig.tight_layout()
    fig.savefig(os.path.join(exp.expr_dir, f"{exp.expr_name}.png"))
    plt.close(fig)


def parse_multiple_experiments(fpath):
    experiment_dict = defaultdict(list)
    for root, dirs, files in os.walk(fpath):
        for f in files:
            if f.endswith(".txt"):
                experiment_dict[root].append(os.path.join(root, f))
    return experiment_dict


def plot_multi_experiments(experiment_dict):
    for exp_dir, exp_files in experiment_dict.items():
        for exp_file in exp_files:
            exp = Experiment(exp_file)
            plot_max_bw(exp)

        exp_dir_name = os.path.basename(exp_dir)
        expr_params = parse_params(exp_dir_name)
        if "sweep" in expr_params:
            sweep_key = expr_params["sweep"]
            records = []
            for exp_file in exp_files:
                record = {}
                exp = Experiment(exp_file)
                record.update(exp.params)
                df = exp.df
                for f in df.columns:
                    if f.startswith("f"):
                        record[f] = df[f].iloc[-1]
                records.append(record)
            df = pd.DataFrame(records).sort_values(by=[sweep_key])

            fig, ax = plt.subplots()
            for f in df.columns:
                if f.startswith("f"):
                    ax.plot(df[sweep_key], df[f], label=f)
            ax.set_xlabel(sweep_key)
            ax.set_ylabel("Bandwidth estimate (Mbps)")
            ax.grid(True)
            ax.legend()
            ax.set_ylim((0, None))
            ax.set_title(exp_dir)
            fig.tight_layout()
            fig.savefig(os.path.join(exp_dir, f"{exp_dir_name}.png"), dpi=300)


def main():
    args = parse_args()
    if os.path.isdir(args.i):
        experiment_dict = parse_multiple_experiments(args.i)
        plot_multi_experiments(experiment_dict)
    else:
        exp = Experiment(args.i)
        plot_max_bw(exp)


if __name__ == "__main__":
    main()