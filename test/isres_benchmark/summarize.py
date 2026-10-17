"""Summarize the ISRES benchmark: success rate and evaluations per variant and problem.

Success = result 2 (NLOPT_STOPVAL_REACHED). Evaluations are compared with a
two-sided Mann-Whitney U test against master (M), over successful runs, and
success counts with Fisher's exact test.
"""
import csv
import glob
import sys

import numpy as np
from scipy import stats

rows = []
for path in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else 'results-*.csv')):
    with open(path) as f:
        rows += [dict(zip(['variant', 'problem', 'seed', 'result', 'evals', 'f'], r)) for r in csv.reader(f)]

variants = ['M', 'R', 'B', 'RB']
problems = list(dict.fromkeys(r['problem'] for r in rows))
print(f"{'problem':11s} {'var':3s} {'success':>8s} {'median':>8s} {'mean':>8s} {'p_evals':>8s} {'p_succ':>8s}")
for p in problems:
    base = [r for r in rows if r['problem'] == p and r['variant'] == 'M']
    base_ok = [int(r['evals']) for r in base if r['result'] == '2']
    for v in variants:
        sel = [r for r in rows if r['problem'] == p and r['variant'] == v]
        ok = [int(r['evals']) for r in sel if r['result'] == '2']
        med = np.median(ok) if ok else float('nan')
        mean = np.mean(ok) if ok else float('nan')
        if v != 'M' and ok and base_ok:
            p_evals = stats.mannwhitneyu(ok, base_ok, alternative='two-sided').pvalue
        else:
            p_evals = float('nan')
        if v != 'M':
            table = [[len(ok), len(sel) - len(ok)], [len(base_ok), len(base) - len(base_ok)]]
            p_succ = stats.fisher_exact(table).pvalue
        else:
            p_succ = float('nan')
        print(f"{p:11s} {v:3s} {len(ok):4d}/{len(sel):<3d} {med:8.0f} {mean:8.0f} {p_evals:8.2g} {p_succ:8.2g}")
