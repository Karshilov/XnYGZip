#!/usr/bin/env python3
"""Summarizes verify.csv and bench.csv of a run_all.sh output directory (standard library only).

usage: summarize.py <dir>      prints the tables and writes <dir>/summary.txt
  1. compression ratio per dataset and bound (default search budget), the strict-bound check of every cell, and
     the comparison with the paper's compression ratios
  2. compression / decompression throughput per dataset and bound (default and fast)
  3. compression throughput of the three search budgets, geometric mean over the bounds (paper Table II), and the
     largest size difference between the default and the exhaustive search
"""
import csv, math, os, sys
from collections import defaultdict

ORDER = ['usgs2', 'hacc', 'warpx', 'usgs', 'vesicles', 'lammps']
EXPECTED = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'expected', 'paper_xnygzip.csv')


def read(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def gmean(v):
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float('nan')


def table(title, rows, cols, fmt, lines):
    lines.append('')
    lines.append(title)
    lines.append('%-10s' % 'dataset' + ''.join('%11s' % c for c in cols))
    for name, vals in rows:
        lines.append('%-10s' % name + ''.join(('%11s' % (fmt % vals[c])) if c in vals else '%11s' % '-' for c in cols))


def main(d):
    ver, ben = read(os.path.join(d, 'verify.csv')), read(os.path.join(d, 'bench.csv'))
    out = []
    sets = [x for x in ORDER if any(r['dataset'] == x for r in ver + ben)]
    sets += sorted({r['dataset'] for r in ver + ben} - set(sets))
    bounds = sorted({r['R'] for r in ver + ben}, key=float)
    if ver:
        bad = [r for r in ver if r['pass'] != 'True']
        out.append('strict error bound: %d of %d cells pass' % (len(ver) - len(bad), len(ver)))
        for r in bad: out.append('  FAIL %s R=%s %s violations=%s mismatches=%s' % (r['dataset'], r['R'], r['search'], r['violations'], r['mismatches']))
        cr = {(r['dataset'], r['R']): float(r['ratio']) for r in ver if r['search'] == 'default'}
        table('compression ratio (default)', [(s, {R: cr[(s, R)] for R in bounds if (s, R) in cr}) for s in sets], bounds, '%.2f', out)
        # compression ratios do not depend on the GPU: compare with the paper's values
        exp = {(r['dataset'], float(r['R']), r['search']): float(r['ratio']) for r in read(EXPECTED) if r['platform'] == 'RTX PRO 6000'}
        cmp = [(r, exp[(r['dataset'], float(r['R']), r['search'])]) for r in ver if (r['dataset'], float(r['R']), r['search']) in exp]
        if cmp:
            dev, worst = max(((abs(float(r['ratio']) / e - 1), r) for r, e in cmp), key=lambda t: t[0])
            out.append('')
            out.append('compression ratio vs paper: %d cells compared, largest relative difference %.2g (%s R=%s %s)'
                       % (len(cmp), dev, worst['dataset'], worst['R'], worst['search']))
    if ben:
        for budget in ('default', 'fast'):
            for col, lab in (('comp_GBps', 'compression'), ('decomp_GBps', 'decompression')):
                v = {(r['dataset'], r['R']): float(r[col]) for r in ben if r['search'] == budget}
                if v: table('%s throughput, GB/s (%s)' % (lab, budget), [(s, {R: v[(s, R)] for R in bounds if (s, R) in v}) for s in sets], bounds, '%.1f', out)
        g = defaultdict(list)
        size = {}
        for r in ben:
            g[(r['dataset'], r['search'])].append(float(r['comp_GBps']))
            size[(r['dataset'], r['R'], r['search'])] = int(r['file_bytes'])
        budgets = [b for b in ('fast', 'default', 'exhaustive') if any(k[1] == b for k in g)]
        rows = []
        for s in sets:
            vals = {b: gmean(g[(s, b)]) for b in budgets if g[(s, b)]}
            diffs = [abs(size[(s, R, 'default')] / size[(s, R, 'exhaustive')] - 1) * 100
                     for R in bounds if (s, R, 'default') in size and (s, R, 'exhaustive') in size]
            if diffs: vals['diff %'] = max(diffs)
            rows.append((s, vals))
        table('compression throughput by search budget, GB/s, geometric mean over bounds (Table II); '
              'diff %: largest size difference, default vs exhaustive', rows, budgets + ['diff %'], '%.2f', out)
    text = '\n'.join(out)
    print(text)
    open(os.path.join(d, 'summary.txt'), 'w').write(text + '\n')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results')
