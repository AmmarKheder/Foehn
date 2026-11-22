#!/usr/bin/env python
"""Portrait, shortened versions of the Supplementary Tables that were set sideways (no rotated
page, a simpler Supplementary Information).

The complete tables are frozen once in source_data/supp_tables_full/ (the LaTeX as it was typeset, and one JSON file per
table) and go into the Source Data file as sheets (source_data/build_source_data.py reads the JSON files). Every number of
the shortened tables is copied from the frozen LaTeX, never retyped.

    python -B tools/si_tables_portrait.py          # freeze (first run), write the JSON files, replace the tables in work/
    python -B tools/si_tables_portrait.py --dry    # print the new LaTeX only
"""
import json
import os
import re
import sys

W = (os.environ["FOEHN_WORK"] + '/eccv/cran-pm/paper/natcomms')
sys.path.insert(0, W + '/tools')
from safe_edit import backup, edit, edit_span  # noqa: E402

FULL = W + '/source_data/supp_tables_full'
SUP = W + '/work/supplementary.tex'
DRY = '--dry' in sys.argv


def safe_write(path, text):
    data = text.encode('utf-8'); tmp = path + f'.tmp{os.getpid()}'
    with open(tmp, 'wb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    assert open(tmp, 'rb').read() == data, path
    os.replace(tmp, path)
    assert open(path, 'rb').read() == data, path


# ------------------------------------------------------------------------------------------------ frozen complete tables
def frozen():
    fn = FULL + '/tables_as_typeset_20261120.tex'
    if not os.path.exists(fn):
        s = open(SUP, encoding='utf-8').read()
        a = s.index('\\begin{sidewaystable}[p]\n\\centering\n\\caption{\\textbf{Sensitivity of the count')
        b = s.index('\\end{sidewaystable}', s.index('\\label{tab:away}')) + len('\\end{sidewaystable}')
        os.makedirs(FULL, exist_ok=True)
        safe_write(fn, '% The five Supplementary Tables as they were typeset on 2026-11-20 (sideways, complete), before their\n'
                       '% shortened portrait versions. Frozen: tools/si_tables_portrait.py reads the numbers from this file.\n' + s[a:b] + '\n')
        print('frozen:', fn, b - a, 'chars')
    return open(fn, encoding='utf-8').read()


def block(s, label):
    i = s.index('\\label{tab:%s}' % label)
    a = s.rindex('\\begin{sidewaystable}', 0, i); b = s.index('\\end{sidewaystable}', i)
    return s[a:b]


def tab_lines(blk):
    a = blk.index('\\begin{tabular}'); b = blk.index('\\end{tabular}')
    body = blk[a:b]; body = body[body.index('\n') + 1:]
    return [x.strip() for x in body.split('\n') if x.strip()]


def parse(blk):
    """-> header row (cells), [(block head, [rows])]; a table without block heads has one block with head ''"""
    header, blocks = None, []
    for line in tab_lines(blk):
        if line in ('\\hline', '\\toprule', '\\midrule', '\\bottomrule') or line.startswith('\\cmidrule'):
            continue
        assert line.endswith('\\\\'), line[:90]
        line = line[:-2].strip()
        m = re.fullmatch(r'\\multicolumn\{\d+\}\{l\}\{\\textit\{(.*)\}\}', line)
        if m:
            blocks.append((m.group(1), [])); continue
        cells = [c.strip() for c in re.split(r'(?<!\\)&', line)]
        if header is None:
            header = cells; continue
        if not blocks:
            blocks.append(('', []))
        blocks[-1][1].append(cells)
    return header, blocks


NUM = r'(?:\\textminus\{\}|-)?\d[\d{},\.]*'


def num(x):
    x = x.replace('\\textminus{}', '-').replace('{,}', '').replace(',', '')
    return int(x) if re.fullmatch(r'-?\d+', x) else float(x)


def plain(x):
    for a, b in (('{,}', ','), ('\\,', ' '), ('\\textmu g', 'µg'), ('m$^{-3}$', 'm-3'), ('$^\\circ$', '°'), ('\\%', '%'), ('\\model{}', 'Foehn'),
                 ('$\\delta$', 'δ'), ('\\textminus{}', '-'), ('--', '–'), ('\\quad', ''), ('~', ' ')):
        x = x.replace(a, b)
    x = re.sub(r'\\text(?:it|bf)\{([^{}]*)\}', r'\1', x)
    return re.sub(r'\s+', ' ', x).strip()


def val3(c):
    """'v [lo, hi]' -> (v, lo, hi); 'v' -> (v, None, None); '--' or '' -> three empty cells; other text -> (text, None, None)"""
    c = c.strip()
    if c in ('', '--'):
        return [None, None, None]
    m = re.fullmatch(r'(%s) \[(%s), (%s)\]' % (NUM, NUM, NUM), c)
    if m:
        return [num(g) for g in m.groups()]
    if re.fullmatch(NUM, c):
        return [num(c), None, None]
    return [plain(c), None, None]


def v(c):        # the value of a cell 'v [lo, hi]', as typeset
    return c.split(' [')[0]


def iv(c):       # its interval, as typeset
    return '[' + c.split(' [')[1] if ' [' in c else ''


def tri(name):
    return [name, name + ', 95% interval low', name + ', 95% interval high']


def dump(name, label, part, what, desc, cols, rows):
    assert all(len(r) == len(cols) for r in rows), (name, len(cols), [len(r) for r in rows][:4])
    safe_write(FULL + '/' + name + '.json', json.dumps(dict(label=label, part=part, what=what, desc=desc, cols=cols, rows=rows), ensure_ascii=False, indent=1) + '\n')
    print('  %-22s %3d rows x %2d columns' % (name + '.json', len(rows), len(cols)))


# ------------------------------------------------------------------------------------------------ LaTeX helpers
def table(caption, label, body, cont=False):
    return ('\\begin{table}[p]\n' + ('\\ContinuedFloat\n' if cont else '') + '\\centering\n\\caption{' + caption + '}\n'
            + ('\\label{tab:%s}\n' % label if label else '') + body + '\n\\end{table}')


def fit(tabular, size='\\footnotesize'):
    return '\\fitwidth{%\n' + size + '%\n\\setlength{\\tabcolsep}{4.5pt}%\n' + tabular + '}'


LAB = {'CAMS forecast, bias-corrected, with the climatological 1\\,km pattern': '\\quad with the climatological 1\\,km pattern'}


# ------------------------------------------------------------------------------------------------ the five tables
def do_sens(s):
    header, blocks = parse(block(s, 'sens'))
    thr = header[1:]
    cols = ['block of rows', 'quantity (millions of people a day, unless a share)']
    for t in thr:
        cols += tri('threshold %s µg m-3' % t)
    rows = [[plain(h), plain(r[0])] + sum((val3(c) for c in r[1:]), []) for h, rs in blocks for r in rs]
    dump('sens', 'sens', '', 'complete table: every row with its 95% interval',
         'Complete Supplementary Table: sensitivity of the count of people hidden by a 40 km grid in 2022 (millions of people a day; five thresholds in '
         'µg m-3; mean over the 365 days, last block over the 179 target days of the forecasts), with the 95% intervals of a moving-block bootstrap over days.',
         cols, rows)
    B = lambda key: next(rs for h, rs in blocks if key in h)                                           # noqa: E731
    row = lambda lab, cells: lab + ' & ' + ' & '.join(cells) + ' \\\\'                                  # noqa: E731
    head = lambda t: '\\multicolumn{6}{@{}l}{\\textit{' + t + '}} \\\\'                                 # noqa: E731

    def rng(c):
        x = [int(t) for t in c.split(' / ')]
        return str(x[0]) if min(x) == max(x) else '%d--%d' % (min(x), max(x))

    def mon(c):
        m = re.fullmatch(r'([\d\.]+) \(([\d\.]+)\\%\)', c)
        return m.group(1), m.group(2)
    c = B('The count'); L = ['\\begin{tabular}{@{}lrrrrr@{}}', '\\toprule', row(header[0], thr), '\\midrule']
    assert c[0][0].startswith('People above') and c[1][0].startswith('Hidden:') and len(c) == 6, [r[0] for r in c]
    L += [head('The count (40\\,km blocks, the definition of the paper)'),
          row(c[0][0], [v(x) for x in c[0][1:]]), row('\\quad 95\\% interval', [iv(x) for x in c[0][1:]]),
          row(c[1][0], [v(x) for x in c[1][1:]]), row('\\quad 95\\% interval', [iv(x) for x in c[1][1:]]),
          row(c[2][0], [v(x) for x in c[2][1:]]), row(c[3][0], c[3][1:]),
          row(c[4][0], [v(x) for x in c[4][1:]]), row(c[5][0], [v(x) for x in c[5][1:]])]
    c = B('Size of the blocks'); L += ['\\addlinespace', head('Size of the blocks')] + [row(r[0], [v(x) for x in r[1:]]) for r in c]
    c = B('Origin of the'); assert len(c) == 1
    L += ['\\addlinespace', head('Origin of the 40\\,km blocks'), row('Shifted by 10, 20 or 30\\,km in both directions (range)', [rng(x) for x in c[0][1:]])]
    c = B('Margin'); assert len(c) == 6
    L += ['\\addlinespace', head('Margin $\\delta$ on both sides of the threshold')] + [row(r[0], [v(x) for x in r[1:]]) for r in c]
    c = B('Monitoring stations'); assert len(c) == 2
    L += ['\\addlinespace', head('Monitoring stations (8,110 monitors of 2022)')]
    for r in c:
        L += [row(r[0], [mon(x)[0] for x in r[1:]]), row('\\quad share of the hidden people (\\%)', [mon(x)[1] for x in r[1:]])]
    c = B('Pixels with at least 100 residents'); assert len(c) == 3
    L += ['\\addlinespace', head('Grid cells with at least 100 residents (94\\% of residents), 179 target days')] + [row(r[0], [v(x) for x in r[1:]]) for r in c]
    L += ['\\bottomrule', '\\end{tabular}']
    cap = ('\\textbf{Sensitivity of the count of people hidden by a 40\\,km grid.} People a day (millions) in 2022, for five thresholds: mean over '
           'the 365 days, with the 95\\% intervals of the two counts (moving-block bootstrap over days, blocks of 7 days). Margin $\\delta$: the 1\\,km '
           'value is above the threshold plus $\\delta$ and the 40\\,km mean is not above the threshold minus $\\delta$. Last block of rows: mean over '
           'the 179 target days of the forecasts (every second day). The intervals of the other rows are given in the Source Data file.')
    return table(cap, 'sens', fit('\n'.join(L)))


def do_warned(s):
    header, blocks = parse(block(s, 'warned'))
    names = ['rightly warned (millions a day)', 'wrongly warned (millions a day)', 'reach (%)', 'correct (%)', 'hidden people warned (millions a day)',
             'hidden people warned (%)', 'hidden people warned, with a 10% margin (%)',
             'wrongly warned where the mean of the 40 km cell is below the threshold (millions a day)', 'the same, per hidden person warned']
    cols = ['threshold, lead time and people exposed', 'system', 'warned (millions a day)'] + sum((tri(n) for n in names), [])
    rows = [[plain(h), plain(r[0]), num(r[1])] + sum((val3(c) for c in r[2:]), []) for h, rs in blocks for r in rs]
    assert len(blocks) == 6 and all(len(rs) == 9 for h, rs in blocks)
    dump('warned', 'warned', '', 'complete table: 1 and 3 days ahead, every count and its 95% interval',
         'Complete Supplementary Table: people warned a day by each system at the 1 km grid cells with at least 100 residents, for three thresholds '
         '(15, 37.5 and 75 µg m-3) and two lead times (1 and 3 days), with 95% intervals over target days. Reach, share of the exposed people who are '
         'warned; correct, share of the warned people who are exposed.', cols, rows)
    L = ['\\begin{tabular}{@{}lrrrrrr@{}}', '\\toprule',
         ' & Warned & Reach & Correct & \\multicolumn{2}{c}{Hidden people warned} & Wrongly warned, \\\\',
         ' & (millions) & (\\%) & (\\%) & (millions) & (\\%) & 40\\,km mean below \\\\', '\\midrule']
    k = 0
    for h, rs in blocks:
        if ', 1 day ahead' not in h:
            continue
        m = re.fullmatch(r'(Above .*?), 1 day ahead: (.*? million hidden), [\d\.]+ million hidden with a 10\\% margin', h)
        if k:
            L.append('\\addlinespace')
        k += 1
        L.append('\\multicolumn{7}{@{}l}{\\textit{%s: %s}} \\\\' % (m.group(1), m.group(2)))
        for r in rs:
            L.append(' & '.join([LAB.get(r[0], r[0]), r[1], v(r[4]), v(r[5]), v(r[6]), v(r[7]), v(r[9])]) + ' \\\\')
    assert k == 3
    L += ['\\bottomrule', '\\end{tabular}']
    cap = ('\\textbf{Who is warned, 1 day ahead.} People a day (millions) warned by each system at the 8.1 million 1\\,km grid cells with at least '
           '100 residents, for three thresholds (means over target days). Reach, share of the exposed people who are warned; Correct, share of the '
           'warned people who are exposed. Bias-corrected: the CAMS forecast mapped towards the 1\\,km estimates in each 0.25$^\\circ$ cell and '
           'calendar month (fitted on 2017--2020), every resident receiving the cell value or this value distributed by the climatological 1\\,km '
           'pattern of the estimates. Climatology alone: a resident is warned when the climatology of the 1\\,km estimates (2017--2020) in their grid '
           'cell exceeds the threshold. Last column: people wrongly warned (millions) where the mean of the 40\\,km cell is below the threshold. '
           'The 95\\% intervals, the values 3 days ahead and the other counts are given in the Source Data file.')
    return table(cap, 'warned', fit('\n'.join(L)))


def do_matched(s):
    header, blocks = parse(block(s, 'matched'))
    names = ['reach (%)', 'correct (%)', 'hidden people warned (millions a day)', 'Foehn minus reference: reach (points)',
             'Foehn minus reference: correct (points)', 'Foehn minus reference: hidden people warned (millions a day)']
    cols = ['threshold, lead time and Foehn', 'reference', 'factor on the threshold'] + sum((tri(n) for n in names), [])
    rows = [[plain(h), plain(r[0]), num(r[1])] + sum((val3(c) for c in r[2:]), []) for h, rs in blocks for r in rs]
    assert len(blocks) == 6 and all(len(rs) == 4 for h, rs in blocks)
    dump('matched', 'matched', '', 'complete table: 1 and 3 days ahead, every 95% interval',
         'Complete Supplementary Table: the reference forecasts made to warn as many people a day as Foehn over 2022 (each warns a resident when its '
         'value exceeds the factor times the threshold), for three thresholds (15, 37.5 and 75 µg m-3) and two lead times, with 95% intervals over '
         'target days (paired for the differences).', cols, rows)
    L = ['\\begin{tabular}{@{}lrrrrcc@{}}', '\\toprule',
         ' & Factor on & Reach & Correct & Hidden warned & \\multicolumn{2}{c}{\\model{} minus reference (points)} \\\\',
         'Reference & the threshold & (\\%) & (\\%) & (millions) & reach & correct \\\\', '\\midrule']
    k = 0
    for h, rs in blocks:
        if ', 1 day ahead' not in h:
            continue
        m = re.fullmatch(r'(Above .*?), 1 day ahead: \\model\{\} warns (.*?) million people a day, reach (.*?)\\%, correct (.*?)\\%, (.*?) million hidden people warned', h)
        if k:
            L.append('\\addlinespace')
        k += 1
        L.append('\\multicolumn{7}{@{}l}{\\textit{%s: \\model{} warns %s million people a day (reach %s\\%%, correct %s\\%%, %s million hidden)}} \\\\' % m.groups())
        for r in rs:
            L.append(' & '.join([LAB.get(r[0], r[0]), r[1], v(r[2]), v(r[3]), v(r[4]), r[5], r[6]]) + ' \\\\')
    assert k == 3
    L += ['\\bottomrule', '\\end{tabular}']
    cap = ('\\textbf{The reference forecasts at the number of people warned by \\model{}, 1 day ahead.} Each reference warns a resident when its '
           'value exceeds the factor times the threshold; the factor is the one for which it warns as many people a day as \\model{} over 2022 '
           '(a choice made in hindsight, which favours the reference). Hidden people warned in millions a day; differences with 95\\% intervals '
           'over target days (paired). The values 3 days ahead and the other intervals are given in the Source Data file.')
    return table(cap, 'matched', fit('\n'.join(L)))


def do_regions(s):
    header, blocks = parse(block(s, 'regions'))
    rs = blocks[0][1]; assert len(blocks) == 1 and len(rs) == 11 and all(len(r) == 11 for r in rs)
    cols = (['region'] + tri('hidden people (millions a day)') + ["share of the world's hidden people (%)"] + tri("share of the region's exposed people (%)")
            + ['hidden by 10 km blocks (millions a day)', 'hidden with a 10% margin (millions a day)']
            + tri('hidden people warned by Foehn, 1 day ahead (%)') + tri('hidden people warned by Foehn, 3 days ahead (%)')
            + tri('hidden people warned by the CAMS forecast, 1 day ahead (%)')
            + ['reach of Foehn (%)', 'reach of the CAMS forecast (%)', 'correct warnings of Foehn (%)', 'correct warnings of the CAMS forecast (%)'])
    rows = []
    for r in rs:
        pr = [num(x) for x in r[9].split(' / ')] + [num(x) for x in r[10].split(' / ')]
        rows.append([plain(r[0])] + val3(r[1]) + [num(r[2])] + val3(r[3]) + [num(r[4]), num(r[5])] + val3(r[6]) + val3(r[7]) + val3(r[8]) + pr)
    dump('regions', 'regions', '', 'complete table: blocks of 10 km, margin, 3 days ahead, every 95% interval',
         'Complete Supplementary Table: hidden people by region above 37.5 µg m-3 (regions are groups of countries that partition the world). Counts '
         'and shares: all 1 km grid cells with residents, 365 days of 2022. Warnings: grid cells with at least 100 residents, 179 target days.', cols, rows)
    L = ['\\begin{tabular}{@{}lcrrrrcc@{}}', '\\toprule',
         ' & Hidden & Share of & Share of & \\multicolumn{2}{c}{Hidden warned} & Reach & Correct \\\\',
         " & (millions & the world's & the region's & \\multicolumn{2}{c}{1 day ahead (\\%)} & (\\%) & (\\%) \\\\",
         'Region & a day) & hidden (\\%) & exposed (\\%) & \\model{} & CAMS & \\model{} / CAMS & \\model{} / CAMS \\\\', '\\midrule']
    for r in rs:
        L.append(' & '.join([r[0], r[1], r[2], v(r[3]), v(r[6]), v(r[8]), r[9], r[10]]) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}']
    cap = ('\\textbf{Hidden people by region, above 37.5\\,\\textmu g\\,m$^{-3}$.} Regions are groups of countries that partition the world. '
           'Columns 2--4: all 1\\,km grid cells with residents, 365 days of 2022 (95\\% interval in brackets). Columns 5--8: grid cells with at least '
           '100 residents, 179 target days, 1 day ahead; CAMS, the CAMS forecast. The counts for blocks of 10\\,km and with a margin, the values '
           '3 days ahead and the other intervals are given in the Source Data file.')
    return table(cap, 'regions', fit('\n'.join(L)))


def do_away(s):
    NET = ['China (CNEMC)', 'USA (EPA)', 'Europe (EEA)', 'Korea (AirKorea)', 'Japan', 'India (CPCB)', 'Six networks', 'Korea, Japan and India']
    rows = []
    for line in tab_lines(block(s, 'away')):
        cells = [c.strip() for c in re.split(r'(?<!\\)&', line.rstrip('\\').strip())]
        if cells[0] in NET:
            rows.append([re.sub(r'\\multicolumn\{2\}\{c\}\{(.*)\}', r'\1', c) for c in cells])
    b1, b2, pb = rows[:7], rows[7:14], rows[14:]
    assert len(rows) == 22 and all(len(r) == 8 for r in b1) and all(len(r) == 7 for r in b2 + pb), [len(r) for r in rows]
    assert [r[0] for r in b1] == NET[:7] == [r[0] for r in b2] and [r[0] for r in pb] == NET
    cols = (['network', 'stations', 'estimate within 0.5 µg m-3 of the measurement, at the station pixel (% of station-days)', 'the same, 3 km away (%)',
             'the same, 10 km away (%)', 'RMSE of Foehn at the station pixel (µg m-3)', 'RMSE of Foehn 3 km away (µg m-3)', 'RMSE of Foehn 10 km away (µg m-3)',
             'RMSE of the CAMS forecast (µg m-3)', 'RMSE of Aurora (µg m-3)']
            + tri('gain over the CAMS forecast at the pixel (%)') + tri('gain over the CAMS forecast 10 km away (%)')
            + tri('gain over Aurora at the pixel (%)') + tri('gain over Aurora 10 km away (%)')
            + tri('gain over the CAMS forecast 3 days ahead, at the pixel (%)') + tri('gain over the CAMS forecast 3 days ahead, 10 km away (%)'))
    full = []
    for r, q in zip(b1, b2):
        full.append([plain(r[0]), num(r[1])] + [num(x) for x in r[2].split(' / ')] + [num(x) for x in r[3].split(' / ')] + [num(r[4]), num(r[5])]
                    + val3(r[6]) + val3(r[7]) + val3(q[3]) + val3(q[4]) + val3(q[5]) + val3(q[6]))
    dump('away_a', 'away', 'a', 'complete panel a: the ring at 3 km, the gains over Aurora and those 3 days ahead',
         'Complete Supplementary Table, panel a: Foehn 1 day ahead in 2022 read at the 1 km pixel of each station and 3 and 10 km away from it (identical '
         'station-days), against the CAMS forecast and Aurora read at the station. Gains are reductions of the RMSE in per cent, with 95% intervals of a '
         'paired moving-block bootstrap over start dates.', cols, full)
    cols = (['network', 'stations'] + tri('gain over the CAMS forecast, 1 day ahead (%)') + tri('gain over the CAMS forecast, 4 days ahead (%)')
            + tri('gain over Aurora, 1 day ahead (%)') + tri('gain over Aurora, 4 days ahead (%)'))
    dump('away_b', 'away', 'b', 'panel b as printed, with numeric columns',
         "Supplementary Table, panel b: the network of the same design trained in Aurora's protocol on the CAMS analyses (no 1 km estimate, no station "
         'measurement), read at the stations (214 initializations, 1 May to 30 November 2022; daily means from three instants of each day for the three '
         'systems). Gains are reductions of the RMSE in per cent, with 95% intervals.', cols,
         [[plain(r[0]), num(r[1])] + val3(r[3]) + val3(r[4]) + val3(r[5]) + val3(r[6]) for r in pb])
    A = ['\\begin{tabular}{@{}lr>{\\centering\\arraybackslash}p{1.0cm}>{\\centering\\arraybackslash}p{1.0cm}rrrrcc@{}}', '\\toprule',
         '\\multicolumn{10}{@{}l}{\\textbf{a}\\quad \\model{} at 1\\,km, 1 day ahead} \\\\',
         ' & & \\multicolumn{2}{c}{Within 0.5 (\\%)} & \\multicolumn{4}{c}{RMSE (\\textmu g\\,m$^{-3}$)} & \\multicolumn{2}{c}{Gain over the CAMS forecast (\\%)} \\\\',
         '\\cmidrule(lr){3-4}\\cmidrule(lr){5-8}\\cmidrule(l){9-10}',
         'Network & Stations & pixel & 10\\,km & \\model{}, pixel & \\model{}, 10\\,km & CAMS & Aurora & at the pixel & 10\\,km away \\\\', '\\midrule']
    for r in b1:
        w = r[2].split(' / '); e = r[3].split(' / ')
        A.append(' & '.join([r[0], r[1], w[0], w[2], e[0], e[2], r[4], r[5], r[6], r[7]]) + ' \\\\')
    A += ['\\bottomrule', '\\end{tabular}']
    Bp = ['\\begin{tabular}{@{}lrcccc@{}}', '\\toprule',
          "\\multicolumn{6}{@{}l}{\\textbf{b}\\quad Same design trained in Aurora's protocol (no 1\\,km estimates, no stations), read at the stations} \\\\",
          ' & & \\multicolumn{2}{c}{Gain over the CAMS forecast (\\%)} & \\multicolumn{2}{c}{Gain over Aurora (\\%)} \\\\',
          '\\cmidrule(lr){3-4}\\cmidrule(l){5-6}',
          'Network & Stations & 1 day ahead & 4 days ahead & 1 day ahead & 4 days ahead \\\\', '\\midrule']
    for r in pb:
        Bp.append(' & '.join([r[0], r[1], r[3], r[4], r[5], r[6]]) + ' \\\\')
    Bp += ['\\bottomrule', '\\end{tabular}']
    both = '\\begin{tabular}{@{}l@{}}\n' + '\n'.join(A) + ' \\\\\n\\noalign{\\vskip 10pt}\n' + '\n'.join(Bp) + '\n\\end{tabular}'
    cap = ('\\textbf{The gain at the stations holds away from the station pixel and without the 1\\,km estimates.} '
           '\\textbf{a}, \\model{} 1 day ahead in 2022, read at the 1\\,km pixel of each station and on the square ring of pixels 9 pixels '
           '(0.09$^\\circ$, about 10\\,km) from it: mean of the ensemble mean over the ring pixels that have at least 100 residents and hold no monitor '
           'of our files; stations with at least four such pixels on the ring and on that at 3\\,km; every second start day; identical station-days in a '
           'row. The CAMS forecast and Aurora, issued at the same time, are read at the station. `Within 0.5\', share of station-days on which '
           'the 1\\,km estimate of the day is within 0.5\\,\\textmu g\\,m$^{-3}$ of the measurement, at the station pixel and at one pixel drawn on the '
           'ring: where the estimates were fitted to a monitor, the share is high at its pixel only. \\textbf{b}, The network of the same design trained '
           'in Aurora\'s protocol on the CAMS analyses, with no 1\\,km estimate and no station measurement in its training, selection or inputs (ensemble '
           'mean of eight members; 214 initializations, 1 May to 30 November 2022; all stations of each network; the three systems read at the same three '
           'instants of each day, Supplementary Note~\\ref{note:stations}). Gains are reductions of the root-mean-square error (RMSE) in per cent, with '
           '95\\% intervals from a paired moving-block bootstrap over start dates (blocks of seven start dates, 10,000 resamples); a positive value means '
           'a lower error than the reference system. The ring at 3\\,km, the gains over Aurora and the gains 3 days ahead are given in the Source Data file.')
    return table(cap, 'away', fit(both))


# ------------------------------------------------------------------------------------------------ the models, side by side
def models():
    col = '>{\\rr}p{3.45cm}'
    pre = ('\\fontsize{7.5}{8.9}\\selectfont\n\\setlength{\\tabcolsep}{3pt}\n\\renewcommand{\\arraystretch}{1.05}\n'
           '\\begin{tabular}{@{}>{\\rr}p{1.5cm}' + col * 4 + '@{}}\n\\toprule\n')
    post = '\\bottomrule\n\\end{tabular}'

    def body(heads, rows):
        out = ' & ' + ' & '.join(heads) + ' \\\\\n\\midrule\n'
        return out + '\\addlinespace\n'.join(name + ' &\n' + ' &\n'.join(cells) + ' \\\\\n' for name, cells in rows)
    F44 = ('32 learned tracers, the share of air from aloft, its survival, the advected \\pmtf{}, three fire terms and six anthropogenic emission terms')
    p1 = body(
        ['\\textbf{i}\\quad \\model{}, the 1\\,km model', '\\textbf{ii}\\quad Foehn-CAMS', '\\textbf{iii}\\quad Foehn-CAMS, real-time version', '\\textbf{iv}\\quad Downscaler'],
        [('Grid and domain', [
            '0.01$^\\circ$ (about 1\\,km); land between 84$^\\circ$N and 60$^\\circ$S; 21 overlapping windows of $4{,}192\\times6{,}992$ grid cells, blended.',
            '0.4$^\\circ$ grid of the CAMS analysis, $361\\times900$ points, 84$^\\circ$N--60$^\\circ$S, land and sea, in one pass.',
            'As model ii.',
            '0.01$^\\circ$; 147.3 million land grid cells between 59.5$^\\circ$S and 83.6$^\\circ$N. Its encoder reads the global 0.25$^\\circ$ grid ($576\\times1{,}440$ points).']),
         ('Initial state and inputs', [
             'At 1\\,km: the estimates of days $d$ and $d-1$, elevation and a land mask. At 0.25$^\\circ$: the estimates of day $d$; 37 weather fields of day $d$ '
             '(ERA5); 31 weather fields of the target day (HRES run of 00\\,UTC on day $d$); three fire fields of day $d$ (GFAS). From FLEXGREEN, driven by the '
             'same HRES run, 44 transport fields: ' + F44 + '.',
             'The CAMS analyses of 00\\,UTC and of 12\\,h earlier (13 fields: PM$_1$, \\pmtf{}, PM$_{10}$, and the total columns and 1000-hPa values of CO, NO, '
             'NO$_2$, O$_3$ and SO$_2$); the six static fields of Aurora; the HRES run of 00\\,UTC (53 fields at the start and at the valid time, and the '
             'precipitation of the 12\\,h before it); the fires of day $d-1$ (GFAS). From FLEXGREEN, run back from each valid time: the same 44 transport '
             'fields as model i.',
             'As model ii, from the data of each morning: the CAMS analyses of 00\\,UTC on day $d$ and of 12\\,UTC on day $d-1$; the forecast of 00\\,UTC of the '
             'Integrated Forecasting System from the open data of the European Centre (0.25$^\\circ$, every 6\\,h); the fires of day $d-1$ from GFAS version '
             '1.4.2, whose injection height replaces the plume-rise height; the anthropogenic inventory of the same month of 2022. FLEXGREEN is unchanged.',
             'No 1\\,km field of any day, no station measurement and no emission inventory. On the 0.25$^\\circ$ grid, 47 fields of the day: the coarse \\pmtf{} '
             '(in training, the first-day mean of the CAMS forecast; in real time, one member of model iii), 37 weather fields, three fire fields and six fields '
             'of a one-day FLEXGREEN transport. For each grid cell: six descriptors (five from the climatology of the estimates of 2017--2020, and its '
             'elevation), the day of the year and a noise value.']),
         ('Output', [
             'Daily mean \\pmtf{} at 0.01$^\\circ$, 1 to 7 days ahead; each lead time is forecast directly from day $d$.',
             'Instantaneous \\pmtf{} near the surface at 0.4$^\\circ$, every 12\\,h from 12 to 120\\,h; each lead time is forecast directly.',
             'As model ii.',
             'Daily mean \\pmtf{} at 0.01$^\\circ$ for the day of the coarse fields. In real time, lead days 1 to 5; lead day 1 is the day of issue.']),
         ('Learned parameters', [
             '14.2 million (14,157,348), of which 10,720 in the tracer encoder.', '9.9 million (9,913,441).', '9.9 million (the weights of model ii, fine-tuned).',
             '3.3 million (3,302,001).']),
         ('Training data and period', [
             'Estimates of 2017--2020 (9,730 pairs of start day and lead time); 2021 for selection; 2022 for the test. Three stages: flow matching (60 epochs); '
             'fine-tuning with the emission terms (third of 12 epochs kept); fine-tuning on the CRPS (second of two epochs kept).',
             'CAMS analyses; 1,667 start days from October 2017 to April 2022, with targets before 1 May 2022 (16,670 pairs of start day and lead time); flow '
             'matching, 35 epochs; last checkpoint, no selection.',
             'Fine-tuned from model ii by flow matching for six epochs on 243 start days, 1 January--31 August 2026 (2,430 pairs), with the CAMS analyses up to '
             '5 September 2026 as targets.',
             'Estimates of 2017--2020 (1,412 days); flow matching, 24,000 steps; checkpoint of step 22,000, selected on 24 days of 2021; noise prior fitted on '
             '48 days of 2021; one test on 2022.']),
         ('Members and sampler', [
             'Four members in the evaluation of 2022 and 16 in the case studies; four Euler steps. Each member has its own noise field and its own half of the '
             '32 particles. The stages before the CRPS fine-tuning are sampled with 12 Euler steps.',
             'Eight members; 12 Euler steps. Each member has its own noise field and its own half of the 32 particles.',
             'Eight members; 12 Euler steps.',
             'Eight members; 10 Euler steps. In real time, member $m$ is drawn from member $m$ of model iii.']),
         ('Shown in', [
             'Figs.~\\ref{M-fig:hidden}f, \\ref{M-fig:overview}, \\ref{M-fig:realism}, \\ref{M-fig:transport}e--g and \\ref{M-fig:smoke}; Supplementary '
             'Figs.~\\ref{fig:exposure}g--j, \\ref{fig:warned}, \\ref{fig:fair}, \\ref{fig:regions}, \\ref{fig:reliability} and \\ref{fig:sedi}. Stages before '
             'the CRPS fine-tuning: Fig.~\\ref{M-fig:transport}j; Supplementary Figs.~\\ref{fig:fair}h and \\ref{fig:emislead}.',
             'Fig.~\\ref{M-fig:auroraset}; Supplementary Figs.~\\ref{fig:fair}h and \\ref{fig:aurorareg}; Supplementary Table~\\ref{tab:away}b (214 '
             'initializations, May--November 2022).',
             'Figs.~\\ref{M-fig:schematic}d and \\ref{M-fig:rtday}c,f,g,i; section `Kilometre-scale forecasts in real time\'.',
             'Figs.~\\ref{M-fig:schematic}e and \\ref{M-fig:rtday}a,d--h,j; section `Kilometre-scale forecasts in real time\'.'])])
    cap1 = ('\\textbf{The models of this study.} One column per model: here the four models of the global system, on the next page the networks of the '
            'European benchmark (Supplementary Note~\\ref{note:benchmark}). Model i is the model of the paper (retrospective ensemble forecasts at 1\\,km); '
            'the name \\model{} alone denotes it. Model ii has the same design in the protocol of Aurora\'s air-pollution model, and none of its inputs is a '
            '1\\,km estimate or a station measurement. Models iii and iv are the two stages of the real-time system: the coarse forecast of each morning, and '
            'the downscaler that draws the 1\\,km field of a day from coarse fields of the same day. $d$, start day; estimates, the 1\\,km \\pmtf{} estimates '
            'of the Global High Air Pollutants dataset; HRES, the high-resolution forecast of the European Centre for Medium-Range Weather Forecasts; GFAS, '
            'the Global Fire Assimilation System; CRPS, continuous ranked probability score. FLEXGREEN, the particle model that computes the transport, has '
            'no learned parameter and is common to the four models.')
    p2 = body(
        ['\\textbf{v}\\quad European 0.25$^\\circ$ network', '\\textbf{vi}\\quad European 1\\,km window network', '\\textbf{vii}\\quad European tile-free 1\\,km network',
         '\\textbf{viii}\\quad Five retrained networks: ConvLSTM, SimVP, Earthformer, ClimaX, TopoFlow'],
        [('Grid and domain', [
            '0.25$^\\circ$, $168\\times280$ points, 30--72$^\\circ$N, 25$^\\circ$W--45$^\\circ$E.',
            '0.01$^\\circ$; a window of $768\\times1{,}536$ grid cells, 41.8--49.5$^\\circ$N, 0--15.4$^\\circ$E (Paris to Florence).',
            '0.01$^\\circ$; the whole European window, $4{,}192\\times6{,}992$ grid cells, 30.1--72$^\\circ$N, 25$^\\circ$W--44.9$^\\circ$E, in one pass.',
            'As model v. For Supplementary Figs.~\\ref{fig:speccost}b and \\ref{fig:km}, SimVP and Earthformer are also trained at 1\\,km on the window of model vi.']),
         ('Initial state and inputs', [
             'The 0.25$^\\circ$ means of the estimates of day $d$; 37 weather fields of day $d$ (ERA5); 31 weather fields of the target day (ERA5 in training; '
             'in the test, the HRES run of 00\\,UTC on day $d$); elevation and a land mask. For its transport layer: the 3-hourly winds at 10\\,m and '
             '925\\,hPa, the daily rain and the stability of the lower troposphere from day $d$ to the target day.',
             'At 1\\,km: the estimates of days $d$ and $d-1$, elevation and a land mask. At 0.25$^\\circ$: the estimates of day $d$, 37 weather fields of day '
             '$d$ and 31 of the target day, as model v. No transport layer.',
             'As model vi, over the whole window. No transport layer.',
             'The 0.25$^\\circ$ means of the estimates of day $d$; 30 weather fields of days $d$ and $d-1$ (ERA5); the surface concentrations of NO$_2$, '
             'O$_3$, SO$_2$, CO and PM$_{10}$ of the CAMS reanalysis on days $d$ and $d-1$; the 31 weather fields of the target day, as model v; elevation.']),
         ('Output', [
             'Daily mean \\pmtf{} at 0.25$^\\circ$, 1 to 4 days ahead; each lead time is forecast directly from day $d$.',
             'Daily mean \\pmtf{} at 0.01$^\\circ$ on the window, 1 to 4 days ahead.',
             'Daily mean \\pmtf{} at 0.01$^\\circ$, 1 to 4 days ahead.',
             'As model v; one forecast, no ensemble.']),
         ('Learned parameters', [
             '4.17 million (4,167,717), of which 4,388 in the transport layer.', '12.4 million.', '14.1 million (14,137,412).',
             'ConvLSTM 8.3, SimVP 5.0, Earthformer 62.3, TopoFlow 62.3 and ClimaX 64.9 million.']),
         ('Training data and period', [
             '0.25$^\\circ$ means of the estimates, 2017--2020 (5,600 pairs); flow matching, 60 epochs; last checkpoint. Three trainings (seeds 42, 43, 44). '
             'The same network without its transport layer (4,163,329 parameters) is trained in the same way.',
             'Estimates of 2017--2020 on the window (5,584 pairs); flow matching, 30 epochs; last checkpoint; one training.',
             'Estimates of 2017--2020 (5,584 pairs); flow matching with the noise schedule of model i, 15 epochs; last checkpoint; one training.',
             '0.25$^\\circ$ means of the estimates, 2017--2020 (5,574 pairs); squared error over land (for TopoFlow, underestimates weigh double); 60 epochs; '
             'checkpoint with the lowest error on 2021. Three trainings each (seeds 42, 43, 44).']),
         ('Members and sampler', [
             'Eight members; eight Euler steps (four for Supplementary Fig.~\\ref{fig:speccost}a).', 'Eight members; eight Euler steps.',
             'Four members, of which the mean is shown; 16 Euler steps.', 'One deterministic forecast.']),
         ('Shown in', [
             'Supplementary Figs.~\\ref{fig:speccost}, \\ref{fig:eu_lead} and \\ref{fig:eu_extremes} (360 start days of 2022; 2,063 stations).',
             'Supplementary Fig.~\\ref{fig:speccost}b (four start days of 2022, four lead times).',
             'Supplementary Fig.~\\ref{fig:km}c,d (four start days of 2022, 1 day ahead).',
             'Supplementary Figs.~\\ref{fig:speccost}, \\ref{fig:eu_lead}, \\ref{fig:eu_extremes} and \\ref{fig:km}b,d.'])])
    cap2 = ('\\textbf{The models of this study (continued).} The networks of the European benchmark (Supplementary Note~\\ref{note:benchmark}). Model v '
            'is the network of the benchmark, which preceded the global model and guided its design; model vi tests a network trained at 1\\,km on one '
            'regional window; model vii tests that one pass over a continent at 1\\,km leaves no seams (its network is the generator of model i, without the '
            'transport and fire fields); the networks of column viii are the comparison networks, trained by regression. In the panels of Supplementary '
            'Figs.~\\ref{fig:speccost}--\\ref{fig:eu_extremes}, the label Foehn denotes model v; `Foehn (1\\,km window)\' denotes model vi and `Foehn, '
            'tile-free\' model vii. None of them is model i. All are trained on 2017--2020, with the reanalysis weather of the target day, and tested on 2022 '
            'with the HRES forecast.')
    return (table(cap1, 'models', pre + p1 + post), table(cap2, '', pre + p2 + post, cont=True))


def main():
    s = frozen()
    print('JSON files of the complete tables:')
    new = dict(sens=do_sens(s), warned=do_warned(s), matched=do_matched(s), regions=do_regions(s), away=do_away(s))
    m1, m2 = models()
    if DRY:
        print(m1, m2, *new.values(), sep='\n\n'); return
    cur = open(SUP, encoding='utf-8').read()
    if 'sidewaystable' not in cur:
        print('work/supplementary.tex holds no sideways table: nothing replaced'); return
    print(backup('before_portrait_tables'))
    E = '\\end{sidewaystable}'
    edit_span('supplementary.tex', '\\begin{sidewaystable}[p]\n\\centering\n\\caption{\\textbf{The models of this study.}', E, m1)
    edit_span('supplementary.tex', '\\begin{sidewaystable}[p]\n\\ContinuedFloat\n\\centering\n\\caption{\\textbf{The models of this study (continued).} The real-time system', E + '\n\\clearpage\n\n', '')
    edit_span('supplementary.tex', '\\begin{sidewaystable}[p]\n\\ContinuedFloat\n\\centering\n\\caption{\\textbf{The models of this study (continued).} The networks', E, m2)
    starts = dict(sens='Sensitivity of the count', warned='Who is warned.', matched='The reference forecasts at the number', regions='Hidden people by region',
                  away='The gain at the stations holds away')
    for k in ('sens', 'warned', 'matched', 'regions', 'away'):
        edit_span('supplementary.tex', '\\begin{sidewaystable}[p]\n\\centering\n\\caption{\\textbf{' + starts[k], E, new[k])
    edit('supplementary.tex', [
        ('\\usepackage{rotating}\n', '\\newcommand{\\fitwidth}[1]{\\resizebox{\\ifdim\\width>\\textwidth\\textwidth\\else\\width\\fi}{!}{#1}}   % a table wider than the text is scaled down to it, never up\n'),
        ('Supplementary Table~\\ref{tab:models}, row v).', 'Supplementary Table~\\ref{tab:models}, column v).'),
        ("is row vi of Supplementary Table~\\ref{tab:models}", "is column vi of Supplementary Table~\\ref{tab:models}"),
        ('Supplementary Table~\\ref{tab:models}, row viii)', 'Supplementary Table~\\ref{tab:models}, column viii)'),
        ('Supplementary Table~\\ref{tab:models}, row vii)', 'Supplementary Table~\\ref{tab:models}, column vii)'),
        ('Supplementary Table~\\ref{tab:models}, rows v--viii)', 'Supplementary Table~\\ref{tab:models}, columns v--viii)'),
        ('their other inputs differ (row viii).', 'their other inputs differ (column viii).'),
        ('rows i and v--vii of Supplementary Table~\\ref{tab:models}; row iv reads', 'columns i and v--vii of Supplementary Table~\\ref{tab:models}; model iv reads'),
        ('Supplementary Table~\\ref{tab:warned} gives, for three thresholds and two lead times, the people warned a day by each system.',
         'Supplementary Table~\\ref{tab:warned} gives, for three thresholds, the people warned a day by each system 1 day ahead (3 days ahead in the Source Data file).'),
    ])
    edit(W + '/work/main.tex', [('\\usepackage{rotating}\n', '')])
    assert 'sideways' not in open(SUP, encoding='utf-8').read()


if __name__ == '__main__':
    main()
