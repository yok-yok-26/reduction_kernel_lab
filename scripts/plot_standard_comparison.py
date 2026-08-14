#!/usr/bin/env python3
import csv
import html
from pathlib import Path

ROOT = Path('.')
TREND = ROOT / 'reports' / 'trends'
NCU = ROOT / 'reports' / 'ncu'
TREND.mkdir(parents=True, exist_ok=True)
MODES = ['library_cub_device_reduce','custom_cuda_atomic','optimized_todo','optimized_tile2','optimized_tile4','optimized_tile8','optimized_2stage','optimized_2stage_tile']
USER = [m for m in MODES if m not in ('library_cub_device_reduce','custom_cuda_atomic')]
IDEA = {
 'library_cub_device_reduce':'L1 generic CUB DeviceReduce baseline',
 'custom_cuda_atomic':'custom per-element atomic comparison, not fair baseline',
 'optimized_todo':'single-kernel user reduction path',
 'optimized_tile2':'user tiled reduction, tile=2',
 'optimized_tile4':'user tiled reduction, tile=4',
 'optimized_tile8':'user tiled reduction, tile=8',
 'optimized_2stage':'user two-stage reduction',
 'optimized_2stage_tile':'user two-stage tiled reduction',
}
COLORS = {
 'library_cub_device_reduce':'#1f77b4','custom_cuda_atomic':'#8c564b','optimized_todo':'#2ca02c','optimized_tile2':'#9467bd',
 'optimized_tile4':'#17becf','optimized_tile8':'#ff7f0e','optimized_2stage':'#d62728','optimized_2stage_tile':'#e377c2'
}

def esc(s): return html.escape(str(s), quote=True)
def num(v, d=0.0):
    try:
        if v in ('', None): return d
        return float(v)
    except Exception: return d

def read_csv(p):
    with p.open() as f: return list(csv.DictReader(f))

def write_csv(path, rows, fields):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def svg_bar(path, title, rows, value_key, label_key, raw_fmt='{:.3f}', x_label=''):
    w = 1100; left = 300; right = 170; top = 70; row_h = 38; bottom = 60
    h = top + len(rows)*row_h + bottom
    maxv = max([num(r[value_key]) for r in rows] + [1e-9])
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">', '<style>text{font-family:DejaVu Sans,Arial,sans-serif}.small{font-size:13px}.title{font-size:24px;font-weight:bold}.label{font-size:12px}</style>', f'<rect width="100%" height="100%" fill="white"/>', f'<text x="24" y="36" class="title">{esc(title)}</text>']
    axis_w = w-left-right
    for i,r in enumerate(rows):
        y = top + i*row_h
        val = num(r[value_key]); bw = 0 if maxv == 0 else val/maxv*axis_w
        mode = r.get('mode', r.get('baseline_mode', ''))
        color = r.get('color') or COLORS.get(mode, '#555')
        parts.append(f'<text x="24" y="{y+22}" class="label">{esc(r[label_key])}</text>')
        parts.append(f'<rect x="{left}" y="{y+8}" width="{bw:.2f}" height="20" fill="{color}" opacity="0.86"/>')
        parts.append(f'<text x="{left+axis_w+18}" y="{y+23}" class="small">{esc(raw_fmt.format(val))}</text>')
    parts.append(f'<text x="{left}" y="{h-22}" class="small">{esc(x_label)}</text>')
    parts.append('</svg>')
    path.write_text('\n'.join(parts))

def svg_ratio(path, title, ratio):
    w=900; h=230; left=190; axis_w=540; y=100
    color = '#2ca02c' if ratio > 1.02 else '#ffbf00' if ratio > 0.98 else '#d62728'
    maxv=max(1.15, ratio*1.15)
    one_x=left + axis_w*(1.0/maxv)
    bw=axis_w*(ratio/maxv)
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">','<style>text{font-family:DejaVu Sans,Arial,sans-serif}.title{font-size:24px;font-weight:bold}.small{font-size:13px}</style>','<rect width="100%" height="100%" fill="white"/>',f'<text x="24" y="36" class="title">{esc(title)}</text>',f'<text x="24" y="{y+17}" class="small">best user / CUB</text>',f'<rect x="{left}" y="{y}" width="{bw:.2f}" height="28" fill="{color}"/>',f'<line x1="{one_x:.2f}" y1="{y-15}" x2="{one_x:.2f}" y2="{y+45}" stroke="black"/>',f'<text x="{one_x+5:.2f}" y="{y-18}" class="small">1.0 baseline</text>',f'<text x="{left+bw+8:.2f}" y="{y+19}" class="small">{ratio:.3f}x</text>','</svg>']
    path.write_text('\n'.join(parts))

def svg_heatmap(path, title, labels, vals, config):
    cell_w=135; w=80+cell_w*len(labels)+60; h=260; maxv=max(vals) or 1
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">','<style>text{font-family:DejaVu Sans,Arial,sans-serif}.title{font-size:22px;font-weight:bold}.small{font-size:12px}</style>','<rect width="100%" height="100%" fill="white"/>',f'<text x="24" y="32" class="title">{esc(title)}</text>',f'<text x="24" y="110" class="small">{esc(config)}</text>']
    x0=80; y0=80; ch=55; winner=max(range(len(vals)), key=lambda i: vals[i])
    for i,(lab,v) in enumerate(zip(labels,vals)):
        norm=v/maxv if maxv else 0; green=int(80+160*norm); red=int(245-120*norm); blue=int(120-80*norm); color=f'rgb({red},{green},{blue})'; x=x0+i*cell_w
        parts.append(f'<rect x="{x}" y="{y0}" width="{cell_w-3}" height="{ch}" fill="{color}" stroke="{"red" if i==winner else "#ddd"}" stroke-width="{"3" if i==winner else "1"}"/>')
        parts.append(f'<text x="{x+8}" y="{y0+24}" class="small">{v:.1f} GB/s</text>')
        parts.append(f'<text transform="translate({x+12},{y0+88}) rotate(35)" class="small">{esc(lab)}</text>')
    parts.append('</svg>'); path.write_text('\n'.join(parts))

def svg_grouped(path, title, groups, normalize_within_group=False, raw_header='Raw value'):
    # groups: [(group_label, [(label, raw_value, color)])]
    left=360; right=170; axis_w=680; top=80; row_h=28; gap=20; w=left+axis_w+right+40
    total_rows=sum(len(items) for _,items in groups); h=top+total_rows*row_h+len(groups)*gap+70
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">','<style>text{font-family:DejaVu Sans,Arial,sans-serif}.title{font-size:24px;font-weight:bold}.small{font-size:12px}.group{font-size:14px;font-weight:bold}</style>','<rect width="100%" height="100%" fill="white"/>',f'<text x="24" y="36" class="title">{esc(title)}</text>',f'<text x="{left+axis_w+18}" y="62" class="group">{esc(raw_header)}</text>']
    y=top
    global_max=max([abs(num(v)) for _,items in groups for _,v,_ in items]+[1e-9])
    for glabel,items in groups:
        parts.append(f'<text x="24" y="{y+16}" class="group">{esc(glabel)}</text>')
        gmax=max([abs(num(v)) for _,v,_ in items]+[1e-9]) if normalize_within_group else global_max
        for label,v,color in items:
            bw=abs(num(v))/gmax*axis_w if gmax else 0
            parts.append(f'<text x="54" y="{y+38}" class="small">{esc(label)}</text>')
            parts.append(f'<rect x="{left}" y="{y+24}" width="{bw:.2f}" height="18" fill="{color}" opacity="0.84"/>')
            parts.append(f'<text x="{left+axis_w+18}" y="{y+38}" class="small">{num(v):.2f}</text>')
            y+=row_h
        y+=gap
    parts.append('</svg>'); path.write_text('\n'.join(parts))

bench = read_csv(ROOT/'reports/benchmark/summary_latest.csv')
for r in bench:
    r['avg_ms']=num(r['avg_ms']); r['effective_gbps']=num(r['effective_gbps']); r['n']=int(num(r.get('n')))
bench_by={r['mode']:r for r in bench}
baseline=bench_by['library_cub_device_reduce']
best=min((bench_by[m] for m in USER), key=lambda r:r['avg_ms'])
config=f"n={baseline['n']}"

best_rows=[{'config':config,'baseline_mode':baseline['mode'],'baseline_avg_ms':baseline['avg_ms'],'baseline_gbps':baseline['effective_gbps'],'best_user_mode':best['mode'],'best_user_avg_ms':best['avg_ms'],'best_user_gbps':best['effective_gbps'],'latency_ratio_user_over_baseline':best['avg_ms']/baseline['avg_ms'],'throughput_ratio_user_over_baseline':best['effective_gbps']/baseline['effective_gbps']}]
write_csv(TREND/'reduction_best_user_vs_baseline_latest.csv', best_rows, list(best_rows[0]))
svg_bar(TREND/'reduction_best_user_vs_baseline.svg', f'Best User vs Fair Baseline ({config})', [{'label':'CUB baseline','value':baseline['avg_ms'],'mode':'library_cub_device_reduce'}, {'label':f"Best user: {best['mode']}", 'value':best['avg_ms'], 'mode':best['mode']}], 'value', 'label', '{:.6f} ms', 'Release benchmark latency')
svg_ratio(TREND/'reduction_best_user_ratio.svg', f'Best User Throughput Ratio ({config})', best['effective_gbps']/baseline['effective_gbps'])
svg_heatmap(TREND/'reduction_full_series_heatmap.svg', 'Full-Series Throughput Heatmap, Row-Normalized', MODES, [bench_by[m]['effective_gbps'] for m in MODES], config)

ncu=read_csv(NCU/'single_mode_summary_latest.csv')
for r in ncu:
    for k in ('duration_us','memory_pct','dram_pct','sm_pct','achieved_occupancy_pct','eligible_warps','warp_cycles_per_issued','registers_per_thread','static_smem_bytes','dynamic_smem_bytes','memory_gbps'):
        r[k]=num(r.get(k))
by={}
for r in ncu: by.setdefault(r['mode'], []).append(r)
metric_rows=[]
for mode in MODES:
    stages=by.get(mode, [])
    if not stages: continue
    dom=max(stages, key=lambda r:r['duration_us'])
    metric_rows.append({'mode':mode,'idea':IDEA[mode],'duration_us_total':sum(s['duration_us'] for s in stages),'memory_throughput_pct_dominant_stage':dom['memory_pct'],'dram_throughput_pct_dominant_stage':dom['dram_pct'],'sm_throughput_pct_dominant_stage':dom['sm_pct'],'achieved_occupancy_pct_dominant_stage':dom['achieved_occupancy_pct'],'eligible_warps_per_sched_dominant_stage':dom['eligible_warps'],'warp_cycles_per_issued_inst_dominant_stage':dom['warp_cycles_per_issued'],'registers_per_thread_dominant_stage':dom['registers_per_thread'],'static_smem_bytes_dominant_stage':dom['static_smem_bytes'],'dynamic_smem_bytes_dominant_stage':dom['dynamic_smem_bytes'],'memory_gbps_dominant_stage':dom['memory_gbps']})
metric_fields=list(metric_rows[0]); write_csv(TREND/'reduction_ncu_metrics_standard_latest.csv', metric_rows, metric_fields)
metrics=[k for k in metric_fields if k not in ('mode','idea')]
metric_names=['Duration us total','Memory throughput %','DRAM throughput %','SM throughput %','Achieved occupancy %','Eligible warps/sched','Warp cycles/issued inst','Registers/thread','Static smem bytes','Dynamic smem bytes','Memory GB/s']
# by algorithm: normalize each metric across algorithms, draw metric index labels
max_by_metric={m:max(abs(num(r[m])) for r in metric_rows) or 1 for m in metrics}
groups=[]
for row in metric_rows:
    items=[]
    for i,m in enumerate(metrics):
        items.append((f'[{i+1}] {metric_names[i]}', num(row[m])/max_by_metric[m], COLORS[row['mode']]))
    groups.append((row['mode'], items))
svg_grouped(TREND/'reduction_ncu_metrics_by_algorithm.svg', f'Combined NCU Metrics by Algorithm ({config})', groups, normalize_within_group=False, raw_header='Normalized value')
# by metric: values normalized inside each metric group
metric_groups=[]
for i,m in enumerate(metrics):
    maxv=max(abs(num(r[m])) for r in metric_rows) or 1
    items=[(r['mode'], num(r[m])/maxv, COLORS[r['mode']]) for r in metric_rows]
    metric_groups.append((f'[{i+1}] {metric_names[i]}', items))
svg_grouped(TREND/'reduction_ncu_metrics_by_metric.svg', f'Combined NCU Metrics by Metric ({config})', metric_groups, normalize_within_group=False, raw_header='Normalized value')

# Stall top5 from raw dominant stage per mode.
def parse_raw_stalls(mode):
    path=NCU/f'reduction_{mode}_single_latest_raw.txt'
    stages=[]; cur=None
    for line in path.read_text(errors='replace').splitlines():
        if 'Context ' in line and ' Stream ' in line and line.startswith('  ') and 'Metric Name' not in line:
            if cur: stages.append(cur)
            cur={'duration':0.0,'stalls':{}}
            continue
        if not cur or not line.startswith('  '): continue
        parts=line.split();
        if not parts: continue
        name=parts[0]; val=num(' '.join(parts[-2:]) if parts[-2:]==['no','data'] else parts[-1], None)
        if name=='gpu__time_duration.sum': cur['duration']=val or 0.0
        pre='smsp__average_warps_issue_stalled_'; suf='_per_issue_active.ratio'
        if name.startswith(pre) and name.endswith(suf) and val is not None:
            cur['stalls'][name[len(pre):-len(suf)]]=val
    if cur: stages.append(cur)
    if not stages: return []
    dom=max(stages, key=lambda s:s['duration'])
    return sorted(dom['stalls'].items(), key=lambda kv:kv[1], reverse=True)[:5]
stall_rows=[]; stall_groups=[]
for mode in MODES:
    items=parse_raw_stalls(mode)
    for rank,(reason,value) in enumerate(items,1):
        stall_rows.append({'mode':mode,'rank':rank,'stall_reason':reason,'per_issue_active_ratio':value})
    stall_groups.append((mode, [(reason,value,COLORS[mode]) for reason,value in items]))
write_csv(TREND/'reduction_stall_top5_standard_latest.csv', stall_rows, ['mode','rank','stall_reason','per_issue_active_ratio'])
svg_grouped(TREND/'reduction_stall_top5_by_algorithm.svg', f'Combined Stall Reason Top 5 by Algorithm ({config})', stall_groups, normalize_within_group=True, raw_header='Real ratio')

(TREND/'reduction_standard_comparison_latest.md').write_text(f'''# Standard Comparison Plots

Config: `{config}`.

Generated from release benchmark CSV and NCU single-mode exports. Release benchmark timings feed best-user-vs-baseline and ratio plots. NCU replay metrics are diagnostic only. For multi-stage modes, `duration_us_total` sums stages; other NCU standard fields use the dominant-duration stage. Full stage detail remains in `reports/ncu/single_mode_summary_latest.csv`.

Best user: `{best['mode']}` at `{best['avg_ms']:.6f}` ms. CUB baseline: `{baseline['avg_ms']:.6f}` ms. Latency ratio user/CUB: `{best['avg_ms']/baseline['avg_ms']:.4f}`.

Files:

- `reports/trends/reduction_best_user_vs_baseline.svg`
- `reports/trends/reduction_best_user_ratio.svg`
- `reports/trends/reduction_full_series_heatmap.svg`
- `reports/trends/reduction_ncu_metrics_by_algorithm.svg`
- `reports/trends/reduction_ncu_metrics_by_metric.svg`
- `reports/trends/reduction_stall_top5_by_algorithm.svg`
- `reports/trends/reduction_best_user_vs_baseline_latest.csv`
- `reports/trends/reduction_ncu_metrics_standard_latest.csv`
- `reports/trends/reduction_stall_top5_standard_latest.csv`
''')
print('wrote standard comparison SVGs and CSVs')
