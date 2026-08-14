#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MODES = [
    "library_cub_device_reduce",
    "custom_cuda_atomic",
    "optimized_todo",
    "optimized_tile2",
    "optimized_tile4",
    "optimized_tile8",
    "optimized_2stage",
    "optimized_2stage_tile",
]
DEFAULT_N = 1 << 24
STAGE_ELEMS = {
    "library_cub_device_reduce": [DEFAULT_N],
    "custom_cuda_atomic": [DEFAULT_N],
    "optimized_todo": [DEFAULT_N],
    "optimized_tile2": [DEFAULT_N],
    "optimized_tile4": [DEFAULT_N],
    "optimized_tile8": [DEFAULT_N],
    "optimized_2stage": [DEFAULT_N, (DEFAULT_N + 255) // 256, 256],
    "optimized_2stage_tile": [DEFAULT_N, (DEFAULT_N + 2047) // 2048, 32],
}


def fonts():
    try:
        return (
            ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19),
            ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14),
            ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28),
        )
    except Exception:
        return None, None, None

FONT, SMALL, TITLE = fonts()


def nums(line: str):
    return re.findall(r"[-+]?\d*\.\d+|\d+", line)


def metric_value(line: str):
    found = nums(line)
    return float(found[-1]) if found else None


def parse_benchmark():
    rows = []
    path = REPORTS / "benchmark" / "latest.txt"
    if not path.exists():
        return rows
    pat = re.compile(r"^(\w+): n=(\d+) avg_ms=([0-9.]+) p50_ms=([0-9.]+) p95_ms=([0-9.]+) max_ms=([0-9.]+) effective_GBps=([0-9.]+)")
    for line in path.read_text().splitlines():
        m = pat.match(line.strip())
        if m:
            rows.append({
                "mode": m.group(1), "n": int(m.group(2)), "avg_ms": float(m.group(3)),
                "p50_ms": float(m.group(4)), "p95_ms": float(m.group(5)),
                "max_ms": float(m.group(6)), "effective_gbps": float(m.group(7)),
            })
    return rows


def parse_ncu_details(mode: str):
    path = REPORTS / "ncu" / f"reduction_{mode}_single_latest_details.txt"
    if not path.exists():
        return []
    rows = []
    current = None
    stage_idx = -1
    header_re = re.compile(r"^\s{2}(.+) \(([^()]*)\)x\(([^()]*)\), Context")
    for line in path.read_text(errors="ignore").splitlines():
        h = header_re.match(line)
        if h:
            if current:
                rows.append(current)
            stage_idx += 1
            current = {"mode": mode, "stage_index": stage_idx, "kernel": h.group(1).strip(), "grid": h.group(2), "block": h.group(3)}
            continue
        if not current:
            continue
        if "Memory Throughput" in line and "%" in line and "memory_pct" not in current:
            current["memory_pct"] = metric_value(line)
        elif "DRAM Throughput" in line and "%" in line and "dram_pct" not in current:
            current["dram_pct"] = metric_value(line)
        elif "Compute (SM) Throughput" in line and "%" in line and "sm_pct" not in current:
            current["sm_pct"] = metric_value(line)
        elif "Duration" in line and "duration_us" not in current:
            value = metric_value(line)
            if value is not None:
                current["duration_us"] = value * 1000.0 if " ms" in line else value
        elif "Memory Throughput" in line and "Gbyte/s" in line and "memory_gbps" not in current:
            current["memory_gbps"] = metric_value(line)
        elif "Eligible Warps Per Scheduler" in line and "eligible_warps" not in current:
            current["eligible_warps"] = metric_value(line)
        elif "Warp Cycles Per Issued Instruction" in line and "warp_cycles_per_issued" not in current:
            current["warp_cycles_per_issued"] = metric_value(line)
        elif "Registers Per Thread" in line and "registers_per_thread" not in current:
            current["registers_per_thread"] = metric_value(line)
        elif "Static Shared Memory Per Block" in line and "static_smem_bytes" not in current:
            current["static_smem_bytes"] = metric_value(line)
        elif "Dynamic Shared Memory Per Block" in line and "dynamic_smem_bytes" not in current:
            current["dynamic_smem_bytes"] = metric_value(line)
        elif "Achieved Occupancy" in line and "%" in line and "achieved_occupancy_pct" not in current:
            current["achieved_occupancy_pct"] = metric_value(line)
        elif "spends" in line and "cycles being stalled" in line and "stall_cycles_hint" not in current:
            m = re.search(r"spends ([0-9.]+) cycles being stalled", line)
            if m:
                current["stall_cycles_hint"] = float(m.group(1))
                current["stall_reason_hint"] = "long_scoreboard/L1TEX dependency"
    if current:
        rows.append(current)
    elems = STAGE_ELEMS.get(mode, [])
    for row in rows:
        idx = row["stage_index"]
        approx_flops = max(1, (elems[idx] - 1) if idx < len(elems) else DEFAULT_N - 1)
        row["approx_flops"] = approx_flops
        dur = row.get("duration_us") or 0
        mem = row.get("memory_gbps") or 0
        row["measured_bytes"] = mem * 1e9 * dur * 1e-6 if dur and mem else 0
        row["estimated_ai"] = approx_flops / row["measured_bytes"] if row["measured_bytes"] else 0
        row["estimated_gflops"] = approx_flops / (dur * 1e-6) / 1e9 if dur else 0
    return rows


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def draw_line_chart(path, title, xs, series, ylabel):
    path.parent.mkdir(parents=True, exist_ok=True)
    W, H = 1500, 900
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    left, top, right, bottom = 120, 95, 1430, 760
    d.text((left, 35), title, fill=(0, 0, 0), font=TITLE)
    d.rectangle([left, top, right, bottom], outline=(30,30,30), width=2)
    values = [v for _, vals, _ in series for v in vals if v is not None]
    ymax = max(values) * 1.15 if values else 1.0
    ymin = 0.0
    for i in range(6):
        yv = ymin + (ymax-ymin)*i/5
        yy = bottom - (yv-ymin)/(ymax-ymin)*(bottom-top)
        d.line([left, yy, right, yy], fill=(225,225,225))
        d.text((40, yy-8), f"{yv:.3g}", fill=(60,60,60), font=SMALL)
    n = max(1, len(xs)-1)
    xcoords = [left + i/n*(right-left) for i in range(len(xs))]
    for x, label in zip(xcoords, xs):
        d.line([x, top, x, bottom], fill=(238,238,238))
        d.text((x-45, bottom+18), label, fill=(45,45,45), font=SMALL)
    colors = [(40,90,190), (220,90,40), (40,150,80), (140,80,180), (120,120,120)]
    for si, (name, vals, style) in enumerate(series):
        col = colors[si % len(colors)]
        pts = []
        for x, v in zip(xcoords, vals):
            if v is None: continue
            y = bottom - (v-ymin)/(ymax-ymin)*(bottom-top)
            pts.append((x,y))
            d.ellipse([x-5,y-5,x+5,y+5], fill=col)
        if len(pts) > 1:
            d.line(pts, fill=col, width=3)
        d.text((left + 20 + si*260, top + 15), name, fill=col, font=SMALL)
    d.text((25, (top+bottom)//2), ylabel, fill=(0,0,0), font=FONT)
    img.save(path)


def draw_roofline(path, rows, global_view=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r for r in rows if r.get("estimated_ai", 0) > 0 and r.get("estimated_gflops", 0) > 0]
    if not rows: return
    peak_sources = [r["memory_gbps"] / (r["memory_pct"] / 100.0) for r in rows if r.get("memory_pct") and r.get("memory_gbps")]
    mem_peak = sorted(peak_sources)[len(peak_sources)//2] if peak_sources else max(r["memory_gbps"] for r in rows)
    compute_peak = 30000.0
    W, H = 1800, 1200
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    left, top, right, bottom = 140, 130, 1280, 980
    if global_view:
        knee = compute_peak / mem_peak
        xmin = max(1e-4, min(r["estimated_ai"] for r in rows) / 8)
        xmax = max(knee * 2.5, max(r["estimated_ai"] for r in rows) * 2)
        ymin, ymax = 0.02, compute_peak * 1.4
        title = "Reduction FLOP Roofline - Global View (estimated FLOPs)"
    else:
        xmin = max(1e-4, min(r["estimated_ai"] for r in rows) / 2)
        xmax = max(r["estimated_ai"] for r in rows) * 2
        ymin = max(0.01, min(r["estimated_gflops"] for r in rows) / 4)
        ymax = max(r["estimated_gflops"] for r in rows) * 2.2
        title = "Reduction FLOP Roofline - Focus View (estimated FLOPs)"
    def lx(x): return left + (math.log10(x)-math.log10(xmin))/(math.log10(xmax)-math.log10(xmin))*(right-left)
    def ly(y): return bottom - (math.log10(y)-math.log10(ymin))/(math.log10(ymax)-math.log10(ymin))*(bottom-top)
    d.text((left, 38), title, fill=(0,0,0), font=TITLE)
    d.text((left, 84), "NCU measured duration/bandwidth; reduction add counts are estimated. Compute roof is visual estimate when not exported.", fill=(45,45,45), font=SMALL)
    d.rectangle([left,top,right,bottom], outline=(20,20,20), width=2)
    for i in range(8):
        x = 10 ** (math.log10(xmin) + i/7*(math.log10(xmax)-math.log10(xmin)))
        xx = lx(x); d.line([xx,top,xx,bottom], fill=(232,232,232)); d.text((xx-35,bottom+12), f"{x:.3g}", fill=(55,55,55), font=SMALL)
    for i in range(9):
        y = 10 ** (math.log10(ymin) + i/8*(math.log10(ymax)-math.log10(ymin)))
        yy = ly(y); d.line([left,yy,right,yy], fill=(232,232,232)); d.text((45,yy-8), f"{y:.3g}", fill=(55,55,55), font=SMALL)
    pts = []
    for x in [xmin, xmax]:
        pts.append((lx(x), ly(max(ymin, min(mem_peak*x, compute_peak)))))
    d.line(pts, fill=(30,100,210), width=4)
    d.text((left+30, top+20), f"Memory roof ~= {mem_peak:.0f} GB/s (NCU SOL estimate)", fill=(30,100,210), font=SMALL)
    if global_view:
        cy = ly(compute_peak)
        for x0 in range(left, right, 28): d.line([x0,cy,x0+16,cy], fill=(180,70,70), width=3)
        d.text((right-440, cy-28), "FP32 roof estimated/visual", fill=(180,70,70), font=SMALL)
    colors = {"optimized_tile8": (0,140,70), "optimized_2stage_tile": (220,90,35), "library_cub_device_reduce": (30,120,180), "custom_cuda_atomic": (160,40,40)}
    for idx, r in enumerate(rows, 1):
        x, y = lx(r["estimated_ai"]), ly(r["estimated_gflops"])
        col = colors.get(r["mode"], (70,80,190) if r["stage_index"] == 0 else (120,120,120))
        d.ellipse([x-7,y-7,x+7,y+7], fill=col, outline=(0,0,0))
        label = f"{idx}. {r['mode']}[{r['stage_index']}] {r['duration_us']:.2f}us {r['memory_gbps']:.0f}GB/s"
        if not global_view or r["mode"] in ("optimized_tile8", "optimized_2stage_tile", "library_cub_device_reduce", "custom_cuda_atomic"):
            d.text((x+10, y-10), label, fill=col, font=SMALL)
    d.text(((left+right)//2-190,H-65), "Estimated arithmetic intensity (FLOP/byte, log)", fill=(0,0,0), font=FONT)
    d.text((20,500), "Estimated GFLOP/s\n(log)", fill=(0,0,0), font=FONT)
    img.save(path)


def main():
    bench = parse_benchmark()
    ncu = []
    for mode in MODES:
        ncu.extend(parse_ncu_details(mode))
    write_csv(REPORTS / "benchmark" / "summary_latest.csv", bench, ["mode","n","avg_ms","p50_ms","p95_ms","max_ms","effective_gbps"])
    ncu_fields = ["mode","stage_index","kernel","grid","block","duration_us","memory_gbps","memory_pct","dram_pct","sm_pct","achieved_occupancy_pct","eligible_warps","warp_cycles_per_issued","registers_per_thread","static_smem_bytes","dynamic_smem_bytes","stall_reason_hint","stall_cycles_hint","approx_flops","measured_bytes","estimated_ai","estimated_gflops"]
    write_csv(REPORTS / "ncu" / "single_mode_summary_latest.csv", ncu, ncu_fields)
    xs = [r["mode"].replace("optimized_", "opt_") for r in bench]
    draw_line_chart(REPORTS / "trends" / "reduction_benchmark_runtime.png", "Release Benchmark Runtime by Mode", xs, [("avg_ms", [r["avg_ms"] for r in bench], "line")], "ms")
    draw_line_chart(REPORTS / "trends" / "reduction_benchmark_throughput.png", "Release Benchmark Effective Throughput by Mode", xs, [("GB/s", [r["effective_gbps"] for r in bench], "line")], "GB/s")
    first_stage = [next((x for x in ncu if x["mode"] == r["mode"] and x["stage_index"] == 0), None) for r in bench]
    draw_line_chart(REPORTS / "trends" / "reduction_ncu_metrics.png", "NCU First-Stage Metrics by Mode", xs, [
        ("duration_us", [r.get("duration_us") if r else None for r in first_stage], "line"),
        ("memory_GBps", [r.get("memory_gbps") if r else None for r in first_stage], "line"),
        ("occupancy_pct", [r.get("achieved_occupancy_pct") if r else None for r in first_stage], "line"),
    ], "mixed units")
    draw_line_chart(REPORTS / "trends" / "reduction_ncu_stall_top5.png", "NCU Stall Hint: Long Scoreboard/L1TEX by Mode", xs, [("stall cycles", [r.get("stall_cycles_hint") if r else None for r in first_stage], "line")], "cycles")
    draw_roofline(REPORTS / "roofline" / "reduction_latest.png", ncu, global_view=False)
    draw_roofline(REPORTS / "roofline" / "reduction_global_latest.png", ncu, global_view=True)
    with (REPORTS / "roofline" / "reduction_latest.md").open("w") as f:
        f.write("# Reduction Roofline Notes\n\n")
        f.write("Type: conventional FLOP roofline for reduction kernels, with FLOP count estimated as reduction add count.\n\n")
        f.write("- Focus PNG: `reports/roofline/reduction_latest.png`\n")
        f.write("- Global PNG: `reports/roofline/reduction_global_latest.png`\n")
        f.write("- NCU summary CSV: `reports/ncu/single_mode_summary_latest.csv`\n")
        f.write("- Benchmark summary CSV: `reports/benchmark/summary_latest.csv`\n\n")
        f.write("The global view includes the estimated bandwidth slope and an estimated compute ceiling. The focus view expands the measured points.\n")
        f.write("Do not interpret estimated GFLOP/s as exact SASS FLOP accounting; use it to locate these reductions relative to the bandwidth roof.\n")
    with (REPORTS / "trends" / "reduction_trends_latest.md").open("w") as f:
        best = min(bench, key=lambda r: r["avg_ms"]) if bench else None
        f.write("# Reduction Trend Notes\n\n")
        if best:
            f.write(f"Best release benchmark mode: `{best['mode']}` at `{best['avg_ms']:.6f}` ms and `{best['effective_gbps']:.3f}` effective GB/s.\n\n")
        f.write("Generated PNGs:\n\n")
        f.write("- `reports/trends/reduction_benchmark_runtime.png`\n")
        f.write("- `reports/trends/reduction_benchmark_throughput.png`\n")
        f.write("- `reports/trends/reduction_ncu_metrics.png`\n")
        f.write("- `reports/trends/reduction_ncu_stall_top5.png`\n")
    print("wrote reports/benchmark/summary_latest.csv")
    print("wrote reports/ncu/single_mode_summary_latest.csv")
    print("wrote reports/roofline/reduction_latest.png")
    print("wrote reports/roofline/reduction_global_latest.png")
    print("wrote reports/trends/*.png")

if __name__ == "__main__":
    main()
