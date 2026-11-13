#!/usr/bin/env python3
"""Parse all EP/FSDP memory analysis reports and output structured JSON.
Fixed: comma-separated numbers, task3/task4 table detection."""
import json, re, sys
from pathlib import Path

def parse_table(lines, start_idx, heading_prefix=None):
    """Parse a markdown table, optionally starting after a heading.
    Returns (headers, rows, next_idx)."""
    idx = start_idx
    # If heading provided, find it first
    if heading_prefix:
        while idx < len(lines):
            if lines[idx].strip().startswith(heading_prefix):
                idx += 1
                break
            idx += 1
    # Find table header row
    while idx < len(lines) and '|' not in lines[idx]:
        idx += 1
    if idx >= len(lines):
        return None, [], idx
    header_line = lines[idx]
    # Skip separator
    sep_idx = idx + 1
    if sep_idx >= len(lines) or '---' not in lines[sep_idx]:
        return None, [], idx
    headers = [h.strip() for h in header_line.split('|')][1:-1]
    rows = []
    idx = sep_idx + 1
    while idx < len(lines) and lines[idx].strip().startswith('|'):
        cells = [c.strip() for c in lines[idx].split('|')][1:-1]
        rows.append(cells)
        idx += 1
    return headers, rows, idx

def parse_value(val_str):
    """Parse '25.00%' -> 25.0, '1,900' -> 1900.0, '+172.9 ✓' -> 172.9, etc."""
    val_str = val_str.strip()
    has_warning = '⚠️' in val_str
    has_check = '✓' in val_str
    # Remove parenthetical content like "25.00% (100%)" -> "25.00%"
    cleaned = re.sub(r'\s*\([^)]*\)', '', val_str)
    cleaned = cleaned.replace(',', '').replace('✓', '').replace('⚠️', '').replace('%', '').replace('+', '').strip()
    try:
        val = float(cleaned)
    except ValueError:
        val = None
    return val, has_warning, has_check

def parse_d2h_col(col_name):
    m = re.match(r'(\d+)G→(\d+)G', col_name)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)

def parse_ddr_col(col_name):
    m = re.match(r'DDR=(\d+)GB', col_name)
    return int(m.group(1)) if m else None

def parse_report(filepath):
    text = Path(filepath).read_text()
    lines = text.split('\n')
    results = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Model section: ## model_name (X.XXXT params)
        if line.startswith('## ') and 'params' in line and not line.startswith('###'):
            m = re.match(r'##\s+(\S+)\s*\(([\d.]+)T\s+params\)', line)
            if m:
                results.append({'model': m.group(1), 'params_t': float(m.group(2)),
                                'hbm_192': None, 'hbm_288': None})
            i += 1
            continue

        # HBM section
        hbm_match = re.match(r'###\s+HBM=(\d+)GB', line)
        if hbm_match and results:
            hbm_key = f'hbm_{hbm_match.group(1)}'
            base = i + 1

            # Find task 1/2 table
            t12_h, t12_rows, next_i = parse_table(lines, base, '#### 任务一')
            # Find task 3 table
            t3_h, t3_rows, next_i = parse_table(lines, next_i, '#### 任务三')
            # Find task 4 table
            t4_h, t4_rows, next_i = parse_table(lines, next_i, '#### 任务四')

            model_data = results[-1]

            # Parse Task 1/2
            t12 = {}
            if t12_rows:
                for row in t12_rows:
                    if len(row) < 13:
                        continue
                    ep = row[0].replace('EP=', '')
                    t12[ep] = {
                        'fsdp_k': parse_value(row[1])[0],
                        'ddr_gb': parse_value(row[8])[0],
                        'd2h_full_gbs': parse_value(row[12])[0],
                    }

            # Parse Task 3 (D2H sweep)
            t3 = {}
            if t3_h and t3_rows:
                d2h_cols = []
                for j, h in enumerate(t3_h):
                    raw_bw, eff_bw = parse_d2h_col(h)
                    if eff_bw:
                        d2h_cols.append((j, eff_bw))
                for row in t3_rows:
                    if len(row) < 2:
                        continue
                    ep = row[0].replace('EP=', '')
                    sweep = {}
                    for col_idx, eff_bw in d2h_cols:
                        if col_idx < len(row):
                            val, _, _ = parse_value(row[col_idx])
                            sweep[str(eff_bw)] = val
                    t3[ep] = sweep

            # Parse Task 4 (DDR sweep)
            t4 = {}
            if t4_h and t4_rows:
                ddr_cols = []
                for j, h in enumerate(t4_h):
                    ddr_gb = parse_ddr_col(h)
                    if ddr_gb:
                        ddr_cols.append((j, ddr_gb))
                for row in t4_rows:
                    if len(row) < 2:
                        continue
                    ep = row[0].replace('EP=', '')
                    sweep = {}
                    for col_idx, ddr_gb in ddr_cols:
                        if col_idx < len(row):
                            val, _, _ = parse_value(row[col_idx])
                            sweep[str(ddr_gb)] = val
                    t4[ep] = sweep

            model_data[hbm_key] = {'task12': t12, 'task3': t3, 'task4': t4}
            i = next_i
        else:
            i += 1

    return results

def main():
    base = Path('/home/shawn/Repos/ModelTrainerSim/reports_analyse')
    all_data = {}
    for f in sorted(base.glob('*_report.md')):
        family = f.stem.replace('_970DT_report', '')
        results = parse_report(f)
        all_data[family] = results
        print(f"Parsed {family}: {len(results)} models")
        for m in results:
            ep128_192 = m.get('hbm_192', {}).get('task12', {}).get('128', {})
            ep128_288 = m.get('hbm_288', {}).get('task12', {}).get('128', {})
            t3_192 = m.get('hbm_192', {}).get('task3', {}).get('128', {})
            t4_192 = m.get('hbm_192', {}).get('task4', {}).get('128', {})
            print(f"  {m['model']} ({m['params_t']}T): K={ep128_192.get('fsdp_k')}, DDR={ep128_192.get('ddr_gb')}GB, D2H={ep128_192.get('d2h_full_gbs')}GB/s")
            print(f"    Task3 D2H keys: {sorted(t3_192.keys()) if t3_192 else 'MISSING'}")
            print(f"    Task4 DDR keys: {sorted(t4_192.keys()) if t4_192 else 'MISSING'}")

    outpath = base / 'parsed_data.json'
    with open(outpath, 'w') as f:
        json.dump(all_data, f, indent=2)
    print(f"\nWritten to {outpath}")

if __name__ == '__main__':
    main()