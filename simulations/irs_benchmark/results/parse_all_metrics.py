"""Parse slurm output files and extract rmse_train, rmse_test, coverage."""
import re
import csv
import sys

def parse_slurm(filepath):
    with open(filepath) as f:
        lines = f.readlines()

    results = []
    i = 0
    while i < len(lines):
        m = re.match(
            r'=== Scenario (\d+) / \d+: (\w+) \| (\w+) \| '
            r'p=([0-9.]+) \| n=(\d+) ===',
            lines[i].strip()
        )
        if m:
            scenario_id = int(m.group(1))
            model = m.group(2)
            pattern = m.group(3)
            p_miss = float(m.group(4))
            n_train = int(m.group(5))

            i += 1
            # Skip blank lines
            while i < len(lines) and lines[i].strip() == '':
                i += 1

            # Check for failure
            if i < len(lines) and 'failed' in lines[i]:
                i += 1
                continue

            # Skip header line (method rmse_train rmse_test ...)
            if i < len(lines) and 'method' in lines[i]:
                i += 1

            # Read method rows
            while i < len(lines):
                line = lines[i].strip()
                if line == '' or line.startswith('===') or 'rmse_train_se' in line:
                    break

                parts = line.split()
                # Last 6 values are metrics
                n_parts = len(parts)
                metrics = parts[n_parts-6:]
                method = ' '.join(parts[:n_parts-6])

                results.append({
                    'scenario_id': scenario_id,
                    'model': model,
                    'pattern': pattern,
                    'p_miss': p_miss,
                    'n_train': n_train,
                    'method': method,
                    'rmse_train': float(metrics[0]),
                    'rmse_test': float(metrics[1]),
                    'bias_test': float(metrics[2]),
                    'mae_test': float(metrics[3]),
                    'coverage': float(metrics[4]),
                    'width': float(metrics[5]),
                })
                i += 1
        else:
            i += 1

    return results

results = []
results.extend(parse_slurm('slurm-20949129.out'))
results.extend(parse_slurm('slurm-20960424.out'))

# Write long-format CSV with all metrics
fields = ['model', 'pattern', 'p_miss', 'n_train', 'method',
          'rmse_train', 'rmse_test', 'coverage']
with open('preliminary_results_full.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    w.writeheader()
    for r in results:
        w.writerow(r)

print(f"Wrote {len(results)} rows to preliminary_results_full.csv")
