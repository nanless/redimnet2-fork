#!/usr/bin/env python3
"""核对两篇官方权重报告中的数字与仓库 CSV；不重新运行模型。"""
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def rows(name):
    with (ROOT / 'checkpoints' / name).open(newline='') as stream:
        return {row['checkpoint'].removesuffix('.pt'): row for row in csv.DictReader(stream)}


def check():
    original = rows('eval_vox1_O_all.csv')
    extended = rows('eval_vox1_EH_all.csv')
    outside = rows('eval_ood_all.csv')
    values, errors = [], []
    jobs = [('02-VoxCeleb1结果.md', original, extended), ('03-域外试验结果.md', outside, None)]
    for name, table, other in jobs:
        for number, line in enumerate((ROOT / 'checkpoints/reports' / name).read_text().splitlines(), 1):
            cells = [part.strip().replace('**', '').replace('−', '-') for part in line.strip('|').split('|')]
            if not cells or cells[0] not in table:
                continue
            key = cells[0]
            if name.startswith('02'):
                fields = [(table[key], 'Vox1-O'), (table[key], 'official-O'), (table[key], 'delta-O'),
                          (other[key], 'Vox1-E'), (other[key], 'official-E'), (other[key], 'delta-E'),
                          (other[key], 'Vox1-H'), (other[key], 'official-H'), (other[key], 'delta-H')]
                if len(cells) == 4:  # 官方未给主训练权重参照数字的表。
                    fields = [fields[index] for index in (0, 3, 6)]
            else:
                fields = [(table[key], item) for item in ('Vox1-B', 'official-Vox1-B', 'delta-Vox1-B',
                          'VOICES', 'official-VOICES', 'delta-VOICES',
                          'CN-Celeb', 'official-CN-Celeb', 'delta-CN-Celeb')]
            if len(cells) - 1 != len(fields):
                errors.append({'document': name, 'line': number, 'reason': '列数不同'})
                continue
            for cell, (source, field) in zip(cells[1:], fields):
                item = {'document': name, 'line': number, 'checkpoint': key, 'field': field,
                        'report': cell, 'csv': source[field]}
                values.append(item)
                if abs(float(cell) - float(source[field])) > 1e-12:
                    errors.append(item)
    return {'schema': 'redimnet2-report-table-check/v1',
            'scope': '两篇历史报告的逐型号表格与现存CSV数字；不证明音频、模型、协议或总体准确率',
            'checked_numeric_cells': len(values), 'errors': errors, 'values': values}


if __name__ == '__main__':
    result = check()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(2 if result['errors'] else 0)
