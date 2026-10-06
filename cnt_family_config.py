#!/usr/bin/env python3
"""计算固定 F=2*m+n 家族的理想 CNT 几何，并输出 CHIRAL_CONFIGS。

约定：配置手性为 (m,n)，m>=n>=0；UC 是最小轴向平移晶胞。
l_def 为散射区的 UC 数；散射区长度 = l_def * UC长度，不含电极。
示例：
  python cnt_family_config.py 22 > chiral_configs.py
无第三方依赖；表格输出到 stderr，Python 配置输出到 stdout。
"""
import argparse
import math
import sys

# 保留输入中的有效组合；取消下面的注释即可启用更多长度。
LENGTH_DEFECT_PAIRS = [
    (3, 1),
    (6, 2),
    (9, 3),
    (12, 4),
    (15, 5),
    (18, 6),
    (21, 7),
    (24, 8),
    (30, 10),
    (36, 12),
    (42, 14),
    (48, 16),
    (54, 18),
    (60, 20),
    (66, 22),
]

def uc_geometry(m, n, bond_angstrom=1.42):
    """返回 (最小 UC 长度/Å, UC 碳原子数)。"""
    d_r = math.gcd(2 * m + n, 2 * n + m)
    q = m * m + m * n + n * n
    return 3 * bond_angstrom * math.sqrt(q) / d_r, 4 * q // d_r


def replicas_for(length_nm):
    if length_nm < 6:
        return 20
    if length_nm <= 10:
        return 40
    return 60


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('F', type=int, help='手性家族 F=2*m+n')
    parser.add_argument('--bond', type=float, default=1.42, help='C-C 键长/Å，默认 1.42')
    args = parser.parse_args()
    if args.F < 2:
        parser.error('F 必须为不小于 2 的整数')
    if not math.isfinite(args.bond) or args.bond <= 0:
        parser.error('bond 必须为有限正数')
    for l_def, n_defects in LENGTH_DEFECT_PAIRS:
        if not isinstance(l_def, int) or l_def <= 0:
            parser.error('l_def 必须为正整数 UC 数')
        if not isinstance(n_defects, int) or n_defects < 0:
            parser.error('N_defects 必须为非负整数')

    # 手性组合在生成脚本中确定；输出配置中 m、n 均为固定数值。
    chiralities = [(m, args.F - 2*m)
                   for m in range(args.F // 2, (args.F + 2) // 3 - 1, -1)]
    print(f'# F = 2*m+n = {args.F}; 散射区长度 = l_def × UC长度')
    print('CHIRAL_CONFIGS = [')
    for m, n in chiralities:
        uc_a, atoms = uc_geometry(m, n, args.bond)
        print(f'    # ({m}, {n}): UC = {uc_a:.6f} Å, {atoms} C/UC')
        print('    *[')
        print('        {')
        print(f'            "m": {m},')
        print(f'            "n": {n},')
        print('            "l_def": l_def,')
        print('            "N_defects": N_defects,')
        print('            "structures": ["MVH"],')
        print('            "replicas": replicas,')
        print('            "conductance_mode": "band_edge_bias",')
        print('        }')
        print('        for l_def, N_defects, replicas in [')
        print('            # UC数, 缺陷数, replicas    散射区长度')
        print(f'({m}, {n}): UC={uc_a:.6f} Å, {atoms} C/UC', file=sys.stderr)
        print(' l_def  N_defects  L_scatter(nm)  density(Å^-1)  replicas', file=sys.stderr)
        for l_def, n_defects in LENGTH_DEFECT_PAIRS:
            length_nm = l_def * uc_a / 10
            replicas = replicas_for(length_nm)
            density = n_defects / (l_def * uc_a)
            print(f'            ({l_def:2}, {n_defects:2}, {replicas:2}),  # {length_nm:.3f} nm')
            print(f'{l_def:6} {n_defects:10} {length_nm:14.3f} {density:14.6f} {replicas:9}',
                  file=sys.stderr)
        print('        ]')
        print('    ],')
    print(']')


if __name__ == '__main__':
    main()
