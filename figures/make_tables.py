from pathlib import Path
import json
d=Path(__file__).resolve().parents[1]
rows=json.loads((d/'results/certificate_and_baseline_summary.json').read_text())
s=r"""\begin{table}[t]
\centering
\caption{Median coarse TV at $k=8$, with 512 rows per family. M4 and S6 add 24 bytes; Uniform adds none. S6 pools three sampling seeds.}
\label{tab:cache}
\begin{tabular}{lrrrr}
\toprule
Family & $T/T_0$ & M4 & S6 & Uniform\\
\midrule
"""
md='\n\n## Baselines and certificate size\n\n| Family | Ratio | Median M4 TV | Median S6 TV | Median uniform TV | Median partition bound | Median bound/error |\n|---|---:|---:|---:|---:|---:|---:|\n'
for r in rows:
    if r['top_k']!=8: continue
    family='Ordinary' if r['domain']=='ordinary' else 'Arithmetic'
    values=[r[k]['median'] for k in ('moment4_tail_tv','sample6_tail_tv','uniform_tail_tv')]
    s+=f"{family} & {r['temperature_ratio']:.1f} & {values[0]:.5f} & {values[1]:.3f} & {values[2]:.3f}"+r"\\"+'\n'
    md+=f"| {family} | {r['temperature_ratio']} | {values[0]:.7f} | {values[1]:.4f} | {values[2]:.4f} | {r['moment4_certificate_bound']['median']:.5f} | {r['moment4_bound_to_error_ratio']['median']:.2f} |\n"
s+=r"""\bottomrule
\end{tabular}
\end{table}
"""
(d/'results/table_cache_accuracy.tex').write_text(s,encoding='utf-8')
