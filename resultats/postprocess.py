import math
import os
from pathlib import Path
import glob
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import to_rgb

COLOR_PALETTE = cm.tab10.colors
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=COLOR_PALETTE)


ROOT = Path(__file__).resolve().parent


def lighten_color(color, amount=0.5):
    r, g, b = to_rgb(color)
    r = 1 - amount * (1 - r)
    g = 1 - amount * (1 - g)
    b = 1 - amount * (1 - b)
    return (r, g, b)


def read_keyvalue_csv(path):
    df = pd.read_csv(path, index_col=0)
    if 'Value' in df.columns:
        vals = df['Value'].to_dict()
    else:
        vals = df.iloc[:, 0].to_dict()
    for k, v in list(vals.items()):
        try:
            vals[k] = float(v)
        except Exception:
            vals[k] = v
    return vals


def find_series_csvs(base_dirs=('.',)):
    files = []
    for base in base_dirs:
        for p in Path(base).rglob('*.csv'):
            try:
                df = pd.read_csv(p, nrows=5)
            except Exception:
                continue
            cols = [c.lower() for c in df.columns]
            if any(c in cols for c in ('h', 'mesh_size', 'lc')) and any(('err' in c or 'dw' in c or 'error' in c) for c in cols):
                files.append(p)
    return files


def plot_validation_convergence(out_dir=None):
    out_dir = Path(out_dir or ROOT)
    val_dir = out_dir / 'validation'
    tests = {
        'traction': ('convergence_traction.csv', 'DeltaW_rel'),
        'flexion': ('convergence_flexion.csv', 'DeltaW_rel'),
        'cisaillement': ('convergence_shear.csv', 'DeltaW_rel'),
    }
    summary_lines = []
    fig, ax = plt.subplots(figsize=(8, 6))
    color_i = 0

    for test_name, (fname, err_col) in tests.items():
        fpath = val_dir / fname
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath)
        h_col = 'h'
        if h_col not in df.columns or err_col not in df.columns:
            continue

        # valeur aberrante
        if 'flexion' in fname:
            df = df.iloc[1:].reset_index(drop=True)

        df = df.dropna(subset=[h_col, err_col])
        h = df[h_col].astype(float).values
        err = df[err_col].astype(float).values
        order = np.argsort(h)
        h = h[order]
        err = err[order]

        valid = (h > 0) & (err > 0)
        if valid.sum() < 2:
            continue
        slope, intercept = np.polyfit(np.log(h[valid]), np.log(err[valid]), 1)

        color = COLOR_PALETTE[color_i % len(COLOR_PALETTE)]
        color_i += 1
        label = f"{test_name} (ordre={slope:.2f})"
        ax.loglog(h, err, 'o-', color=color, label=label, linewidth=2, markersize=6)

        hs = np.linspace(h.min(), h.max(), 100)
        pred = np.exp(intercept) * hs ** slope
        ax.loglog(hs, pred, linestyle='--', color=color, alpha=0.4, linewidth=1.5)

        summary_lines.append(f"Fichier: {fpath.name}\nTest: {test_name}\nPoints: {len(h)}\nPente (log-log): {slope:.3f}\nOrdonnée: {intercept:.3e}\n")
        tbl = '\n'.join([f"{hv:.6g}\t{ev:.6g}" for hv, ev in zip(h, err)])
        summary_lines.append("h\terreur\n" + tbl + "\n---\n")

    if len(ax.lines) == 0:
        print('Pas de séries de validation trouvées dans resultats/validation/')
        return

    ax.set_xlabel('h (taille caractéristique de maille)', fontsize=12)
    ax.set_ylabel('erreur relative', fontsize=12)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, which='both', ls='--', alpha=0.3)
    fig_path = out_dir / 'validation_convergence.png'
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)

    summary_path = out_dir / 'validation_summary.txt'
    with open(summary_path, 'w') as fh:
        fh.write('\n'.join(summary_lines))

    print(f'Écrit graphique de validation -> {fig_path} et résumé -> {summary_path}')


def mesh_convergence_tables_and_plots(out_dir=None):
    out_dir = Path(out_dir or ROOT)
    csvs = list(Path(out_dir).rglob('*.csv'))
    groups = {}
    for p in csvs:
        name = p.stem
        import re
        m = re.match(r'(.+)_([0-9]+)$', name)
        if m:
            base = m.group(1)
        else:
            base = name
        groups.setdefault(base, []).append(p)

    for base, files in groups.items():
        items = []
        for f in files:
            try:
                vals = read_keyvalue_csv(f)
                if 'h' in vals:
                    items.append((f, vals))
            except Exception:
                continue
        if len(items) < 2:
            continue
        # build dataframe
        rows = []
        for f, vals in items:
            row = {
                'file': f.name,
                'h': float(vals.get('h', math.nan)),
                'nelems': int(vals.get('n_elems', vals.get('n_elems', 0))) if vals.get('n_elems', None) is not None else int(vals.get('n_elems', 0)),
                'E1': float(vals.get('E1', math.nan)),
                'E2': float(vals.get('E2', math.nan)),
                'v12': float(vals.get('v12', math.nan)),
                'v21': float(vals.get('v21', math.nan)),
                'G12': float(vals.get('G12', math.nan)),
                'tcpumax': float(vals.get('tcpumax', math.nan)),
            }
            rows.append(row)
        df = pd.DataFrame(rows).sort_values('h')

        ref = df.iloc[0]
        norm = df[['h', 'E1', 'E2', 'G12', 'v12', 'v21']].copy()
        norm[['E1', 'E2', 'G12', 'v12', 'v21']] = norm[['E1', 'E2', 'G12', 'v12', 'v21']].div(ref[['E1', 'E2', 'G12', 'v12', 'v21']].values)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(norm['h'], norm['E1'], '-o', label=r'$E_{11}$ (norm)', linewidth=2, markersize=6, color=COLOR_PALETTE[0])
        ax.plot(norm['h'], norm['E2'], '-o', label=r'$E_{22}$ (norm)', linewidth=2, markersize=6, color=COLOR_PALETTE[1])
        ax.plot(norm['h'], norm['G12'], '-o', label=r'$G_{12}$ (norm)', linewidth=2, markersize=6, color=COLOR_PALETTE[2])
        ax.plot(norm['h'], norm['v12'], '-o', label=r'$\nu_{12}$ (norm)', linewidth=2, markersize=6, color=COLOR_PALETTE[3])
        ax.plot(norm['h'], norm['v21'], '-o', label=r'$\nu_{21}$ (norm)', linewidth=2, markersize=6, color=COLOR_PALETTE[4])
        ax.set_xscale('log')
        ax.set_xlabel('h (taille de maille)', fontsize=12)
        ax.set_ylabel('valeur normalisée', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, which='both', ls='--', alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f'conv_maillage_{base}.png', dpi=200)


def etude_vf_plots(out_dir=None):
    out_dir = Path(out_dir or ROOT) / 'etude_vf'
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(out_dir.glob('*.csv'))
    if not files:
        print('No CSVs found in etude_vf/')
        return
    rows = []
    for f in files:
        vals = read_keyvalue_csv(f)
        vf = float(vals.get('V_fiber', vals.get('V_f', vals.get('vf', math.nan))))
        E1 = float(vals.get('E1', math.nan))
        E2 = float(vals.get('E2', E1))
        G12 = float(vals.get('G12', math.nan))
        nu12 = float(vals.get('v12', vals.get('nu12', math.nan)))
        nu21 = float(vals.get('v21', vals.get('nu21', math.nan)))
        rows.append({'file': f.name, 'vf': vf, 'E1': E1, 'E2': E2, 'G12': G12, 'v12': nu12, 'v21': nu21})
    df = pd.DataFrame(rows).sort_values('vf')

    #On fait la moyenne entre E1 et E2 pour ET, pareil pour v12 et v21 pour vTT

    df['ET'] = (df['E1'] + df['E2']) / 2
    df['nuTT'] = (df['v12'] + df['v21']) / 2
    comp_cfg = Path(__file__).resolve().parents[1] / 'FEM' / 'config' / 'composite.txt'
    E_m = 12e9
    E_f = 34e9
    nu_m = 0.3
    nu_f = 0.25
    if comp_cfg.exists():
        for line in comp_cfg.read_text().splitlines():
            if '=' in line:
                k, v = line.split('=', 1)
                k = k.strip()
                v = v.split('#', 1)[0].strip()
                try:
                    if k == 'E':
                        E_m = float(v)
                    if k == 'E_fiber':
                        E_f = float(v)
                    if k == 'nu':
                        nu_m = float(v)
                    if k == 'nu_fiber':
                        nu_f = float(v)
                except Exception:
                    pass
    if E_m is None or E_f is None:
        print('Warning: could not read E (matrix) and E_fiber from config; theoretical bounds will be skipped')

    vf_vals = np.linspace(0.0, 0.7, 200)

    def voigt_E(vf):
        return vf * E_f + (1 - vf) * E_m

    def reuss_E(vf):
        return 1.0 / (vf / E_f + (1 - vf) / E_m)

    def hill_E(vf):
        return 0.5 * (voigt_E(vf) + reuss_E(vf))

    def halpin_tsai_E(vf, xi=1.0):
        r = E_f / E_m
        eta = (r - 1) / (r + xi)
        return E_m * (1 + xi * eta * vf) / (1 - eta * vf)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df['vf'], df['ET']/1e9, 'o--', label=r'$E_{T}$ (données)', linewidth=1.5, markersize=7, color="black", zorder=10)
    if E_m is not None and E_f is not None:
        ax.plot(vf_vals, voigt_E(vf_vals)/1e9, '-', color=COLOR_PALETTE[1], label='Voigt', linewidth=2)
        ax.plot(vf_vals, reuss_E(vf_vals)/1e9, '-', color=COLOR_PALETTE[2], label='Reuss', linewidth=2)
        ax.plot(vf_vals, hill_E(vf_vals)/1e9, '-', color=COLOR_PALETTE[3], label='Hill', linewidth=2)
        ax.plot(vf_vals, halpin_tsai_E(vf_vals, xi=0.6)/1e9, '-', color=COLOR_PALETTE[4], label=r'Halpin-Tsai ($\xi=0.6$)', linewidth=2)
    ax.set_xlim(0.2, 0.6)

    y_data = df['ET']/1e9
    y_min_data = y_data.min()
    y_max_data = y_data.max()
    y_margin = 2.0
    ax.set_ylim(y_min_data - y_margin, y_max_data + y_margin)
    ax.set_xlabel('Fraction volumique de fibre', fontsize=12)
    ax.set_ylabel(r'$E_T$ (GPa)', fontsize=12)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, ls='--', alpha=0.3)
    fig.tight_layout()
    fig.savefig('etude_vf_E.png', dpi=200)

    G_m = None
    G_f = None
    if E_m is not None and nu_m is not None:
        G_m = E_m / (2 * (1 + nu_m))
    if E_f is not None and nu_f is not None:
        G_f = E_f / (2 * (1 + nu_f))

    def voigt_G(vf):
        return vf * G_f + (1 - vf) * G_m

    def reuss_G(vf):
        return 1.0 / (vf / G_f + (1 - vf) / G_m)

    def hill_G(vf):
        return 0.5 * (voigt_G(vf) + reuss_G(vf))

    def halpin_tsai_G(vf, xi=1.0):
        r = G_f / G_m
        eta = (r - 1) / (r + xi)
        return G_m * (1 + xi * eta * vf) / (1 - eta * vf)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df['vf'], df['G12']/1e9, 'o--', label=r'$G_{LT}$ (données)', linewidth=1.5, markersize=7, color="black", zorder=10)
    if G_m is not None and G_f is not None:
        ax.plot(vf_vals, voigt_G(vf_vals)/1e9, '-', color=COLOR_PALETTE[1], label='Voigt (G)', linewidth=2)
        ax.plot(vf_vals, reuss_G(vf_vals)/1e9, '-', color=COLOR_PALETTE[2], label='Reuss (G)', linewidth=2)
        ax.plot(vf_vals, hill_G(vf_vals)/1e9, '-', color=COLOR_PALETTE[3], label='Hill (G)', linewidth=2)
        ax.plot(vf_vals, halpin_tsai_G(vf_vals, xi=0.8)/1e9, '-', color=COLOR_PALETTE[4], label=r'Halpin-Tsai ($\xi=0.8$)', linewidth=2)
    ax.set_xlim(0.2, 0.6)
    y_data = df['G12']/1e9
    y_min_data = y_data.min()
    y_max_data = y_data.max()
    y_margin = 2.0
    ax.set_ylim(y_min_data - y_margin, y_max_data + y_margin)
    ax.set_xlabel('Fraction volumique de fibre', fontsize=12)
    ax.set_ylabel(r'$G_{LT}$ (GPa)', fontsize=12)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, ls='--', alpha=0.3)
    fig.tight_layout()
    fig.savefig('etude_vf_G.png', dpi=200)

    print(f'Graphiques etude_vf écrits dans {out_dir}')

    def voigt_nu(vf):
        return vf * nu_f + (1 - vf) * nu_m
    
    def reuss_nu(vf):
        return 1.0 / (vf / nu_f + (1 - vf) / nu_m)
    
    def hill_nu(vf):
        return 0.5 * (voigt_nu(vf) + reuss_nu(vf))
    
    def halpin_tsai_nu(vf, xi=1.0):
        r = nu_f / nu_m
        eta = (r - 1) / (r + xi)
        return nu_m * (1 + xi * eta * vf) / (1 - eta * vf)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df['vf'], df['nuTT'], 'o--', label=r'$\nu_{TT}$ (données)', linewidth=1.5, markersize=7, color="black", zorder=10)
    if nu_m is not None and nu_f is not None:
        ax.plot(vf_vals, voigt_nu(vf_vals), '-', color=COLOR_PALETTE[1], label='Voigt (nu)', linewidth=2)
        ax.plot(vf_vals, reuss_nu(vf_vals), '-', color=COLOR_PALETTE[2], label='Reuss (nu)', linewidth=2)
        ax.plot(vf_vals, hill_nu(vf_vals), '-', color=COLOR_PALETTE[3], label='Hill (nu)', linewidth=2)
        ax.plot(vf_vals, halpin_tsai_nu(vf_vals, xi=0.8), '-', color=COLOR_PALETTE[4], label=r'Halpin-Tsai ($\xi=0.8$)', linewidth=2)
    ax.set_xlim(0.2, 0.6)
    y_data = df['nuTT']
    y_min_data = y_data.min()
    y_max_data = y_data.max()
    y_margin = 0.01
    ax.set_ylim(max(0, y_min_data - y_margin), min(1, y_max_data + y_margin))
    ax.set_xlabel('Fraction volumique de fibre', fontsize=12)
    ax.set_ylabel(r'$\nu_{TT}$', fontsize=12)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, ls='--', alpha=0.3)
    fig.tight_layout()
    fig.savefig('etude_vf_nu.png', dpi=200) 
    # --- Metrics comparing data to theoretical bounds ---
    def compute_metrics(data_vals, model_vals):
        data = np.array(data_vals, dtype=float)
        model = np.array(model_vals, dtype=float)
        eps = 1e-30
        rel = (data - model) / np.maximum(np.abs(model), eps)
        mae = np.mean(np.abs(rel))
        rmse = np.sqrt(np.mean(rel**2))
        mean_rel = np.mean(rel)
        std_rel = np.std(rel)
        max_abs = np.max(np.abs(rel))
        return {'mean_rel': mean_rel, 'std_rel': std_rel, 'mae': mae, 'rmse': rmse, 'max_abs': max_abs}

    # Prepare arrays
    vf_data = df['vf'].values
    ET_data = df['ET'].values
    G_data = df['G12'].values

    # Models evaluated at data vf points
    E_voigt_at = voigt_E(vf_data)
    E_reuss_at = reuss_E(vf_data)
    E_hill_at = hill_E(vf_data)
    E_halpin_at = halpin_tsai_E(vf_data, xi=0.6)

    G_voigt_at = voigt_G(vf_data)
    G_reuss_at = reuss_G(vf_data)
    G_hill_at = hill_G(vf_data)
    G_halpin_at = halpin_tsai_G(vf_data, xi=0.8)

    # Compute metrics for E_T
    metrics_E_voigt = compute_metrics(ET_data, E_voigt_at)
    metrics_E_reuss = compute_metrics(ET_data, E_reuss_at)
    metrics_E_hill = compute_metrics(ET_data, E_hill_at)
    metrics_E_halpin = compute_metrics(ET_data, E_halpin_at)

    # Compute metrics for G_LT
    metrics_G_voigt = compute_metrics(G_data, G_voigt_at)
    metrics_G_reuss = compute_metrics(G_data, G_reuss_at)
    metrics_G_hill = compute_metrics(G_data, G_hill_at)
    metrics_G_halpin = compute_metrics(G_data, G_halpin_at)

    def print_metrics(label, m):
        print(f"{label}: mean_rel={m['mean_rel']:.6f}, std_rel={m['std_rel']:.6f}, mae={m['mae']:.6f}, rmse={m['rmse']:.6f}, max_abs_rel={m['max_abs']:.6f}")

    print('\nMetrics comparant E_T aux bornes (relatif):')
    print_metrics('E_T vs Voigt', metrics_E_voigt)
    print_metrics('E_T vs Reuss', metrics_E_reuss)
    print_metrics('E_T vs Hill', metrics_E_hill)
    print_metrics('E_T vs Halpin-Tsai', metrics_E_halpin)

    print('\nMetrics comparant G_LT aux bornes (relatif):')
    print_metrics('G_LT vs Voigt', metrics_G_voigt)
    print_metrics('G_LT vs Reuss', metrics_G_reuss)
    print_metrics('G_LT vs Hill', metrics_G_hill)
    print_metrics('G_LT vs Halpin-Tsai', metrics_G_halpin)


def main():
    plot_validation_convergence(ROOT)
    mesh_convergence_tables_and_plots(ROOT)
    etude_vf_plots(ROOT)


if __name__ == '__main__':
    main()
