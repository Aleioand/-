"""
================================================================
Synthetic Validation: DI vs PMIME  (ΝΟΥΜΕΡΑ + ΣΧΗΜΑΤΑ σε ένα)
================================================================
Τρέχει ΚΑΙ τις δύο μεθόδους (DI, PMIME) στο ΙΔΙΟ συνθετικό σύστημα
Boolean δομής (Εξ. 19 του Li et al. 2022), όπου μόνο οι σειρές
c0, c1, c2 προκαλούν την ανταμοιβή R (ground truth γνωστό).

`python synthetic_all.py`):
  1. Τρέχει N_RUNS επαναλήψεις, εκτυπώνει πίνακα ανά run + σύνοψη.
  2. Αποθηκεύει δύο σχήματα (PNG + PDF) στο στυλ του paper:
       - fig_synthetic_heatmaps : vector heatmaps (κίτρινο-μπλε) DI/PMIME
       - fig_synthetic_bars     : grouped bars με ground-truth highlight
  3. Αποθηκεύει synthetic_results.csv με τα σκορ κάθε run.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from DI_func import Directed_Information, build_DI_inputs
from PMIMEsigClass import PMIMEsig
from PMIMEsigClass import run_PMIMEsig as pmimesigclass


# ================================================================
#  ΡΥΘΜΙΣΕΙΣ
# ================================================================
N = 1024              # μήκος χρονοσειράς (ίδιο με το paper)
M = 6                 # πλήθος σειρών c
N_RUNS = 30           # επαναλήψεις για στατιστική
GROUND_TRUTH = [0, 1, 2]   # οι αιτιακές σειρές
L_DI = 2              # lag για DI
ALPHA_DI = 1.01       # Renyi alpha

PMIME_PARAMS = dict(
    Lmax=3, T=1, nsur=100, alpha=0.10, showtxt=0, nnei=5,
    type='rp', mi_method='nonexact', cmi_method='nonexact',
    full_logs=False, cpu_pool=3, execution='serial',
    include_interactions=False, use_backward_revision=False, method=0
)

COLOR_DI = '#2166AC'
COLOR_PMIME = '#B2182B'
PAPER_CMAP = 'viridis'   # κίτρινο-μπλε-μωβ όπως το paper


def set_thesis_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
        'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
        'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })


# ================================================================
#  ΓΕΝΝΗΤΡΙΑ ΣΥΝΘΕΤΙΚΟΥ ΣΥΣΤΗΜΑΤΟΣ (Boolean, Εξ. 19)
# ================================================================
def generate_boolean_system(N=1024, M=6, seed=None):
    if seed is not None:
        np.random.seed(seed)
    c = np.zeros((M, N), dtype=float)
    for n in range(N - 1):
        c[:, n + 1] = ((c[:, n] + np.random.randn(M)) > 0).astype(float)
    R = np.zeros(N, dtype=float)
    R[:2] = 0.01 * np.random.randn(2)
    for n in range(1, N - 1):
        term1 = 0.1 * R[n]
        term2 = float(c[0, n] != c[0, n - 1])
        term3 = float(c[1, n] != c[1, n - 1]) ** c[2, n]
        noise = 0.01 * np.random.randn()
        R[n + 1] = term1 + term2 + term3 + noise
    R = (R - R.min()) / (R.max() - R.min() + 1e-12)
    return c, R, [0, 1, 2]


# ================================================================
#  ΜΕΘΟΔΟΙ
# ================================================================
def run_di(c, R, L=L_DI, alpha=ALPHA_DI):
    M = c.shape[0]
    di = np.zeros(M)
    for m in range(M):
        X, Y, Z = build_DI_inputs(c[m], R, L)
        with torch.no_grad():
            di[m] = float(Directed_Information(X, Y, Z, alpha).cpu().item())
    return di


def run_pmime(c, R, params=PMIME_PARAMS):
    M = c.shape[0]
    data_matrix = np.hstack([R.reshape(-1, 1)] + [c[m].reshape(-1, 1) for m in range(M)])
    allM = PMIMEsig.normalize_data(data_matrix)
    tp = params.copy()
    tp.update({"data": data_matrix, "allM": allM, "seeds": {0: 12345}})
    out = pmimesigclass(0, **tp)
    K = data_matrix.shape[1]
    rm_col = None
    for item in out:
        if hasattr(item, "shape") and item.shape in [(K, 1), (K,)]:
            rm_col = np.asarray(item).flatten()
            break
    scores = np.zeros(M)
    if rm_col is not None:
        for m in range(M):
            scores[m] = float(rm_col[1 + m])
    return scores


def evaluate(scores, ground_truth, top_k=3):
    top_k_idx = list(np.argsort(scores)[::-1][:top_k])
    found = len(set(top_k_idx) & set(ground_truth))
    return top_k_idx, found


def _norm(v):
    v = np.asarray(v, dtype=float)
    vmin, vmax = v.min(), v.max()
    return (v - vmin) / (vmax - vmin) if (vmax - vmin) > 1e-12 else v * 0.0


# ================================================================
#  ΣΧΗΜΑ A: Vector heatmaps (τύπου Fig. 6 του paper)
# ================================================================
def make_vector_heatmaps(di_scores, pmime_scores, ground_truth, M,
                         out="fig_synthetic_heatmaps"):
    gt_vec = np.array([[1.0 if m in ground_truth else 0.0] for m in range(M)])
    di_disp = _norm(di_scores).reshape(-1, 1)
    pmime_disp = _norm(pmime_scores).reshape(-1, 1)

    fig, axes = plt.subplots(1, 3, figsize=(8, 4.5))
    titles = ['(a) Ground Truth', '(b) DI', '(c) PMIME']
    data = [gt_vec, di_disp, pmime_disp]
    im = None
    for ax, title, d in zip(axes, titles, data):
        im = ax.imshow(d, cmap=PAPER_CMAP, aspect='auto', vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks(range(M))
        ax.set_yticklabels([f'c{m}' for m in range(M)])
        for m in ground_truth:
            ax.add_patch(plt.Rectangle((-0.5, m - 0.5), 1, 1, fill=False,
                                        edgecolor='red', lw=2))
    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.046, pad=0.08,
                 label='Normalized causality score')
    fig.suptitle('Synthetic system: causality $c_m \\rightarrow R$\n'
                 '(red box = ground-truth causal series)', y=1.03)
    fig.savefig(f'{out}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{out}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {out}.png / .pdf")


# ================================================================
#  ΣΧΗΜΑ B: Grouped bars με ground-truth highlight
# ================================================================
def make_bar_comparison(di_mean, di_std, pmime_mean, pmime_std, ground_truth, M,
                        out="fig_synthetic_bars"):
    x = np.arange(M)
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width/2, di_mean, width, yerr=di_std, capsize=3,
           color=COLOR_DI, label='DI', edgecolor='white', linewidth=0.5)
    ax.bar(x + width/2, pmime_mean, width, yerr=pmime_std, capsize=3,
           color=COLOR_PMIME, label='PMIME', edgecolor='white', linewidth=0.5)
    for m in ground_truth:
        ax.axvspan(m - 0.5, m + 0.5, color='gold', alpha=0.15, zorder=0)
    ax.set_xlabel('Series $c_m$ (gold = ground-truth causal series)')
    ax.set_ylabel('Mean normalized causality score')
    ax.set_title('Synthetic system: mean DI and PMIME scores per series')
    ax.set_xticks(x)
    ax.set_xticklabels([f'c{m}' for m in range(M)])
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    fig.savefig(f'{out}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{out}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {out}.png / .pdf")


# ================================================================
#  MAIN
# ================================================================
def main():
    set_thesis_style()

    di_found_list, pmime_found_list, pmime_detected_list = [], [], []
    di_all = np.zeros((N_RUNS, M))
    pmime_all = np.zeros((N_RUNS, M))
    csv_rows = []

    print(f"Synthetic validation: {N_RUNS} runs (N={N}, M={M}, ground truth={GROUND_TRUTH})\n")
    print(f"{'Run':>4} | {'DI top-3':>18} | {'DI ok':>6} | {'PMIME top-3':>18} | {'PMIME ok':>8}")
    print("-" * 72)

    for run in range(N_RUNS):
        c, R, gt = generate_boolean_system(N=N, M=M, seed=run)

        di_scores = run_di(c, R)
        di_top, di_found = evaluate(di_scores, gt, top_k=3)

        pmime_scores = run_pmime(c, R)
        pmime_top, pmime_found = evaluate(pmime_scores, gt, top_k=3)
        pmime_detected = int(np.max(pmime_scores) > 0)

        di_found_list.append(di_found)
        pmime_found_list.append(pmime_found)
        pmime_detected_list.append(pmime_detected)

        di_all[run] = _norm(di_scores)
        pmime_all[run] = _norm(pmime_scores) if pmime_scores.max() > 0 else pmime_scores

        row = {'run': run, 'di_found': di_found, 'pmime_found': pmime_found,
               'pmime_detected': pmime_detected}
        for m in range(M):
            row[f'di_c{m}'] = round(float(di_scores[m]), 5)
            row[f'pmime_c{m}'] = round(float(pmime_scores[m]), 5)
        csv_rows.append(row)

        print(f"{run:>4} | {str(di_top):>18} | {di_found:>4}/3 | "
              f"{str(pmime_top):>18} | {pmime_found:>4}/3")

    # ---- ΣΥΝΟΨΗ ----
    di_found_arr = np.array(di_found_list)
    pmime_found_arr = np.array(pmime_found_list)
    pmime_det_arr = np.array(pmime_detected_list)

    print("\n" + "=" * 60)
    print(f"ΣΥΝΟΨΗ (μέσος όρος {N_RUNS} runs)")
    print("=" * 60)
    print(f"DI:    μέσος αριθμός ground-truth σειρών στο top-3: {di_found_arr.mean():.2f}/3")
    print(f"       runs με ΚΑΙ ΤΙΣ 3 σωστές: {(di_found_arr == 3).mean()*100:.0f}%")
    print(f"       runs με >=2 σωστές:        {(di_found_arr >= 2).mean()*100:.0f}%")
    print()
    print(f"PMIME: μέσος αριθμός ground-truth σειρών στο top-3: {pmime_found_arr.mean():.2f}/3")
    print(f"       runs με ΚΑΙ ΤΙΣ 3 σωστές: {(pmime_found_arr == 3).mean()*100:.0f}%")
    print(f"       runs με >=2 σωστές:        {(pmime_found_arr >= 2).mean()*100:.0f}%")
    print(f"       runs όπου ανίχνευσε ΟΤΙΔΗΠΟΤΕ (max R>0): {pmime_det_arr.mean()*100:.0f}%")
    print("=" * 60)

    # ---- ΣΧΗΜΑΤΑ ----
    print("\nΠαράγω σχήματα...")
    # Heatmaps: αντιπροσωπευτικό run (seed=0)
    c0, R0, _ = generate_boolean_system(N=N, M=M, seed=0)
    make_vector_heatmaps(run_di(c0, R0), run_pmime(c0, R0), GROUND_TRUTH, M)
    # Bars: μέσοι όροι όλων των runs
    make_bar_comparison(di_all.mean(0), di_all.std(0),
                        pmime_all.mean(0), pmime_all.std(0), GROUND_TRUTH, M)

    # ---- CSV ----
    try:
        import pandas as pd
        pd.DataFrame(csv_rows).to_csv('synthetic_results.csv', index=False)
        print("  [OK] synthetic_results.csv")
    except Exception as e:
        print(f"  [WARN] CSV skipped: {e}")

    print("\nΟλοκληρώθηκε. Στείλε μου τη ΣΥΝΟΨΗ και τα 2 PNG για το Κεφ. 4.")


if __name__ == "__main__":
    main()
