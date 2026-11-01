"""
Signal Detection Theory (SDT) Core Analysis Module for Parapsychology Testing
Assignment 2 - Centre for Neuroscience, Indian Institute of Science
Author: Shubham Yadav
"""

import math
import numpy as np
from scipy.stats import norm

def compute_sdt_metrics(hits, false_alarms, n_signal, n_noise):
    """
    Computes SDT metrics according to Prof. Sridhar Devarajan's lecture formulas:
      d' = Φ^-1(H) - Φ^-1(F)
      c  = -Φ^-1(F)
      b  = -(Φ^-1(H) + Φ^-1(F)) / 2

    Applies the log-linear correction (Hautus, 1995) to prevent infinite z-scores:
      H = (hits + 0.5) / (n_signal + 1)
      F = (false_alarms + 0.5) / (n_noise + 1)
    """
    n_signal = max(1, int(n_signal))
    n_noise = max(1, int(n_noise))
    hits = max(0, min(hits, n_signal))
    false_alarms = max(0, min(false_alarms, n_noise))

    # Log-linear Hautus correction
    h_rate = (hits + 0.5) / (n_signal + 1.0)
    fa_rate = (false_alarms + 0.5) / (n_noise + 1.0)

    z_h = float(norm.ppf(h_rate))
    z_fa = float(norm.ppf(fa_rate))

    d_prime = float(z_h - z_fa)
    c = float(-z_fa)
    b = float(-(z_h + z_fa) / 2.0)

    # Standard error of d' (Miller 1996, Gourevitch & Galanter 1967)
    phi_h = norm.pdf(z_h)
    phi_fa = norm.pdf(z_fa)
    
    var_dp = (h_rate * (1.0 - h_rate)) / (n_signal * (phi_h ** 2)) + \
             (fa_rate * (1.0 - fa_rate)) / (n_noise * (phi_fa ** 2))
    se_dp = float(np.sqrt(max(1e-9, var_dp)))

    # Wald z-score for H0: d' = 0
    z_score = d_prime / se_dp
    p_value = float(2.0 * (1.0 - norm.cdf(abs(z_score))))

    # 95% Confidence Interval for d'
    ci_lower = d_prime - 1.96 * se_dp
    ci_upper = d_prime + 1.96 * se_dp

    # Bayes Factor BF01 favoring H0: d' = 0 over H1: d' ~ N(0, 1)
    sigma_prior = 1.0
    var_prior = sigma_prior ** 2
    var_total = var_prior + var_dp
    
    # Marginal likelihood ratio
    log_bf01 = 0.5 * (np.log(var_total) - np.log(var_dp)) - \
               0.5 * (d_prime ** 2 / var_dp) + \
               0.5 * (d_prime ** 2 / var_total)
    bf01 = float(np.exp(np.clip(log_bf01, -20, 20)))
    bf10 = float(1.0 / bf01) if bf01 > 1e-12 else float('inf')

    return {
        'hits': hits,
        'misses': n_signal - hits,
        'false_alarms': false_alarms,
        'correct_rejections': n_noise - false_alarms,
        'n_signal': n_signal,
        'n_noise': n_noise,
        'total_trials': n_signal + n_noise,
        'raw_hit_rate': hits / n_signal,
        'raw_fa_rate': false_alarms / n_noise,
        'accuracy': (hits + (n_noise - false_alarms)) / (n_signal + n_noise),
        'corrected_hit_rate': h_rate,
        'corrected_fa_rate': fa_rate,
        'z_h': z_h,
        'z_fa': z_fa,
        'd_prime': d_prime,
        'c': c,
        'b': b,
        'se_dp': se_dp,
        'ci_95': (ci_lower, ci_upper),
        'z_score': z_score,
        'p_value': p_value,
        'bf01': bf01,
        'bf10': bf10
    }

def run_permutation_test(y_true, y_pred, n_permutations=5000, seed=42):
    """
    Non-parametric permutation test for d'.
    Shuffles y_true while holding y_pred fixed to construct the exact empirical null distribution.
    """
    np.random.seed(seed)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_trials = len(y_true)
    
    n_signal = np.sum(y_true == 1)
    n_noise = np.sum(y_true == 0)

    # Observed d'
    obs_metrics = compute_sdt_metrics(
        hits=np.sum((y_true == 1) & (y_pred == 1)),
        false_alarms=np.sum((y_true == 0) & (y_pred == 1)),
        n_signal=n_signal,
        n_noise=n_noise
    )
    obs_dp = obs_metrics['d_prime']

    perm_dps = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled_true = np.random.permutation(y_true)
        h = np.sum((shuffled_true == 1) & (y_pred == 1))
        fa = np.sum((shuffled_true == 0) & (y_pred == 1))
        h_r = (h + 0.5) / (n_signal + 1.0)
        fa_r = (fa + 0.5) / (n_noise + 1.0)
        perm_dps[i] = norm.ppf(h_r) - norm.ppf(fa_r)

    # Two-sided empirical p-value
    emp_p = float((1.0 + np.sum(np.abs(perm_dps) >= np.abs(obs_dp))) / (n_permutations + 1.0))
    
    return {
        'observed_dp': obs_dp,
        'perm_dps': perm_dps,
        'empirical_p': emp_p,
        'perm_mean': float(np.mean(perm_dps)),
        'perm_std': float(np.std(perm_dps)),
        'perm_ci': (float(np.percentile(perm_dps, 2.5)), float(np.percentile(perm_dps, 97.5)))
    }

def compute_roc_from_ratings(ratings, y_true):
    """
    Computes multi-point ROC from confidence ratings (e.g., 1 to 4).
    Ratings: 1=Definitely Absent, 2=Probably Absent, 3=Probably Present, 4=Definitely Present
    Thresholds:
      Criterion 1: Respond 1 vs (2,3,4) -> Most liberal (c_lib)
      Criterion 2: Respond (1,2) vs (3,4) -> Standard balanced binary threshold
      Criterion 3: Respond (1,2,3) vs 4 -> Most conservative (c_cons)
    """
    ratings = np.array(ratings)
    y_true = np.array(y_true)
    n_signal = np.sum(y_true == 1)
    n_noise = np.sum(y_true == 0)

    unique_thresholds = [4, 3, 2]  # Decision criteria: rating >= threshold
    points = [{'threshold_desc': 'Anchor (0,0)', 'fa': 0.0, 'hit': 0.0, 'c': float('inf'), 'b': float('nan')}]

    for thresh in unique_thresholds:
        # Binary decision: rating >= thresh means 'Signal Present'
        y_binary = (ratings >= thresh).astype(int)
        hits = np.sum((y_true == 1) & (y_binary == 1))
        fas = np.sum((y_true == 0) & (y_binary == 1))
        
        m = compute_sdt_metrics(hits, fas, n_signal, n_noise)
        points.append({
            'threshold': thresh,
            'threshold_desc': f'Rating ≥ {thresh}',
            'raw_hit': hits / n_signal,
            'raw_fa': fas / n_noise,
            'hit': m['corrected_hit_rate'],
            'fa': m['corrected_fa_rate'],
            'd_prime': m['d_prime'],
            'c': m['c'],
            'b': m['b']
        })

    points.append({'threshold_desc': 'Anchor (1,1)', 'fa': 1.0, 'hit': 1.0, 'c': float('-inf'), 'b': float('nan')})
    
    # Calculate empirical AUC using trapezoidal rule
    fa_coords = [p['fa'] for p in points]
    hit_coords = [p['hit'] for p in points]
    sorted_pairs = sorted(zip(fa_coords, hit_coords), key=lambda x: x[0])
    auc = 0.0
    for i in range(len(sorted_pairs) - 1):
        x1, y1 = sorted_pairs[i]
        x2, y2 = sorted_pairs[i+1]
        auc += (x2 - x1) * (y1 + y2) / 2.0

    return {
        'roc_points': points,
        'empirical_auc': float(auc)
    }

# ==============================================================================
# Pure Python SVG Plot Generators (Crisp, High-Resolution, No Matplotlib Needed)
# ==============================================================================

def generate_sdt_distributions_svg(d_prime, c, b, width=640, height=340):
    """
    Generates a clear SVG diagram of Signal & Noise Gaussian distributions,
    showing decision criterion c and relative bias b.
    """
    x_min, x_max = -3.5, max(4.0, d_prime + 3.5)
    margin_l, margin_r, margin_t, margin_b = 60, 40, 40, 50
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    def to_screen_x(val):
        return margin_l + (val - x_min) / (x_max - x_min) * plot_w

    def to_screen_y(val):
        max_y = 0.45
        return margin_t + plot_h - (val / max_y) * plot_h

    # Sample points for Gaussian curves
    xs = np.linspace(x_min, x_max, 250)
    noise_pdf = norm.pdf(xs, loc=0, scale=1)
    signal_pdf = norm.pdf(xs, loc=d_prime, scale=1)

    noise_path = "M " + " L ".join([f"{to_screen_x(x):.1f},{to_screen_y(y):.1f}" for x, y in zip(xs, noise_pdf)])
    signal_path = "M " + " L ".join([f"{to_screen_x(x):.1f},{to_screen_y(y):.1f}" for x, y in zip(xs, signal_pdf)])

    # Shaded False Alarm region: Noise pdf where x >= c
    fa_xs = xs[xs >= c]
    if len(fa_xs) > 1:
        fa_pts = [(to_screen_x(fa_xs[0]), to_screen_y(0))]
        fa_pts += [(to_screen_x(x), to_screen_y(norm.pdf(x, 0, 1))) for x in fa_xs]
        fa_pts.append((to_screen_x(fa_xs[-1]), to_screen_y(0)))
        fa_fill_path = "M " + " L ".join([f"{x:.1f},{y:.1f}" for x, y in fa_pts]) + " Z"
    else:
        fa_fill_path = ""

    # Shaded Hit region: Signal pdf where x >= c
    hit_xs = xs[xs >= c]
    if len(hit_xs) > 1:
        hit_pts = [(to_screen_x(hit_xs[0]), to_screen_y(0))]
        hit_pts += [(to_screen_x(x), to_screen_y(norm.pdf(x, d_prime, 1))) for x in hit_xs]
        hit_pts.append((to_screen_x(hit_xs[-1]), to_screen_y(0)))
        hit_fill_path = "M " + " L ".join([f"{x:.1f},{y:.1f}" for x, y in hit_pts]) + " Z"
    else:
        hit_fill_path = ""

    x_c = to_screen_x(c)
    x_ideal = to_screen_x(d_prime / 2.0)
    y_base = margin_t + plot_h

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="background:#0f172a; border-radius:12px; font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;">
      <!-- Title -->
      <text x="{width/2}" y="25" fill="#f8fafc" font-size="14" font-weight="600" text-anchor="middle">Signal Detection Theory Model (d' = {d_prime:.2f}, c = {c:.2f}, b = {b:.2f})</text>
      
      <!-- Shaded Areas -->
      {f'<path d="{hit_fill_path}" fill="rgba(239, 68, 68, 0.25)" />' if hit_fill_path else ''}
      {f'<path d="{fa_fill_path}" fill="rgba(59, 130, 246, 0.25)" />' if fa_fill_path else ''}

      <!-- Baseline Axis -->
      <line x1="{margin_l}" y1="{y_base}" x2="{width - margin_r}" y2="{y_base}" stroke="#475569" stroke-width="2" />
      
      <!-- Noise Distribution (N(0,1)) -->
      <path d="{noise_path}" fill="none" stroke="#38bdf8" stroke-width="2.5" />
      <text x="{to_screen_x(0)}" y="{to_screen_y(0.40) - 8}" fill="#38bdf8" font-size="12" font-weight="600" text-anchor="middle">Noise (Absent)</text>
      <line x1="{to_screen_x(0)}" y1="{to_screen_y(0.40)}" x2="{to_screen_x(0)}" y2="{y_base}" stroke="#38bdf8" stroke-dasharray="3,3" stroke-width="1.5"/>

      <!-- Signal Distribution (N(d',1)) -->
      <path d="{signal_path}" fill="none" stroke="#f43f5e" stroke-width="2.5" />
      <text x="{to_screen_x(d_prime)}" y="{to_screen_y(0.40) - 8}" fill="#f43f5e" font-size="12" font-weight="600" text-anchor="middle">Signal (Present)</text>
      <line x1="{to_screen_x(d_prime)}" y1="{to_screen_y(0.40)}" x2="{to_screen_x(d_prime)}" y2="{y_base}" stroke="#f43f5e" stroke-dasharray="3,3" stroke-width="1.5"/>

      <!-- Criterion Line c -->
      <line x1="{x_c}" y1="{margin_t + 10}" x2="{x_c}" y2="{y_base}" stroke="#eab308" stroke-width="2.5" stroke-dasharray="6,4" />
      <rect x="{x_c - 30}" y="{margin_t}" width="60" height="20" rx="4" fill="#854d0e" />
      <text x="{x_c}" y="{margin_t + 14}" fill="#fef08a" font-size="11" font-weight="bold" text-anchor="middle">c = {c:.2f}</text>

      <!-- Ideal Unbiased Criterion Line (d'/2) -->
      <line x1="{x_ideal}" y1="{margin_t + 25}" x2="{x_ideal}" y2="{y_base}" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="2,2" />
      <text x="{x_ideal}" y="{y_base + 30}" fill="#94a3b8" font-size="10" text-anchor="middle">Neutral (d'/2)</text>

      <!-- Legend & Axis Labels -->
      <text x="{margin_l}" y="{y_base + 18}" fill="#64748b" font-size="10" text-anchor="middle">{x_min:.1f}</text>
      <text x="{width - margin_r}" y="{y_base + 18}" fill="#64748b" font-size="10" text-anchor="middle">{x_max:.1f}</text>
      <text x="{width/2}" y="{y_base + 38}" fill="#cbd5e1" font-size="12" text-anchor="middle">Internal Decision Variable (Evidence Axis)</text>
      <text x="{margin_l + 10}" y="{margin_t + 20}" fill="#38bdf8" font-size="11">■ False Alarms (P(Yes|Noise))</text>
      <text x="{margin_l + 10}" y="{margin_t + 36}" fill="#f43f5e" font-size="11">■ Hits (P(Yes|Signal))</text>
    </svg>"""
    return svg

def generate_roc_svg(roc_points, theoretical_dp=0.0, width=440, height=380):
    """
    Generates an SVG Receiver Operating Characteristic (ROC) curve plot.
    """
    margin_l, margin_r, margin_t, margin_b = 55, 30, 40, 50
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    def to_screen_x(fa):
        return margin_l + fa * plot_w

    def to_screen_y(hit):
        return margin_t + plot_h - hit * plot_h

    # Theoretical ROC curve for d'
    fa_smooth = np.linspace(0.001, 0.999, 100)
    hit_smooth = norm.cdf(norm.ppf(fa_smooth) + theoretical_dp)
    theo_path = "M " + " L ".join([f"{to_screen_x(f):.1f},{to_screen_y(h):.1f}" for f, h in zip(fa_smooth, hit_smooth)])

    # Empirical points
    points_markup = ""
    poly_pts = []
    sorted_pts = sorted([p for p in roc_points if 0 <= p.get('fa', -1) <= 1], key=lambda x: x['fa'])
    for pt in sorted_pts:
        sx = to_screen_x(pt['fa'])
        sy = to_screen_y(pt['hit'])
        poly_pts.append(f"{sx:.1f},{sy:.1f}")
        label = pt.get('threshold_desc', '')
        if label and 'Anchor' not in label:
            points_markup += f"""
              <circle cx="{sx:.1f}" cy="{sy:.1f}" r="6" fill="#38bdf8" stroke="#ffffff" stroke-width="2" />
              <text x="{sx + 10:.1f}" y="{sy - 8:.1f}" fill="#38bdf8" font-size="10" font-weight="bold">{label}</text>
            """

    emp_poly = "M " + " L ".join(poly_pts) if poly_pts else ""

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="background:#0f172a; border-radius:12px; font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;">
      <!-- Title -->
      <text x="{width/2}" y="24" fill="#f8fafc" font-size="13" font-weight="600" text-anchor="middle">Receiver Operating Characteristic (ROC)</text>

      <!-- Axes Grid -->
      <rect x="{margin_l}" y="{margin_t}" width="{plot_w}" height="{plot_h}" fill="#1e293b" stroke="#334155" />
      <line x1="{to_screen_x(0.5)}" y1="{margin_t}" x2="{to_screen_x(0.5)}" y2="{margin_t + plot_h}" stroke="#334155" stroke-dasharray="2,2"/>
      <line x1="{margin_l}" y1="{to_screen_y(0.5)}" x2="{margin_l + plot_w}" y2="{to_screen_y(0.5)}" stroke="#334155" stroke-dasharray="2,2"/>

      <!-- Chance Diagonal Line (d' = 0) -->
      <line x1="{to_screen_x(0)}" y1="{to_screen_y(0)}" x2="{to_screen_x(1)}" y2="{to_screen_y(1)}" stroke="#64748b" stroke-width="2" stroke-dasharray="5,5" />
      <text x="{to_screen_x(0.7)}" y="{to_screen_y(0.65)}" fill="#64748b" font-size="10" transform="rotate(-45 {to_screen_x(0.7)} {to_screen_y(0.65)})">Chance (d' = 0)</text>

      <!-- Theoretical Curve for d' -->
      {f'<path d="{theo_path}" fill="none" stroke="#f43f5e" stroke-width="2" />' if abs(theoretical_dp) > 0.05 else ''}

      <!-- Empirical ROC Trajectory -->
      {f'<path d="{emp_poly}" fill="none" stroke="#38bdf8" stroke-width="2.5" />' if emp_poly else ''}
      {points_markup}

      <!-- Labels -->
      <text x="{to_screen_x(0)}" y="{margin_t + plot_h + 18}" fill="#94a3b8" font-size="10" text-anchor="middle">0.0</text>
      <text x="{to_screen_x(0.5)}" y="{margin_t + plot_h + 18}" fill="#94a3b8" font-size="10" text-anchor="middle">0.5</text>
      <text x="{to_screen_x(1.0)}" y="{margin_t + plot_h + 18}" fill="#94a3b8" font-size="10" text-anchor="middle">1.0</text>
      <text x="{width/2}" y="{margin_t + plot_h + 36}" fill="#cbd5e1" font-size="11" text-anchor="middle">False Alarm Rate (F)</text>

      <text x="{margin_l - 8}" y="{to_screen_y(0.0)}" fill="#94a3b8" font-size="10" text-anchor="end">0.0</text>
      <text x="{margin_l - 8}" y="{to_screen_y(0.5)}" fill="#94a3b8" font-size="10" text-anchor="end">0.5</text>
      <text x="{margin_l - 8}" y="{to_screen_y(1.0)}" fill="#94a3b8" font-size="10" text-anchor="end">1.0</text>
      <text x="18" y="{height/2}" fill="#cbd5e1" font-size="11" text-anchor="middle" transform="rotate(-90 18 {height/2})">Hit Rate (H)</text>
    </svg>"""
    return svg

def generate_permutation_histogram_svg(perm_dps, obs_dp, p_val, width=520, height=300):
    """
    Generates an SVG histogram for the permutation test null distribution vs observed d'.
    """
    margin_l, margin_r, margin_t, margin_b = 55, 30, 40, 50
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    # Histogram bins
    bins = 30
    counts, edges = np.histogram(perm_dps, bins=bins)
    max_c = max(1, np.max(counts))

    min_x = min(edges[0], obs_dp - 0.2)
    max_x = max(edges[-1], obs_dp + 0.2)

    def to_screen_x(val):
        return margin_l + (val - min_x) / (max_x - min_x) * plot_w

    def to_screen_y(cnt):
        return margin_t + plot_h - (cnt / max_c) * plot_h

    bars_svg = ""
    for i in range(len(counts)):
        x1 = to_screen_x(edges[i])
        x2 = to_screen_x(edges[i+1])
        y = to_screen_y(counts[i])
        h = (margin_t + plot_h) - y
        w = max(1, x2 - x1 - 1)
        bars_svg += f'<rect x="{x1:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="#334155" stroke="#475569" />\n'

    x_obs = to_screen_x(obs_dp)
    y_base = margin_t + plot_h

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="background:#0f172a; border-radius:12px; font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;">
      <!-- Title -->
      <text x="{width/2}" y="24" fill="#f8fafc" font-size="13" font-weight="600" text-anchor="middle">Permutation Null Distribution (B={len(perm_dps)} shuffles)</text>

      <!-- Bars -->
      {bars_svg}

      <!-- Baseline -->
      <line x1="{margin_l}" y1="{y_base}" x2="{margin_l + plot_w}" y2="{y_base}" stroke="#64748b" stroke-width="2" />

      <!-- Null Center Line (0) -->
      <line x1="{to_screen_x(0)}" y1="{margin_t}" x2="{to_screen_x(0)}" y2="{y_base}" stroke="#94a3b8" stroke-dasharray="3,3" stroke-width="1.5"/>

      <!-- Observed d' Line -->
      <line x1="{x_obs}" y1="{margin_t}" x2="{x_obs}" y2="{y_base}" stroke="#f43f5e" stroke-width="3" />
      <rect x="{x_obs - 45}" y="{margin_t - 5}" width="90" height="20" rx="4" fill="#881337" />
      <text x="{x_obs}" y="{margin_t + 9}" fill="#fecdd3" font-size="10" font-weight="bold" text-anchor="middle">Obs d'={obs_dp:.2f}</text>

      <!-- Subtitle / p-value annotation -->
      <text x="{margin_l + 10}" y="{margin_t + 25}" fill="#a5f3fc" font-size="11">Empirical p = {p_val:.4f}</text>
      <text x="{margin_l + 10}" y="{margin_t + 42}" fill="#94a3b8" font-size="10">95% Null Range: [{np.percentile(perm_dps, 2.5):.2f}, {np.percentile(perm_dps, 97.5):.2f}]</text>

      <!-- Axis Labels -->
      <text x="{to_screen_x(min_x)}" y="{y_base + 18}" fill="#94a3b8" font-size="10" text-anchor="start">{min_x:.2f}</text>
      <text x="{to_screen_x(0)}" y="{y_base + 18}" fill="#94a3b8" font-size="10" text-anchor="middle">0.0</text>
      <text x="{to_screen_x(max_x)}" y="{y_base + 18}" fill="#94a3b8" font-size="10" text-anchor="end">{max_x:.2f}</text>
      <text x="{width/2}" y="{y_base + 36}" fill="#cbd5e1" font-size="11" text-anchor="middle">Permuted d' Values</text>
    </svg>"""
    return svg

def generate_comparison_bar_svg(participants_summary, width=540, height=320):
    """
    Generates comparison bar chart across tested participants with 95% Confidence Intervals.
    """
    margin_l, margin_r, margin_t, margin_b = 60, 40, 45, 60
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    n_p = len(participants_summary)
    if n_p == 0:
        return ""

    y_min, y_max = -1.0, 1.5
    for p in participants_summary:
        ci_l, ci_u = p['metrics']['ci_95']
        y_min = min(y_min, ci_l - 0.2)
        y_max = max(y_max, ci_u + 0.3)

    def to_screen_y(val):
        return margin_t + plot_h - ((val - y_min) / (y_max - y_min)) * plot_h

    y_zero = to_screen_y(0.0)

    bar_group_w = plot_w / n_p
    bar_w = min(50, bar_group_w * 0.5)

    bars_svg = ""
    for idx, p in enumerate(participants_summary):
        name = p.get('name', f"Subject {idx+1}")
        m = p['metrics']
        dp = m['d_prime']
        ci_l, ci_u = m['ci_95']

        center_x = margin_l + (idx + 0.5) * bar_group_w
        bar_x = center_x - bar_w / 2.0

        sy_dp = to_screen_y(dp)
        if dp >= 0:
            by = sy_dp
            bh = y_zero - sy_dp
            color = "#38bdf8"
        else:
            by = y_zero
            bh = sy_dp - y_zero
            color = "#f43f5e"

        sy_cil = to_screen_y(ci_l)
        sy_ciu = to_screen_y(ci_u)

        bars_svg += f"""
          <!-- Bar -->
          <rect x="{bar_x:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{max(2, bh):.1f}" fill="{color}" rx="4" />
          
          <!-- 95% CI Error bar -->
          <line x1="{center_x:.1f}" y1="{sy_cil:.1f}" x2="{center_x:.1f}" y2="{sy_ciu:.1f}" stroke="#f8fafc" stroke-width="2" />
          <line x1="{center_x - 6:.1f}" y1="{sy_cil:.1f}" x2="{center_x + 6:.1f}" y2="{sy_cil:.1f}" stroke="#f8fafc" stroke-width="2" />
          <line x1="{center_x - 6:.1f}" y1="{sy_ciu:.1f}" x2="{center_x + 6:.1f}" y2="{sy_ciu:.1f}" stroke="#f8fafc" stroke-width="2" />

          <!-- Value label -->
          <text x="{center_x:.1f}" y="{min(by, sy_ciu) - 8:.1f}" fill="#f8fafc" font-size="11" font-weight="bold" text-anchor="middle">d'={dp:.2f}</text>

          <!-- X Label -->
          <text x="{center_x:.1f}" y="{margin_t + plot_h + 20:.1f}" fill="#f1f5f9" font-size="11" font-weight="600" text-anchor="middle">{name}</text>
          <text x="{center_x:.1f}" y="{margin_t + plot_h + 34:.1f}" fill="#94a3b8" font-size="9" text-anchor="middle">{p.get('paradigm', 'Standard')}</text>
        """

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="background:#0f172a; border-radius:12px; font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;">
      <!-- Title -->
      <text x="{width/2}" y="24" fill="#f8fafc" font-size="13" font-weight="600" text-anchor="middle">Sensitivity Comparison Across Subjects (d' with 95% CI)</text>

      <!-- Reference Zero Line (Chance ESP) -->
      <line x1="{margin_l}" y1="{y_zero}" x2="{margin_l + plot_w}" y2="{y_zero}" stroke="#e2e8f0" stroke-width="2" stroke-dasharray="4,4" />
      <text x="{margin_l + plot_w - 5}" y="{y_zero - 6}" fill="#cbd5e1" font-size="10" text-anchor="end">H0: d'=0 (No ESP)</text>

      <!-- Bars and Error Bars -->
      {bars_svg}

      <!-- Y Axis -->
      <line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}" stroke="#475569" stroke-width="1.5" />
      <text x="{margin_l - 8}" y="{to_screen_y(0.0)}" fill="#94a3b8" font-size="10" text-anchor="end">0.0</text>
      <text x="{margin_l - 8}" y="{to_screen_y(1.0)}" fill="#94a3b8" font-size="10" text-anchor="end">1.0</text>
      <text x="16" y="{height/2}" fill="#cbd5e1" font-size="11" text-anchor="middle" transform="rotate(-90 16 {height/2})">Sensitivity (d')</text>
    </svg>"""
    return svg

