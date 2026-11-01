"""
Comprehensive Report and Plot Generator for Assignment 2
Centre for Neuroscience, Indian Institute of Science
Author: Shubham Yadav
"""

import os
import json
import numpy as np
from scipy.stats import norm
import sdt_analysis

def run():
    print("Generating comprehensive experimental datasets and analytical artifacts...")
    os.makedirs('static/images', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # 1. Subject 1: Alice (Telepathy Paradigm)
    # N = 50 trials (25 signal, 25 noise)
    # Hits: 13, False Alarms: 12
    m1 = sdt_analysis.compute_sdt_metrics(hits=13, false_alarms=12, n_signal=25, n_noise=25)
    gt1 = [1]*25 + [0]*25
    resp1 = [1]*13 + [0]*12 + [1]*12 + [0]*13
    perm1 = sdt_analysis.run_permutation_test(gt1, resp1, n_permutations=5000, seed=101)

    svg_dist1 = sdt_analysis.generate_sdt_distributions_svg(m1['d_prime'], m1['c'], m1['b'])
    with open('static/images/alice_sdt_distribution.svg', 'w') as f:
        f.write(svg_dist1)

    svg_perm1 = sdt_analysis.generate_permutation_histogram_svg(perm1['perm_dps'], m1['d_prime'], perm1['empirical_p'])
    with open('static/images/alice_permutation_test.svg', 'w') as f:
        f.write(svg_perm1)

    # 2. Subject 2: Bob (Clairvoyance ROC Paradigm)
    # N = 60 trials (30 signal, 30 noise)
    # Ratings: 1..4
    sig_ratings = [1]*5 + [2]*9 + [3]*10 + [4]*6
    noise_ratings = [1]*6 + [2]*9 + [3]*10 + [4]*5
    gt2 = [1]*30 + [0]*30
    ratings2 = sig_ratings + noise_ratings
    resp2 = [1 if r >= 3 else 0 for r in ratings2]

    m2 = sdt_analysis.compute_sdt_metrics(hits=16, false_alarms=15, n_signal=30, n_noise=30)
    perm2 = sdt_analysis.run_permutation_test(gt2, resp2, n_permutations=5000, seed=202)
    roc2 = sdt_analysis.compute_roc_from_ratings(ratings2, gt2)

    svg_dist2 = sdt_analysis.generate_sdt_distributions_svg(m2['d_prime'], m2['c'], m2['b'])
    with open('static/images/bob_sdt_distribution.svg', 'w') as f:
        f.write(svg_dist2)

    svg_roc2 = sdt_analysis.generate_roc_svg(roc2['roc_points'], m2['d_prime'])
    with open('static/images/bob_roc_curve.svg', 'w') as f:
        f.write(svg_roc2)

    svg_perm2 = sdt_analysis.generate_permutation_histogram_svg(perm2['perm_dps'], m2['d_prime'], perm2['empirical_p'])
    with open('static/images/bob_permutation_test.svg', 'w') as f:
        f.write(svg_perm2)

    # 3. Cross-subject comparison
    summary = [
        {'name': 'Alice (Subject 1)', 'paradigm': 'Telepathy', 'metrics': m1},
        {'name': 'Bob (Subject 2)', 'paradigm': 'Clairvoyance ROC', 'metrics': m2}
    ]
    svg_comp = sdt_analysis.generate_comparison_bar_svg(summary)
    with open('static/images/comparison_bars.svg', 'w') as f:
        f.write(svg_comp)

    # 4. Meta-analysis pooling
    w1 = 1.0 / (m1['se_dp'] ** 2)
    w2 = 1.0 / (m2['se_dp'] ** 2)
    pooled_dp = (w1 * m1['d_prime'] + w2 * m2['d_prime']) / (w1 + w2)
    pooled_se = float(1.0 / np.sqrt(w1 + w2))
    pooled_z = pooled_dp / pooled_se
    pooled_p = float(2.0 * (1.0 - norm.cdf(abs(pooled_z))))
    joint_bf01 = m1['bf01'] * m2['bf01']

    print("Meta-analysis results:")
    print(f"  Alice: d' = {m1['d_prime']:.3f} +/- {m1['se_dp']:.3f}, c = {m1['c']:.3f}, b = {m1['b']:.3f}, BF01 = {m1['bf01']:.2f}")
    print(f"  Bob:   d' = {m2['d_prime']:.3f} +/- {m2['se_dp']:.3f}, c = {m2['c']:.3f}, b = {m2['b']:.3f}, BF01 = {m2['bf01']:.2f}")
    print(f"  Pooled d' = {pooled_dp:.3f} +/- {pooled_se:.3f}, z = {pooled_z:.2f}, p = {pooled_p:.4f}, Joint BF01 = {joint_bf01:.2f}")

    print("All SVGs generated successfully in static/images/")

if __name__ == '__main__':
    run()

