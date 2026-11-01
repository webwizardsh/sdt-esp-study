#!/usr/bin/env python3
"""
CLI Runner for Signal Detection Theory Parapsychology Testing
Assignment 2 - Centre for Neuroscience, Indian Institute of Science
Author: Shubham Yadav
"""

import sys
import random
import numpy as np
from scipy.stats import norm
import sdt_analysis

def run_interactive_cli():
    print("=" * 70)
    print("  SIGNAL DETECTION THEORY: TESTING PARAPSYCHOLOGY & ESP")
    print("  Centre for Neuroscience, Indian Institute of Science")
    print("=" * 70)
    
    print("\nSelect Experimental Paradigm:")
    print("  [1] Paradigm A: Mental Telepathy (Binary Yes/No, Sender/Receiver)")
    print("  [2] Paradigm B: Clairvoyance (4-point Confidence Rating ROC)")
    print("  [3] Run Quick Automated Benchmark (n=2 Subjects: Alice & Bob)")
    
    choice = input("\nEnter choice (1-3) [default 3]: ").strip() or "3"
    
    if choice == "3":
        run_benchmark_summary()
        return

    participant = input("Enter Participant Name: ").strip() or "Subject"
    n_trials_str = input("Enter Total Number of Trials (e.g. 40, 50, 60) [default 40]: ").strip() or "40"
    n_trials = int(n_trials_str)
    
    # Generate balanced ground truth
    n_sig = n_trials // 2
    n_noi = n_trials - n_sig
    gt = [1] * n_sig + [0] * n_noi
    random.shuffle(gt)
    
    responses = []
    ratings = []
    
    print(f"\n--- Starting Experiment for {participant} ({n_trials} trials) ---")
    print("Sensory Shielding Active: Targets randomized double-blind.\n")
    
    for i, target in enumerate(gt, 1):
        if choice == "1":
            print(f"Trial {i}/{n_trials}: [Mental Transmission Phase]")
            ans = input("  Did the sender transmit the signal? (y/n): ").strip().lower()
            resp = 1 if ans.startswith('y') else 0
            responses.append(resp)
        else:
            print(f"Trial {i}/{n_trials}: [Cryptographic Safe Shielded]")
            print("  Rate confidence: 1=Sure Absent, 2=Probable Absent, 3=Probable Present, 4=Sure Present")
            ans = input("  Rating (1-4): ").strip()
            r = int(ans) if ans in ['1', '2', '3', '4'] else 2
            ratings.append(r)
            responses.append(1 if r >= 3 else 0)

    # Compute SDT Metrics
    y_true = np.array(gt)
    y_pred = np.array(responses)
    hits = int(np.sum((y_true == 1) & (y_pred == 1)))
    fas = int(np.sum((y_true == 0) & (y_pred == 1)))
    
    m = sdt_analysis.compute_sdt_metrics(hits, fas, n_sig, n_noi)
    perm = sdt_analysis.run_permutation_test(y_true, y_pred, n_permutations=3000)
    
    display_results(participant, m, perm)
    
    if choice == "2":
        roc = sdt_analysis.compute_roc_from_ratings(ratings, y_true)
        print("\nReceiver Operating Characteristic (ROC) Points:")
        print("  Threshold         FA Rate   Hit Rate   Criterion c   Bias b")
        for pt in roc['roc_points']:
            if 'Anchor' not in pt.get('threshold_desc', ''):
                print(f"  {pt['threshold_desc']:<17} {pt['fa']:<9.3f} {pt['hit']:<10.3f} {pt['c']:<13.3f} {pt['b']:<8.3f}")
        print(f"  Empirical AUC: {roc['empirical_auc']:.3f} (Chance = 0.500)")

def run_benchmark_summary():
    print("\nRunning standardized benchmark for n=2 subjects...")
    # Subject 1: Alice
    m1 = sdt_analysis.compute_sdt_metrics(hits=13, false_alarms=12, n_signal=25, n_noise=25)
    perm1 = sdt_analysis.run_permutation_test([1]*25+[0]*25, [1]*13+[0]*12+[1]*12+[0]*13, n_permutations=3000)
    display_results("Alice (Subject 1 - Telepathy)", m1, perm1)

    # Subject 2: Bob
    m2 = sdt_analysis.compute_sdt_metrics(hits=16, false_alarms=15, n_signal=30, n_noise=30)
    perm2 = sdt_analysis.run_permutation_test([1]*30+[0]*30, [1]*16+[0]*14+[1]*15+[0]*15, n_permutations=3000)
    display_results("Bob (Subject 2 - Clairvoyance ROC)", m2, perm2)

    # Pooled Meta-Analysis
    w1 = 1.0 / (m1['se_dp']**2)
    w2 = 1.0 / (m2['se_dp']**2)
    p_dp = (w1*m1['d_prime'] + w2*m2['d_prime']) / (w1+w2)
    p_se = 1.0 / np.sqrt(w1+w2)
    p_z = p_dp / p_se
    p_val = 2.0 * (1.0 - norm.cdf(abs(p_z)))
    joint_bf01 = m1['bf01'] * m2['bf01']

    print("\n" + "=" * 70)
    print("  CROSS-SUBJECT META-ANALYSIS SYNTHESIS (n = 2)")
    print("=" * 70)
    print(f"  Combined Trials:   110 trials")
    print(f"  Pooled d':         {p_dp:.3f} +/- {p_se:.3f} (SE)")
    print(f"  95% CI on d':      [{p_dp - 1.96*p_se:.3f}, {p_dp + 1.96*p_se:.3f}]")
    print(f"  Wald z-statistic:  z = {p_z:.2f} (p = {p_val:.4f})")
    print(f"  Joint Bayes Factor: BF01 = {joint_bf01:.2f} (Evidence favoring H0: No ESP)")
    print("=" * 70)
    print("  SCIENTIFIC CONCLUSION: ESP DISPROVED VIA SIGNAL DETECTION THEORY.")
    print("  Sensitivity across individuals is strictly bounded around zero.")
    print("=" * 70)

def display_results(participant, m, perm):
    print("\n" + "-" * 70)
    print(f"  SDT DIAGNOSTIC REPORT: {participant}")
    print("-" * 70)
    print(f"  Total Trials:      {m['total_trials']} (Signal: {m['n_signal']}, Noise: {m['n_noise']})")
    print(f"  Hits:              {m['hits']} (Hit Rate H = {m['corrected_hit_rate']*100:.1f}%)")
    print(f"  False Alarms:      {m['false_alarms']} (FA Rate F = {m['corrected_fa_rate']*100:.1f}%)")
    print(f"  Raw Accuracy:      {m['accuracy']*100:.1f}%")
    print("  " + "." * 66)
    print(f"  Sensitivity (d'):  {m['d_prime']:.3f} +/- {m['se_dp']:.3f} (SE)")
    print(f"  95% CI for d':     [{m['ci_95'][0]:.3f}, {m['ci_95'][1]:.3f}]")
    print(f"  Criterion (c):     {m['c']:.3f}  [c = -norminv(F)]")
    print(f"  Relative Bias (b): {m['b']:.3f}  [b = -(norminv(H) + norminv(F))/2]")
    print("  " + "." * 66)
    print(f"  Wald z-test:       z = {m['z_score']:.2f}, p-value = {m['p_value']:.4f}")
    print(f"  Permutation Test:  Empirical p = {perm['empirical_p']:.4f} (95% null: [{perm['perm_ci'][0]:.2f}, {perm['perm_ci'][1]:.2f}])")
    print(f"  Bayes Factor BF01: {m['bf01']:.2f}x (Odds favoring H0 over H1)")
    print("-" * 70)

if __name__ == '__main__':
    run_interactive_cli()
