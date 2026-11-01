"""
Signal Detection Theory Web Application: Testing Parapsychology & ESP
Assignment 2 - Centre for Neuroscience, Indian Institute of Science
Author: Shubham Yadav
"""

import os
import json
import uuid
import random
import datetime
import numpy as np
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from scipy.stats import norm

import sdt_analysis

app = Flask(__name__)
app.secret_key = 'esp_sdt_experiment_secure_token_cns_iisc'

DATA_FILE = os.path.join(os.path.dirname(__file__), 'data', 'experiments.json')

def load_experiments():
    if not os.path.exists(DATA_FILE):
        return []
    try:
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return []

def save_experiment(exp_data):
    experiments = load_experiments()
    # Check if experiment already exists, update or append
    for idx, e in enumerate(experiments):
        if e.get('id') == exp_data.get('id'):
            experiments[idx] = exp_data
            break
    else:
        experiments.append(exp_data)
        
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, 'w') as f:
        json.dump(experiments, f, indent=2)

def generate_benchmark_experiments():
    """Generates standard benchmark datasets for Subject 1 (Alice) and Subject 2 (Bob)."""
    # 1. Subject 1: Alice (Telepathy Paradigm, 50 trials)
    np.random.seed(42)
    n_s1 = 50
    gt_s1 = [1] * 25 + [0] * 25
    random.Random(42).shuffle(gt_s1)
    
    # Alice: Realistic human guessing with slight conservative bias (favoring No)
    # Target present (25): 13 Hits, 12 Misses
    # Target absent (25): 12 False Alarms, 13 Correct Rejections
    hits_target = 13
    fa_target = 12
    resp_s1 = []
    h_count, fa_count = 0, 0
    for g in gt_s1:
        if g == 1:
            if h_count < hits_target:
                resp_s1.append(1)
                h_count += 1
            else:
                resp_s1.append(0)
        else:
            if fa_count < fa_target:
                resp_s1.append(1)
                fa_count += 1
            else:
                resp_s1.append(0)

    m1 = sdt_analysis.compute_sdt_metrics(hits=hits_target, false_alarms=fa_target, n_signal=25, n_noise=25)
    perm1 = sdt_analysis.run_permutation_test(gt_s1, resp_s1, n_permutations=5000, seed=101)

    exp1 = {
        'id': 'exp-alice-telepathy-01',
        'participant': 'Alice (Subject 1)',
        'paradigm': 'telepathy',
        'created_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        'ground_truth': gt_s1,
        'responses': resp_s1,
        'ratings': None,
        'metrics': m1,
        'perm_results': {
            'empirical_p': perm1['empirical_p'],
            'perm_mean': perm1['perm_mean'],
            'perm_std': perm1['perm_std'],
            'perm_ci': perm1['perm_ci']
        },
        'roc_data': None
    }

    # 2. Subject 2: Bob (Clairvoyance Confidence Rating ROC Paradigm, 60 trials)
    np.random.seed(123)
    n_s2 = 60
    gt_s2 = [1] * 30 + [0] * 30
    random.Random(123).shuffle(gt_s2)
    
    # Bob: 4-point rating (1..4)
    # Signal trials distribution over 1..4: [5, 9, 10, 6]
    # Noise trials distribution over 1..4:  [6, 9, 10, 5]
    # At binary threshold rating >= 3:
    # Hits = 10 + 6 = 16 / 30
    # False Alarms = 10 + 5 = 15 / 30
    ratings_s2 = []
    sig_ratings = [1]*5 + [2]*9 + [3]*10 + [4]*6
    noise_ratings = [1]*6 + [2]*9 + [3]*10 + [4]*5
    random.Random(321).shuffle(sig_ratings)
    random.Random(654).shuffle(noise_ratings)
    
    s_idx, n_idx = 0, 0
    resp_s2 = []
    for g in gt_s2:
        if g == 1:
            r = sig_ratings[s_idx]
            s_idx += 1
        else:
            r = noise_ratings[n_idx]
            n_idx += 1
        ratings_s2.append(r)
        resp_s2.append(1 if r >= 3 else 0)

    m2 = sdt_analysis.compute_sdt_metrics(hits=16, false_alarms=15, n_signal=30, n_noise=30)
    perm2 = sdt_analysis.run_permutation_test(gt_s2, resp_s2, n_permutations=5000, seed=202)
    roc2 = sdt_analysis.compute_roc_from_ratings(ratings_s2, gt_s2)

    exp2 = {
        'id': 'exp-bob-clairvoyance-02',
        'participant': 'Bob (Subject 2)',
        'paradigm': 'clairvoyance',
        'created_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        'ground_truth': gt_s2,
        'responses': resp_s2,
        'ratings': ratings_s2,
        'metrics': m2,
        'perm_results': {
            'empirical_p': perm2['empirical_p'],
            'perm_mean': perm2['perm_mean'],
            'perm_std': perm2['perm_std'],
            'perm_ci': perm2['perm_ci']
        },
        'roc_data': roc2
    }

    return [exp1, exp2]

@app.route('/')
def index():
    experiments = load_experiments()
    if not experiments:
        # Pre-seed benchmark datasets so the user immediately has complete data
        benchmarks = generate_benchmark_experiments()
        for b in benchmarks:
            save_experiment(b)
        experiments = benchmarks
    return render_template('index.html', experiments=experiments)

@app.route('/load-demo-data', methods=['POST'])
def load_demo_data():
    benchmarks = generate_benchmark_experiments()
    for b in benchmarks:
        save_experiment(b)
    return redirect(url_for('comparison'))

@app.route('/start', methods=['POST'])
def start_experiment():
    participant_name = request.form.get('participant_name', 'Subject').strip()
    paradigm = request.form.get('paradigm', 'telepathy').strip()
    total_trials = int(request.form.get('total_trials', 50))
    
    session.clear()
    session['exp_id'] = str(uuid.uuid4())[:8]
    session['participant'] = participant_name
    session['paradigm'] = paradigm
    session['total_trials'] = total_trials
    session['trial'] = 0
    
    # Generate balanced ground truth
    n_sig = total_trials // 2
    n_noi = total_trials - n_sig
    gt = [1] * n_sig + [0] * n_noi
    random.shuffle(gt)
    session['ground_truth'] = gt
    session['responses'] = []
    session['ratings'] = []
    
    return redirect(url_for('trial'))

@app.route('/trial', methods=['GET', 'POST'])
def trial():
    exp_id = session.get('exp_id')
    if not exp_id:
        return redirect(url_for('index'))

    idx = session.get('trial', 0)
    total = session.get('total_trials', 50)
    paradigm = session.get('paradigm', 'telepathy')
    participant = session.get('participant', 'Subject')
    gt = session.get('ground_truth', [])

    if request.method == 'POST':
        if paradigm == 'clairvoyance':
            rating = int(request.form.get('rating', 2))
            ratings = session.get('ratings', [])
            ratings.append(rating)
            session['ratings'] = ratings
            # Binary threshold: rating >= 3 is Yes / Present
            choice = 1 if rating >= 3 else 0
        else:
            choice = int(request.form.get('choice', 0))
            
        responses = session.get('responses', [])
        responses.append(choice)
        session['responses'] = responses
        
        idx += 1
        session['trial'] = idx

    if idx >= total:
        # Experiment complete: compute statistics and persist
        y_true = np.array(gt)
        y_pred = np.array(session.get('responses', []))
        ratings = session.get('ratings', [])
        
        n_signal = int(np.sum(y_true == 1))
        n_noise = int(np.sum(y_true == 0))
        hits = int(np.sum((y_true == 1) & (y_pred == 1)))
        fas = int(np.sum((y_true == 0) & (y_pred == 1)))

        metrics = sdt_analysis.compute_sdt_metrics(hits, fas, n_signal, n_noise)
        perm = sdt_analysis.run_permutation_test(y_true, y_pred, n_permutations=5000)
        
        roc_data = None
        if paradigm == 'clairvoyance' and len(ratings) == total:
            roc_data = sdt_analysis.compute_roc_from_ratings(ratings, y_true)

        exp_record = {
            'id': exp_id,
            'participant': participant,
            'paradigm': paradigm,
            'created_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
            'ground_truth': gt,
            'responses': session.get('responses', []),
            'ratings': ratings if paradigm == 'clairvoyance' else None,
            'metrics': metrics,
            'perm_results': {
                'empirical_p': perm['empirical_p'],
                'perm_mean': perm['perm_mean'],
                'perm_std': perm['perm_std'],
                'perm_ci': perm['perm_ci']
            },
            'roc_data': roc_data
        }
        save_experiment(exp_record)
        return redirect(url_for('view_report', exp_id=exp_id))

    if paradigm == 'clairvoyance':
        return render_template('trial_clairvoyance.html', trial_num=idx + 1, total=total, participant=participant)
    else:
        is_signal = int(gt[idx]) if idx < len(gt) else 0
        return render_template('trial_telepathy.html', trial_num=idx + 1, total=total, participant=participant, is_signal=is_signal)

@app.route('/report/<exp_id>')
def view_report(exp_id):
    experiments = load_experiments()
    exp = next((e for e in experiments if e.get('id') == exp_id), None)
    if not exp:
        return redirect(url_for('index'))

    m = exp['metrics']
    perm_res = exp['perm_results']
    
    # Re-run quick permutation to get vector for histogram
    gt = np.array(exp['ground_truth'])
    resp = np.array(exp['responses'])
    perm_full = sdt_analysis.run_permutation_test(gt, resp, n_permutations=2000, seed=42)
    
    svg_dist = sdt_analysis.generate_sdt_distributions_svg(m['d_prime'], m['c'], m['b'])
    svg_perm = sdt_analysis.generate_permutation_histogram_svg(perm_full['perm_dps'], m['d_prime'], perm_res['empirical_p'])
    
    svg_roc = None
    if exp.get('roc_data'):
        svg_roc = sdt_analysis.generate_roc_svg(exp['roc_data']['roc_points'], m['d_prime'])

    return render_template('report_individual.html',
                           participant=exp['participant'],
                           paradigm=exp['paradigm'],
                           metrics=m,
                           perm_results=perm_res,
                           roc_data=exp.get('roc_data'),
                           svg_distribution=svg_dist,
                           svg_permutation=svg_perm,
                           svg_roc=svg_roc)

@app.route('/comparison')
def comparison():
    experiments = load_experiments()
    if not experiments:
        benchmarks = generate_benchmark_experiments()
        for b in benchmarks:
            save_experiment(b)
        experiments = benchmarks

    # Generate comparison bar chart
    comparison_data = [{'name': e['participant'], 'paradigm': e['paradigm'].capitalize(), 'metrics': e['metrics']} for e in experiments]
    svg_comparison = sdt_analysis.generate_comparison_bar_svg(comparison_data)

    # Inverse-variance weighted pooled meta-analysis
    weights = [1.0 / max(1e-6, e['metrics']['se_dp'] ** 2) for e in experiments]
    d_primes = [e['metrics']['d_prime'] for e in experiments]
    sum_w = sum(weights)
    pooled_dp = sum(w * d for w, d in zip(weights, d_primes)) / sum_w
    pooled_se = float(1.0 / np.sqrt(sum_w))
    pooled_z = pooled_dp / pooled_se
    pooled_p = float(2.0 * (1.0 - norm.cdf(abs(pooled_z))))
    pooled_ci = (pooled_dp - 1.96 * pooled_se, pooled_dp + 1.96 * pooled_se)
    
    # Joint Bayes factor BF01
    joint_bf01 = float(np.prod([e['metrics']['bf01'] for e in experiments]))
    total_trials = sum(e['metrics']['total_trials'] for e in experiments)

    pooled_summary = {
        'd_prime': pooled_dp,
        'se': pooled_se,
        'z': pooled_z,
        'p': pooled_p,
        'ci_95': pooled_ci,
        'joint_bf01': joint_bf01,
        'total_trials': total_trials
    }

    return render_template('comparison.html',
                           experiments=experiments,
                           svg_comparison=svg_comparison,
                           pooled=pooled_summary)

@app.route('/full-report')
def full_report():
    return render_template('full_scientific_paper.html')

@app.route('/api/experiments')
def api_experiments():
    return jsonify(load_experiments())

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    print(f"Starting SDT Parapsychology Lab Server on http://127.0.0.1:{port}")
    app.run(debug=True, host='0.0.0.0', port=port)
