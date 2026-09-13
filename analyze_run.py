#!/usr/bin/env python3
"""Paper-style analysis and plots for one archived model run.

Reads raw archived outputs without modifying them. Primary uncertainty uses
1,000 bootstrap draws of 10 experimental units, matching the manuscript.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import math
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


MODEL = "qwen3-vl-32b"
DISPLAY_MODEL = "Qwen3-VL-32B"
BOOTSTRAP_SIZE = 10
RECOURSE_BOOTSTRAP_SIZE = 30
ANALYSIS_SEED = 42
COND = ["control", "causal", "cfs"]
TEACH_GT = np.array([1.0, 2.0, 3.0])
REL_COND = ["no_ai", "saliency", "example_based"]
REL_COL = {"no_ai": "control_labels", "saliency": "saliency_labels", "example_based": "nns_labels"}
REL_GT = {"adult": np.array([2.0, 1.0, 3.0]), "bios": np.array([1.0, 2.0, 3.0])}
MI_COND = ["DCE", "MCE", "HYB"]
MI_GT = np.array([2.0, 1.0, 3.0])
SYS_KEYS = ["Explanation 1", "Explanation 2", "Explanation 4", "Explanation 5", "Explanation 11", "Explanation 21"]
SYS_LABELS = ["X1", "X2", "X3", "X4", "X5", "X6"]
SYS_HUMAN = np.array([5.25, 5.19, 4.97, 4.97, 4.53, 3.94])
WARREN = np.array([0.590, 0.614, 0.636])
GOYAL = np.array([0.7109, 0.7429, 0.7877])
CHEN = {
    "adult": {"mean": np.array([.611, .606, .711]), "se": np.array([.029, .042, .029])},
    "bios": {"mean": np.array([.600, .644, .712]), "se": np.array([.044, .043, .046])},
}

# Directed (better, worse) comparisons supported by the original human studies
# and the prespecified author-conclusion ground truths. Absence is unresolved.
SIGNIFICANT_CONSTRAINTS = {
    "model_improvement": [("HYB", "DCE"), ("HYB", "MCE")],
    "knowledge_extraction": [("cfs", "control")],
    "reliance_tabular": [("example_based", "no_ai"),
                           ("example_based", "saliency")],
    "reliance_nlp": [("example_based", "no_ai"),
                       ("example_based", "saliency")],
    "system_engagement": [
        ("X1", "X5"), ("X2", "X5"), ("X3", "X5"), ("X4", "X5"),
        ("X5", "X6"),
    ],
}


def rankdata(x):
    s = pd.Series(np.asarray(x, dtype=float))
    return s.rank(method="average").to_numpy()


def correlations(gt, observed):
    t = kendalltau(gt, observed, variant="b").statistic
    r = spearmanr(gt, observed).statistic
    return float(t), float(r)


def rng_for(seed, label):
    value = int.from_bytes(hashlib.sha256(f"{seed}:{label}".encode()).digest()[:8], "big")
    return np.random.default_rng(value)


def significance_aware(observed, labels, constraints, boot_observed=None,
                       marginal_constraints=None):
    """Score a model against a human partial order and its linear extensions."""
    observed = np.asarray(observed, float)
    index = {label: i for i, label in enumerate(labels)}
    pairs = [(index[better], index[worse]) for better, worse in constraints]
    marginal_pairs = [(index[better], index[worse])
                      for better, worse in (marginal_constraints or [])]

    def directions(values, selected_pairs):
        values = np.asarray(values, float)
        diff = np.column_stack([values[..., b] - values[..., w]
                                for b, w in selected_pairs])
        return np.where(np.isclose(diff, 0.0, rtol=1e-9, atol=1e-12),
                        0.0, np.sign(diff))
    valid_orders = []
    for order in itertools.permutations(range(len(labels))):  # worst -> best
        position = {item: rank for rank, item in enumerate(order)}
        if all(position[better] > position[worse] for better, worse in pairs):
            valid_orders.append(order)
    extension_stats = []
    observed_for_rank = np.round(observed, 12)
    for order in valid_orders:
        gt = np.empty(len(labels), float)
        for rank, item in enumerate(order, 1):
            gt[item] = rank
        extension_stats.append(correlations(gt, observed_for_rank))
    extension_stats = np.asarray(extension_stats, float)
    out = {
        "interpretation": "Only prespecified human-supported pairwise directions are constraints; unresolved pairs are not ties.",
        "constraints": [{"better": b, "worse": w} for b, w in constraints],
        "marginal_constraints": [{"better": b, "worse": w}
                                 for b, w in (marginal_constraints or [])],
        "n_constraints": len(pairs),
        "n_admissible_total_rankings": len(valid_orders),
        "linear_extension_tau": {
            "mean": float(np.nanmean(extension_stats[:, 0])),
            "range": [float(np.nanmin(extension_stats[:, 0])),
                      float(np.nanmax(extension_stats[:, 0]))],
        },
        "linear_extension_rho": {
            "mean": float(np.nanmean(extension_stats[:, 1])),
            "range": [float(np.nanmin(extension_stats[:, 1])),
                      float(np.nanmax(extension_stats[:, 1]))],
        },
    }
    if marginal_pairs:
        marginal_signs = directions(observed, marginal_pairs).reshape(-1)
        marginal = {
            "partial_tau": float(marginal_signs.mean()),
            "pairwise_agreement": float(((marginal_signs + 1) / 2).mean()),
            "all_constraints_satisfied": bool(np.all(marginal_signs > 0)),
        }
        if boot_observed is not None:
            marginal_boot = directions(boot_observed, marginal_pairs)
            marginal["bootstrap_all_constraints_probability"] = float(
                np.mean(np.all(marginal_boot > 0, axis=1)))
            marginal["bootstrap_partial_tau_ci95"] = ci(
                marginal_boot.mean(axis=1))
        out["marginal_sensitivity"] = marginal
    if not pairs:
        out.update({"partial_tau": None, "pairwise_agreement": None,
                    "all_constraints_satisfied": None,
                    "bootstrap_all_constraints_probability": None,
                    "bootstrap_partial_tau_ci95": [None, None]})
        return out
    point_directions = directions(observed, pairs).reshape(-1)
    for constraint, direction in zip(out["constraints"], point_directions):
        constraint["point_direction"] = int(direction)
    out["partial_tau"] = float(point_directions.mean())
    out["pairwise_averaged_tau"] = {
        "point": float(point_directions.mean()),
    }
    out["pairwise_averaged_rho"] = {
        "point": float(point_directions.mean()),
    }
    out["pairwise_agreement"] = float(((point_directions + 1) / 2).mean())
    out["all_constraints_satisfied"] = bool(np.all(point_directions > 0))
    if boot_observed is not None:
        boot_observed = np.asarray(boot_observed, float)
        signs = directions(boot_observed, pairs)
        boot_tau = signs.mean(axis=1)
        for j, constraint in enumerate(out["constraints"]):
            constraint["bootstrap_probability_correct"] = float(np.mean(signs[:, j] > 0))
            constraint["bootstrap_probability_tied"] = float(np.mean(signs[:, j] == 0))
            constraint["bootstrap_probability_reversed"] = float(np.mean(signs[:, j] < 0))
        out["bootstrap_partial_tau_mean"] = float(boot_tau.mean())
        out["bootstrap_partial_tau_ci95"] = ci(boot_tau)
        distribution = [float(v) for v in boot_tau]
        for metric in ("pairwise_averaged_tau", "pairwise_averaged_rho"):
            out[metric].update({
                "bootstrap_mean": float(boot_tau.mean()),
                "ci95": ci(boot_tau),
                "distribution": distribution,
            })
        out["bootstrap_all_constraints_probability"] = float(
            np.mean(np.all(signs > 0, axis=1)))
    return out


def promote_constraint_metrics(result):
    """Make partial-order correlations primary while retaining total-rank results."""
    sig = result["significance_aware"]
    result["forced_total_ranking_sensitivity"] = {
        "tau": result["tau"],
        "rho": result["rho"],
        "exact_ranking_probability": result["exact_ranking_probability"],
    }
    if "full_sample_sensitivity" in result:
        result["forced_total_ranking_sensitivity"]["full_sample_bootstrap"] = result.pop(
            "full_sample_sensitivity")
    result["tau"] = sig["pairwise_averaged_tau"]
    result["rho"] = sig["pairwise_averaged_rho"]
    result["exact_ranking_probability"] = sig["bootstrap_all_constraints_probability"]
    result["ground_truth_kind"] = "author-supported partial order"
    return result


def ci(x):
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else [None, None]


def pack(point, tau, rho, exact, n_units, raw, full_tau=None, full_rho=None,
         extra=None, bootstrap_size=None):
    size = BOOTSTRAP_SIZE if bootstrap_size is None else bootstrap_size
    out = {
        "n_units": int(n_units), "bootstrap_iterations": int(len(tau)), "bootstrap_size": int(size),
        "tau": {"point": point[0], "bootstrap_mean": float(np.nanmean(tau)), "ci95": ci(tau),
                "distribution": [None if not np.isfinite(v) else float(v) for v in tau]},
        "rho": {"point": point[1], "bootstrap_mean": float(np.nanmean(rho)), "ci95": ci(rho),
                "distribution": [None if not np.isfinite(v) else float(v) for v in rho]},
        "exact_ranking_probability": float(np.mean(exact)), "raw": raw,
    }
    if full_tau is not None:
        out["full_sample_sensitivity"] = {
            "bootstrap_size": int(n_units),
            "tau_mean": float(np.nanmean(full_tau)), "tau_ci95": ci(full_tau),
            "rho_mean": float(np.nanmean(full_rho)), "rho_ci95": ci(full_rho),
        }
    if extra:
        out.update(extra)
    return out


def bootstrap_mean_rank(matrix, gt, rng, n_boot, size=10):
    matrix = np.asarray(matrix, float); n = len(matrix)
    point = correlations(gt, matrix.mean(axis=0))
    tau, rho, exact = np.empty(n_boot), np.empty(n_boot), np.empty(n_boot, bool)
    gt_r = rankdata(gt)
    for b in range(n_boot):
        means = matrix[rng.integers(0, n, size=size)].mean(axis=0)
        tau[b], rho[b] = correlations(gt, means)
        exact[b] = np.array_equal(rankdata(means), gt_r)
    return point, tau, rho, exact


def bootstrap_participant_corr(matrix, gt, rng, n_boot, size=10):
    matrix = np.asarray(matrix, float); n = len(matrix)
    per = np.array([correlations(gt, row) for row in matrix])
    point = tuple(np.nanmean(per, axis=0))
    tau, rho = np.empty(n_boot), np.empty(n_boot)
    exact = np.empty(n_boot, bool); gt_r = rankdata(gt)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=size)
        tau[b], rho[b] = np.nanmean(per[idx], axis=0)
        exact[b] = np.array_equal(rankdata(matrix[idx].mean(axis=0)), gt_r)
    return point, tau, rho, exact


def literal_list(v):
    if isinstance(v, list): return v
    if not isinstance(v, str): return []
    try:
        x = ast.literal_eval(v)
        return x if isinstance(x, list) else []
    except Exception:
        return []


def mode(v):
    xs = [x for x in literal_list(v) if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs: return None
    counts = pd.Series(xs).value_counts()
    return sorted(counts[counts == counts.iloc[0]].index, key=str)[0]


def style():
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 240, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .2, "figure.facecolor": "white"})


def save(fig, out, stem):
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def analyze_teaching(run, out, rng, n_boot):
    files = list((run / "knowledge_extraction").glob(f"experiment_{MODEL}_n*_r1_final.csv"))
    p = max(files, key=lambda x: int(x.name.split("_n")[1].split("_")[0]))
    d = pd.read_csv(p); parsed = {}
    for c in COND: parsed[c] = d[f"llm_labels_{c}"].map(mode)
    valid = {c: int(parsed[c].notna().sum()) for c in COND}
    keep = np.logical_and.reduce([parsed[c].notna().to_numpy() for c in COND])
    y = pd.to_numeric(d.ground_truth_label, errors="coerce").to_numpy()
    keep &= np.isfinite(y)
    mat = np.column_stack([(parsed[c].to_numpy()[keep] == y[keep]).astype(float) for c in COND])
    point, t, r, ex = bootstrap_mean_rank(mat, TEACH_GT, rng, n_boot, BOOTSTRAP_SIZE)
    _, ft, fr, _ = bootstrap_mean_rank(mat, TEACH_GT, rng, n_boot, len(mat))
    means = mat.mean(0)
    # Accuracy uncertainty under the same paired configured-size draws.
    accboot = np.array([mat[rng.integers(0, len(mat), BOOTSTRAP_SIZE)].mean(0) for _ in range(n_boot)])
    raw = {"attempted": len(d), "valid_by_condition": valid, "complete_cases": int(len(mat)),
           "excluded_incomplete": int(len(d)-len(mat)), "accuracies": dict(zip(COND, means.tolist())),
           "source": str(p)}
    res = pack(point, t, r, ex, len(mat), raw, ft, fr,
               {"ground_truth": "counterfactual > control; causal unresolved"})
    res["significance_aware"] = significance_aware(
        means, COND, SIGNIFICANT_CONSTRAINTS["knowledge_extraction"], accboot)
    promote_constraint_metrics(res)
    x = np.arange(3); order = [2,1,0]
    fig, ax = plt.subplots(figsize=(8,5))
    q = means[order]; lo=np.percentile(accboot[:,order],2.5,0); hi=np.percentile(accboot[:,order],97.5,0)
    ax.errorbar(x, q, yerr=[q-lo,hi-q], marker="o", lw=2.5, capsize=5, label=DISPLAY_MODEL)
    ax.plot(x, WARREN[order], "x--", lw=2, ms=8, label="Warren et al. (2024)")
    ax.plot(x, GOYAL[order], "^-.", lw=2, ms=7, label="Goyal et al. (2019)")
    ax.set_xticks(x, ["Counterfactual","Causal","Control"]); ax.set_ylabel("Accuracy")
    ax.set_ylim(0,1); ax.set_title(f"Human-machine teaching: {DISPLAY_MODEL} vs human studies")
    ax.legend(); save(fig,out,"teaching_human_comparison")
    return res


def analyze_mi(run, out, rng, n_boot):
    p=run/"model_improvement"/"experiment_iterations_all.csv"; d=pd.read_csv(p)
    idx=d.groupby(["option","random_state"])["accuracy"].idxmax(); b=d.loc[idx].copy()
    b["delta"]=b.accuracy-b.baseline_accuracy
    piv=b.pivot(index="random_state",columns="option",values="delta")[MI_COND].dropna()
    mat=piv.to_numpy(); point,t,r,ex=bootstrap_participant_corr(mat,MI_GT,rng,n_boot,BOOTSTRAP_SIZE)
    _,ft,fr,_=bootstrap_participant_corr(mat,MI_GT,rng,n_boot,len(mat))
    raw={"complete_seeds":len(mat),"best_delta_means":dict(zip(MI_COND,mat.mean(0).tolist())),"source":str(p)}
    res=pack(point,t,r,ex,len(mat),raw,ft,fr,
             {"ground_truth":"HYB > DCE and HYB > MCE; DCE/MCE unresolved"})
    srng = rng_for(ANALYSIS_SEED, "significance-model-improvement")
    sigboot = np.array([mat[srng.integers(0, len(mat), BOOTSTRAP_SIZE)].mean(0)
                        for _ in range(n_boot)])
    res["significance_aware"] = significance_aware(
        mat.mean(0), MI_COND, SIGNIFICANT_CONSTRAINTS["model_improvement"], sigboot)
    promote_constraint_metrics(res)
    fig,ax=plt.subplots(figsize=(7,5)); bp=ax.boxplot([mat[:,i] for i in range(3)],tick_labels=MI_COND,patch_artist=True,showmeans=True)
    for patch,c in zip(bp["boxes"],["#e76f51","#457b9d","#2a9d8f"]): patch.set_facecolor(c); patch.set_alpha(.7)
    ax.axhline(0,color="firebrick",ls="--",lw=1); ax.set_ylabel("Best accuracy improvement")
    ax.set_title("Model improvement - ground truth order: MCE < DCE < HYB")
    save(fig,out,"model_improvement")
    return res


def reliance_matrix(run, domain):
    folder=run/("reliance_tabular" if domain=="adult" else "reliance_nlp")
    prefix="results_adult_seed_" if domain=="adult" else "results_bios_seed_"
    rows=[]; all_frames=[]; coverage={c:[0,0] for c in REL_COND}
    for p in sorted(folder.glob(f"{prefix}*_{MODEL}.csv")):
        d=pd.read_csv(p); truth=d.test_true_label.astype(str); vals=[]
        for c in REL_COND:
            pred=d[REL_COL[c]].map(mode); coverage[c][0]+=int(pred.notna().sum()); coverage[c][1]+=len(pred)
            vals.append(float((truth==pred).mean()))  # paper behavior: parse failures score incorrect
        rows.append(vals); d=d.copy(); d["__seed"]=p.stem.split("_seed_")[1].split("_")[0]; all_frames.append(d)
    return np.array(rows,float),pd.concat(all_frames,ignore_index=True),coverage


def analyze_reliance(run,out,rng,n_boot,domain):
    mat,d,cov=reliance_matrix(run,domain); gt=REL_GT[domain]
    point,t,r,ex=bootstrap_participant_corr(mat,gt,rng,n_boot,BOOTSTRAP_SIZE)
    _,ft,fr,_=bootstrap_participant_corr(mat,gt,rng,n_boot,len(mat))
    means=mat.mean(0); baseline=float((d.test_true_label.astype(str)==d.test_predicted_label.astype(str)).mean())
    raw={"seeds":len(mat),"rows":len(d),"parse_coverage":{c:{"valid":v[0],"total":v[1]} for c,v in cov.items()},
         "overall_accuracy":dict(zip(REL_COND,means.tolist())),"ai_baseline":baseline,
         "source_directory":str(run/("reliance_tabular" if domain=="adult" else "reliance_nlp"))}
    res=pack(point,t,r,ex,len(mat),raw,ft,fr,
             {"ground_truth":"example_based > no_ai and example_based > saliency"})
    # Paired seed bootstrap at the configured simulated-participant sample size.
    mb=np.array([mat[rng.integers(0,len(mat),BOOTSTRAP_SIZE)].mean(0) for _ in range(n_boot)])
    key = "reliance_tabular" if domain == "adult" else "reliance_nlp"
    res["significance_aware"] = significance_aware(
        means, REL_COND, SIGNIFICANT_CONSTRAINTS[key], mb)
    promote_constraint_metrics(res)
    qlo,qhi=np.percentile(mb,[2.5,97.5],axis=0)
    human=CHEN[domain]; x=np.arange(3); w=.36
    fig,ax=plt.subplots(figsize=(8,5))
    ax.bar(x-w/2,human["mean"],w,yerr=human["se"],capsize=4,label="Chen et al. humans",color="#4c78a8")
    ax.bar(x+w/2,means,w,yerr=[means-qlo,qhi-means],capsize=4,label=DISPLAY_MODEL,color="#f28e2b")
    ax.axhline(.625,color="firebrick",ls="--",label="Original AI baseline (62.5%)")
    ax.set_xticks(x,["No AI","Feature-based","Example-based"]); ax.set_ylim(.35,1); ax.set_ylabel("Overall accuracy")
    ax.set_title(f"Reliance calibration - {'Adult / income' if domain=='adult' else 'BIOS / biography'}")
    ax.legend(); save(fig,out,f"reliance_{domain}_human_comparison")
    return res


def recourse_corr(rating,cost,idx):
    return correlations(rankdata(cost[idx]),rankdata(-rating[idx]))


def analyze_recourse(run,out,seed,n_boot):
    p=run/"actionable_recommendation"/f"german_experiment_results_{MODEL}.csv"; d=pd.read_csv(p)
    d=d.dropna(subset=["actionability_score","sparsity","proximity"]).copy()
    d=d[d.actionability_score.between(1,7)]; rating=d.actionability_score.to_numpy(float)
    result={}
    for label,col in [("L0","sparsity"),("L1","proximity")]:
        rng = rng_for(seed, f"recourse-{label}")
        cost=d[col].to_numpy(float); n=len(d); t=np.empty(n_boot); r=np.empty(n_boot); ex=np.empty(n_boot,bool)
        for b in range(n_boot):
            draw_size = min(RECOURSE_BOOTSTRAP_SIZE, n)
            idx=rng.choice(n,draw_size,replace=False); t[b],r[b]=recourse_corr(rating,cost,idx); ex[b]=np.array_equal(rankdata(cost[idx]),rankdata(-rating[idx]))
        point=(float(np.nanmean(t)),float(np.nanmean(r)))
        # all-recourse sensitivity bootstraps n with replacement
        ft=np.empty(n_boot); fr=np.empty(n_boot)
        for b in range(n_boot):
            idx=rng.integers(0,n,n); ft[b],fr[b]=recourse_corr(rating,cost,idx)
        raw={"valid_ratings":n,"mean_rating":float(rating.mean()),"cost_column":col,"source":str(p)}
        result[label]=pack(
            point,t,r,ex,n,raw,ft,fr,{"ground_truth":"lower cost = more actionable"},
            bootstrap_size=RECOURSE_BOOTSTRAP_SIZE)
        result[label]["significance_aware"] = {
            "applicable": False,
            "reason": "The original human result is a significant nonlinear decline-then-plateau association, not a discrete total order.",
            "human_test": {"association": "nonlinear", "p_value": "<0.001"},
        }
        fig,ax=plt.subplots(figsize=(7,5)); jitter=rng_for(seed, f"jitter-{label}").normal(0,.045,len(rating))
        ax.scatter(cost,rating+jitter,s=18,alpha=.28,color="#4c78a8",edgecolor="none")
        # Paper-like smooth trend using binned means and a cubic fit when possible.
        order=np.argsort(cost); xs=cost[order]; ys=rating[order]
        bins=np.array_split(np.arange(len(xs)),min(12,len(xs)))
        bx=np.array([xs[z].mean() for z in bins]); by=np.array([ys[z].mean() for z in bins])
        deg=min(3,len(bx)-1); coef=np.polyfit(bx,by,deg); grid=np.linspace(xs.min(),xs.max(),200)
        ax.plot(grid,np.polyval(coef,grid),color="black",lw=2.2,label=f"Smoothed {DISPLAY_MODEL} trend")
        ax.set_xlabel(f"{label} cost ({'sparsity' if label=='L0' else 'proximity'})"); ax.set_ylabel("Actionability rating (1-7)")
        ax.set_title(f"Actionable recommendation ({label})\nGround truth: ratings should decrease as cost increases")
        ax.legend(); save(fig,out,f"actionable_{label.lower()}")
    return result


def analyze_sys(run,out,rng,n_boot):
    p=run/"system_engagement"/f"results_recommender_{MODEL}.pkl"
    with open(p,"rb") as f: blob=pickle.load(f)
    rr=blob.get("results",blob); arr=[np.asarray(rr[k],float) for k in SYS_KEYS]
    means=np.array([x.mean() for x in arr]); point=correlations(SYS_HUMAN,means)
    t=np.empty(n_boot); r=np.empty(n_boot); ex=np.empty(n_boot,bool); gt_r=rankdata(SYS_HUMAN)
    for b in range(n_boot):
        bm=np.array([x[rng.integers(0,len(x),BOOTSTRAP_SIZE)].mean() for x in arr]); t[b],r[b]=correlations(SYS_HUMAN,bm); ex[b]=np.array_equal(rankdata(bm),gt_r)
    ft=np.empty(n_boot); fr=np.empty(n_boot)
    for b in range(n_boot):
        bm=np.array([x[rng.integers(0,len(x),len(x))].mean() for x in arr]); ft[b],fr[b]=correlations(SYS_HUMAN,bm)
    raw={"valid_by_condition":dict(zip(SYS_LABELS,map(len,arr))),"qwen_means":dict(zip(SYS_LABELS,means.tolist())),
         "human_means":dict(zip(SYS_LABELS,SYS_HUMAN.tolist())),"source":str(p)}
    res=pack(point,t,r,ex,min(map(len,arr)),raw,ft,fr,{"ground_truth":"Herlocker et al. means; X3/X4 tied"})
    srng = rng_for(ANALYSIS_SEED, "significance-system-engagement")
    sigboot = np.array([
        [x[srng.integers(0, len(x), BOOTSTRAP_SIZE)].mean() for x in arr]
        for _ in range(n_boot)
    ])
    res["significance_aware"] = significance_aware(
        means, SYS_LABELS, SIGNIFICANT_CONSTRAINTS["system_engagement"], sigboot)
    promote_constraint_metrics(res)
    fig,ax=plt.subplots(figsize=(7,5)); ax.scatter(SYS_HUMAN,means,s=75,color="#4c78a8")
    for x,y,l in zip(SYS_HUMAN,means,SYS_LABELS): ax.annotate(l,(x,y),xytext=(5,5),textcoords="offset points",weight="bold")
    coef=np.polyfit(SYS_HUMAN,means,1); grid=np.linspace(SYS_HUMAN.min()-.05,SYS_HUMAN.max()+.05,100)
    ax.plot(grid,np.polyval(coef,grid),"r--",lw=1.5); ax.set_xlabel("Herlocker human mean"); ax.set_ylabel("Qwen mean rating")
    ax.set_title(f"System engagement: {DISPLAY_MODEL} vs human means\nKendall tau={point[0]:.2f}, Spearman rho={point[1]:.2f}")
    save(fig,out,"system_engagement_human_comparison")
    return res


def overview(results,out):
    rows=[]
    keys=[("Model improvement",results["model_improvement"]),("Teaching",results["knowledge_extraction"]),
          ("Reliance - Adult",results["reliance_tabular"]),("Reliance - BIOS",results["reliance_nlp"]),
          ("Actionable - L0",results["actionable_recommendation"]["L0"]),("Actionable - L1",results["actionable_recommendation"]["L1"]),
          ("System engagement",results.get("system_engagement"))]
    keys=[x for x in keys if x[1] is not None]
    fig,(a,b)=plt.subplots(1,2,figsize=(12,5.8),sharey=True); y=np.arange(len(keys))
    for ax,metric in [(a,"tau"),(b,"rho")]:
        val=np.array([x[1][metric]["bootstrap_mean"] for x in keys]); lo=np.array([x[1][metric]["ci95"][0] for x in keys]); hi=np.array([x[1][metric]["ci95"][1] for x in keys])
        ax.hlines(y, lo, hi, color="#2a6fbb", lw=1.5)
        ax.plot(lo, y, "|", color="#2a6fbb", ms=8)
        ax.plot(hi, y, "|", color="#2a6fbb", ms=8)
        ax.scatter(val, y, color="#2a6fbb", zorder=3)
        ax.axvline(0,color="firebrick",ls="--",lw=1)
        ax.set_xlim(-1.05,1.05); ax.set_xlabel("Kendall tau-b" if metric=="tau" else "Spearman rho"); ax.set_yticks(y,[x[0] for x in keys]); ax.invert_yaxis()
    fig.suptitle(
        f"{DISPLAY_MODEL} agreement with human-supported ground truths\n"
        f"{results['configuration']['n_bootstrap']:,} bootstrap samples; "
        f"n={BOOTSTRAP_SIZE} discrete, n={RECOURSE_BOOTSTRAP_SIZE} actionable")
    save(fig,out,"correlation_overview")
    for name,res in keys:
        rows.append({"domain":name,"tau":res["tau"]["bootstrap_mean"],"tau_lo":res["tau"]["ci95"][0],"tau_hi":res["tau"]["ci95"][1],
                     "rho":res["rho"]["bootstrap_mean"],"rho_lo":res["rho"]["ci95"][0],"rho_hi":res["rho"]["ci95"][1],"n_units":res["n_units"]})
    pd.DataFrame(rows).to_csv(out/"correlation_overview.csv",index=False)
    return rows


def main():
    global MODEL, DISPLAY_MODEL, BOOTSTRAP_SIZE, RECOURSE_BOOTSTRAP_SIZE, ANALYSIS_SEED
    ap=argparse.ArgumentParser(); ap.add_argument("--run-dir",type=Path,required=True); ap.add_argument("--output-dir",type=Path)
    ap.add_argument("--model",default="qwen3-vl-32b"); ap.add_argument("--display-name")
    ap.add_argument("--n-bootstrap",type=int,default=1000); ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--bootstrap-size",type=int,default=30)
    ap.add_argument("--recourse-bootstrap-size",type=int,default=30)
    ap.add_argument("--skip-system-engagement",action="store_true"); args=ap.parse_args()
    MODEL=args.model; DISPLAY_MODEL=args.display_name or args.model; BOOTSTRAP_SIZE=args.bootstrap_size
    RECOURSE_BOOTSTRAP_SIZE=args.recourse_bootstrap_size; ANALYSIS_SEED=args.seed
    if args.n_bootstrap < 1 or BOOTSTRAP_SIZE < 2 or RECOURSE_BOOTSTRAP_SIZE < 2:
        ap.error("bootstrap counts must be positive and bootstrap sizes >= 2")
    run=args.run_dir.resolve(); out=(args.output_dir or run/"analysis").resolve(); out.mkdir(parents=True,exist_ok=True)
    style()
    results={"model":MODEL,"display_name":DISPLAY_MODEL,"run":str(run),"configuration":{
        "n_bootstrap":args.n_bootstrap,"bootstrap_size":BOOTSTRAP_SIZE,
        "recourse_bootstrap_size":RECOURSE_BOOTSTRAP_SIZE,"seed":args.seed}}
    results["model_improvement"]=analyze_mi(run,out,rng_for(args.seed,"model-improvement"),args.n_bootstrap)
    results["knowledge_extraction"]=analyze_teaching(run,out,rng_for(args.seed,"teaching"),args.n_bootstrap)
    results["reliance_tabular"]=analyze_reliance(run,out,rng_for(args.seed,"reliance-adult"),args.n_bootstrap,"adult")
    results["reliance_nlp"]=analyze_reliance(run,out,rng_for(args.seed,"reliance-bios"),args.n_bootstrap,"bios")
    results["actionable_recommendation"]=analyze_recourse(run,out,args.seed,args.n_bootstrap)
    if not args.skip_system_engagement:
        results["system_engagement"]=analyze_sys(run,out,rng_for(args.seed,"system-engagement"),args.n_bootstrap)
    rows=overview(results,out); results["table_rows"]=rows
    with open(out/"results.json","w") as f: json.dump(results,f,indent=2,allow_nan=False)
    def fmt(x): return f"{x['bootstrap_mean']:.2f} [{x['ci95'][0]:.2f}, {x['ci95'][1]:.2f}]"
    with open(out/"main_table_row.md","w") as f:
        f.write("| Model | Model Imp. | Knowl. Ext. | Reliance (Tab) | Reliance (NLP) | Act. Rec. | System Eng. | Mean |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        cells=[]
        specs=[("model_improvement",None),("knowledge_extraction",None),("reliance_tabular",None),("reliance_nlp",None),("actionable_recommendation","L1")]
        if "system_engagement" in results: specs.append(("system_engagement",None))
        for k,sub in specs:
            z=results[k] if sub is None else results[k][sub]; cells.append(f"tau {fmt(z['tau'])}; rho {fmt(z['rho'])}")
        base=["model_improvement","knowledge_extraction","reliance_tabular","reliance_nlp"]+(["system_engagement"] if "system_engagement" in results else [])
        mt=np.mean([results[k]["tau"]["bootstrap_mean"] for k in base]+[results["actionable_recommendation"]["L1"]["tau"]["bootstrap_mean"]])
        mr=np.mean([results[k]["rho"]["bootstrap_mean"] for k in base]+[results["actionable_recommendation"]["L1"]["rho"]["bootstrap_mean"]])
        f.write(f"| {DISPLAY_MODEL} | "+" | ".join(cells)+f" | tau {mt:.2f}; rho {mr:.2f} |\n")
    paper_rows=[]
    for name,k,sub in [("Model Imp.","model_improvement",None),("Knowl. Ext.","knowledge_extraction",None),("Reliance (Tab)","reliance_tabular",None),("Reliance (NLP)","reliance_nlp",None),("Act. Rec.","actionable_recommendation","L1"),("System Eng.","system_engagement",None)]:
        if k not in results: continue
        z=results[k] if sub is None else results[k][sub]
        paper_rows.append({"domain":name,"tau":z["tau"]["bootstrap_mean"],"tau_lo":z["tau"]["ci95"][0],"tau_hi":z["tau"]["ci95"][1],"rho":z["rho"]["bootstrap_mean"],"rho_lo":z["rho"]["ci95"][0],"rho_hi":z["rho"]["ci95"][1]})
    pd.DataFrame(paper_rows).to_csv(out/"main_table_row.csv",index=False)
    sig_rows=[]
    for name,key in [("Model improvement","model_improvement"),("Teaching","knowledge_extraction"),
                     ("Reliance - Adult","reliance_tabular"),("Reliance - BIOS","reliance_nlp"),
                     ("System engagement","system_engagement")]:
        if key not in results: continue
        z=results[key]["significance_aware"]
        sig_rows.append({"domain":name,"n_constraints":z["n_constraints"],
                         "partial_tau":z["partial_tau"],
                         "bootstrap_tau_mean":z["pairwise_averaged_tau"]["bootstrap_mean"],
                         "bootstrap_tau_lo":z["pairwise_averaged_tau"]["ci95"][0],
                         "bootstrap_tau_hi":z["pairwise_averaged_tau"]["ci95"][1],
                         "bootstrap_rho_mean":z["pairwise_averaged_rho"]["bootstrap_mean"],
                         "bootstrap_rho_lo":z["pairwise_averaged_rho"]["ci95"][0],
                         "bootstrap_rho_hi":z["pairwise_averaged_rho"]["ci95"][1],
                         "pairwise_agreement":z["pairwise_agreement"],
                         "all_constraints_satisfied":z["all_constraints_satisfied"],
                         "bootstrap_all_constraints_probability":z["bootstrap_all_constraints_probability"],
                         "n_admissible_total_rankings":z["n_admissible_total_rankings"],
                         "linear_extension_tau_mean":z["linear_extension_tau"]["mean"],
                         "linear_extension_tau_min":z["linear_extension_tau"]["range"][0],
                         "linear_extension_tau_max":z["linear_extension_tau"]["range"][1],
                         "linear_extension_rho_mean":z["linear_extension_rho"]["mean"],
                         "linear_extension_rho_min":z["linear_extension_rho"]["range"][0],
                         "linear_extension_rho_max":z["linear_extension_rho"]["range"][1]})
    pd.DataFrame(sig_rows).to_csv(out/"significance_aware.csv",index=False)
    print(json.dumps({"output":str(out),"table":rows},indent=2))


if __name__=="__main__": main()
