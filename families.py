"""Method families for the fair selection protocol (one place, used by select_report.py,
pick_selected.py and review_report.py). A family is a merging operator; inside a family every
method may pick tau (deferral) and the schedule (const / geo / probe allocation) on the training subset.
"""
OM = "0.02"                       # outlier mask: top-2% highest-norm tokens (same definition as the probe)

# family -> (method, prior field written by eval_imagenet.py, exact match?)
FAMILIES = {
    "tome": ("tome", "-", True),
    "sadtm": ("sadtm", "uniform", True),
    "pitome": ("pitome", "official", True),        # v22 rows ("-") used a non-official energy/margin
    "calib": ("calib", "soft-g4", True),
    "hybrid": ("hybrid", "hyb-soft-g4", True),
    "rpm": ("rpm", "-", True),                         # readout-preserving merging, CLS query (ours, new)
    "rpm_all": ("rpm_all", "-", True),                 # readout-preserving merging, all queries (ours, new)
    "rpm_wo": ("rpm_wo", "-", True),                   # ... with values through W_o and layer-scale
    "rpm2": ("rpm2", "-", True),                       # ... plus the second-order attention-mass term
    "mctf": ("mctf", "-", True),
    "evit": ("evit", "-", True),                       # training-free EViT (prune + fuse by [CLS] attention)
    "rpmq": ("rpmq", "-", True),
    "rpmk": ("rpmk", "-", True),
    "rpmf": ("rpmf", "-", True),
    "rpmf2": ("rpmf2", "-", True),                     # RPM-f with sqrt(size) query weights (consistent Frobenius)
    "rpms8": ("rpms", "k8", True),                    # RPM-f with a Gaussian sketch over queries (fast)
    "rpms16": ("rpms", "k16", True),
    "rpms32": ("rpms", "k32", True),
    "rpms64": ("rpms", "k64", True),
    "rpmsh16": ("rpmsh", "k16", True),               # ... sketch of the head-averaged attention
    "rpmsh32": ("rpmsh", "k32", True),
    "hybrid_rpm": ("hybrid", "hyb-soft-g4-k32", True),  # probe hybrid with sketched RPM instead of class attention
    "swrpm": ("hybrid", "rfix-sw", False),             # ToMe in the first blocks, sketched RPM after (switch point is a candidate)
    "rpml": ("rpml", "-", True),                      # RPM-lite: closed form, no attention matrix (ours)
    "rpmlnw": ("rpmlnw", "-", True),                  # ... without query whitening
    "rpmlkv": ("rpmlkv", "-", True),                  # ... key distance x value distance only
    "rpmlc": ("rpmlc", "m8", True),                  # RPM-lite on 8 key-similar candidates (fast; ours)
    "rpmlnwc": ("rpmlnwc", "m8", True),
    "rpmlkvc": ("rpmlkvc", "m8", True),
    "rpmlc4": ("rpmlc", "m4", True),
    "rpmlc16": ("rpmlc", "m16", True),
    "hybrid_lite": ("hybrid", "hyb-soft-g4-rpmlnw", True),  # probe hybrid with RPM-lite as the accurate branch
    "rpmsum": ("rpmsum", "a", False),                  # control: weighted SUM of RPM-lite's two factors (alpha is a candidate)
    "adamerge": ("adamerge", "-", True),              # AdaMerge (2026): salience-weighted similarity, re-implemented
    "adamerge2": ("adamerge2", "-", True),            # AdaMerge pair score, size-weighted averaging (stronger variant)
    "rpmd": ("rpmd", "-", True),                      # RPM-lite + replacement error (dense outputs)
    "rpmlnw_u0c": ("rpmlnw", "u0c", False),           # CLIP diagnosis: attention level capped at a quantile
    "mctf_noinfo": ("mctf", "noinfo", True),           # MCTF without informativeness
    "mctf_nosize": ("mctf", "nosize", True),           # MCTF without the size term
    "rpmsuml": ("rpmsum", "L", False),                 # control: weighted sum, ONE normalisation per layer (alpha candidate)
    "rpmgeo": ("rpmsum", "geo", True),                 # control: sum over heads of the per-head geometric mean
    "rpmnprod": ("rpmsum", "nprod", True),             # control: product of per-head normalised factors
    "rpmlnwq": ("rpmlnwq", "-", True),                # diagnostic: RPM-lite with attention level from all queries
    "sadtm_pub": ("sadtm", "saliency", True),         # SAD-TM exactly as published: saliency prior, importance weights
    "pitome_np": ("pitome", "official-noprop", True),  # PiToMe without proportional attention (reviewer M6)                       # RPM, Frobenius norm over all queries (fast; ours)                       # RPM over each token's 4 key-nearest partners (fast; ours)                       # RPM with [CLS] + 16 sampled queries (fast; ours)                       # MCTF criteria (CVPR 2024), training-free re-implementation
    # reviewer controls
    "fixedge": ("hybrid", "fix-edge2", True),          # fixed rule: ToMe in first 2 + last 2 blocks, attention elsewhere
    "fixsw": ("hybrid", "fix-sw", False),              # RAPID-style early/late switch, switch point is a candidate
    "sadtm_om": ("sadtm", f"uniform-om{OM}", True),    # SAD-TM with outlier-masked class attention
    "hybrid_om": ("hybrid", f"hyb-soft-g4-om{OM}", True),
}
BASELINES_MAIN = ("tome", "sadtm", "pitome", "mctf")
BASELINES_ALL = ("tome", "sadtm", "pitome", "mctf", "fixedge", "fixsw", "sadtm_om")

ARGS = {"tome": "--method tome", "pitome": "--method pitome", "rpm": "--method rpm", "rpm_all": "--method rpm_all", "rpm_wo": "--method rpm_wo", "rpm2": "--method rpm2", "mctf": "--method mctf", "evit": "--method evit", "rpmq": "--method rpmq", "rpmk": "--method rpmk", "rpmf": "--method rpmf", "rpmf2": "--method rpmf2", "rpml": "--method rpml", "rpmlc": "--method rpmlc --cand-m 8", "rpmlnwc": "--method rpmlnwc --cand-m 8", "rpmlkvc": "--method rpmlkvc --cand-m 8", "rpmlc4": "--method rpmlc --cand-m 4", "rpmlc16": "--method rpmlc --cand-m 16", "rpmlnw": "--method rpmlnw", "rpmsum": "--method rpmsum", "rpmsuml": "--method rpmsum --sum-mode layer", "rpmlnw_u0c": "--method rpmlnw", "mctf_noinfo": "--method mctf --mctf-drop info", "mctf_nosize": "--method mctf --mctf-drop size", "rpmgeo": "--method rpmsum --sum-mode geo", "rpmnprod": "--method rpmsum --sum-mode nprod", "adamerge": "--method adamerge", "adamerge2": "--method adamerge2", "rpmd": "--method rpmd", "rpmlnwq": "--method rpmlnwq", "rpmlkv": "--method rpmlkv", "rpms8": "--method rpms --sketch-k 8", "rpms16": "--method rpms --sketch-k 16", "rpms32": "--method rpms --sketch-k 32", "rpms64": "--method rpms --sketch-k 64", "rpmsh16": "--method rpmsh --sketch-k 16", "rpmsh32": "--method rpmsh --sketch-k 32", "pitome_np": "--method pitome --prop-attn off",
        "sadtm": "--method sadtm --weight-mode size --prop-attn on --prior uniform",
        "sadtm_pub": "--method sadtm --prior saliency",                       # + --prior-file priors/<model>.pt
        "sadtm_om": f"--method sadtm --weight-mode size --prop-attn on --prior uniform --outlier-mask {OM}",
        "calib": "--method calib --weight-mode size --prop-attn on --blend soft --gamma 4",
        "hybrid": "--method hybrid --weight-mode size --prop-attn on --blend soft --gamma 4",
        "hybrid_om": f"--method hybrid --weight-mode size --prop-attn on --blend soft --gamma 4 --outlier-mask {OM}",
        "fixedge": "--method hybrid --weight-mode size --prop-attn on --fixed-ops edge2",
        "fixsw": "--method hybrid --weight-mode size --prop-attn on",
        "hybrid_rpm": "--method hybrid --weight-mode size --prop-attn on --blend soft --gamma 4 --attn-op rpms --sketch-k 32",
        "hybrid_lite": "--method hybrid --weight-mode size --prop-attn on --blend soft --gamma 4 --attn-op rpmlnw",
        "swrpm": "--method hybrid --weight-mode size --prop-attn on --attn-op rpms --sketch-k 32"}      # + --fixed-ops swX (from the row)


def fam_of(r):
    for k, (m, p, exact) in FAMILIES.items():
        if r["method"] != m:
            continue
        pr = r.get("prior") or ""
        if not (pr == p if exact else pr.startswith(p)):
            continue
        if m in ("tome", "pitome", "rpm", "rpm_all", "rpm_wo", "rpm2", "mctf", "rpmq", "rpmk", "rpmf", "rpmf2", "rpms", "rpmsh", "rpml", "rpmlkv", "rpmlnw", "rpmlc", "rpmlnwc", "rpmlkvc", "rpmsum", "adamerge", "rpmlnwq", "adamerge2", "rpmd", "evit") or r.get("weight_mode") == "size" or k == "sadtm_pub":
            return k
    return None


def cfg_of(r):
    """Configuration a family selects among (prior distinguishes e.g. the switch point of fixsw)."""
    return (r["tau"], r.get("schedule", "const"), r.get("prior"))


def args_of(k, r):
    """eval_imagenet.py arguments that reproduce a selected row of family k (budget not included)."""
    sch = r.get("schedule", "const")
    extra = " --schedule geo" if sch == "geo" else (f" --schedule rel --rel-kappa {sch[5:]}" if sch.startswith("rel") else "")
    if k == "fixsw":
        extra += f" --fixed-ops {r['prior'][4:]}"
    if k == "rpmlnw_u0c":
        extra += f" --u0-clip {r['prior'][3:]}"
    if k in ("rpmsum", "rpmsuml"):
        extra += f" --sum-alpha {r['prior'][1:]}"
    if k == "swrpm":
        extra += f" --fixed-ops {r['prior'][5:].split('-k')[0]}"
    return f"--tau {r['tau']} {ARGS[k]}{extra}"
