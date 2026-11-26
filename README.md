# RPM-lite: readout-preserving token merging for Vision Transformers

Code for the paper *"What Does Merging Two Tokens Cost? A Readout-Preserving Criterion for Token Merging in Vision
Transformers"* (Shubneet, Department of Mechanical Engineering, National Central University, Taiwan).

Token merging speeds up a pretrained ViT without retraining by averaging tokens inside each block. RPM-lite chooses the
pairs to merge with a closed-form cost derived from the first-order change of the attention output:

```
C(i, j) = s_i s_j / (s_i + s_j) * sum_h sqrt(u0_i^h u0_j^h) * ||k_i^h - k_j^h|| * ||v_i^h - v_j^h||
```

(sizes `s`, per-patch [CLS] attention `u0`, per-head keys `k` and values `v`). It needs no attention matrix, adds only a
few percent to the cost of ToMe, and works with any timm ViT; for ViTs without a [CLS] token the mean query is used.
The repository also contains every baseline we compare with (ToMe, SAD-TM, PiToMe, MCTF, AdaMerge, training-free EViT),
the label-free reliability probe, and all scripts behind the tables and figures of the paper.

## Install

```bash
conda create -n tm python=3.11 -y && conda activate tm
pip install -r requirements.txt            # PyTorch with CUDA: see https://pytorch.org for your CUDA version
```

Tested with Python 3.11, PyTorch 2.11 (CUDA 12.8), timm 1.0.31 on an RTX 5060 Ti (16 GB).

## Data

ImageNet-1k in the usual folder layout (`train/<wnid>/*.JPEG`, `val/<wnid>/*.JPEG`; `prepare_imagenet.py` sorts the
validation images into class folders). ADE20K (`ADEChallengeData2016`) for the segmentation experiment.

```bash
export IMAGENET=/path/to/imagenet
```

## Quick start

Evaluate RPM-lite on DeiT-S at 2.6 GFLOPs (merging starts after one block, constant schedule):

```bash
python eval_imagenet.py --data $IMAGENET/val --model deit_small_patch16_224 \
       --method rpmlnw --target-gflops 2.6 --tau 1 --out results/demo.jsonl
```

Other methods: `--method tome | sadtm | pitome | mctf | adamerge | evit | rpmlnw (RPM-lite) | rpmf2 (exact RPM) | hybrid`
(probe-guided; add `--weight-mode size --prop-attn on --blend soft --gamma 4 --attn-op rpmlnw`). Budgets:
`--target-gflops G` or `--target-ratio R` (share of the unmerged FLOPs). Schedules: `--tau T` (first merging block),
`--schedule const | geo | rel` (`rel` = probe-based allocation, with `--rel-kappa 1|2|4`).

Using RPM-lite in your own code:

```python
import timm, torch
from tm.patch import apply_patch, configure
from tm.flops import find_r_for_gflops

model = apply_patch(timm.create_model("deit_small_patch16_224", pretrained=True).eval().cuda())
r, gflops = find_r_for_gflops(model, "rpmlnw", 2.6, 1)     # merges per block for a 2.6 GFLOP budget
configure(model, "rpmlnw", r, tau=1)
logits = model(torch.randn(8, 3, 224, 224).cuda())
```

## Reproducing the paper

| Paper | Script |
|---|---|
| Reliability of [CLS] attention and key similarity (Fig. 2), probe | `measure_reliability.py` |
| Linear probes for DINOv2 / CLIP | `train_probe.py` |
| Selection on training images, evaluation on validation | `eval_imagenet.py`, `pick_selected.py` (grid: `scripts/run_*.sh`) |
| Theory check, assumptions, damage study (Sec. 6.2-6.4) | `rev_diag.py` |
| Factor scales during merging (Fig. factors) | `rev_factors.py` |
| Latency, throughput, memory (Tables 7, B) | `bench_latency.py` |
| Accuracy at equal measured speed | `bench_selected.py` |
| KL / JS / flip rate / ECE | `fidelity.py` |
| ADE20K segmentation | `seg_ade20k.py` |
| Merge visualisations | `vis_merge.py` |
| Statistics, tables, figures | `analysis/` (see below) |

The batch scripts in `scripts/` are the exact runs behind the paper (they expect a `venv/bin/activate` that activates
the environment and `IMAGENET` set; all are resumable).

`results/` contains the raw result files: one JSON line per run with the configuration, FLOPs and accuracy, for every
selection run (training images) and every validation evaluation, plus the latency, fidelity, segmentation, probe and
factor-scale measurements. The per-image predictions used for the McNemar tests are too large for the repository and are
provided as a release archive (`preds.zip`); unpack it into `results/preds/`.

### Analysis scripts

Run from the repository root with `PYTHONPATH=.`, e.g. `PYTHONPATH=. python analysis/b6.py results`.

| Script | Output |
|---|---|
| `load.py` | shared loader: selection of each method's configuration on training images, McNemar, Holm |
| `b6.py` | main comparison (Tables main, per-baseline), hierarchical bootstrap, severity, SAD-TM |
| `b10.py` | Friedman test, Nemenyi critical difference, effect sizes, McNemar odds ratios |
| `b11.py` | product vs. weighted-sum controls |
| `b2.py`, `b7.py` | accuracy at equal measured speed |
| `b4.py` | per-setting table (appendix) |
| `rd1.py` | theory check, assumptions and damage study from `rev_diag.py` output |
| `figs*.py`, `figcd.py`, `figvis.py` | figures (300 dpi) |

## Citation

```bibtex
@article{shubneet2026rpm,
  title   = {What Does Merging Two Tokens Cost? A Readout-Preserving Criterion for Token Merging in Vision Transformers},
  author  = {Shubneet},
  journal = {Submitted to Neurocomputing},
  year    = {2026}
}
```

## License

MIT (see `LICENSE`). Model weights come from timm and keep their own licenses.
