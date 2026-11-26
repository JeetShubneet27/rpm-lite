"""Latency, throughput, memory and scoring overhead (mentor point 4).

All merging methods use the SAME merge schedule (ToMe's r for the budget), so they keep exactly the same number of
tokens in every block: any difference to ToMe is the cost of computing the merge score itself. Everything is
torch.compile'd and timed in alternating random order over several rounds on the same GPU.

  throughput  batch 256, images/s
  latency     batch 1 and batch 8, median milliseconds per forward pass
  memory      peak GPU memory during a forward pass (batch 256 and batch 1)

    python bench_latency.py --setting 0 --out results/latency.jsonl      (one setting per process keeps RAM flat)
"""
import argparse, json, random, statistics as st, time
import torch
import torch._dynamo
from tm.data import load_model
from tm.patch import apply_patch, configure, simulate_counts
from tm.flops import find_r_for_gflops, vit_gflops

torch._dynamo.config.cache_size_limit = 64
SETTINGS = [("deit_small_patch16_224", 2.6, None), ("deit_base_patch16_224", 9.0, None),
            ("vit_base_patch14_dinov2.lvd142m", None, 0.6), ("vit_base_patch16_clip_224.laion2b", None, 0.6),
            ("vit_large_patch14_dinov2.lvd142m", None, 0.6),
            # full appendix (revision): every remaining model at a middle budget
            ("deit_tiny_patch16_224", 0.8, None), ("deit3_small_patch16_224.fb_in1k", 2.6, None),
            ("deit3_base_patch16_224.fb_in1k", 9.0, None), ("vit_small_patch14_dinov2.lvd142m", None, 0.6),
            ("vit_small_patch14_reg4_dinov2.lvd142m", None, 0.6), ("vit_base_patch14_reg4_dinov2.lvd142m", None, 0.6),
            ("vit_large_patch14_reg4_dinov2.lvd142m", None, 0.6), ("vit_small_patch16_224.augreg_in21k_ft_in1k", None, 0.6),
            ("vit_base_patch16_224.augreg_in21k_ft_in1k", None, 0.6)]
WARMUP, TPUT_ITERS, LAT1_REPS, LAT8_REPS = 2, 10, 50, 30


def env_info():
    import platform, subprocess, timm
    try:
        drv = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        drv = ""
    return dict(gpu=torch.cuda.get_device_name(0), nvidia_smi=drv, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                torch=torch.__version__, timm=timm.__version__, python=platform.python_version(),
                precision="fp32 weights, default matmul precision", compile="torch.compile(mode='default')",
                warmup_calls=WARMUP, tput_iters=TPUT_ITERS, lat1_reps=LAT1_REPS, lat8_reps=LAT8_REPS)
METHODS = ("none", "tome", "mctf", "rpmlnw", "rpmf2")


def sync():
    torch.cuda.synchronize()


def throughput(net, x, iters):
    with torch.no_grad():
        net(x); sync()
        t = time.perf_counter()
        for _ in range(iters):
            net(x)
        sync()
    return x.shape[0] * iters / (time.perf_counter() - t)


def latency(net, x, reps):
    ts = []
    with torch.no_grad():
        net(x); sync()
        for _ in range(reps):
            t = time.perf_counter(); net(x); sync(); ts.append((time.perf_counter() - t) * 1000)
    return st.median(ts)


def peak_mem(net, x):
    with torch.no_grad():
        net(x); sync()
        torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
        net(x); sync()
    return (torch.cuda.max_memory_allocated() - base) / 2**20, torch.cuda.max_memory_allocated() / 2**20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out", default="results/latency.jsonl")
    ap.add_argument("--model", default=None, help="override: model name (with --ratio)")
    ap.add_argument("--ratio", type=float, default=0.6)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    name, gf, ratio = SETTINGS[a.setting] if a.model is None else (a.model, None, a.ratio)
    nets = {}
    for m in METHODS:
        model = apply_patch(load_model(name, "cuda", pretrained=False, img_size=a.img_size if ("dinov2" in name or a.img_size != 224) else None)).eval()
        if a.dtype == "bf16":
            model = model.to(torch.bfloat16)
        depth = len(model.blocks)
        if not nets:
            base = vit_gflops(model, simulate_counts(model, "none", [0] * depth))
            target = gf if gf else round(ratio * base, 3)
            r = find_r_for_gflops(model, "tome", target, 1)[0]
        configure(model, m, 0 if m == "none" else r, tau=1)
        nets[m] = (model, torch.compile(model))
    counts = {m: simulate_counts(nets[m][0], m, nets[m][0]._tm["r_list"]) for m in METHODS if m != "none"}
    assert all(c == counts["tome"] for c in counts.values()), "token counts differ between methods"
    S = int(nets["none"][0].patch_embed.img_size[0])
    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    xs = {a.batch: torch.randn(a.batch, 3, S, S, device="cuda", dtype=dt), 8: torch.randn(8, 3, S, S, device="cuda", dtype=dt),
          1: torch.randn(1, 3, S, S, device="cuda", dtype=dt)}
    for m, (_, c) in nets.items():                     # compile every shape outside the timed rounds
        for x in xs.values():
            with torch.no_grad():
                c(x); c(x)
    res = {m: dict(tput=[], lat1=[], lat8=[]) for m in METHODS}
    for rd in range(a.rounds):
        order = list(METHODS); random.Random(rd).shuffle(order)
        for m in order:
            c = nets[m][1]
            res[m]["tput"].append(throughput(c, xs[a.batch], TPUT_ITERS))
            res[m]["lat1"].append(latency(c, xs[1], LAT1_REPS))
            res[m]["lat8"].append(latency(c, xs[8], LAT8_REPS))
    out = dict(model=name, img_size=S, dtype=a.dtype, batch=a.batch, target_gflops=target, r=r, gpu=torch.cuda.get_device_name(0), rounds=a.rounds, methods={},
               env=env_info(), raw={m: res[m] for m in METHODS})
    for m in METHODS:
        mem256 = peak_mem(nets[m][1], xs[a.batch]); mem1 = peak_mem(nets[m][1], xs[1])
        q = lambda v: [round(x, 3) for x in st.quantiles(v, n=4)] if len(v) > 1 else [v[0]] * 3
        out["methods"][m] = dict(img_s=round(st.median(res[m]["tput"]), 1), img_s_iqr=q(res[m]["tput"]),
                                 lat1_iqr=q(res[m]["lat1"]), lat8_iqr=q(res[m]["lat8"]),
                                 lat1_ms=round(st.median(res[m]["lat1"]), 3), lat8_ms=round(st.median(res[m]["lat8"]), 3),
                                 act_mem256_mb=round(mem256[0], 1), peak_mem256_mb=round(mem256[1], 1),
                                 act_mem1_mb=round(mem1[0], 1))
    t = out["methods"]["tome"]
    for m in METHODS:
        o = out["methods"][m]
        o["tput_vs_tome"] = round(o["img_s"] / t["img_s"], 4)
        o["lat1_vs_tome"] = round(o["lat1_ms"] / t["lat1_ms"], 4)
        o["lat8_vs_tome"] = round(o["lat8_ms"] / t["lat8_ms"], 4)
    with open(a.out, "a") as f:
        f.write(json.dumps(out) + "\n")
    print(f"{name} @ {target} GFLOPs (r={r})")
    print(f"  {'method':8s} {'img/s':>8s} {'x ToMe':>7s} {'lat b1 ms':>10s} {'lat b8 ms':>10s} {'act mem b256 MB':>16s}")
    for m in METHODS:
        o = out["methods"][m]
        print(f"  {m:8s} {o['img_s']:8.1f} {o['tput_vs_tome']:7.3f} {o['lat1_ms']:10.2f} {o['lat8_ms']:10.2f} {o['act_mem256_mb']:16.0f}")


if __name__ == "__main__":
    main()
