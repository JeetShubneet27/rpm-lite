"""Fair throughput of every SELECTED configuration (saved by eval_imagenet.py --save-config), eager and
torch.compile'd, timed in alternating random order over several rounds. Within one (model, budget) group all
methods and the unmerged model are timed together, so each result is reported as a ratio to the unmerged model
measured in the same rounds (robust to GPU warm-up and throttling).

    python bench_selected.py results/cfgs/list.txt --out results/speed_sel.jsonl
Run it only when nothing else uses the GPU. Weights are random: speed depends only on token counts.
"""
import argparse, json, os, random, statistics as st, time
from collections import defaultdict
import torch
import torch._dynamo
from tm.data import load_model
from tm.patch import apply_patch

torch._dynamo.config.cache_size_limit = 64


def build(cfg, method_state):
    m = apply_patch(load_model(cfg["model"], "cuda", pretrained=False, headless=cfg["headless"],
                               img_size=cfg["img_size"])).eval()
    if method_state is not None:
        m._tm.update({k: (v.cuda() if torch.is_tensor(v) else v) for k, v in method_state.items()})
        m._tm["size"] = None
    return m


def timed(net, x, iters):
    with torch.no_grad():
        net(x)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(iters):
            net(x)
        torch.cuda.synchronize()
    return x.shape[0] * iters / (time.time() - t)


def run_group(items, rounds, iters, batch, compiled_only=False):
    first = items[0][1]
    base = build(first, None)
    nets = {} if compiled_only else {"none (eager)": base}
    nets["none (compiled)"] = torch.compile(base)
    for fam, cfg in items:
        eager = build(cfg, cfg["tm"])
        if not compiled_only:
            nets[f"{fam} (eager)"] = eager
        nets[f"{fam} (compiled)"] = torch.compile(eager)
    x = torch.randn(batch, 3, first["img_size"], first["img_size"], device="cuda")
    tput = {}
    for lab, net in nets.items():                      # warm-up and compilation outside the timed rounds
        try:
            timed(net, x, 2)
            tput[lab] = []
        except Exception as e:
            print(f"    {lab}: failed ({type(e).__name__}: {str(e)[:100]}); dropped", flush=True)
    for rd in range(rounds):
        order = list(tput)
        random.Random(rd).shuffle(order)
        for lab in order:
            tput[lab].append(timed(nets[lab], x, iters))
    del nets
    torch.cuda.empty_cache()
    return tput


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("listfile")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default="results/speed_sel.jsonl")
    ap.add_argument("--compiled-only", action="store_true", help="time only the torch.compile'd versions")
    ap.add_argument("--max-groups", type=int, default=0,
                    help="measure at most this many unfinished (model, budget) groups, then exit (0 = all). Exit code 3 "
                         "when nothing is left. Running one group per process keeps RAM flat: compiled graphs are "
                         "not freed between groups inside one process.")
    a = ap.parse_args()
    groups = defaultdict(list)
    for line in open(a.listfile):
        line = line.strip()
        if not line:
            continue
        fam, path = line.split("|")
        cfg = torch.load(path, map_location="cpu", weights_only=False)
        groups[(cfg["model"], cfg["ident"]["target_gflops"])].append((fam, cfg))
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            if l.strip():
                o = json.loads(l)
                done.add((o["model"], o["target_gflops"]))
    todo = [(gi, key, items) for gi, (key, items) in enumerate(sorted(groups.items())) if key not in done]
    if not todo:
        print("nothing left to measure", flush=True)
        raise SystemExit(3)
    if a.max_groups:
        todo = todo[:a.max_groups]
    for gi, key, items in todo:
        print(f"[{gi + 1}/{len(groups)}] {key[0]} @ {key[1]} GFLOPs: {', '.join(f for f, _ in items)}", flush=True)
        batch = a.batch
        while True:
            try:
                tput = run_group(items, a.rounds, a.iters, batch, a.compiled_only)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch <= 32:
                    raise
                batch //= 2
                print(f"    out of memory, retrying with batch {batch}", flush=True)
        res = {}
        for lab, v in tput.items():
            base = tput["none (compiled)" if "compiled" in lab else "none (eager)"]
            rel = [p / q for p, q in zip(v, base)]
            res[lab] = dict(img_s=round(st.median(v), 1), x_unmerged=round(st.median(rel), 4),
                            spread=round(max(rel) - min(rel), 4))
        cfgs = {fam: dict(gflops=cfg["gflops"], tau=cfg["ident"]["tau"], schedule=cfg["ident"]["schedule"],
                          prior=cfg["ident"]["prior"]) for fam, cfg in items}
        rec = dict(model=key[0], target_gflops=key[1], target_ratio=items[0][1]["target_ratio"], batch=batch,
                   rounds=a.rounds, gpu=torch.cuda.get_device_name(0), configs=cfgs, results=res)
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("    " + "  ".join(f"{k}: {v['x_unmerged']:.2f}" for k, v in res.items()), flush=True)


if __name__ == "__main__":
    main()
