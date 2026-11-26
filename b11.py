import sys, numpy as np
sys.path.insert(0, sys.path[0]); exec(open(sys.path[0] + "/load.py").read())
S = sorted({(m, b) for (m, b, k) in val if k == "rpmnprod"})
print(len(S), "settings with controls")
for k in ("rpmsum", "rpmsuml", "rpmgeo", "rpmnprod"):
    d = []
    for m, b in S:
        x, y = chosen(m, b, "rpmlnw"), chosen(m, b, k)
        if x and y: d.append((x["acc1"] - y["acc1"], pv(x, y), m, b))
    v = np.array([t[0] for t in d])
    hs = holm([t[1] for t in d])
    print(f"product vs {k:9s} n={len(d)} W/T/L {sum(p<.05 and a>0 for a,p,_,_ in d)}/{sum(p>=.05 for a,p,_,_ in d)}/{sum(p<.05 and a<0 for a,p,_,_ in d)} "
          f"Holm {sum(i in hs and d[i][0]>0 for i in range(len(d)))}/{sum(i in hs and d[i][0]<0 for i in range(len(d)))} mean {v.mean():+.2f} median {np.median(v):+.2f} max {v.max():+.2f} min {v.min():+.2f}")
    for a, p, m, b in sorted(d)[-3:]: print("      largest:", m[:26], b, round(a, 2))
