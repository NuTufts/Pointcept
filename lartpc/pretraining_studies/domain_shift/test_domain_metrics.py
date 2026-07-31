#!/usr/bin/env python3
"""
Synthetic calibration tests for domain_metrics.py + bootstrap.py.

Run anywhere with numpy/sklearn (e.g. inside the pointcept container):
  python3 lartpc/pretraining_studies/domain_shift/test_domain_metrics.py

Checks (doubles as the methods-calibration evidence for the proposal):
  1. Identical Gaussians  -> AUC ~ 0.5, PAD ~ 0, MMD p-value not small.
  2. 0.5-sigma mean shift in 8/64 dims -> AUC >> 0.5, MMD p < 0.01, and the
     bootstrap CI excludes the null-split band.
  3. proto_jsd: identical multinomials ~ 0; disjoint supports = 1.
  4. CKA: self = 1; orthogonal rotation = 1 (linear invariance);
     independent features ~ 0.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bootstrap import bootstrap_ci, null_split  # noqa: E402
from domain_metrics import cka, mmd2, pad, proto_jsd  # noqa: E402

RNG = np.random.default_rng(7)
N, D = 500, 64
FAILURES = []


def check(name, cond, detail=""):
    status = "OK  " if cond else "FAIL"
    print(f"  [{status}] {name}  {detail}")
    if not cond:
        FAILURES.append(name)


print("== 1. identical Gaussians (null) ==")
Xa, Xb = RNG.normal(size=(N, D)), RNG.normal(size=(N, D))
r = pad(Xa, Xb)
check("null AUC(linear) ~ 0.5", abs(r["auc_linear"] - 0.5) < 0.06,
      f"auc={r['auc_linear']:.3f}")
check("null AUC(knn) ~ 0.5", abs(r["auc_knn"] - 0.5) < 0.06,
      f"auc={r['auc_knn']:.3f}")
# Single null p-values are uniform by construction (any value is a valid
# draw), so calibration is checked over repeats: p should look uniform.
null_ps = []
for i in range(30):
    ra, rb = (RNG.normal(size=(200, D)) for _ in range(2))
    null_ps.append(mmd2(ra, rb, n_perm=200, seed=i)["mmd2_p"])
null_ps = np.array(null_ps)
check("null MMD p uniform: mean in [0.35,0.65]",
      0.35 < null_ps.mean() < 0.65, f"mean_p={null_ps.mean():.3f}")
check("null MMD p uniform: P(p<0.1) < 0.3",
      np.mean(null_ps < 0.1) < 0.3,
      f"frac_below_0.1={np.mean(null_ps < 0.1):.2f}")

print("== 2. mean shift 0.5 sigma in 8 dims ==")
Xb2 = RNG.normal(size=(N, D))
Xb2[:, :8] += 0.5
r = pad(Xa, Xb2)
check("shift AUC(linear) > 0.75", r["auc_linear"] > 0.75,
      f"auc={r['auc_linear']:.3f}")
check("shift PAD > 0", r["pad_linear"] > 0.2, f"pad={r['pad_linear']:.3f}")
m = mmd2(Xa, Xb2, n_perm=300)
check("shift MMD p < 0.01", m["mmd2_p"] < 0.01, f"p={m['mmd2_p']:.4f}")

print("== 2b. bootstrap + null split separate signal from null ==")
bs = bootstrap_ci(pad, Xa, Xb2, n_boot=50, seed=1)
nl = null_split(pad, np.concatenate([Xa, Xb]), n_splits=8, seed=1)
lo = bs["auc_linear"]["lo"]
null_hi = nl["auc_linear"]["mean"] + 3 * nl["auc_linear"]["std"]
check("boot CI(low) above null band", lo > null_hi,
      f"ci_lo={lo:.3f} null_hi={null_hi:.3f}")

print("== 3. prototype JSD ==")
P = 256
base = RNG.dirichlet(np.ones(P))
ha = RNG.multinomial(20000, base, size=100)
hb = RNG.multinomial(20000, base, size=100)
r = proto_jsd(ha, hb)
check("same multinomial JSD ~ 0", r["proto_jsd"] < 0.02,
      f"jsd={r['proto_jsd']:.4f}")
r_dis = proto_jsd(
    np.concatenate([ha[:, :P // 2], np.zeros_like(ha[:, :P // 2])], axis=1),
    np.concatenate([np.zeros_like(ha[:, :P // 2]), ha[:, :P // 2]], axis=1))
check("disjoint support JSD = 1", abs(r_dis["proto_jsd"] - 1.0) < 1e-6,
      f"jsd={r_dis['proto_jsd']:.4f}")
check("exclusive counts symmetric",
      r_dis["proto_excl_a"] > 0 and r_dis["proto_excl_b"] > 0,
      f"excl_a={r_dis['proto_excl_a']} excl_b={r_dis['proto_excl_b']}")

print("== 4. CKA ==")
F = RNG.normal(size=(300, 32))
Q, _ = np.linalg.qr(RNG.normal(size=(32, 32)))
check("CKA(self) = 1", abs(cka(F, F)["cka_linear"] - 1.0) < 1e-8)
check("CKA rotation-invariant",
      abs(cka(F, F @ Q)["cka_linear"] - 1.0) < 1e-8)
c_ind = cka(F, RNG.normal(size=(300, 32)))["cka_linear"]
check("CKA(independent) ~ 0", c_ind < 0.2, f"cka={c_ind:.3f}")

print()
if FAILURES:
    print(f"*** {len(FAILURES)} FAILURES: {FAILURES}")
    sys.exit(1)
print("ALL METRIC CALIBRATION TESTS PASSED")
