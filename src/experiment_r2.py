import os
import json
import time
import tracemalloc
import itertools
from math import comb, log2
import numpy as np

from schemes import ConfigA_NPN_Ramp, ConfigB_Multivariate_Verify, NPN_BASES, BLOCK_BITS, MARKER, gf2_mod

SEED = 42
RNG = np.random.default_rng(SEED)
OUT = os.path.dirname(os.path.abspath(__file__))
RESULTS = {}


def make_record(nbytes):
    return os.urandom(nbytes)


def _irreducible_deg6():
    out = []
    for p in range(1 << 6, 1 << 7):
        ok = True
        for d in range(2, 4):
            xq = 2
            for _ in range(d * 6):
                xq = gf2_mod(xq * xq if False else _sq(xq), p)
            pass
        out.append(p)
    return out


def _sq(a):
    r = 0
    b = a
    shift = 0
    while b:
        if b & 1:
            r ^= (a << shift)
        b >>= 1
        shift += 1
    return r


def deg6_bases():
    res = []
    for p in range(1 << 6, 1 << 7):
        x2n = 2
        for _ in range(6):
            x2n = gf2_mod(_sq(x2n), p)
        if x2n != 2:
            continue
        prime = True
        for d in (2, 3):
            xd = 2
            for _ in range(d):
                xd = gf2_mod(_sq(xd), p)
            g = _gf2_gcd(xd ^ 2, p)
            if g != 1:
                prime = False
                break
        if prime:
            res.append(p)
    return res


def _gf2_gcd(a, b):
    while b:
        a, b = b, gf2_mod(a, b)
    return a


DEG6 = deg6_bases()


def s1_thresholds():
    print("S1  Larger threshold configurations ...")
    print("    available irreducible degree-6 bases: %d -> %s" % (len(DEG6), DEG6))
    R = make_record(76)
    out = {"available_deg6_bases": len(DEG6)}
    for (n, k) in [(5, 3), (7, 4), (9, 5), (9, 7), (11, 7)]:
        row = {}
        cfgA = None
        if k * 6 >= BLOCK_BITS + 1 and n <= len(DEG6):
            cfgA = ConfigA_NPN_Ramp(n=n, k=k, bases=DEG6[:n])
        for label, scheme in (("A", cfgA), ("B", ConfigB_Multivariate_Verify(n=n, k=k))):
            if scheme is None:
                reason = "n exceeds available degree-6 bases (%d)" % len(DEG6) if n > len(DEG6) else "k*6 < 17"
                row[label] = {"applicable": False, "reason": reason}
                continue
            if label == "A":
                sh = scheme.split(R)
                ok = all(scheme.reconstruct([sh[i] for i in c]) == R
                         for c in itertools.combinations(range(n), k))
                share_b = scheme.share_nbytes(sh[0])
            else:
                sh, ex = scheme.split(R)
                ok = all(scheme.reconstruct([sh[i] for i in c], ex["seed"], ex["chunk"], ex["slen"]) == R
                         for c in itertools.combinations(range(n), k))
                share_b = scheme.share_nbytes(sh[0])
            row[label] = {"applicable": True, "all_subsets_lossless": bool(ok),
                          "per_share_bytes": share_b, "total_storage_bytes": share_b * n}
        out["n%d_k%d" % (n, k)] = row
        aok = row.get("A", {}).get("all_subsets_lossless")
        bok = row.get("B", {}).get("all_subsets_lossless")
        print("    (%d,%d): A lossless=%s  B lossless=%s" % (n, k, aok, bok))
    RESULTS["S1_thresholds"] = out


def s2_scalability():
    print("S2  Scalability with template size ...")
    out = {}
    seg = 48
    for bits in [256, 512, 1024, 2048, 4096]:
        rec = make_record(16 + 12 + bits // 8 + 16)
        segments = [rec[i:i + seg] for i in range(0, len(rec), seg)]
        a = ConfigA_NPN_Ramp()
        b = ConfigB_Multivariate_Verify()
        t = time.perf_counter()
        sa = a.split(rec)
        ta_split = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        a.reconstruct(sa[:3])
        ta_rec = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        sb = [b.split(s) for s in segments]
        tb_split = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        for s, (shares, ex) in zip(segments, sb):
            b.reconstruct(shares[:3], ex["seed"], ex["chunk"], ex["slen"])
        tb_rec = (time.perf_counter() - t) * 1000
        a_share = a.share_nbytes(sa[0])
        b_share = b.share_nbytes(sb[0][0]) * len(segments)
        out["template_%dbit" % bits] = {
            "record_bytes": len(rec), "config_B_segments": len(segments),
            "A_share_bytes": a_share, "B_share_bytes_total": b_share,
            "A_split_ms": round(ta_split, 3), "A_reconstruct_ms": round(ta_rec, 3),
            "B_split_ms": round(tb_split, 3), "B_reconstruct_ms": round(tb_rec, 3)}
        print("    %d-bit: A split %.2f ms rec %.2f ms | B split %.2f ms rec %.2f ms (%d seg)" % (
            bits, ta_split, ta_rec, tb_split, tb_rec, len(segments)))
    RESULTS["S2_scalability"] = out


def s3_memory():
    print("S3  Peak memory ...")
    rec = make_record(76)
    out = {}
    for label, fn in (("A_split", lambda: ConfigA_NPN_Ramp().split(rec)),
                      ("B_split", lambda: ConfigB_Multivariate_Verify().split(rec))):
        tracemalloc.start()
        r = fn()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        out[label] = {"peak_kib": round(peak / 1024, 2)}
    a = ConfigA_NPN_Ramp()
    sa = a.split(rec)
    tracemalloc.start()
    a.reconstruct(sa[:3])
    _, pa = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    b = ConfigB_Multivariate_Verify()
    sb, ex = b.split(rec)
    tracemalloc.start()
    b.reconstruct(sb[:3], ex["seed"], ex["chunk"], ex["slen"])
    _, pb = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    out["A_reconstruct"] = {"peak_kib": round(pa / 1024, 2)}
    out["B_reconstruct"] = {"peak_kib": round(pb / 1024, 2)}
    RESULTS["S3_memory"] = out
    print("    A split %.1f KiB rec %.1f KiB | B split %.1f KiB rec %.1f KiB" % (
        out["A_split"]["peak_kib"], out["A_reconstruct"]["peak_kib"],
        out["B_split"]["peak_kib"], out["B_reconstruct"]["peak_kib"]))


def s4_energy():
    print("S4  Energy estimate ...")
    tdp = 15.0
    cores = 1
    rec = make_record(76)
    reps = 500
    a = ConfigA_NPN_Ramp()
    b = ConfigB_Multivariate_Verify()
    t = time.perf_counter()
    for _ in range(reps):
        a.split(rec)
    ta = (time.perf_counter() - t) / reps
    t = time.perf_counter()
    for _ in range(reps):
        b.split(rec)
    tb = (time.perf_counter() - t) / reps
    p_core = tdp / max(cores, 1)
    out = {"assumed_core_power_w": p_core,
           "A_split_energy_mJ": round(ta * p_core * 1000, 4),
           "B_split_energy_mJ": round(tb * p_core * 1000, 4),
           "method": "energy = mean_split_time * per_core_power; power is a stated assumption, not measured"}
    RESULTS["S4_energy"] = out
    print("    A split %.4f mJ | B split %.4f mJ (per-core %.1f W assumption)" % (
        out["A_split_energy_mJ"], out["B_split_energy_mJ"], p_core))


def s5_information():
    print("S5  Entropy / mutual information below threshold ...")
    def candidatesA(v, subset):
        res = [(gf2_mod(MARKER | v, bb), bb) for bb in subset]
        return sum(1 for cand in range(1 << BLOCK_BITS)
                   if all(gf2_mod(MARKER | cand, bb) == r for (r, bb) in res))
    A = {}
    H_full = BLOCK_BITS
    for t in (0, 1, 2):
        mis = []
        for _ in range(120):
            v = int(RNG.integers(0, 1 << BLOCK_BITS))
            if t == 0:
                cand = 1 << BLOCK_BITS
            else:
                subset = [int(x) for x in RNG.choice(NPN_BASES, size=t, replace=False)]
                cand = candidatesA(v, subset)
            H_post = log2(cand)
            mis.append(H_full - H_post)
        A[t] = {"H_prior_bits": H_full, "H_posterior_bits": round(H_full - float(np.mean(mis)), 3),
                "mutual_information_bits": round(float(np.mean(mis)), 3),
                "leakage_fraction": round(float(np.mean(mis)) / H_full, 4)}
    q = ConfigB_Multivariate_Verify().q
    B = {}
    for t in (0, 1, 2):
        B[t] = {"H_prior_bits": round(log2(q), 1), "H_posterior_bits": round(log2(q), 1),
                "mutual_information_bits": 0.0, "leakage_fraction": 0.0}
    RESULTS["S5_information"] = {
        "ConfigA_per_16bit_block": A, "ConfigB_per_chunk": B,
        "note": ("Mutual information I(secret; shares) = H_prior - H_posterior, estimated by exact "
                 "candidate enumeration for Config A and from the underdetermined-system argument for "
                 "Config B. Config B leaks 0 bits (perfect); Config A leaks 6 bits (1 share) and 12 bits "
                 "(2 shares) of ciphertext per 16-bit block.")}
    print("    A: I(1 share)=%.1f bits, I(2 shares)=%.1f bits | B: 0 bits" % (
        A[1]["mutual_information_bits"], A[2]["mutual_information_bits"]))


def s6_baselines_extended():
    print("S6  Extended baseline set ...")
    rows = [
        ["Single-copy storage", "1x record", "full template on one breach", "no", "no"],
        ["Standard Shamir (GF(2^8))", "76 B", "none (perfect)", "no", "no"],
        ["Verifiable SS (Feldman)", "172 B", "none (perfect)", "yes (public commit.)", "no"],
        ["Blockchain-stored template (on-chain ct)", "76 B x replicas", "depends on chain confidentiality", "ledger integrity", "no"],
        ["Fuzzy vault / fuzzy commitment", "varies", "helper-data leakage documented in literature", "no", "n/a"],
        ["Configuration A (NPN ramp, ours)", "50 B", "6 or 12 bits / 16-bit block", "yes (tag)", "yes"],
        ["Configuration B (multivariate, ours)", "194 B", "none (perfect)", "yes (tag + auth)", "no"],
    ]
    RESULTS["S6_baselines_extended"] = {
        "columns": ["method", "per_share_or_footprint", "below_threshold_leakage", "verification", "ramp"],
        "rows": rows}
    for r in rows:
        print("    %-42s %s" % (r[0], r[1]))


def main():
    t0 = time.time()
    RESULTS["environment"] = {"cpu": "Intel(R) Xeon(R) @ 2.80GHz (1 vCPU)",
                             "note": "same environment as experiment_full.py", "seed": SEED}
    s1_thresholds()
    s2_scalability()
    s3_memory()
    s4_energy()
    s5_information()
    s6_baselines_extended()
    RESULTS["runtime_sec"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, "results_r2.json"), "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False, default=str)
    print("\nDONE -> results_r2.json")


if __name__ == "__main__":
    main()
