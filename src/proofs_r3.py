import os
import json
import hashlib
import itertools
from collections import Counter
import numpy as np
from schemes import ConfigB_Multivariate_Verify, ConfigA_NPN_Ramp, N_SHARES, K_THRESHOLD

OUT = os.path.dirname(os.path.abspath(__file__))
RESULTS = {}


def perfect_secrecy_small_field():
    q = 101
    k = 3
    n = 5

    def row(pid, seed):
        return [int.from_bytes(hashlib.sha512(seed + bytes([pid, j])).digest(), "big") % q for j in range(k)]

    seed = os.urandom(16)
    A = [row(i, seed) for i in range(1, n + 1)]
    true_secret = [37, 88, 5]

    below = {}
    for t in (1, 2):
        obs = list(range(t))
        y_obs = {i: sum(A[i][j] * true_secret[j] for j in range(k)) % q for i in obs}
        per_s1 = Counter()
        for s1 in range(q):
            for s2 in range(q):
                for s3 in range(q):
                    sv = (s1, s2, s3)
                    if all(sum(A[i][j] * sv[j] for j in range(k)) % q == y_obs[i] for i in obs):
                        per_s1[s1] += 1
        counts = [per_s1[s1] for s1 in range(q)]
        below[t] = {"observed_shares": t,
                    "total_consistent_vectors": int(sum(counts)),
                    "expected_total_q_pow": q ** (k - t),
                    "solutions_per_secret_value": counts[0],
                    "uniform_over_secret": len(set(counts)) == 1}
    RESULTS["perfect_secrecy_enumeration"] = {
        "field_q": q, "k": k, "n": n,
        "statement": "For any t < k observed shares, the number of full secret vectors consistent with the "
                     "observation is q^(k-t), and each value of the secret component is consistent with exactly "
                     "q^(k-t-1) of them; hence every secret value is equally likely and I(secret; shares)=0.",
        "below_threshold": below}
    print("Perfect secrecy (q=%d): t=1 total=%d (=q^2=%d) per-secret=%d uniform=%s | t=2 total=%d (=q^1=%d) per-secret=%d uniform=%s" % (
        q, below[1]["total_consistent_vectors"], q ** 2, below[1]["solutions_per_secret_value"], below[1]["uniform_over_secret"],
        below[2]["total_consistent_vectors"], q, below[2]["solutions_per_secret_value"], below[2]["uniform_over_secret"]))


def posterior_equals_prior():
    q = 101
    k = 3
    n = 5

    def row(pid, seed):
        return [int.from_bytes(hashlib.sha512(seed + bytes([pid, j])).digest(), "big") % q for j in range(k)]

    seed = os.urandom(16)
    A = [row(i, seed) for i in range(1, n + 1)]
    obs = [0, 1]
    true_secret = [37, 88, 5]
    y_obs = {i: sum(A[i][j] * true_secret[j] for j in range(k)) % q for i in obs}

    posterior_s1 = Counter()
    total = 0
    for s1 in range(q):
        for s2 in range(q):
            for s3 in range(q):
                sv = [s1, s2, s3]
                if all(sum(A[i][j] * sv[j] for j in range(k)) % q == y_obs[i] for i in obs):
                    posterior_s1[s1] += 1
                    total += 1
    probs = np.array([posterior_s1[s1] / total for s1 in range(q)])
    RESULTS["posterior_equals_prior"] = {
        "observed_shares": len(obs), "field_q": q,
        "prior_prob": 1.0 / q, "posterior_min": float(probs.min()),
        "posterior_max": float(probs.max()),
        "posterior_is_uniform": bool(np.allclose(probs, 1.0 / q)),
        "max_abs_deviation_from_uniform": float(np.max(np.abs(probs - 1.0 / q)))}
    print("Posterior over secret with 2 shares uniform=%s (max dev %.2e, prior=1/%d)" % (
        RESULTS["posterior_equals_prior"]["posterior_is_uniform"],
        RESULTS["posterior_equals_prior"]["max_abs_deviation_from_uniform"], q))


def verification_forgery():
    scheme = ConfigB_Multivariate_Verify()
    R = os.urandom(76)
    shares, ex = scheme.split(R)
    trials = 20000
    accepted_forgeries = 0
    for _ in range(trials):
        forged = dict(shares[0])
        forged["share"] = int.from_bytes(os.urandom(66), "big") % scheme.q
        exp = hashlib.sha256(forged["share"].to_bytes(scheme._qb, "big") + forged["SK"]).digest()
        if exp == forged["SH"]:
            accepted_forgeries += 1
    RESULTS["verification_forgery"] = {
        "trials": trials, "accepted_forgeries": accepted_forgeries,
        "empirical_forgery_rate": accepted_forgeries / trials,
        "theoretical_bound": "2^-256 (SHA-256 preimage/collision resistance)",
        "statement": "A forged share y' passes verification only if SHA-256(y'||SK) equals the stored SH; "
                     "without SK this succeeds with probability at most 2^-256."}
    print("Verification forgery: %d/%d accepted (bound 2^-256)" % (accepted_forgeries, trials))


def tamper_detection_completeness():
    out = {}
    for scheme, key in ((ConfigA_NPN_Ramp(), "A"), (ConfigB_Multivariate_Verify(), "B")):
        R = os.urandom(76)
        if key == "A":
            shares = scheme.split(R)
        else:
            shares, ex = scheme.split(R)
        trials = 5000
        detected = 0
        for _ in range(trials):
            bad = [dict(s) for s in shares]
            if key == "A":
                d = bytearray(bad[0]["data"])
                pos = int.from_bytes(os.urandom(2), "big") % len(d)
                d[pos] ^= (1 + (int.from_bytes(os.urandom(1), "big") % 63))
                bad[0]["data"] = bytes(d)
            else:
                bad[0]["share"] = (bad[0]["share"] + 1 + int.from_bytes(os.urandom(66), "big")) % scheme.q
            try:
                if key == "A":
                    scheme.reconstruct([bad[0]] + shares[1:K_THRESHOLD])
                else:
                    scheme.reconstruct([bad[0]] + shares[1:K_THRESHOLD], ex["seed"], ex["chunk"], ex["slen"])
            except ValueError:
                detected += 1
        out[key] = {"trials": trials, "detected": detected, "detection_rate": detected / trials}
    RESULTS["tamper_detection"] = out
    print("Tamper detection: A %d/%d, B %d/%d" % (
        out["A"]["detected"], out["A"]["trials"], out["B"]["detected"], out["B"]["trials"]))


def main():
    perfect_secrecy_small_field()
    posterior_equals_prior()
    verification_forgery()
    tamper_detection_completeness()
    with open(os.path.join(OUT, "results_r3.json"), "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False, default=str)
    print("\nDONE -> results_r3.json")


if __name__ == "__main__":
    main()
