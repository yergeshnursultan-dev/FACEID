import argparse
import numpy as np
from sklearn.datasets import fetch_lfw_pairs
from schemes import ConfigB_Multivariate_Verify, K_THRESHOLD
from experiment_full import biohash, enroll, recover_template


def embed_images(images, encoder):
    from deepface import DeepFace
    embs = []
    for img in images:
        rep = DeepFace.represent(img_path=img, model_name=encoder, enforce_detection=False)[0]["embedding"]
        v = np.asarray(rep, dtype=float)
        embs.append(v / np.linalg.norm(v))
    return np.array(embs)


def eer_from_scores(genuine, impostor):
    thr = np.linspace(0, 1, 400)
    far = np.array([np.mean(impostor <= t) for t in thr])
    frr = np.array([np.mean(genuine > t) for t in thr])
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2)


def main():
    ap = argparse.ArgumentParser(description="Run the FaceVault pipeline on the LFW verification benchmark")
    ap.add_argument("--encoder", default="ArcFace")
    args = ap.parse_args()

    pairs = fetch_lfw_pairs(subset="10_folds", color=True, resize=1.0)
    n = len(pairs.target)
    fold = n // 10
    scheme = ConfigB_Multivariate_Verify()

    prot_g, prot_i, unprot_g, unprot_i, fold_eers = [], [], [], [], []
    for f in range(10):
        seg = slice(f * fold, (f + 1) * fold)
        pg, pi, ug, ui = [], [], [], []
        for (imA, imB), same in zip(pairs.pairs[seg], pairs.target[seg]):
            eA = embed_images([imA], args.encoder)[0]
            eB = embed_images([imB], args.encoder)[0]
            sh, meta, B = enroll(scheme, eA)
            Bstore, _ = recover_template(scheme, sh[:K_THRESHOLD], meta)
            Bprobe = biohash(eB, meta["salt"])
            Bdirect = biohash(eA, meta["salt"])
            dp = np.mean(np.unpackbits(np.frombuffer(Bstore[:meta["Blen"]], np.uint8))
                         != np.unpackbits(np.frombuffer(Bprobe[:meta["Blen"]], np.uint8)))
            du = np.mean(np.unpackbits(np.frombuffer(Bdirect[:meta["Blen"]], np.uint8))
                         != np.unpackbits(np.frombuffer(Bprobe[:meta["Blen"]], np.uint8)))
            (pg if same else pi).append(dp)
            (ug if same else ui).append(du)
        prot_g += pg
        prot_i += pi
        unprot_g += ug
        unprot_i += ui
        fold_eers.append(eer_from_scores(np.array(pg), np.array(pi)))

    fe = np.array(fold_eers)
    ci = (fe.mean() - 1.96 * fe.std(ddof=1) / np.sqrt(10),
          fe.mean() + 1.96 * fe.std(ddof=1) / np.sqrt(10))
    eer_p = eer_from_scores(np.array(prot_g), np.array(prot_i))
    eer_u = eer_from_scores(np.array(unprot_g), np.array(unprot_i))
    print("Encoder         : %s" % args.encoder)
    print("LFW pairs       : %d (10 folds of %d)" % (n, fold))
    print("EER protected   : %.3f%%" % (eer_p * 100))
    print("EER unprotected : %.3f%%" % (eer_u * 100))
    print("Per-fold EER    : %.3f%%  95%% CI [%.3f%%, %.3f%%]" % (fe.mean() * 100, ci[0] * 100, ci[1] * 100))


if __name__ == "__main__":
    main()
