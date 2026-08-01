import os
import hmac
import struct
import hashlib

N_SHARES = 5
K_THRESHOLD = 3

NPN_BASES = [67, 87, 103, 109, 115]
BLOCK_BITS = 16
MARKER = 1 << BLOCK_BITS
Q_B = (1 << 521) - 1


def poly_deg(p):
    return p.bit_length() - 1 if p > 0 else -1


def gf2_mul(a, b):
    r = 0
    while b:
        if b & 1:
            r ^= a
        a <<= 1
        b >>= 1
    return r


def gf2_divmod(a, b):
    db = poly_deg(b)
    q = 0
    r = a
    while poly_deg(r) >= db and r:
        s = poly_deg(r) - db
        q ^= (1 << s)
        r ^= (b << s)
    return q, r


def gf2_mod(a, b):
    return gf2_divmod(a, b)[1]


def _ext_gcd(a, b):
    if b == 0:
        return a, 1, 0
    g, x1, y1 = _ext_gcd(b, gf2_mod(a, b))
    q = gf2_divmod(a, b)[0]
    return g, y1, x1 ^ gf2_mul(q, y1)


def gf2_inv(a, m):
    g, x, _ = _ext_gcd(a, m)
    if g != 1:
        raise ValueError("no inverse")
    return gf2_mod(x, m)


class ConfigA_NPN_Ramp:
    name = "Configuration A (NPN + CRT, ramp)"

    def __init__(self, n=N_SHARES, k=K_THRESHOLD, bases=NPN_BASES):
        self.n = n
        self.k = k
        self.bases = bases
        assert k * 6 >= BLOCK_BITS + 1

    def _blocks(self, data):
        pad = data.ljust((len(data) + 1) // 2 * 2, b"\x00")
        return [int.from_bytes(pad[i:i + 2], "big") for i in range(0, len(pad), 2)], len(data)

    def split(self, secret):
        blocks, raw_len = self._blocks(secret)
        residues = [[] for _ in range(self.n)]
        for v in blocks:
            S = MARKER | v
            for i in range(self.n):
                residues[i].append(gf2_mod(S, self.bases[i]))
        shares = []
        for i in range(self.n):
            payload = bytes(residues[i])
            tag = hashlib.sha256(payload + struct.pack(">I", self.bases[i])).digest()[:4]
            shares.append({"id": i + 1, "base": self.bases[i], "data": payload,
                           "tag": tag, "nblocks": len(blocks), "raw_len": raw_len})
        return shares

    def _crt_block(self, residues, bases):
        P = 1
        for p in bases:
            P = gf2_mul(P, p)
        acc = 0
        for a, p in zip(residues, bases):
            Mi, _ = gf2_divmod(P, p)
            Ni = gf2_inv(gf2_mod(Mi, p), p)
            acc ^= gf2_mul(gf2_mul(a, Mi), Ni)
        return gf2_mod(acc, P) & (MARKER - 1)

    def reconstruct(self, selected):
        if len(selected) < self.k:
            raise ValueError("threshold not met")
        for sh in selected:
            exp = hashlib.sha256(sh["data"] + struct.pack(">I", sh["base"])).digest()[:4]
            if not hmac.compare_digest(sh["tag"], exp):
                raise ValueError("share %d integrity check failed" % sh["id"])
        chosen = selected[:self.k]
        bases = [sh["base"] for sh in chosen]
        nb = chosen[0]["nblocks"]
        raw_len = chosen[0]["raw_len"]
        out = bytearray()
        for b in range(nb):
            v = self._crt_block([sh["data"][b] for sh in chosen], bases)
            out += int(v).to_bytes(2, "big")
        return bytes(out[:raw_len])

    def share_nbytes(self, sh):
        return len(sh["data"]) + len(sh["tag"]) + 4 + 4


class ConfigB_Multivariate_Verify:
    name = "Configuration B (multivariate linear + verification, perfect)"

    def __init__(self, n=N_SHARES, k=K_THRESHOLD, q=Q_B):
        self.n = n
        self.k = k
        self.q = q
        self._qb = (q.bit_length() + 7) // 8

    def _row(self, pid, seed):
        return [int.from_bytes(hashlib.sha512(seed + struct.pack(">II", pid, j)).digest(), "big") % self.q
                for j in range(self.k)]

    def split(self, secret):
        chunk = (len(secret) + self.k - 1) // self.k
        padded = secret.ljust(chunk * self.k, b"\x00")
        s_vec = [int.from_bytes(padded[j * chunk:(j + 1) * chunk], "big") for j in range(self.k)]
        assert all(sv < self.q for sv in s_vec)
        seed = os.urandom(32)
        msk = os.urandom(32)
        shares = []
        for i in range(1, self.n + 1):
            row = self._row(i, seed)
            yi = sum(row[j] * s_vec[j] for j in range(self.k)) % self.q
            cb = b"".join(c.to_bytes(self._qb, "big") for c in row)
            UH = hashlib.sha256(cb).digest()
            SK = hashlib.sha256(msk + struct.pack(">I", i)).digest()
            signUH = hashlib.sha256(UH + SK).digest()
            SH = hashlib.sha256(yi.to_bytes(self._qb, "big") + SK).digest()
            shares.append({"id": i, "share": yi, "UH": UH, "signUH": signUH, "SH": SH, "SK": SK})
        extra = {"seed": seed, "chunk": chunk, "slen": len(secret)}
        return shares, extra

    def _gauss(self, A, y):
        k = len(y)
        q = self.q
        M = [A[i][:] + [y[i]] for i in range(k)]
        for c in range(k):
            piv = next((r for r in range(c, k) if M[r][c] % q), None)
            if piv is None:
                raise ValueError("singular")
            M[c], M[piv] = M[piv], M[c]
            inv = pow(M[c][c] % q, q - 2, q)
            M[c] = [v * inv % q for v in M[c]]
            for r in range(k):
                if r != c and M[r][c] % q:
                    f = M[r][c]
                    M[r] = [(M[r][j] - f * M[c][j]) % q for j in range(k + 1)]
        return [M[i][k] % q for i in range(k)]

    def reconstruct(self, selected, seed, chunk, slen):
        if len(selected) < self.k:
            raise ValueError("threshold not met")
        chosen = selected[:self.k]
        rows = []
        for sh in chosen:
            exp = hashlib.sha256(sh["share"].to_bytes(self._qb, "big") + sh["SK"]).digest()
            if not hmac.compare_digest(sh["SH"], exp):
                raise ValueError("share %d verification failed" % sh["id"])
            rows.append(self._row(sh["id"], seed))
        s_vec = self._gauss(rows, [sh["share"] for sh in chosen])
        raw = b"".join(sv.to_bytes(chunk, "big") for sv in s_vec)
        return raw[:slen]

    def share_nbytes(self, sh):
        return self._qb + 32 + 32 + 32 + 32
