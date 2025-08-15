import os, random, hashlib
from math import gcd

def H_bytes(x: str) -> bytes:
    return hashlib.sha256(x.encode('utf-8')).digest()

def H_int(x: str, mod: int) -> int:
    h = int.from_bytes(H_bytes(x), 'big')
    return 1 + (h % (mod - 2))

def inv_mod(a, m):
    def eg(a,b):
        if b==0: return (1,0,a)
        x,y,g = eg(b, a%b)
        return (y, x - (a//b)*y, g)
    x, y, g = eg(a, m)
    if g != 1: raise ValueError("no inverse")
    return x % m

def gen_prime(bits=256, rounds=16):
    def is_probable_prime(n):
        if n < 4: return n in (2,3)
        small = [2,3,5,7,11,13,17,19,23,29]
        for p in small:
            if n % p == 0:
                return n == p
        r, d = 0, n-1
        while d % 2 == 0:
            r += 1; d //= 2
        for _ in range(rounds):
            a = random.randrange(2, n-2)
            x = pow(a, d, n)
            if x == 1 or x == n-1:
                continue
            for __ in range(r-1):
                x = pow(x, 2, n)
                if x == n-1:
                    break
            else:
                return False
        return True
    while True:
        n = random.getrandbits(bits) | 1 | (1 << (bits-1))
        if is_probable_prime(n):
            return n

def gen_group(bits=256):
    p = gen_prime(bits)
    while True:
        g = random.randrange(2, p-1)
        if pow(g, 2, p) != 1 and pow(g, (p-1)//2, p) != 1:
            return p, g

class Paillier:
    def __init__(self, bits=512):
        self.p = self._gen_prime(bits//2)
        self.q = self._gen_prime(bits//2)
        self.n = self.p * self.q
        self.n2 = self.n * self.n
        self.lmbda = (self.p-1)*(self.q-1) // gcd(self.p-1, self.q-1)
        self.g = self.n + 1
        mu_num = self._L(pow(self.g, self.lmbda, self.n2))
        self.mu = inv_mod(mu_num, self.n)

    def _gen_prime(self, bits):
        return gen_prime(bits)

    def _L(self, u):
        return (u - 1) // self.n

    def enc(self, m: int) -> int:
        m %= self.n
        while True:
            r = random.randrange(1, self.n)
            if gcd(r, self.n) == 1:
                break
        return ( (1 + m*self.n) % self.n2 ) * pow(r, self.n, self.n2) % self.n2

    def add(self, c1: int, c2: int) -> int:
        return (c1 * c2) % self.n2

    def add_plain(self, c: int, m: int) -> int:
        return (c * self.enc(m)) % self.n2

    def refresh(self, c: int) -> int:
        while True:
            r = random.randrange(1, self.n)
            if gcd(r, self.n) == 1:
                break
        return (c * pow(r, self.n, self.n2)) % self.n2

    def dec(self, c: int) -> int:
        u = pow(c, self.lmbda, self.n2)
        L = self._L(u)
        m = (L * self.mu) % self.n
        return m

class PIDDH:
    def __init__(self, group_bits=256, paillier_bits=512):
        self.p, self.g = gen_group(group_bits)
        self.paillier = Paillier(paillier_bits)

    def h2g(self, x: str) -> int:
        return pow(self.g, H_int(x, self.p-1), self.p)

    def round1_from_P1(self, V: list, k1: int):
        elems = [ pow(self.h2g(v), k1, self.p) for v in V ]
        random.shuffle(elems)
        return elems

    def round2_from_P2(self, recv_from_P1_R1: list, W: list, k2: int):
        Z = [ pow(x, k2, self.p) for x in recv_from_P1_R1 ]
        random.shuffle(Z)
        tagged = []
        for (w, t) in W:
            h = self.h2g(w)
            hk2 = pow(h, k2, self.p)
            ct = self.paillier.enc(int(t))
            tagged.append((hk2, ct))
        random.shuffle(tagged)
        return Z, tagged

    def round3_from_P1(self, Z: list, tagged: list, k1: int):
        mapped = [ (pow(hk2, k1, self.p), ct) for (hk2, ct) in tagged ]
        Zset = set(Z)
        sum_ct = None
        count = 0
        for (h12, ct) in mapped:
            if h12 in Zset:
                count += 1
                sum_ct = ct if sum_ct is None else self.paillier.add(sum_ct, ct)
        if sum_ct is None:
            sum_ct = self.paillier.enc(0)
        sum_ct = self.paillier.refresh(sum_ct)
        return sum_ct, count

