"""Compute the first K nontrivial zeros of zeta (imaginary parts) with mpmath; save to out/zeta_zeros.txt."""
import sys, time, mpmath
mpmath.mp.dps = 20
K = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
t0 = time.time()
with open("out/zeta_zeros.txt", "w") as f:
    for k in range(1, K + 1):
        z = mpmath.zetazero(k)
        assert abs(z.real - 0.5) < 1e-15
        f.write(f"{k} {mpmath.nstr(z.imag, 18)}\n")
        if k % 250 == 0:
            f.flush(); print(k, round(time.time() - t0, 1), flush=True)
print("done", K, round(time.time() - t0, 1))
