# ISRES variants benchmark (issue #590)

`src/algs/isres/isres.c` on this branch has two compile-time switches:

- `ISRES_BEST_FIX`: `x0` is the best-ranked individual, `xs + irank[0]*n`, instead of storage slot 0.
- `ISRES_RANK_FIX`: the differential variation uses `xs[irank[k+1]*n+j]`, the (k+1)-th ranked individual, instead of storage slot k+1.

With both switches the update is `x_k' = x_k + gamma (x_1 - x_{k+1})` over ranked individuals. Without them it is master.

Variants: M (master), R (`ISRES_RANK_FIX`), B (`ISRES_BEST_FIX`), RB (both).

## Running

1. Build the nlopt library four times, defining the switches for R, B and RB (for example `-DISRES_RANK_FIX` in `CFLAGS`, or a `#define` at the top of `isres.c`).
2. Link `bench_isres.cxx` against each library.
3. Run `bench_isres <variant> 300 350000 > results-<variant>.csv` for each variant (seeds 1 to 300, at most 350000 evaluations).
4. Run `python summarize.py` in the folder with the four CSV files.

Each run uses `nlopt::srand(seed)` and `set_stopval(f* + tol)`; a run succeeds if it returns `NLOPT_STOPVAL_REACHED`. The program first checks each problem's transcription at its known minimizer.

`summary-300.txt` is the output for 300 seeds (MSVC 2022, Windows, master 6e6593f). `p_evals` is a two-sided Mann-Whitney test of the evaluation counts of successful runs against M; `p_succ` is Fisher's exact test of the success counts against M.
