import time
import numpy as np
import pandas as pd

# Original version extracted from Electricity Data.py

def periodic_transform_original(df, variable):
    # Uses global df_scaled in the original code; we adapt it to use df here
    df[f"{variable}_SIN"] = np.sin(df[variable] / df[variable].max() * 2 * np.pi)
    df[f"{variable}_COS"] = np.cos(df[variable] / df[variable].max() * 2 * np.pi)
    return df

# Optimized version from Electricity Data.py

def periodic_transform_optimized(df, variables):
    if isinstance(variables, str):
        variables = [variables]
    for variable in variables:
        col = df[variable].to_numpy(copy=False)
        max_val = np.nanmax(col)
        denom = max_val if max_val != 0 else 1.0
        angle = (col / denom) * (2 * np.pi)
        df[f"{variable}_SIN"] = np.sin(angle)
        df[f"{variable}_COS"] = np.cos(angle)
    return df


def bench(fn, df, variables, repeat=5):
    times = []
    for _ in range(repeat):
        df_copy = df.copy(deep=True)
        t0 = time.perf_counter()
        fn(df_copy, variables if isinstance(variables, list) else variables)
        times.append(time.perf_counter() - t0)
    return min(times)


def main():
    n = 2_000_000
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        'DayOfWeek': rng.integers(0, 7, size=n),
        'Day': rng.integers(1, 32, size=n),
        'Month': rng.integers(1, 13, size=n),
        'PeriodOfDay': rng.integers(0, 24, size=n),
    })

    variables = ['DayOfWeek', 'Day', 'Month', 'PeriodOfDay']

    t_orig = bench(lambda d, _: [periodic_transform_original(d, v) for v in variables], df, variables)
    t_opt = bench(periodic_transform_optimized, df, variables)

    print(f"Periodic transform original min time: {t_orig:.4f}s")
    print(f"Periodic transform optimized min time: {t_opt:.4f}s")
    if t_opt > 0:
        print(f"Speedup: {t_orig / t_opt:.2f}x")


if __name__ == "__main__":
    main()
