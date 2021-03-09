def compute_schedule(start, stock, end=None):
    """
    Compute daily list of vaccination from starting rate to final rate limiting 
    the total stock.
    """
    if end is None or start == end:
        n = stock // start
        if stock % start == 0:
            return [start] * n  
        else: 
            return [*compute_schedule(start, n * start), stock - n * start]
    
    delta = end - start
    N = stock / (start + delta/2)
    d = delta / (N - 1)
    
    res = [int(start + d * n) for n in range(int(N))]
    r = stock - sum(res)
    if r:
        res.append(r)
    return res    


def population_80_plus(data, *, by_decade=False):
    """
    Normalize population age distribution.
    """
    data = data.copy()
    data.loc[80] = data.loc[80:].sum()
    out = data.loc[20:80].iloc[::-1]
    if by_decade:
        for age in range(20, 71, 10):
            out.loc[age] += out.loc[age + 5]
        out = out.iloc[::2]
    return out


def by_periods(n: int, period: int) -> int:
    """
    Return the smallest multiple of period that fits n.
    """
    n = int(n)
    if n % period == 0:
        return n
    months = n // period
    return period * (months + 1)
    