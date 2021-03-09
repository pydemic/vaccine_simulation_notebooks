from collections import deque, defaultdict, Counter
import pandas as pd
from typing import List, Tuple
import numpy as np
import pandas as pd


def parse_plan(st: str, age_distribution: pd.Series) -> List[Tuple[int, int]]:
    """
    Parse vaccination plan.
    """
    plan = []
    lines = deque(filter(None, map(strip_comments, st.splitlines())))
    age_levels = sorted(age_distribution.index)
    while lines:
        line = lines.popleft()
        if ":" not in line:
            lines.appendleft(f"global: {line}")
            continue

        key, _, value = map(str.strip, line.partition(":"))
        if key == "global":
            if value.endswith("%"):
                lines.extendleft(f"{age}: {value}" for age in age_levels)
                continue
            else:
                N = age_distribution.sum()
                M = float(value)
                lines.extendleft(
                    f"{age}: {int(M / N * n)}"
                    for age, n in zip(age_levels, age_distribution)
                )
        key = int(key)

        if value.endswith("%"):
            value = int(float(value[:-1]) / 100 * age_distribution.loc[key])
        else:
            value = int(value)

        plan.append((key, value))
    return plan


def strip_comments(st):
    return st.partition("#")[0].strip()


def execute_plan_hasty(
    plan, rate, age_distribution, initial=None, *, delay, final_rate=None
):
    index = age_distribution.index
    population = np.asarray(age_distribution)
    nbins = len(population)

    events = defaultdict(lambda: np.array([0.0] * 2))
    schedule = defaultdict(Counter)

    acc = np.array([[0, 0]] * nbins)
    if initial is not None:
        if np.ndim(initial) == 1:
            acc[:, 0] = initial
        else:
            acc[:] = initial
    acc = pd.DataFrame(acc, columns=["first", "second"], index=index)

    for day, daily_stock in enumerate(rate):
        stock = daily_stock
        while stock > 0:
            for age, goal in sorted(schedule[day].items(), reverse=True):
                if not stock:
                    break

                if goal <= 0:
                    del schedule[day][age, goal]
                    continue

                applied = min(goal, stock)
                stock -= applied
                acc.loc[age, "second"] += applied
                schedule[day][age] -= applied
                events[day, 2, age] += [applied, float(acc.loc[age, "second"])]

                if goal > applied:
                    schedule[day + 1][age] += goal - applied

            if not stock or not plan:
                break

            age, goal = plan[0]
            if not goal:
                del plan[0]
                continue

            pop = age_distribution.loc[age]
            applied = min(goal, stock, pop - acc.loc[age, "first"])

            if applied == 0:
                del plan[0]
                continue

            schedule[day + delay][age] += applied
            plan[0] = age, goal - applied
            stock -= applied
            acc.loc[age, "first"] += applied

            events[day, 1, age] += [applied, acc.loc[age, "first"]]

    events = pd.DataFrame(
        events.values(),
        columns=["applied", "acc"],
        index=pd.MultiIndex.from_tuples(events.keys(), names=["day", "dose", "age"]),
    ).reset_index()
    age_distribution.name = "population"

    events["population"] = events["age"].apply(age_distribution.loc.__getitem__)
    events["day"] += 1

    events["fraction"] = events["acc"] / events["population"]
    events = events[["day", "dose", "age", "applied", "acc", "fraction"]]
    events = events.reset_index(drop=True)
    return events


def execute_plan_safe(
    plan,
    rate,
    age_distribution,
    initial=None,
    *,
    delay,
    final_rate=None,
    max_doses=None,
):
    index = age_distribution.index
    population = np.asarray(age_distribution)
    nbins = len(population)
    if max_doses is None:
        max_doses = sum(rate) / 2

    events = defaultdict(lambda: np.array([0.0] * 2))
    schedule = defaultdict(Counter)

    acc = np.array([[0, 0]] * nbins)
    if initial is not None:
        if np.ndim(initial) == 1:
            acc[:, 0] = initial
        else:
            acc[:] = initial
    acc = pd.DataFrame(acc, columns=["first", "second"], index=index)
    rates = list(rate)
    rates.extend([final_rate or rates[-1]] * 100)

    for day, daily_stock in enumerate(rates):
        stock = daily_stock
        schedule.pop(day - 1, None)

        while stock > 0:
            for age, goal in sorted(schedule[day].items(), reverse=True):
                if not stock:
                    break

                if goal <= 0:
                    del schedule[day][age, goal]
                    continue

                applied = min(goal, stock)
                stock -= applied
                acc.loc[age, "second"] += applied
                schedule[day][age] -= applied
                events[day, 2, age] += [applied, float(acc.loc[age, "second"])]

                if goal > applied:
                    schedule[day + 1][age] += goal - applied

            if not stock or not plan:
                break

            age, goal = plan[0]
            if not goal:
                del plan[0]
                continue

            pop = age_distribution.loc[age]
            applied = min(goal, stock, pop - acc.loc[age, "first"], max_doses)

            if applied == 0:
                del plan[0]
                continue

            schedule[day + delay][age] += applied
            plan[0] = age, goal - applied
            stock -= applied
            max_doses -= applied
            acc.loc[age, "first"] += applied

            events[day, 1, age] += [applied, acc.loc[age, "first"]]
            break

    events = pd.DataFrame(
        events.values(),
        columns=["applied", "acc"],
        index=pd.MultiIndex.from_tuples(events.keys(), names=["day", "dose", "age"]),
    ).reset_index()
    age_distribution.name = "population"

    events["population"] = events["age"].apply(age_distribution.loc.__getitem__)
    events["day"] += 1

    events["fraction"] = events["acc"] / events["population"]
    events = events[["day", "dose", "age", "applied", "acc", "fraction"]]
    events = events.reset_index(drop=True)
    return events
