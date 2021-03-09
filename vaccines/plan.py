from collections import deque, defaultdict, Counter
import pandas as pd
from typing import List, Tuple, NamedTuple, Dict, Deque, Iterator
import numpy as np
import pandas as pd

EMPTY_DOSE = (None, None, None, 0)


class Event(NamedTuple):
    day: int
    age: int
    doses: int
    phase: int


class FullEvent(NamedTuple):
    day: int
    age: int
    doses: int
    phase: int
    vaccine_type: int


class Plan:
    """
    Base class for executing a vaccination plan
    """
    steps: Tuple[Tuple[int, int]]
    age_distribution: pd.Series
    pending: Deque[Tuple[int, int]]
    events: List[Event]
    day: int

    _column_names = ['day', 'age', 'doses', 'phase']
    _event = Event

    @classmethod
    def from_source(cls, src: str, age_distribution: pd.Series, *args, **kwargs):
        steps = parse_plan(src, age_distribution)
        return cls(steps, age_distribution, *args, **kwargs)

    def __init__(self, steps, age_distribution):
        self.steps = tuple(steps)
        self.age_distribution = age_distribution
        self.pending = deque(self.steps)
        self.events = []
        self.day = 0

    def execute(self, max_iter=10_000):
        """
        Execute plan
        """
        for _ in range(max_iter):
            if self.is_complete():
                return
            self.execute_step()
        raise ValueError('maximum number of iterations reached')

    def execute_step(self):
        """
        Execute a single day of vaccination from plan.
        """
        raise NotImplementedError

    def is_complete(self) -> bool:
        """
        Return True if vaccination plan is complete.
        """
        return not bool(self.pending)

    def accumulate(self) -> Iterator[float]:
        """
        Accumulate doses by age and phase.

        Return an iterator synchronized with events. 
        """
        acc = Counter()
        for (_, age, doses, phase) in self.events:
            acc[age, phase] += doses
            yield acc[age, phase]

    def summary(self) -> pd.DataFrame:
        """
        Return dataframe summary.
        """
        df = pd.DataFrame(self.events, columns=self._column_names)
        df['acc'] = list(self.accumulate())
        total = df["age"].apply(self.age_distribution.loc.__getitem__)
        df["fraction"] = df['acc'] / total
        return df[df['doses'] > 0]


class SimpleRatePlan(Plan):
    """
    A simple fixed rate of vaccination.

    Proceed until all population is vaccinated or the stock of doses is depleted.
    """
    rate: int
    max_doses: int
    given_doses: int
    phase: int

    def __init__(self, steps, age_distribution, *, rate, phase=1, max_doses=float('inf')):
        super().__init__(steps, age_distribution)
        self.rate = rate
        self.max_doses = max_doses
        self.given_doses = 0
        self.phase = phase

    def is_complete(self) -> bool:
        return super().is_complete() or self.given_doses >= self.max_doses

    def execute_step(self):
        if not self.pending:
            self.day += 1
            return []

        age, n = self.pending.popleft()
        rate = min(self.rate, self.max_doses - self.given_doses)

        if n > rate:
            n -= rate
            applied = rate
            self.pending.appendleft((age, n))
        else:
            applied = n

        ev = self._event(self.day, age, applied, self.phase)
        self.given_doses += applied
        self.events.append(ev)
        self.day += 1
        return [ev]


class SimpleDosesRatePlan(SimpleRatePlan):
    """
    A simple fixed rate of vaccination, with more than one dose per individual.

    Proceed until all population is vaccinated or the stock of doses is depleted.
    The limit "max_doses" refers to the full vaccination scheme. 
    """
    delay: int
    schedule: Dict[int, List[Event]]

    def __init__(self, steps, age_distribution, *, delay, **kwargs):
        super().__init__(steps, age_distribution, **kwargs)
        self.delay = delay
        self.schedule = defaultdict(list)

    def is_complete(self) -> bool:
        return super().is_complete() and not bool(self.schedule)

    def execute_step(self):
        events = self.schedule.pop(self.day, ())
        if events:
            events = self.execute_scheduled_events(events)
        if events:
            self.day += 1
            return events

        events = super().execute_step()
        for (day, age, applied, dose) in events:
            day_ = day + self.delay
            ev = Event(day_, age, applied, dose + 1)
            self.schedule[day_].append(ev)
        return events

    def execute_scheduled_events(self, events):
        self.events.extend(events)
        return events


class MultipleVaccinesRatePlan(Plan):
    """
    Multiple vaccines
    """
    rates: List[int]
    max_doses: List[int]
    given_doses: List[int]
    delays: List[int]
    schedule: Dict[int, List[FullEvent]]
    phase: int
    _column_names = [*Plan._column_names, 'vaccine_type']
    _event = FullEvent

    @property
    def vac_types(self):
        return len(self.rates)

    def __init__(self, steps, age_distribution, *, rates, max_doses=None, delays=None, phase=1):
        super().__init__(steps, age_distribution)
        self.phase = 1
        self.rates = list(rates)

        if max_doses is None:
            self.max_doses = [float('inf') for _ in self.rates]
        else:
            self.max_doses = list(max_doses)
        if delays is None:
            self.delays = [0.0 for _ in self.rates]
        else:
            self.delays = list(delays)
        self.delays = list(delays)
        self.schedule = defaultdict(list)
        self.given_doses = [0] * self.vac_types

    def is_complete(self) -> bool:
        return len(self.schedule) == 0 and (
            not self.pending
            or all(x >= y for (x, y) in zip(self.given_doses, self.max_doses))
        )

    def execute_step(self):
        events = self.schedule.pop(self.day, ())
        if events:
            events = self._execute_scheduled_events(events)
        if not events:
            age, n = self.pending.popleft()
            events = self._execute_plan_step(age, n)

        self.day += 1
        return events

    def _execute_plan_step(self, age, n):
        events = []
        for idx, rate in enumerate(self.rates):
            rate = min(rate, self.max_doses[idx] - self.given_doses[idx])

            if n == 0:
                break
            elif n > rate:
                n -= rate
                applied = rate
            else:
                applied, n = n, 0

            if not applied:
                continue

            ev = FullEvent(self.day, age, applied, self.phase, idx)
            self.given_doses[idx] += applied
            events.append(ev)

            # Schedule second dose
            next_day = self.day + self.delays[idx]
            ev = FullEvent(next_day, age, applied, self.phase + 1, idx)
            self.schedule[next_day].append(ev)
        else:
            if n > 0:
                self.pending.appendleft((age, n))

        self.events.extend(events)
        return events

    def _execute_scheduled_events(self, events):
        self.events.extend(events)
        return events

    def accumulate(self) -> Iterator[float]:
        """
        Accumulate doses by age and phase.

        Return an iterator synchronized with events. 
        """
        acc = Counter()
        for (_, age, doses, phase, ref) in self.events:
            acc[age, phase, ref] += doses
            yield acc[age, phase, ref]

    def summary(self, keep_type=False) -> pd.DataFrame:
        df = super().summary()
        if keep_type:
            return df
        return df.groupby(['day', 'age', 'phase']).sum().drop(columns='vaccine_type').reset_index()


def parse_plan(st: str, age_distribution: pd.Series) -> List[Tuple[int, int]]:
    """
    Parse vaccination plan. 

    The plan is a list of tuples with (age category, number of doses).
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
        index=pd.MultiIndex.from_tuples(
            events.keys(), names=["day", "dose", "age"]),
    ).reset_index()
    age_distribution.name = "population"

    events["population"] = events["age"].apply(
        age_distribution.loc.__getitem__)
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
    rates.extend([final_rate or rates[-1]] * 365)

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
        index=pd.MultiIndex.from_tuples(
            events.keys(), names=["day", "dose", "age"]),
    ).reset_index()
    age_distribution.name = "population"

    events["population"] = events["age"].apply(
        age_distribution.loc.__getitem__)
    events["day"] += 1

    events["fraction"] = events["acc"] / events["population"]
    events = events[["day", "dose", "age", "applied", "acc", "fraction"]]
    events = events.reset_index(drop=True)
    return events
