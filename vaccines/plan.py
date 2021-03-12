from collections import deque, defaultdict, Counter
import pandas as pd
from typing import List, Tuple, NamedTuple, Dict, Deque, Iterator, TypeVar, Type, Any, Union, cast
import numpy as np
import pandas as pd
from numbers import Real

from .campaign import VaccinationCampaign, MultiVaccineCampaign
from .vaccine import Vaccine

Number = Union[int, float]
EMPTY_DOSE = (None, None, None, 0)
T = TypeVar("T", bound="Plan", covariant=True)


class Event(NamedTuple):
    day: int
    age: int
    doses: Real
    phase: int


class FullEvent(NamedTuple):
    day: int
    age: int
    doses: Real
    phase: int
    vaccine_type: int


class Plan:
    """
    Base class for executing a vaccination plan
    """

    steps: Tuple[Tuple[int, Real]]
    age_distribution: pd.Series
    pending: Deque[Tuple[int, Real]]
    events: List[Event]
    day: int

    _column_names = ["day", "age", "doses", "phase"]
    _event = Event
    _result = VaccinationCampaign

    @classmethod
    def from_source(
        cls: Type[T], src: str, age_distribution: pd.Series, *args, **kwargs
    ) -> T:
        initial = kwargs.pop("initial", None)
        steps = parse_plan(src, age_distribution, initial=initial)
        return cls(steps, age_distribution, *args, **kwargs)

    def __init__(self, steps, age_distribution):
        self.steps = tuple(steps)
        self.age_distribution = age_distribution
        self.pending = deque(self.steps)
        self.events = []
        self.day = 0

    def execute(self, max_iter=10_000) -> VaccinationCampaign:
        """
        Execute plan and return results
        """
        for _ in range(max_iter):
            if self.is_complete():
                return self._create_result()
            self._execute_step()
        raise ValueError("maximum number of iterations reached")

    def _execute_step(self):
        """
        Execute a single day of vaccination from plan.
        """
        raise NotImplementedError

    def _create_result(self):
        events = self.summary()
        return self._result(events, self.age_distribution)

    def is_complete(self) -> bool:
        """
        Return True if vaccination plan is complete.
        """
        return not bool(self.pending)

    def summary(self) -> pd.DataFrame:
        """
        Return dataframe summary.
        """
        df = pd.DataFrame(self.events, columns=self._column_names)
        df["acc"] = list(self._accumulate())
        total = df["age"].apply(self.age_distribution.loc.__getitem__)
        df["fraction"] = df["acc"] / total
        return df[df["doses"] > 0]

    def _accumulate(self) -> Iterator[float]:
        """
        Accumulate doses by age and phase.

        Return an iterator synchronized with events.
        """
        acc = Counter()
        for (_, age, doses, phase) in self.events:
            acc[age, phase] += doses
            yield acc[age, phase]


class SimpleRatePlan(Plan):
    """
    A simple fixed rate of vaccination.

    Proceed until all population is vaccinated or the stock of doses is depleted.
    """

    rate: Real
    max_doses: Real
    given_doses: Real
    phase: int

    def __init__(
        self,
        steps,
        age_distribution,
        *,
        rate,
        phase=1,
        max_doses=float("inf"),
        given_doses=0,
    ):
        super().__init__(steps, age_distribution)
        self.rate = rate
        self.max_doses = max_doses
        self.given_doses = given_doses
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

    rates: List[Real]
    max_doses: List[Real]
    given_doses: List[Real]
    delays: List[int]
    schedule: Dict[int, List[FullEvent]]
    phase: int
    events: List[FullEvent]
    vaccines: List[Vaccine]
    _column_names = [*Plan._column_names, "vaccine_type"]
    _result = MultiVaccineCampaign
    _event = FullEvent

    @property
    def vac_types(self):
        return len(self.rates)

    def __init__(
        self,
        steps,
        age_distribution,
        vaccines,
        *,
        rates,
        max_doses=float("inf"),
        phase=1,
        num_phases=2,
        given_doses=0,
    ):
        super().__init__(steps, age_distribution)
        self.phase = phase
        self.num_phases = num_phases
        self.rates = list(rates)
        self.vaccines = list(vaccines)
        self.delays = [v.second_dose_delay for v in self.vaccines]
        self.schedule = defaultdict(list)

        if isinstance(max_doses, Real):
            self.max_doses = [max_doses for _ in self.rates]
        else:
            self.max_doses = list(max_doses)

        if isinstance(given_doses, Real):
            self.given_doses = [given_doses for _ in self.rates]
        else:
            self.given_doses = list(given_doses)

    def is_complete(self) -> bool:
        return len(self.schedule) == 0 and (
            not self.pending
            or all(x >= y for (x, y) in zip(self.given_doses, self.max_doses))
        )

    def _execute_step(self):
        events = self.schedule.pop(self.day, ())
        if events:
            events = self._execute_scheduled_events(events)
        if not events and  self.pending:
            age, n = self.pending.popleft()
            events = self._execute_plan_step(age, n)

        self.day += 1
        return events

    def _execute_plan_step(self, age: int, n: Real):
        events = []
        for idx, rate in enumerate(self.rates):
            rate = min(rate, self.max_doses[idx] - self.given_doses[idx])

            if n == 0:
                break
            elif n > rate:
                n -= rate
                applied = rate
            else:
                applied = n
                n *= 0

            if not applied:
                continue

            ev = FullEvent(self.day, age, applied, self.phase, idx)
            self.given_doses[idx] += applied
            events.append(ev)

            # Schedule second dose
            if ev.phase < self.num_phases:
                next_day = self.day + self.delays[idx]
                ev = FullEvent(next_day, age, applied, ev.phase + 1, idx)
                self.schedule[next_day].append(ev)
        else:
            if n > cast(Real, 0):
                self.pending.appendleft((age, n))

        self.events.extend(events)
        return events

    def _execute_scheduled_events(self, events):
        self.events.extend(events)
        return events

    def _create_result(self):
        events = self.summary()
        return self._result(
            events, age_distribution=self.age_distribution, vaccines=self.vaccines
        )

    def _accumulate(self) -> Iterator[float]:
        """
        Accumulate doses by age and phase.

        Return an iterator synchronized with events.
        """
        acc = Counter()
        for (_, age, doses, phase, ref) in self.events:
            acc[age, phase, ref] += doses
            yield acc[age, phase, ref]

    def summary(self, group_type=False) -> pd.DataFrame:
        df = super().summary()
        if group_type:
            df = (
                df.groupby(["day", "age", "phase"])
                .sum()
                .drop(columns="vaccine_type")
                .reset_index()
            )
        return df


def parse_plan(src: str, age_distribution: pd.Series, initial:pd.Series=None) -> List[Tuple[int, int]]:
    """
    Parse vaccination plan.

    The plan is a list of tuples with (age category, number of doses).
    """
    
    plan = []
    age_levels = sorted(age_distribution.index)
    age_acc = Counter()
    
    if initial is not None and len(initial) > 0:
        for k, v in zip(initial.index, initial.values):
            if v:
                age_acc[k] += int(v)
    
    lines = deque(validate_plan(src))
    while lines:
        key, value, is_relative = lines.popleft()
        if key == "global":
            if is_relative:
                lines.extendleft((age, value, True) for age in age_levels)
                continue
            else:
                N = age_distribution.sum()
                M = float(value)
                lines.extendleft((age, int(M / N * n), False)
                    for age, n in zip(age_levels, age_distribution)
                )
        key = int(key)

        if is_relative:
            value = int(value * age_distribution.loc[key])
            value -= age_acc[key]
        else:
            value = int(value)

        age_acc[key] += value
        if value > 0:
            plan.append((key, value))

    return plan


def validate_plan(st) -> Iterator[Tuple[Any, Number, bool]]:
    def error(msg):
        return SyntaxError(f'Erro linha {ln}: {msg}')
    
    def parse_num(num):
        if num.endswith('%'):
            try:
                return float(num[:-1]) / 100, True
            except ValueError:
                raise error('porcentagem inválida!')
        else:
            try:
                return int(num), False
            except ValueError:
                raise error('número inválido')
    
    for ln, line in enumerate(st.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        if ":" not in line:
            if not line.endswith('%'):
                raise error('esperava uma porcentagem!')
            parse_num(line)
            line = f'global:{line}'
        
        key, _, number = map(str.strip, line.partition(":"))
        
        if not (key == 'global' or key.isdigit()):
            raise SyntaxError(f'Erro linha {ln}: população inválida: {key}')
        if key.isdigit():
            key = int(key)
        
        value, is_relative = parse_num(number) 
        if value:
            yield key, value, is_relative


def strip_comments(st):
    return st.partition("#")[0].strip()
