from dataclasses import dataclass, field
from typing import Optional, Sequence, List, Union, TypeVar, cast

import numpy as np
from numpy.ma import asarray
import pandas as pd
from pandas.api.extensions import register_dataframe_accessor
from matplotlib.axes import Axes

from .vaccine import Vaccine


class DataFrameEx(pd.DataFrame):
    events: "Events"


Pandas = Union[pd.Series, DataFrameEx]
T = TypeVar("T")


@register_dataframe_accessor("events")
class Events:
    def __init__(self, data):
        self.data: pd.DataFrame = data

    @property
    def campaign_duration(self) -> int:
        return self.data["day"].max()  # type: ignore

    @property
    def applied_doses(self) -> float:
        return self.data["doses"].sum()

    def vaccination_start(self, tol=0.0, delay=0) -> pd.Series:
        """
        Return a Series with duration to start vaccination of each age cohoort.
        """
        events = self.data
        df = events.loc[events["phase"] == 1, ["day", "age", "fraction"]]
        df = df[df["fraction"] > tol].drop(columns="fraction")
        return df.groupby("age").min()["day"] + delay

    def vaccination_end(self, minimum=0.0, delay=0) -> pd.Series:
        """
        Return a Series with duration to start vaccination of each age cohoort.
        """
        events = self.data
        df = events.loc[events["phase"] == 2, ["day", "age", "fraction"]]
        df = df[df["fraction"] > minimum].drop(columns="fraction")
        return df.groupby("age").max()["day"] + delay

    def vaccination_curves(self, phase) -> pd.DataFrame:
        """
        Return a dataframe with vaccination curves for each age class.

        Return age groups as columns and days as rows.
        """
        events = self.data
        columns = ["day", "age", "fraction"]
        filtered = events.loc[events["phase"] == phase, columns]
        return filtered.pivot(*columns).fillna(method="pad").fillna(0.0)


@dataclass(frozen=False)
class VaccinationCampaign:
    """
    Represent results of a vaccination campaign.
    """

    events: DataFrameEx
    duration: int
    style: dict = field(default_factory=dict)

    @property
    def campaign_duration(self) -> int:
        return self.events["day"].max()  # type: ignore

    @property
    def applied_doses(self) -> float:
        return self.events["doses"].sum()

    @property
    def _title_fontsize(self):
        return self.style.get("title", {}).get("font_size", 16)

    @property
    def _days_index(self):
        return pd.RangeIndex(0, self.duration + 1)

    def __init__(self, events, duration=None, style=None):
        self.events = events
        if duration is None:
            duration = self.campaign_duration
        self.duration = duration
        self.style = {} if style is None else style

    def copy(self: T, **kwargs) -> T:
        """
        Return copy possibly overriding arguments.
        """
        data = {k: getattr(self, k) for k in self.__annotations__}
        data.update(kwargs)
        new = object.__new__(type(self))
        new.__init__(**data)
        return cast(T, new)

    def vaccination_start(self, tol=0.0, delay=0) -> pd.Series:
        return self.events.events.vaccination_start(tol, delay)

    def vaccination_end(self, minimum=0.0, delay=0) -> pd.Series:
        return self.events.events.vaccination_end(minimum, delay)

    def vaccination_curves(self, phase) -> pd.DataFrame:
        return self.events.events.vaccination_curves(phase)

    def damage_curve(
        self, damage, phase=2, delay=0, efficiency=1.0, initial=None
    ) -> pd.Series:
        """
        Return the damage curve.
        """
        if initial is not None:
            raise NotImplementedError

        curves = self.vaccination_curves(phase)
        if len(curves) == 0:
            return pd.Series([], dtype=float)

        total = damage.sum()
        reduction = (curves * efficiency * damage.loc[curves.columns]).sum(1)
        curve = (total - reduction) / total

        curve.index += delay
        return curve.reindex(self._days_index).fillna(method="pad").fillna(1.0)

    #
    # Plot vaccine curves
    #
    def plot_vaccination_schedule(self, ax: Axes = None, **kwargs):
        """
        Plot campaign schedule.
        """
        duration = self.duration
        start = self.vaccination_start(tol=kwargs.pop("tol", 0))
        end = self.vaccination_end(minimum=kwargs.pop("minimum", 0))
        end = end.reindex(start.index).fillna(duration)
        if kwargs:
            raise TypeError(f"invalid arguments: {set(kwargs)}")

        df = pd.DataFrame(
            {
                "sem vacinação": start,
                "vacinado (1a dose)": end - start,
                "vacinado (2 doses)": duration - end,  # type: ignore
            }
        ).sort_index(ascending=False)

        ax = df.plot.barh(stacked=True, grid=True, color=["0.9", "0.7", "g"], ax=ax)
        ax.set_xlim(0, duration)
        ax.set_title("Calendário de vacinação", fontsize=self._title_fontsize)
        ax.set_xlabel("Etapa da campanha (dias)")
        ax.set_ylabel("Faixa etária (anos)")
        ax.legend(bbox_to_anchor=(1.33, 1), loc="upper right")
        return ax

    def plot_hospitalization_pressure_curve(
        self, pressure: Pandas, ax: Axes = None
    ) -> Axes:
        title = "Estimativa de redução de hospitalizações"
        ax = self._plot_pressure_curve(pressure, title, ax)
        ax.set_ylabel("pressão hospitalar (%)")
        return ax

    def plot_death_pressure_curve(self, pressure: Pandas, ax: Axes = None) -> Axes:
        title = "Estimativa de redução de mortalidade"
        ax = self._plot_pressure_curve(pressure, title, ax)
        ax.set_ylabel("mortalidade (%)")
        return ax

    def _plot_pressure_curve(
        self, pressure: Pandas, title: str, ax: Axes = None, styles: dict = None
    ) -> Axes:
        df = np.minimum(100 * pressure, 99.5)

        if isinstance(df, pd.DataFrame):
            if styles is None:
                styles = {pressure.columns[0]: {"lw": 2}}

            for name, col in df.items():
                opts = styles.get(name, {"ls": "--"})  #  type: ignore
                ax = col.plot(ax=ax, label=name, **opts)
        else:
            ax = df.plot(lw=2, ax=ax)

        if ax is None:
            raise ValueError("cannot plot empty dataframe")

        ax.set_title(title, fontsize=self._title_fontsize)
        ax.set_xlabel("tempo (dias)")
        ax.set_xlabel("pressão (%)")
        ax.grid(True)
        ax.set_ylim(0, 100)
        ax.set_xlim(0, self.duration)
        return ax


class MultiVaccineCampaign(VaccinationCampaign):
    events: DataFrameEx
    duration: int
    age_distribution: pd.Series
    vaccines: Sequence["Vaccine"]

    def __init__(self, events, age_distribution, vaccines, **kwargs):
        super().__init__(events, **kwargs)
        self.age_distribution = age_distribution
        self.vaccines = vaccines

    def damage_curve(
        self,
        damage: pd.Series,
        phase=2,
        delay: Union[Sequence[int], int, None] = None,
        initial=None,
        efficiency: Union[Sequence[int], int, None] = None,
    ) -> pd.Series:
        """
        Compute damage function for vaccination with multiple vaccines.
        """
        res = self._init_damage_curve_params(damage, delay, initial, efficiency)
        reduction_acc, delays, effs = res
        events = self.events
        # dbg(effs)

        mask = events["phase"] == phase
        rows = events.loc[mask, ["day", "age", "doses", "vaccine_type"]]
        for (day, age, doses, key) in rows.values:
            day, age, key = map(int, [day, age, key])
            ratio = doses / self.age_distribution.loc[age]
            value = damage.loc[age] * ratio * effs[key]
            vaccine_delay = delays[key]
            reduction_acc[day + vaccine_delay :] += value

        reduction_acc /= -damage.sum()
        reduction_acc += 1

        return pd.Series(reduction_acc, index=range(len(reduction_acc)))

    def expected_damage_curve(
        self,
        damage: pd.Series,
        single_dose=False,
        delay: Union[Sequence[int], int, None] = None,
        initial=None,
        kernel=None,
        efficiency: Union[Sequence[int], int, None] = None,
    ) -> pd.Series:

        events = self.events
        res = self._init_damage_curve_params(damage, delay, initial, efficiency)
        reduction_acc, delays, effs = res

        if kernel is None and delay is not None:
            kernel = simple_kernel(delays, effs, self.vaccines)
        elif kernel is None and single_dose:
            kernel = single_dose_vaccination_kernel(self.vaccines)
        elif kernel is None:
            kernel = full_vaccination_scheme_kernel(self.vaccines)
        else:
            kernel = np.asarray(kernel)

        mask = events["phase"] == 1
        rows = events.loc[mask, ["day", "age", "doses", "vaccine_type"]]

        for (day, age, doses, key) in rows.values:
            day, age, key = map(int, [day, age, key])
            ratio = doses / self.age_distribution.loc[age]
            value = damage.loc[age] * ratio

            interval = min(len(kernel), self.duration - day)
            acc = (kernel[:, key] * value)[:interval]
            reduction_acc[day : day + interval] += acc
            reduction_acc[day + interval :] += kernel[-1, key] * value

        reduction_acc /= -damage.sum()
        reduction_acc += 1

        return pd.Series(reduction_acc, index=range(len(reduction_acc)))

    def _init_damage_curve_params(
        self,
        damage: pd.Series,
        delay: Union[Sequence[int], int, None],
        initial,
        efficiency: Union[Sequence[int], int, None],
    ):

        delays = normalize_delay(delay, self.vaccines)
        eff = normalize_efficiency(efficiency, self.vaccines)

        # We accumulate the contribution of each vaccine in this array
        reduction_acc = np.zeros(self.duration or 0, dtype=float)

        if initial is not None:
            col = "vaccine_type"
            weights = self.events[["doses", col]].groupby(col).sum().values
            weights /= weights.sum()
            mean_eff = np.dot(eff, weights)

            value = 0.0
            for age, doses in zip(initial.index, initial.values):
                ratio = doses / self.age_distribution.loc[age]
                value += damage.loc[age] * ratio * mean_eff

            reduction_acc += value

        return (reduction_acc, delays, eff)


def normalize_efficiency(
    eff: Union[Sequence[int], int, None], vaccines
) -> Sequence[float]:
    if eff is None:
        res = np.array([v.efficiency for v in vaccines])
    elif isinstance(eff, float):
        res = np.ones(len(vaccines), dtype=float)
        res *= eff
    else:
        res = np.asarray(eff)
    return cast(Sequence[float], res)


def normalize_delay(delay: Union[Sequence[int], int, None], vaccines) -> Sequence[int]:
    if delay is None:
        res = np.array([v.immunization_delay for v in vaccines], dtype=int)
    elif isinstance(delay, int):
        res = np.ones(len(vaccines), dtype=int)
        res *= delay
    else:
        res = asarray(delay, dtype=int)
    return cast(Sequence[int], res)


def simple_kernel(delays, effs, vaccines):
    if effs is None:
        effs = [v.efficiency for v in vaccines]

    kernel = np.zeros((max(delays), len(vaccines)))
    for i, (step, eff) in enumerate(zip(delays, effs)):
        kernel[:step, i] = np.linspace(0, eff, step)
        kernel[step:, i] = eff
    return kernel


def full_vaccination_scheme_kernel(vaccines: Sequence[Vaccine]):
    max_delay = max(map(lambda v: v.full_immunization_delay, vaccines))
    kernel = np.zeros((max_delay, len(vaccines)))
    for i, v in enumerate(vaccines):
        step = v.second_dose_delay
        kernel[: step + 1, i] = np.linspace(0, v.single_dose_efficiency, step + 1)

        start, step = step, v.immunization_delay
        kernel[start : start + step, i] = np.linspace(
            v.single_dose_efficiency, v.efficiency, step
        )

        kernel[start + step :, i] = v.efficiency
    return kernel


def single_dose_vaccination_kernel(vaccines: Sequence[Vaccine]):
    max_delay = max(map(lambda v: v.immunization_delay, vaccines))
    kernel = np.zeros((max_delay, len(vaccines)))
    for i, v in enumerate(vaccines):
        step = v.immunization_delay
        kernel[:step, i] = np.linspace(0, v.single_dose_efficiency, step)
        kernel[step:, i] = v.single_dose_efficiency
    return kernel
