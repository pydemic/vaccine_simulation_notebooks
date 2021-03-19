from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from .vaccine import Vaccine

Pandas = Union[pd.Series, pd.DataFrame]


@dataclass(frozen=False)
class VaccinationCampaign:
    """
    Represent results of a vaccination campaign.
    """

    events: pd.DataFrame
    duration: Optional[int] = None
    style: dict = field(default_factory=dict)

    @property
    def _duration(self) -> int:
        return self.campaign_duration if self.duration is None else self.duration

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
        return pd.RangeIndex(0, self._duration + 1)

    def copy(self, **kwargs):
        """
        Return copy possibly overriding arguments.
        """
        data = {k: getattr(self, k) for k in self.__dataclass_fields__}
        data.update(kwargs)
        return type(self)(**data)

    def vaccination_start(self, tol=0.0, delay=0) -> pd.Series:
        """
        Return a Series with duration to start vaccination of each age cohoort.
        """
        events = self.events
        df = events.loc[events["phase"] == 1, ["day", "age", "fraction"]]
        df = df[df["fraction"] > tol].drop(columns="fraction")
        return df.groupby("age").min()["day"] + delay

    def vaccination_end(self, minimum=0.0, delay=0) -> pd.Series:
        """
        Return a Series with duration to start vaccination of each age cohoort.
        """
        events = self.events
        df = events.loc[events["phase"] == 2, ["day", "age", "fraction"]]
        df = df[df["fraction"] > minimum].drop(columns="fraction")
        return df.groupby("age").max()["day"] + delay

    def vaccination_curves(self, phase) -> pd.DataFrame:
        """
        Return a dataframe with vaccination curves for each age class.

        Return age groups as columns and days as rows.
        """
        events = self.events
        columns = ["day", "age", "fraction"]
        filtered = events.loc[events["phase"] == phase, columns]
        return filtered.pivot(*columns).fillna(method="pad").fillna(0.0)

    def damage_curve(
        self, damage, phase=2, delay=0, efficiency=1.0, initial=0
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
        duration = self._duration
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
        ax.set_xlim(0, self._duration)
        return ax


class MultiVaccineCampaign(VaccinationCampaign):
    __dataclass_fields__ = {
        **VaccinationCampaign.__dataclass_fields__,  # type: ignore
        "age_distribution": field(),
        "vaccines": field(),
    }

    age_distribution: pd.Series
    vaccines: Sequence["Vaccine"]

    def __init__(self, events, age_distribution, vaccines, **kwargs):
        super().__init__(events, **kwargs)
        self.age_distribution = age_distribution
        self.vaccines = vaccines

    @property
    def n_vaccines(self):
        return len(self.vaccines)

    def damage_curve(
        self,
        damage: pd.Series,
        phase=2,
        delay: Union[Sequence[int], int] = None,
        initial=None,
        efficiency: Union[Sequence[int], int] = None,
    ) -> pd.Series:
        """
        Compute damage function for events computed with multiple vaccines.
        """

        res = self._init_damage_curve_params(damage, delay, initial, efficiency)
        reduction_acc, delay, efficiency = res
        events = self.events

        mask = events["phase"] == phase
        rows = events.loc[mask, ["day", "age", "doses", "vaccine_type"]]
        for (day, age, doses, key) in rows.values:
            day, age, key = map(int, [day, age, key])
            ratio = doses / self.age_distribution.loc[age]
            value = damage.loc[age] * ratio * efficiency[key]
            vaccine_delay = delay[key]
            reduction_acc[day + vaccine_delay :] += value
        
        reduction_acc /= -damage.sum()
        reduction_acc += 1

        return pd.Series(reduction_acc, index=range(len(reduction_acc)))

    def expected_damage_curve(
        self,
        damage,
        phase,
        delay: Union[Sequence[int], int],
        initial,
        efficiency: Union[Sequence[int], int],
    ) -> pd.Series:
        res = self._init_damage_curve_params(damage, delay, initial, efficiency)
        reduction_acc, delay, efficiency = res
        events = self.events

        kernel = np.zeros((max(delay), self.n_vaccines))
        for key, step in enumerate(delay):
            kernel[: delay[key], key] = np.linspace(0, 1, step)
        # st.line_chart(kernel)

        mask = events["phase"] == phase
        rows = events.loc[mask, ["day", "age", "doses", "vaccine_type"]]
        for (day, age, doses, key) in rows.values:
            day, age, key = map(int, [day, age, key])
            ratio = doses / self.age_distribution.loc[age]
            value = damage.loc[age] * ratio * efficiency[key]

            interval = min(delay[key], self._duration - day)
            acc = (kernel[:, key] * value)[:interval]
            reduction_acc[day : day + interval] += acc
            reduction_acc[day + interval :] += value

        reduction_acc /= -damage.sum()
        reduction_acc += 1

        return pd.Series(reduction_acc, index=range(len(reduction_acc)))

    def _init_damage_curve_params(
        self,
        damage: pd.Series,
        delay: Union[Sequence[int], int],
        initial,
        efficiency: Union[Sequence[int], int],
    ):
        if delay is None:
            delay = [v.immunization_delay for v in self.vaccines]
        if isinstance(delay, int):
            delay = [delay for _ in range(self.n_vaccines)]

        if efficiency is None:
            efficiency = np.array([v.efficiency for v in self.vaccines])
        efficiency = np.asarray(efficiency)

        # We accumulate the contribution of each vaccine in this array
        reduction_acc = np.zeros(self._duration or 0, dtype=float)

        if initial is not None:
            col = "vaccine_type"
            weights = self.events[["doses", col]].groupby(col).sum().values
            weights /= weights.sum()
            eff = efficiency.dot(weights)

            value = 0.0
            for age, doses in zip(initial.index, initial.values):
                ratio = doses / self.age_distribution.loc[age]
                value += damage.loc[age] * ratio * eff

            reduction_acc += value

        return (reduction_acc, delay, efficiency)
