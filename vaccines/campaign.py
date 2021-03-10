from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Optional, Tuple, Sequence
from numbers import Real
from .vaccine import Vaccine


@dataclass(frozen=False)
class VaccinationCampaign:
    """
    Represent results of a vaccination campaign.
    """

    events: pd.DataFrame
    duration: Optional[int] = None
    style: dict = field(default_factory=dict)

    @property
    def _duration(self):
        return self.campaign_duration if self.duration is None else self.duration

    @property
    def campaign_duration(self):
        return self.events["day"].max()

    @property
    def applied_doses(self):
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
        df = events[["day", "age", "fraction"]][events["phase"] == 1]
        df = df[df["fraction"] > tol].drop(columns="fraction")
        return df.groupby("age").min()["day"] + delay

    def vaccination_end(self, minimum=0.0, delay=0) -> pd.Series:
        """
        Return a Series with duration to start vaccination of each age cohoort.
        """
        events = self.events
        df = events[["day", "age", "fraction"]][events["phase"] == 2]
        df = df[df["fraction"] > minimum].drop(columns="fraction")
        return df.groupby("age").max()["day"] + delay

    def vaccination_curves(self, phase) -> pd.DataFrame:
        """
        Return a dataframe with vaccination curves for each age class.

        Return age groups as columns and days as rows.
        """
        events = self.events
        columns = ["day", "age", "fraction"]
        filtered = events[columns][events["phase"] == phase]
        return filtered.pivot(*columns).fillna(method="pad").fillna(0.0)

    def damage_curve(self, damage, phase=2, delay=0, efficiency=1.0) -> pd.DataFrame:
        """
        Return the damage curve
        """
        curves = self.vaccination_curves(phase)
        if len(curves) == 0:
            return pd.Series([], dtype=float)
        total = damage.sum()
        curve = (
            total - (curves * efficiency * damage.loc[curves.columns]).sum(1)
        ) / total
        curve.index += delay
        return curve.reindex(self._days_index).fillna(method="pad").fillna(1.0)

    #
    # Plot vaccine curves
    #
    def plot_vaccination_schedule(self, ax: Axes=None, **kwargs):
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
                "vacinado (2 doses)": duration - end,
            }
        ).sort_index(ascending=False)

        ax = df.plot.barh(
            stacked=True, grid=True, color=["0.9", "0.7", "g"], ax=ax
        )
        ax.set_xlim(0, duration)
        ax.set_title("Calendário de vacinação", fontsize=self._title_fontsize)
        ax.set_xlabel("Etapa da campanha (dias)")
        ax.set_ylabel("Faixa etária (anos)")
        ax.legend(bbox_to_anchor=(1.33, 1), loc="upper right")
        return ax

    def plot_hospitalization_pressure_curve(
        self, severe, as_pressure=False, ax: Axes=None, **kwargs
    ) -> Tuple[pd.DataFrame, Axes]:
        pressure = severe if as_pressure else self.damage_curve(severe, **kwargs)
        ax = np.minimum(100 * pressure, 99.5).plot(lw=2, ax=ax)
        ax.grid(True)
        ax.set_title(
            "Estimativa de redução de hospitalizações", fontsize=self._title_fontsize
        )
        ax.set_xlabel("tempo (dias)")
        ax.set_ylabel("pressão hospitalar (%)")
        ax.set_ylim(0, 100)
        ax.set_xlim(0, self._duration)
        return ax

    def plot_death_pressure_curve(
        self, deaths, as_pressure=False, ax: Axes=None, **kwargs
    ) -> Tuple[pd.DataFrame, Axes]:
        pressure = deaths if as_pressure else self.damage_curve(deaths, **kwargs)
        ax = np.minimum(100 * pressure, 99.5).plot(lw=2, ax=ax)
        ax.grid(True)
        ax.set_title(
            "Estimativa de redução de mortalidade", fontsize=self._title_fontsize
        )
        ax.set_xlabel("tempo (dias)")
        ax.set_ylabel("mortalidade (%)")
        ax.set_ylim(0, 100)
        ax.set_xlim(0, self._duration)
        return ax

@dataclass
class MultiVaccineCampaign(VaccinationCampaign):
    vaccines: Sequence['Vaccine'] = ()
    age_distribution: pd.Series = None

    @property
    def n_vaccines(self):
        return len(self.vaccines)

    def damage_curve(self, damage, phase=2, delay=None, smooth=False, initial=None, efficiency=None) -> pd.Series:
        """
        Compute damage function for events computed with multiple vaccines.
        """
        events = self.events
        if delay is None:
            delay = [v.immunization_delay for v in self.vaccines]
        if isinstance(delay, Real):
            delay = [delay for _ in range(self.n_vaccines)]
        
        efficiency = np.array([v.efficiency for v in self.vaccines])
        damages_acc = np.zeros(self.duration, dtype=float)

        if initial is not None:
            col = 'vaccine_type'
            weights = events[['doses', col]].groupby(col).sum().values
            eff = (efficiency * weights / weights.sum()).sum()
            
            value = 0.0
            for age, doses in zip(initial.index, initial.values):
                ratio = doses / self.age_distribution.loc[age]
                value += damage.loc[age] * ratio * eff
            
            damages_acc += value

        if smooth:
            kernel = np.zeros((max(delay), self.n_vaccines))
            for key, step in enumerate(delay):
                kernel[:delay[key], key] = np.linspace(0, 1, step)
        else:
            kernel = None

        rows = events[events["phase"] == phase][["day", "age", "doses", "vaccine_type"]]
        for (day, age, doses, key) in rows.values:
            day, age, key = map(int, [day, age, key])
            ratio = doses / self.age_distribution.loc[age]
            value = damage.loc[age] * ratio * efficiency[key]
            vaccine_delay = delay[key]

            if kernel is None:
                damages_acc[day + vaccine_delay:] += value
            else:
                interval = min(delay[key], self.duration - day)
                damages_acc[day:day + interval] += (kernel[:, key] * value)[:interval]
                damages_acc[day + interval:] += value

        damages_acc /= -damage.sum()
        damages_acc += 1

        return pd.Series(damages_acc, index=range(len(damages_acc)))
