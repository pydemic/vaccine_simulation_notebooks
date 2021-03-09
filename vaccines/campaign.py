from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Optional, Tuple


@dataclass(frozen=False)
class VaccinationCampaign:
    """
    Represent results of a vaccination campaign.
    """

    events: pd.DataFrame
    duration: Optional[int] = None
    # age_distribution: Optional[pd.DataFrame] = field(repr=False, default=None)
    # hospitalizations: Optional[pd.DataFrame] = field(repr=False, default=None)
    style: dict = field(default_factory=dict)

    @property
    def _duration(self):
        return self.campaign_duration if self.duration is None else self.duration

    @property
    def campaign_duration(self):
        return self.events["day"].max()

    @property
    def vaccines(self):
        return self.events["applied"].sum()

    @property
    def _title_fontsize(self):
        return self.style.get("title", {}).get("font_size", 16)

    @property
    def _days_index(self):
        return pd.RangeIndex(0, self._duration + 1)

    def vaccination_start(self, tol=0.0, delay=0) -> pd.Series:
        """
        Return a Series with duration to start vaccination of each age cohoort.
        """
        events = self.events
        df = events[["day", "fraction", "age"]][events["dose"] == 1]
        df = df[df["fraction"] > tol].drop(columns="fraction")
        return df.groupby("age").min()["day"] + delay

    def vaccination_end(self, minimum=0.0, delay=0) -> pd.Series:
        """
        Return a Series with duration to start vaccination of each age cohoort.
        """
        events = self.events
        df = events[["day", "fraction", "age"]][events["dose"] == 2]
        df = df[df["fraction"] > minimum].drop(columns="fraction")
        return df.groupby("age").max()["day"] + delay

    def vaccination_curves(self, dose) -> pd.DataFrame:
        """
        Return a dataframe with vaccination curves for each age class.

        Return age groups as columns and days as rows.
        """
        events = self.events
        columns = ["day", "age", "fraction"]
        filtered = events[columns][events["dose"] == dose]
        return filtered.pivot(*columns).fillna(method="pad").fillna(0.0)

    def damage_curve(self, damage, dose=2, delay=0, efficiency=1.0) -> pd.DataFrame:
        """
        Return the damage curve
        """
        curves = self.vaccination_curves(dose)
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
    def plot_vaccination_schedule(self, **kwargs):
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

        df.plot.barh(stacked=True, grid=True, color=["0.9", "0.7", "g"])
        plt.xlim(0, duration)
        plt.title("Calendário de vacinação", fontsize=self._title_fontsize)
        plt.xlabel("Etapa da campanha (dias)")
        plt.ylabel("Faixa etária (anos)")
        plt.legend(bbox_to_anchor=(1.33, 1), loc="upper right")

    def plot_hospitalization_pressure_curve(self, severe, as_pressure=False, **kwargs) -> Tuple[pd.DataFrame, Axes]:
        if as_pressure:
            pressure = severe
        else:
            pressure = self.damage_curve(severe, dose=2, **kwargs)

        np.minimum(100 * pressure, 99.5).plot(lw=2)
        plt.grid(True)
        plt.title("Estimativa de redução de hospitalizações", fontsize=self._title_fontsize)
        plt.xlabel("tempo (dias)")
        plt.ylabel("pressão hospitalar (%)")
        plt.ylim(0, 100)
        plt.xlim(0, self._duration)
        return pressure, plt.gca()