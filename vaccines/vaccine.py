from typing import NamedTuple


# TODO: move kernel calculations to this class!
# We are possibly computing the kernel incorrectly by making wrong
# extrapolations for single dose efficacy
class Vaccine(NamedTuple):
    name: str
    second_dose_delay: int
    immunization_delay: int
    efficiency: float
    single_dose_efficiency: float

    @property
    def full_immunization_delay(self):
        return self.second_dose_delay + self.immunization_delay

    def __str__(self):
        return self.name


VACCINE_DB = [
    Vaccine(
        name="Coronavac/sinovac/butantan",
        second_dose_delay=28,
        immunization_delay=28,
        efficiency=0.95,
        single_dose_efficiency=0.76,
    ),
    Vaccine(
        name="Astrazeneca/Fiocruz",
        second_dose_delay=90,
        immunization_delay=28,
        efficiency=0.95,
        single_dose_efficiency=0.76,
    ),
]
