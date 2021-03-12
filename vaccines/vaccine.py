from typing import NamedTuple


class Vaccine(NamedTuple):
    name: str
    second_dose_delay: int
    immunization_delay: int
    efficiency: float
    single_dose_efficiency: float

    def __str__(self):
        return self.name


VACCINE_DB = [
    Vaccine("Coronavac/sinovac/butantan", 22, 28, 0.95, 0.76),
    Vaccine("Astrazeneca/Fiocruz", 90, 28, 0.95, 0.76),
]
