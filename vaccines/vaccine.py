from typing import NamedTuple


class Vaccine(NamedTuple):
    name: str
    second_dose_delay: int
    immunization_delay: int
    efficiency: float

    def __str__(self):
        return self.name


VACCINE_DB = [
    Vaccine("Coronavac/sinovac/butantan", 22, 28, 0.95),
    Vaccine("Astrazeneca/Fiocruz", 90, 28, 0.95),
]
