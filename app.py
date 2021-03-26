from dataclasses import dataclass, field
import locale
import sys
import os
from typing import (
    Callable,
    Generic,
    List,
    Tuple,
    TypeVar,
    Dict,
    NamedTuple,
    overload,
    cast,
)
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import vaccines as lib

sys.path.append(".")

DEVELOP = os.environ.get('DEVELOP', 'false').lower() == 'true'
cte = lambda x: lambda *args: x


T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")


class cached_property(Generic[T1, T2]):
    """A quick and dirt cached property"""

    _name: str

    def __init__(self, func: Callable[[T1], T2]):
        self._func = func

    @overload
    def __get__(self, obj: None, cls=None) -> "cached_property":
        ...

    @overload
    def __get__(self, obj: T1, cls=None) -> T2:
        ...

    def __get__(self, obj, cls=None):
        if obj is None:
            return self
        try:
            return obj.__dict__[self._name]
        except KeyError:
            value = self._func(obj)
            setattr(obj, self._name, value)
            return value

    def __set_name__(self, cls, name):
        if not hasattr(self, "_name"):
            self._name = name


def simple(n):
    st = str(int(n))
    tail = max(len(st) - 2, 0)
    return int(st[:2] + "0" * tail)


@dataclass()
class InputReader:
    """
    Mostra sidebar e lê entradas do usuário
    """

    data: dict = field(default_factory=dict)
    exclude: set = field(default_factory=set)
    FIELDS = [
        "region",
        "stocks",
        "rate",
        "coarse",
        # "smooth",
        "single_dose",
        "vaccine_plan",
        "initial_plan",
    ]
    PLACEHOLDER_VACCINE_PLAN = (
        "# Preencha meta de vacinação de cada faixa etária como porcentagem\n"
        "# da população em cada faixa.\n"
    )

    def ask(self):
        st.sidebar.header("Painel de controle")
        for field in self.FIELDS:
            if field not in self.exclude:
                fn = getattr(self, f"read_{field}")
                self.data[field] = fn()
        return self.data

    def __getattr__(self, key):
        return self.data[key]

    def __getitem__(self, key):
        return self.data[key]

    def check_plan(self, plan, full=True):
        try:
            _ = [*lib.validate_plan(plan)]
        except SyntaxError as ex:
            if full:
                st.error(f"{ex} Considerando cobertura de 100%")
                return "global: 100%"
            else:
                st.error(f"{ex} Considerando cobertura de 0%")
                return "80: 0"
        return plan

    def read_region(self):
        opts = load_regions()
        return st.sidebar.selectbox(
            "UF", [*opts.keys()], format_func=opts.get, index=len(opts) - 1
        )

    def read_stocks(self):
        population = load_population(self["region"])
        st.sidebar.subheader("Estoque (número de doses)")
        stock = simple(population * 0.5 / len(lib.VACCINE_DB))
        return {
            vaccine: st.sidebar.number_input(vaccine.name, min_value=0, value=stock)
            for vaccine in lib.VACCINE_DB
        }

    def read_rate(self):
        population = load_population(self["region"])
        st.sidebar.subheader("Capacidade de vacinação")
        msg = "Capacidade de vacinação (doses/dia)"
        return st.sidebar.number_input(
            msg, min_value=0, value=simple(0.005 * population)
        )

    def read_coarse(self):
        st.sidebar.subheader("Opções")
        return st.sidebar.checkbox("Agrupar de 10 em 10 anos")

    def read_smooth(self):
        return st.sidebar.checkbox("Considera aumento gradual da imunidade")

    def read_single_dose(self):
        return st.sidebar.checkbox("Impacto com a primeira dose")

    def read_vaccine_plan(self):
        st.header("Planos de vacinação")
        with st.beta_expander(
            "Vacinação por faixa etária (clique para expandir)", expanded=True
        ):
            fn = lambda x: "95%" if x >= 40 else "0%"
            msg = "Metas de vacinação por faixa etária"
            return self.read_plan(msg, fn, self.PLACEHOLDER_VACCINE_PLAN)

    def read_initial_plan(self):
        with st.beta_expander("Vacinas já aplicadas (clique para expandir)"):
            placeholder = (
                "# Preencha a quantidade de pessoas já vacinadas por faixa etária.\n"
            )
            return self.read_plan("Vacinados", 0, placeholder)

    def read_plan(self, msg, value, placeholder=""):
        coarse = self.data.get("coarse", False)
        step = 10 if coarse else 5
        value = value if callable(value) else cte(value)
        steps: List[Tuple[int, str]] = [(80, "80+")]
        steps.extend((n, f"{n}-{n + step - 1}") for n in range(80 - step, 19, -step))
        placeholder += "\n".join(f"{step}: {value(n)}" for n, step in steps)
        height = 28 * max(4, placeholder.count("\n"))
        plan = st.text_area(msg, placeholder, height=height)
        return self.check_plan(plan, full=False)


def read() -> dict:
    return InputReader().ask()


def config():
    import builtins

    plt.rcParams["figure.figsize"] = (10, 6.66)
    pd.set_option("display.max_rows", None)

    for loc in ["pt_BR.UTF-8", ""]:
        try:
            locale.setlocale(locale.LC_ALL, loc)
            break
        except:
            pass

    setattr(builtins, "st", st)
    setattr(builtins, "dbg", st.write)


# A hack to keep types of cached functions consistent
def cache(fn: T) -> T:
    return st.cache(fn)  # type: ignore


@cache
def load_population(region) -> pd.Series:
    return load_age_distribution(region).sum()


@cache
def load_age_distribution(region, coarse=False) -> pd.Series:
    db = pd.read_pickle("data.pkl.gz")
    out = db.loc[region, "age_distribution"]
    return lib.population_80_plus(out, coarse=coarse)


@cache
def load_hospitalizations(region, coarse=False) -> pd.Series:
    db = pd.read_pickle("hospitalization.pkl.gz")
    db.columns = [*db.columns]
    out = db.loc[region].iloc[::-1]
    if coarse:
        out = lib.coarse_distribution(out)
    return out


@cache
def load_deaths(region, coarse=False) -> pd.Series:
    db = pd.read_pickle("deaths.pkl.gz")
    db.columns = [*db.columns]
    out = db.loc[region].iloc[::-1]
    if coarse:
        out = lib.coarse_distribution(out)
    return out


@cache
def load_regions():
    db = pd.read_pickle("data.pkl.gz")
    return db["name"].to_dict()


# TODO: move to campaign?
class ResultStats(NamedTuple):
    """
    Simple statistics derived from simulation.
    """

    duration: int
    applied_doses: int
    num_vaccinated: int
    initial_doses: int = 0
    expected_deaths: int = 0
    expected_deaths_max: int = 0
    expected_hospitalizations: int = 0
    expected_hospitalizations_max: int = 0
    reduced_deaths: float = 0.0
    reduced_hospitalizations: float = 0.0


class ResultReport(NamedTuple):
    """
    Full report derived from simulation.
    """

    stats: ResultStats
    age_distribution: pd.Series
    hospitalizations: pd.Series
    deaths: pd.Series
    pressure: pd.Series
    vaccinated: pd.Series
    initial: pd.Series
    error: bool = False


class Job:
    result: lib.MultiVaccineCampaign

    def __init__(
        self,
        rate,
        region,
        stocks,
        initial_plan,
        vaccine_plan,
        coarse=False,
        smooth=True,
        single_dose=False,
    ):
        self.coarse: bool = coarse
        self.rate = rate
        self.region = region
        self.stocks = stocks
        self.initial_plan = initial_plan
        self.vaccine_plan = vaccine_plan
        self.smooth = smooth
        self.single_dose = single_dose

        self.error = False
        self.age_distribution = load_age_distribution(self.region, coarse=self.coarse)
        self.hospitalizations = load_hospitalizations(self.region, coarse=self.coarse)
        self.deaths = load_deaths(self.region, self.coarse)

        # Processa valores iniciais de vacina
        initial = lib.parse_plan(self.initial_plan, self.age_distribution)
        initial = pd.DataFrame(initial, columns=["age", "value"]).set_index("age")
        self.initial = initial

        # Inicializa simulação
        self._has_init = False

    def _force_init(self):
        if not self._has_init:
            self._has_init = True
            self.result = self._run_simulation()
            self.pressure = self._compute_pressure(self.result)
            self.vaccinated = self._compute_vaccine_distribution(self.result)
            self.stats = ResultStats(
                duration=self.result.duration,
                applied_doses=int(self.result.applied_doses),
                **self._compute_expected_damage(self.pressure, self.result.duration),
                **self._compute_vaccinations(self.result),
            )

    def __getattr__(self, attr):
        if not self._has_init:
            self._force_init()
        return self.__dict__[attr]

    def _run_simulation(self) -> lib.MultiVaccineCampaign:
        # Prepara entradas da simulação
        num_phases = 1 if self.single_dose else 2
        max_doses = [self.vaccine_stocks[k] // num_phases for k in self.vaccines]
        total_doses = sum(max_doses)

        # Simula plano e calcula resultados
        plan = lib.MultipleVaccinesRatePlan.from_source(
            self.vaccine_plan,
            self.age_distribution,
            self.vaccines,
            initial=self.initial,
            num_phases=num_phases,
            rates=[self.rate * n / total_doses for n in max_doses],
            max_doses=max_doses,
        )
        result = plan.execute()

        # Recria resultado com duração desejada da simulação
        delay = self.get_delay(self.vaccines)
        max_delay = sum(max(x) for x in delay.values())
        max_day = cast(int, result.events["day"].max())
        duration = max_day + max_delay
        duration = lib.by_periods(duration, 30)

        # Resultado e propriedades derivadas
        return result.copy(duration=duration)

    def _compute_pressure(self, result) -> pd.DataFrame:
        if self.single_dose:
            eff = [v.single_dose_efficiency for v in self.vaccines]
        else:
            eff = [v.efficiency for v in self.vaccines]

        delay = self.get_delay(self.vaccines)
        kwds = {
            "initial": self.initial,
            "efficiency": eff,
        }
        kwds_max = {
            **kwds,
            "delay": delay["immunization"],
            "phase": 1 if self.single_dose else 2,
        }
        kwds_min = {**kwds, "delay": 0, "phase": 1}
        kwds["single_dose"] = self.single_dose

        damage = result.damage_curve
        expected = result.expected_damage_curve

        df_hosp = pd.DataFrame(
            {
                "expected": expected(self.hospitalizations, **kwds),
                "max": damage(self.hospitalizations, **kwds_max),
                "min": damage(self.hospitalizations, **kwds_min),
            }
        )
        df_deaths = pd.DataFrame(
            {
                "expected": expected(self.deaths, **kwds),
                "max": damage(self.deaths, **kwds_max),
                "min": damage(self.deaths, **kwds_min),
            }
        )
        return pd.concat({"hospitalizations": df_hosp, "deaths": df_deaths}, axis=1)

    def _compute_expected_damage(self, pressure, duration) -> dict:
        """
        Derive hospitalizations/deaths from pressure curves.
        """
        out = {}
        d_pressure = pressure["deaths", "expected"]
        h_pressure = pressure["hospitalizations", "expected"]

        deaths = self.deaths.sum()
        hospitalizations = self.hospitalizations.sum()

        out["expected_deaths_max"] = int(deaths)
        out["expected_hospitalizations_max"] = int(hospitalizations)

        n = self.expected_damage(d_pressure, duration, deaths)
        out["expected_deaths"] = int(n)

        n = self.expected_damage(h_pressure, duration, hospitalizations)
        out["expected_hospitalizations"] = int(n)

        out["reduced_deaths"] = 1 - d_pressure.iloc[-1]
        out["reduced_hospitalizations"] = 1 - h_pressure.iloc[-1]
        return out

    def _compute_vaccinations(self, result) -> Dict[str, int]:
        return {
            "num_vaccinated": int(
                result.applied_doses / (1 if self.single_dose else 2)
            ),
            "initial_doses": int(self.initial.values.sum()),
        }

    def _compute_vaccine_distribution(self, result) -> pd.Series:
        df = result.events.drop(columns=["day", "fraction", "acc"])
        df = df.loc[df["phase"] == 2].groupby("age").sum()
        return df["doses"]

    def get_delay(self, vaccines):
        return {
            "immunization": [v.immunization_delay for v in vaccines],
            "second_dose": [v.second_dose_delay for v in vaccines],
        }

    # Simulation inputs
    @cached_property
    def vaccine_stocks(self) -> dict:
        return {k: v for k, v in self.stocks.items() if v}

    @cached_property
    def vaccines(self) -> list:
        return [k for k, _ in sorted(self.vaccine_stocks.items(), reverse=True)]

    def expected_damage(self, pressure, duration, scale=1):
        """
        Danos esperados com/sem vacina
        """
        res = (pressure / 365).sum()
        if duration >= 365:
            res *= 365 / duration
        else:
            dt = 365 - duration
            res += dt / 365 * pressure.iloc[-1]
        return scale * res


# @st.cache(hash_funcs={Job: id})
def compute(**kwargs) -> Tuple[lib.MultiVaccineCampaign, ResultReport]:
    public_fields = (
        "pressure",
        "hospitalizations",
        "deaths",
        "age_distribution",
        "vaccinated",
        "initial",
    )
    data = Job(**kwargs)
    fields = {k: getattr(data, k) for k in public_fields}
    return data.result, ResultReport(stats=data.stats, **fields)


#
# Application
#
config()
st.title(
    "Ferramenta para determinação do impacto da vacinação nas internações por COVID-19"
)
campaign, r = compute(**(InputReader().ask()))
st.header("Resultados")
st.markdown(
    f"""
* **Total de doses:** {int(r.stats.applied_doses):n}
* **Pessoas vacinadas:** {int(r.stats.num_vaccinated):n} + {r.stats.initial_doses:n} (inicial)
* **Dias de vacinação:** {campaign.campaign_duration}"""
    #     f"""
    # * **Óbitos anuais projetados*: ** {r.stats.expected_deaths:n} (com vacina) / {r.stats.expected_deaths_max:n} (sem vacinação)
    # * **Hospitalizações anuais projetadas*: ** {r.stats.expected_hospitalizations:n} (com vacina) / {r.stats.expected_hospitalizations_max:n} (sem vacinação)"""
    f"""
* **Redução na hospitalização:** {100 * r.stats.reduced_hospitalizations:.1f}%
* **Redução dos óbitos:** {100 * r.stats.reduced_deaths:.1f}%

&ast;  Óbitos e hospitalizações foram projetados a partir de dados do 
SIVEP/gripe, que é atualizado semanalmente e disponibilizado no link: https://opendatasus.saude.gov.br/dataset?tags=SRAG.
Os resultados apresentam maior acurácia quanto maior for a qualidade e 
oportunidade de registro dos dados nos sistemas oficiais.
"""
)

#
# Gráficos
#
fig_names = {"expected": "Esperado"}
if DEVELOP:
    select = ["expected", "min", "max"]
else:
    select = ['expected']

if not r.error:
    df = r.pressure.loc[:, "hospitalizations"].loc[:, select]
    df.rename(columns=fig_names, inplace=True)

    fig, ax = plt.subplots()
    campaign.plot_hospitalization_pressure_curve(df)
    ax.legend()
    st.pyplot(fig)

    df = r.pressure.loc[:, "deaths"].loc[:, select]
    df.rename(columns=fig_names, inplace=True)

    fig, ax = plt.subplots()
    campaign.plot_death_pressure_curve(df)
    ax.legend()
    st.pyplot(fig)

    fig, ax = plt.subplots()
    campaign.plot_vaccination_schedule(ax=ax)
    st.pyplot(fig)
else:
    st.error(f"Erro durante execução da simulação: {r.error}")

#
# Observações
#
st.markdown(
    f"""
## Observações

O primeiro gráfico mostra a estimativa de redução nas hospitalizações esperadas 
enquanto o segundo gráfico mostra a estimativa de redução de mortalidade, em 
função da estratégia de imunização, ou seja devido à proteção conferida pelas 
vacinas. 

O terceiro gráfico mostra a cobertura vacinal por faixa etária ao longo do tempo. 
A faixa cinza mais clara representa o percentual da população que ainda não foi 
imunizada, a faixa cinza mais escura mostra a população que está aguardando a 
segunda dose e a faixa verde mostra a população na qual foram aplicadas as 
duas doses da vacina. 

A simulação considera que a taxa de infecção se mantêm constante, o que é uma
suposição conservadora, especialmente em níveis mais altos de vacinação.

## Parâmetros utilizados 

1. Eficácia para formas graves da doença COVID-19 = 100% [1,2]
2. Para o esquema vacinal foram considerados os seguintes intervalos:
- **Butantan:** O esquema de imunização é de 2 doses com intervalo máximo de 28 dias entre as doses [1].
- **Astrazeneca/Fiocruz:** O esquema de imunização é de 2 doses com intervalo máximo de 90 dias entre as doses [2].
3. Foram considerados imunizados apenas os indivíduos que tiverem as duas doses da vacina [1,2].  
4. A imunização (soroconversão) ocorre em um período de 28 dias após aplicação da segunda dose [1,2].

## Referências

1. Parecer Público de avaliação de solicitação de autorização temporária de uso
emergencial, em caráter experimental, da vacina adsorvida covid-19 (inativada) –
[Instituto Butantan](https://www.gov.br/anvisa/pt-br/assuntos/noticias-anvisa/2021/confira-materiais-da-reuniao-extraordinaria-da-dicol/ppam-final-vacina-adsorvida-covid-19-inativada-butantan.pdf)

2. Parecer Público de avaliação de solicitação de autorização temporária de uso
emergencial, em caráter experimental, da vacina covid-19 (recombinante) –
[Fundação Oswaldo Cruz (Fiocruz)](https://www.gov.br/anvisa/pt-br/assuntos/noticias-anvisa/2021/confira-materiais-da-reuniao-extraordinaria-da-dicol/ppam-final-vacina-covid-19-recombinante-fiocruz.pdf)
"""
)

# Imprime dados adicionais e possivelmente locais para baixar CSVs.
st.subheader("Dados adicionais")

with st.beta_expander("Dados demográficos"):
    st.markdown(
        """Conjunto de dados do SIVEP-GRIPE e IBGE que foram utilizados para 
cálculo de parâmetros epidemiológicos e  populacionais nas projeções de óbitos 
e hospitalizações estimadas pela ferramenta. Os dados de hospitalização e óbitos 
do  SIVEP-GRIPE abrangem os dados desde 01 de janeiro a 31 de dezembro de 2020. 
Para os dados de população por faixa etária foi utilizada a projeção do IBGE 
para 2020. A coluna de vacinado (planejado), corresponde ao número de doses 
planejadas para aplicação  em cada uma das faixa etárias, dado que deve variar 
conforme simulação. 
"""
    )
    df = pd.DataFrame(
        {
            "Distribuição etária": r.age_distribution,
            "Hospitalizações": r.hospitalizations,
            "Vacinados (planejado)": r.vaccinated,
            "Óbitos": r.deaths,
        }
    ).dropna()
    st.dataframe(df.iloc[::-1])

with st.beta_expander("Simulação da estratégia de vacinação planejada por dia"):
    st.markdown(
        """
Esta tabela apresenta de forma detalhada o planejamento informado no painel de 
controle para simulação da estratégia de vacinação, apresentando os 
resultados por dia, faixa etária e etapa da vacinação.
"""
    )
    df = campaign.events.copy()
    st.dataframe(
        campaign.events.rename(
            columns={
                "day": "Dia",
                "age": "Idade",
                "doses": "Doses aplicadas",
                "phase": "Dose",
                "vaccine_type": "Vacina",
                "acc": "Doses acumuladas",
                "fraction": "% de cobertura",
            }
        )
    )
    codes = "; ".join(f"{i} = {v.name}" for i, v in enumerate(campaign.vaccines))
    st.markdown(f"Obs.: código das vacinas: {codes}")

if r.stats.initial_doses > 0:
    with st.beta_expander("Vacinas iniciais"):
        init = r.initial
        totals = r.age_distribution.loc[init.index].values
        st.bar_chart(100 * init.iloc[:, 0] / totals)

st.text("\n")
st.image("logo-opas.png")
st.markdown("Mais informações sobre vacinação COVID-19 [Our World in Data](https://ourworldindata.org/covid-vaccinations)")
