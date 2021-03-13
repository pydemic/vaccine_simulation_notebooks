from types import SimpleNamespace
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
import operator
import locale
from typing import List, Tuple
from dataclasses import dataclass, field

sys.path.append(".")
import vaccines as lib

cte = lambda x: lambda *args: x


def simple(n):
    st = str(int(n))
    tail = max(len(st) - 2, 0)
    return int(st[:2] + "0" * tail)


@dataclass()
class InputReader:
    data: dict = field(default_factory=dict)
    exclude: set = field(default_factory=set)
    FIELDS = [
        "region",
        "stocks",
        "rate",
        "coarse",
        "smooth",
        "single_dose",
        "vaccine_plan",
        "initial_plan",
    ]
    PLACEHOLDER_VACCINE_PLAN = (
        "# Preencha meta de vacinação de cada faixa etária como porcentagem\n"
        "# da população em cada faixa.\n"
    )

    def ask(self):
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
        st.sidebar.header("Configurações")
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
        with st.beta_expander("Vacinação por faixa etária (clique para expandir)", expanded=True):
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
        height = 225 if coarse else 375
        plan = st.text_area(msg, placeholder, height=height)
        return self.check_plan(plan, full=False)


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


@st.cache
def load_population(region) -> pd.Series:
    return load_age_distribution(region).sum()


@st.cache
def load_age_distribution(region, coarse=False) -> pd.Series:
    db = pd.read_pickle("data.pkl.gz")
    out = db.loc[region, "age_distribution"]
    return lib.population_80_plus(out, coarse=coarse)


@st.cache
def load_hospitalizations(region, coarse=False) -> pd.Series:
    db = pd.read_pickle("hospitalization.pkl.gz")
    db.columns = [*db.columns]
    out = db.loc[region].iloc[::-1]
    if coarse:
        out = lib.coarse_distribution(out)
    return out


@st.cache
def load_deaths(region, coarse=False) -> pd.Series:
    db = pd.read_pickle("deaths.pkl.gz")
    db.columns = [*db.columns]
    out = db.loc[region].iloc[::-1]
    if coarse:
        out = lib.coarse_distribution(out)
    return out


@st.cache
def load_regions():
    db = pd.read_pickle("data.pkl.gz")
    return db["name"].to_dict()


@st.cache
def compute(
    coarse, rate, region, stocks, initial_plan, vaccine_plan, smooth, single_dose
):
    age_distribution = load_age_distribution(region, coarse)
    hospitalizations = load_hospitalizations(region, coarse)
    deaths = load_deaths(region, coarse)
    error = None

    # Prepara entradas
    num_phases = 1 if single_dose else 2
    vaccine_stocks = {k: v for k, v in stocks.items() if v}
    vaccines = [k for k, v in sorted(vaccine_stocks.items(), reverse=True)]
    max_doses = [vaccine_stocks[k] // num_phases for k in vaccines]
    total_doses = sum(max_doses)

    # Processa valores iniciais de vacina
    initial = lib.parse_plan(initial_plan, age_distribution)
    initial = pd.DataFrame(initial, columns=["age", "value"]).set_index("age")

    # Simula plano e calcula resultados
    plan = lib.MultipleVaccinesRatePlan.from_source(
        vaccine_plan,
        age_distribution,
        vaccines,
        initial=initial,
        rates=[rate * n / total_doses for n in max_doses],
        max_doses=max_doses,
    )
    result = plan.execute()

    # Recria resultado com duração desejada da simulação
    delay = {
        "immunization": [v.immunization_delay for v in vaccines],
        "second_dose": [v.second_dose_delay for v in vaccines],
    }
    duration = result.events["day"].max() + sum(max(x) for x in delay.values())
    duration = lib.by_periods(duration, 30)
    result = result.copy(duration=duration)

    # Danos esperados com/sem vacina
    def expected(pressure, scale=1):
        res = (pressure / 365).sum()
        if duration >= 365:
            res *= 365 / duration
        else:
            dt = 365 - duration
            res += dt / 365 * pressure.iloc[-1]
        return scale * res

    # Curvas de redução de pressão
    kwds = {
        "delay": delay["immunization"],
        "smooth": smooth,
        "initial": initial,
    }
    eff = [v.single_dose_efficiency if single_dose else v.efficiency for v in vaccines]

    try:
        hospital_pressure = result.damage_curve(
            hospitalizations, efficiency=eff, phase=2, **kwds
        )
        hospital_pressure_min = result.damage_curve(
            hospitalizations, efficiency=eff, phase=1, **kwds
        )

        death_pressure = result.damage_curve(deaths, phase=2, **kwds)
        death_pressure_min = result.damage_curve(deaths, phase=1, **kwds)

        expected_deaths = expected(death_pressure, deaths.sum())
        expected_hospitalizations = expected(hospital_pressure, hospitalizations.sum())

        reduced_deaths = 1 - death_pressure.iloc[-1]
        reduced_hospitalizations = 1 - hospital_pressure.iloc[-1]

    except Exception as error:
        hospital_pressure = hospital_pressure_min = None
        death_pressure = death_pressure_min = None
        expected_deaths = deaths.sum()
        expected_hospitalizations = hospitalizations.sum()
        reduced_deaths = 0
        reduced_hospitalizations = 0

    # Vacinados por faixa etária
    df = result.events.drop(columns=["day", "fraction", "acc"])
    df = df[df["phase"] == 2].groupby("age").sum()
    vaccinated = df["doses"]

    # Saída
    applied_doses = result.applied_doses
    duration = result.campaign_duration
    events = result.events
    initial_doses = initial.values.sum()
    expected_deaths_max = int(deaths.sum())
    expected_hospitalizations_max = int(hospitalizations.sum())
    vaccines = plan.vaccines
    del df, kwds, eff, expected
    return SimpleNamespace(plots=result, **locals())


#
# Application
#
config()
st.title(
    "Ferramenta para determinação do impacto da vacinação nas internações por COVID-19"
)
r = compute(**(InputReader().ask()))

st.header("Resultados")
st.markdown(
    f"""
* **Total de doses:** {int(r.applied_doses):n}
* **Pessoas vacinadas:** {int(r.applied_doses // 2):n} + {r.initial_doses:n} (inicial)
* **Óbitos anuais projetados*: ** {r.expected_deaths:n} (com vacina) / {r.expected_deaths_max:n} (sem vacina)
* **Hospitalizações anuais projetadas*: ** {r.expected_hospitalizations:n} (com vacina) / {r.expected_hospitalizations_max:n} (sem vacina)
* **Dias de vacinação:** {r.duration}
* **Redução na hospitalização:** {100 * r.reduced_hospitalizations:.1f}%
* **Redução dos óbitos:** {100 * r.reduced_deaths:.1f}%

&ast;  Óbitos e hospitalizações foram projetados a partir de dados do 
SIVEP/gripe, que é atualizado semanalmente e disponibilizado no link: https://opendatasus.saude.gov.br/dataset?tags=SRAG.
Os resultados apresentam maior acurácia quanto maior for a qualidade e 
oportunidade de registro dos dados nos sistemas oficiais.
"""
)

#
# Gráficos
#
if not r.error:
    fig, ax = plt.subplots()
    r.plots.plot_hospitalization_pressure_curve(
        r.hospital_pressure,
        as_pressure=True,
        # minimum=r.hospital_pressure_min,
    )
    st.pyplot(fig)

    fig, ax = plt.subplots()
    r.plots.plot_death_pressure_curve(
        r.death_pressure,
        as_pressure=True,
        # minimum=r.death_pressure_min,
    )
    st.pyplot(fig)

    fig, ax = plt.subplots()
    r.plots.plot_vaccination_schedule(ax=ax)
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
- **Butantan:** O esquema de imunização é de 2 doses com intervalo máximo de 22 dias entre as doses [1].
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

# Imprime dados avançados e possivelmente locais para baixar CSVs.
st.subheader("Avançado")

with st.beta_expander("Dados demográficos"):
    df = pd.DataFrame(
        {
            "Distribuição etária": r.age_distribution,
            "Hospitalizações": r.hospitalizations,
            "Vacinados (estimado)": r.vaccinated,
            "Óbitos": r.deaths,
        }
    ).dropna()
    st.dataframe(df.iloc[::-1])

with st.beta_expander("Programa vacinal detalhado"):
    df = r.events.copy()
    st.dataframe(
        r.events.rename(
            columns={
                "day": "dia",
                "age": "idade",
                "doses": "doses",
                "phase": "fase",
                "vaccine_type": "vacina",
                "acc": "doses acumuladas",
                "fraction": "fração da faixa etária",
            }
        )
    )
    codes = "; ".join(f"{i} = {v.name}" for i, v in enumerate(r.vaccines))
    st.markdown(f"Obs.: código das vacinas: {codes}")

if r.initial_doses > 0:
    with st.beta_expander("Vacinas iniciais"):
        init = r.initial_distribution
        totals = r.age_distribution.loc[init.index].values
        st.bar_chart(100 * init.iloc[:, 0] / totals)
