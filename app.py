from vaccines.utils import compute_schedule
from types import SimpleNamespace
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import locale
from typing import List, Dict

sys.path.append(".")
import vaccines as lib

def simple(n):
    st = str(int(n))
    return int(st[:2] + '0' * max(len(st) - 2, 0))

def read_inputs(sb=st.sidebar, st=st) -> dict:
    sb.header("Configurações")
    opts = load_regions()
    region = sb.selectbox("UF", [*opts.keys()], format_func=opts.get, index=len(opts) - 1)
    population = load_population(region)

    sb.subheader("Estoque de doses")
    stock = simple(population * 0.5 / len(lib.VACCINE_DB)) // 1000
    stocks = {
        vaccine: 1000 * sb.number_input(f"Mil doses ({vaccine})", min_value=0, value=stock)
        for vaccine in lib.VACCINE_DB
    }

    sb.subheader("Capacidade de vacinação")
    msg = "Capacidade de vacinação (doses/dia)"
    rate = sb.number_input(msg, min_value=0, value=simple(0.005 * population))
    
    sb.subheader("Opções")
    coarse = sb.checkbox("Agrupar de 10 em 10 anos")
    smooth=sb.checkbox("Imunidade gradual")

    st.header("Planos de vacinação")
    vaccine_plan = st.text_area("Metas de vacinação por faixa etária", "95%")
    with st.beta_expander('Vacinas já aplicadas'):
        step = 10 if coarse else 5
        placeholder = "\n".join(f"{n}: 0" for n in range(80, 19, -step))
        initial_plan = st.text_area(
            "Vacinados",
            f"""
# Preencha a quantidade de pessoas vacinadas por faixa etária.
{placeholder}
        """.strip(),
            height=225 if coarse else 375,
        )

    return {
        "region": region,
        "stocks": stocks,
        "smooth": smooth,
        "rate": rate,
        "coarse": coarse,
        "initial_plan": initial_plan,
        "vaccine_plan": vaccine_plan,
    }


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

    setattr(builtins, 'st', st)


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
def compute(coarse, rate, region, stocks, initial_plan, vaccine_plan, smooth):
    age_distribution = load_age_distribution(region, coarse)
    hospitalizations = load_hospitalizations(region, coarse)
    deaths = load_deaths(region, coarse)

    # Prepara entradas
    vaccine_stocks = {k: v for k, v in stocks.items() if v}
    vaccines = [k for k, v in sorted(vaccine_stocks.items(), reverse=True)]
    max_doses = [vaccine_stocks[k] // 2 for k in vaccines]
    total_doses = sum(max_doses)

    delay = {
        "immunization": [v.immunization_delay for v in vaccines],
        "second_dose": [v.second_dose_delay for v in vaccines],
    }

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

    duration = result.events["day"].max() + sum(max(x) for x in delay.values())
    duration = lib.by_periods(duration, 30)
    result = result.copy(duration=duration)

    kwds = {
        "delay": delay["immunization"],
        "smooth": smooth,
        "initial": initial,
    }
    eff = [v.efficiency for v in vaccines]
    hospital_pressure = result.damage_curve(hospitalizations, efficiency=eff, **kwds)
    death_pressure = result.damage_curve(deaths, **kwds)

    def expected(pressure, scale=1):
        res = (pressure / 365).sum()
        if duration >= 365:
            res *= 365 / duration
        else:
            dt = 365 - duration
            res += dt / 365 * pressure.iloc[-1]
        return scale * res
    
    expected_deaths = expected(death_pressure, deaths.sum())
    expected_hospitalizations = expected(hospital_pressure, hospitalizations.sum())

    return SimpleNamespace(
        age_distribution=age_distribution,
        applied_doses=result.applied_doses,
        death_pressure=death_pressure,
        deaths=deaths,
        duration=result.campaign_duration,
        expected_deaths=int(expected_deaths),
        expected_deaths_max=int(deaths.sum()),
        expected_hospitalizations=int(expected_hospitalizations),
        expected_hospitalizations_max=int(hospitalizations.sum()),
        events=result.events,
        hospital_pressure=hospital_pressure,
        hospitalizations=hospitalizations,
        initial_distribution=initial,
        initial_doses=initial.values.sum(),
        plots=result,
        reduced_deaths=1 - death_pressure.iloc[-1],
        reduced_hospitalizations=1 - hospital_pressure.iloc[-1],
        vaccines=plan.vaccines,
    )


#
# Application
#
config()
st.title(
    "Ferramenta para determinação do impacto da vacinação nas internações por COVID-19"
)
r = compute(**read_inputs())

st.header('Resultados')
st.markdown(
    f"""
* **Total de doses:** {int(r.applied_doses):n}
* **Pessoas vacinadas:** {int(r.applied_doses // 2):n} + {r.initial_doses:n} (inicial)
* **Óbitos anuais projetados*: ** {r.expected_deaths:n} (com vacina) / {r.expected_deaths_max:n} (sem vacina)
* **Hospitalizações anuais projetadas*: ** {r.expected_hospitalizations:n} (com vacina) / {r.expected_hospitalizations_max:n} (sem vacina)
* **Dias de vacinação:** {r.duration}
* **Redução na hospitalização:** {100 * r.reduced_hospitalizations:.1f}%
* **Redução dos óbitos:** {100 * r.reduced_deaths:.1f}%

&ast; Óbitos e hospitalizações foram projetadas a partir de dados do 
SIVEP/gripe. Alguns estados não possuem dados confiáveis nestas bases.
"""
)

#
# Gráficos
#
fig, ax = plt.subplots()
r.plots.plot_hospitalization_pressure_curve(r.hospital_pressure, as_pressure=True)
st.pyplot(fig)

fig, ax = plt.subplots()
r.plots.plot_death_pressure_curve(r.death_pressure, as_pressure=True)
st.pyplot(fig)

fig, ax = plt.subplots()
r.plots.plot_vaccination_schedule(ax=ax)
st.pyplot(fig)


#
# Observações
#
st.markdown(
    f"""
## Observações

O primeiro gráfico mostra a estimativa de redução nas hospitalizações esperadas enquanto o segundo gráfico mostra a estimativa de redução de mortalidade, em função da estratégia de imunização, ou seja devido à proteção conferida pelas vacinas. 

O terceiro gráfico mostra a cobertura vacinal por faixa etária ao longo do tempo. A faixa cinza mais clara representa o percentual da população que ainda não foi imunizada, 
a faixa cinza mais escura mostra a população que está aguardando a segunda dose e a faixa verde mostra a população que na qual foram aplicadas as duas
doses da vacina. 

A simulação considera que a taxa de infecção se mantêm constante, o que é uma suposição conservadora, 
especialmente em níveis mais altos de vacinação.

## Parâmetros utilizados 

1. Eficácia para formas graves da doença COVID-19 = 100%
2. Para o esquema vacinal foram considerados os seguintes intervalos:
- **Astrazeneca/Fiocruz:** O esquema de imunização é de 2 doses com intervalo máximo de 90 dias entre as doses.
- **Butantan:** O esquema de imunização é de 2 doses com intervalo máximo de 22 dias entre as doses.
3. Foram considerados imunizados apenas os indivíduos que tiverem as duas doses da vacina, 
4. A imunização (soroconversão) ocorre em um período de 28 dias após aplicação da segunda dose

## Referências

Parecer Público de avaliação de solicitação de autorização temporária de uso
emergencial, em caráter experimental, da vacina adsorvida covid-19 (inativada) –
[Instituto Butantan](https://www.gov.br/anvisa/pt-br/assuntos/noticias-anvisa/2021/confira-materiais-da-reuniao-extraordinaria-da-dicol/ppam-final-vacina-adsorvida-covid-19-inativada-butantan.pdf)

Parecer Público de avaliação de solicitação de autorização temporária de uso
emergencial, em caráter experimental, da vacina covid-19 (recombinante) –
[Fundação Oswaldo Cruz (Fiocruz)](https://www.gov.br/anvisa/pt-br/assuntos/noticias-anvisa/2021/confira-materiais-da-reuniao-extraordinaria-da-dicol/ppam-final-vacina-covid-19-recombinante-fiocruz.pdf)

## Avançado
"""
)

# Imprime dados
with st.beta_expander("Dados demográficos"):
    df = pd.DataFrame(
        {
            "Distribuição etária": r.age_distribution,
            "Hospitalizações": r.hospitalizations,
            "Óbitos": r.deaths,
        }
    ).dropna()
    st.dataframe(df.iloc[::-1])

with st.beta_expander("Programa vacinal detalhado"):
    df = r.events.copy()
    st.dataframe(r.events.rename(columns={
        'day': 'dia', 'age': 'idade', 'doses': 'doses', 'phase': 'fase',
        'vaccine_type': 'vacina', 'acc': 'doses acumuladas', 
        'fraction': 'fração da faixa etária',
    }))
    codes = '; '.join(f'{i} = {v.name}' for i, v in enumerate(r.vaccines))
    st.markdown(f'Obs.: código das vacinas: {codes}')

if r.initial_doses > 0:
    with st.beta_expander("Vacinas iniciais"):
        init = r.initial_distribution
        totals = r.age_distribution.loc[init.index].values
        st.bar_chart(100 * init.iloc[:, 0] / totals)


st.write()