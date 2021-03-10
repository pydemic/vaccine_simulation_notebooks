import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import locale
from typing import List, Dict

sys.path.append(".")
import vaccines as lib


def read_inputs(st=st.sidebar) -> dict:
    st.header("Opções")
    opts = load_regions()
    region = st.selectbox("UF", [*opts.keys()], format_func=opts.get)

    st.subheader("Estoque de doses")
    stocks = {
        vaccine: st.number_input(f"Doses ({vaccine})", min_value=0, value=200_000)
        for vaccine in lib.VACCINE_DB
    }
    rate = st.number_input(
        "Capacidade de vacinação (vacinas/dia)", min_value=0, value=6_000
    )

    st.subheader("Planos de vacinação")
    coarse = st.checkbox("Agrupar de 10 em 10 anos")

    vaccine_plan = st.text_area("Metas", "95%")
    step = 10 if coarse else 5
    placeholder = "\n".join(f"{n}: 0" for n in range(80, 19, -step))
    initial_plan = st.text_area(
        "Vacinados",
        f"""
# Preencha a quantidade de 
# pessoas vacinadas por 
# faixa etária.
{placeholder}
    """.strip(),
        height=400,
    )

    st.markdown("---")

    return {
        "region": region,
        "stocks": stocks,
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

    builtins.st = st


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


#
# Application
#
config()
st.markdown(
    "## Ferramenta para determinação do impacto da "
    "vacinação nas internações por COVID-19"
)

opts = read_inputs()
coarse, rate, region, stocks = (opts[k] for k in ["coarse", "rate", "region", "stocks"])
age_distribution = load_age_distribution(region, coarse)
hospitalizations = load_hospitalizations(region, coarse)
deaths = load_deaths(region, coarse)


#
# Prepara entradas
#
vaccine_stocks = {k: v for k, v in opts["stocks"].items() if v}
vaccines = [k for k, v in sorted(vaccine_stocks.items(), reverse=True)]
max_doses = [vaccine_stocks[k] // 2 for k in vaccines]
total_doses = sum(max_doses)

delay = {
    "immunization": [v.immunization_delay for v in vaccines],
    "second_dose": [v.second_dose_delay for v in vaccines],
}

initial = lib.parse_plan(opts["initial_plan"], age_distribution)
initial = pd.DataFrame(initial, columns=["age", "value"]).set_index("age")


#
# Simula plano e calcula resultados
#
with st.spinner():
    plan = lib.MultipleVaccinesRatePlan.from_source(
        opts["vaccine_plan"],
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
    "smooth": st.checkbox("Imunidade gradual"),
    "initial": initial,
}
eff = [v.efficiency for v in vaccines]
hospital_pressure = result.damage_curve(hospitalizations, efficiency=eff, **kwds)
death_pressure = result.damage_curve(deaths, **kwds)

st.markdown(
    f"""
## Resultados

* **Total de doses:** {int(result.applied_doses):n}
* **Pessoas vacinadas:** {int(result.applied_doses // 2):n} + {initial.values.sum():n} (inicial)
* **Dias de vacinação:** {result.campaign_duration}
* **Redução na hospitalização:** {100 - 100 * hospital_pressure.iloc[-1]:.1f}%
* **Redução dos óbitos:** {100 - 100 * death_pressure.iloc[-1]:.1f}%
"""
)

#
# Gráficos
#
fig, ax = plt.subplots()
result.plot_hospitalization_pressure_curve(hospital_pressure, as_pressure=True)
st.pyplot(fig)

fig, ax = plt.subplots()
result.plot_death_pressure_curve(death_pressure, as_pressure=True)
st.pyplot(fig)

fig, ax = plt.subplots()
result.plot_vaccination_schedule(ax=ax)
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
            "Distribuição etária": age_distribution,
            "Hospitalizações": hospitalizations,
            "Óbitos": deaths,
        }
    ).dropna()
    st.dataframe(df.iloc[::-1])
