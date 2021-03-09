import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import asyncio
from time import time
from collections import defaultdict, deque, Counter
import operator
from typing import List, Tuple
from IPython.display import display, Latex, Markdown


def population_80_plus(data):
    data.loc[80] = data.loc[80:].sum()
    return data.loc[20:80].iloc[::-1]




def vaccine_curves(events, dose=2, extra=0):
    df = (
        events[['day', 'fraction', 'age']][events['dose'] == dose]
            .pivot('day', 'age', 'fraction')
            .fillna(method='pad')
            .fillna(0.0)
    )
    if extra:
        n = df.index.max() + 1
        df = pd.concat([df, pd.DataFrame([df.iloc[-1]], index=range(n, n + extra))])
    return df


def vaccination_day(events, end=True, delay=20):
    if end:
        c = vaccine_curves(events, 2, delay)
        return pd.Series(
            c.index[np.argmax((c == c.max()).values, axis=0)],
            index=curve.columns,
        ).sort_index(ascending=False)
    else:
        c = vaccine_curves(events, 1, delay)
        return pd.Series(
            c.index[np.argmin((c > 0).values, axis=0)],
            index=curve.columns,
        ).sort_index(ascending=False)


def severe_rate(curves, severe, delay=10, efficiency=1.0):
    N = severe.sum()
    rate = (N - (curves * efficiency * severe.loc[curves.columns]).sum(1)) / N
    rate.index += delay
    rate = rate.reindex(pd.RangeIndex(rate.index.max())).fillna(method='pad').fillna(1.0)
    return rate


def vaccination_day(events, start=False, delay=0, start_threshold=0.0):
    if start:
        c = vaccine_curves(events, 1, delay)
        return pd.Series(
            c.index[np.argmax((c > start_threshold).values, axis=0)],
            index=c.columns,
        ).sort_index(ascending=False)
    else:
        c = vaccine_curves(events, 2, delay)
        return pd.Series(
            c.index[np.argmax((c == c.max()).values, axis=0)],
            index=c.columns,
        ).sort_index(ascending=False)

    
def plot_vaccine_curves(events, severe, delay, efficiency=1.0, duration=None):
    fast = vaccine_curves(events, 1, delay)
    slow = vaccine_curves(events, 2, delay)
    
    curves = fast.copy().fillna(0.0)
    n = curves.index.min() - 1
    m = curves.index.max() + 1
    curves = pd.concat([
        pd.DataFrame([curves.iloc[0] * 0] * n, index=range(1, n + 1)),
        curves,
        pd.DataFrame([curves.iloc[0] * float('nan')] * delay, index=range(m, m + delay)),
    ]).fillna(method='pad')
    # curves.plot(alpha=0.5, ls='--', legend=True)

    c1 = severe_rate(slow, severe, delay, efficiency)
    c2 = severe_rate(fast, severe, delay, efficiency).reindex(c1.index).fillna(method='pad')
    
    # plt.fill_between(c1.index, c1.values, c2.reindex(c1.index).values, color='k', alpha=0.1)
    (100 * c1).plot(lw=2)
    # c2.plot(color='k', lw=2, ls='--', label='Primeira dose')
    plt.grid(True)
    plt.title('Estimativa de redução de hospitalizações', fontsize=20)
    plt.xlabel('tempo (dias)')
    plt.ylabel('pressão hospitalar (%)')
    plt.ylim(0, 100)
    plt.xlim(0, duration)
    #plt.legend(bbox_to_anchor=(1.21, 1), loc='upper right')
    
    return c1
    

def execute_plan(plan, rate, age_distribution, initial=None, *, delay):
    index = age_distribution.index
    population = np.asarray(age_distribution)
    nbins = len(population)
    
    events = defaultdict(lambda: np.array([0.0] * 2))
    schedule = defaultdict(Counter)

    acc = np.array([[0, 0]] * nbins)
    if initial is not None:
        if np.ndim(initial) == 1:
            acc[:, 0] = initial
        else:
            acc[:] = initial
    acc = pd.DataFrame(acc, columns=['first', 'second'], index=index)

    for day, daily_stock in enumerate(rate):
        stock = daily_stock
        while stock > 0:
            for age, goal in sorted(schedule[day].items(), reverse=True):
                if not stock:
                    break
                    
                if goal <= 0:
                    del schedule[day][age, goal]
                    continue

                applied = min(goal, stock)
                stock -= applied
                acc.loc[age, 'second'] += applied
                schedule[day][age] -= applied
                events[day, 2, age] += [applied, float(acc.loc[age, 'second'])]
                
                if goal > applied:
                    schedule[day + 1][age] += goal - applied

            if not stock or not plan:
                break

            age, goal = plan[0]
            if not goal:
                del plan[0]
                continue 

            pop = age_distribution.loc[age]
            applied = min(goal, stock, pop - acc.loc[age, 'first'])

            if applied == 0:
                del plan[0]
                continue

            schedule[day + delay][age] += applied
            plan[0] = age, goal - applied
            stock -= applied
            acc.loc[age, 'first'] += applied

            events[day, 1, age] += [applied, acc.loc[age, 'first']]
            break

    events = pd.DataFrame(
        events.values(), 
        columns=['applied', 'acc'],
        index=pd.MultiIndex.from_tuples(events.keys(), names=['day', 'dose', 'age']),
    ).reset_index()
    age_distribution.name = 'population'
    
    events['population'] = events['age'].apply(age_distribution.loc.__getitem__)
    events['day'] += 1
    
    events['fraction'] = events['acc'] / events['population']
    events = events[['day', 'dose', 'age', 'applied', 'acc', 'fraction']]
    events = events.reset_index(drop=True)
    return events



@widgets.interact(
    duration=widgets.IntSlider(6 * 30, 10, 500, deion="duração"), 
    region=widgets.Dropdown(options=region_options, deion="UF"),
    rate=widgets.IntText(6_000, deion="dose/dia"),
    # rate_increase=widgets.IntText(2_000, deion="incremento/dia"),
    vaccine=widgets.Dropdown(options=vaccine_options, deion="tipo vacina"),
    plan=widgets.Textarea(default_plan, deion="metas"),
    # vaccine_start=widgets.FloatSlider(0, min=0, max=1),
)
def run(duration, region, rate, vaccine, plan, vaccine_start=fixed(0.1), rate_increase=fixed(0), coarse=fixed(False)):
    global events, age_distribution, severe
    
    second_dose_delay, immunization_delay, eff = vaccine
    rate = [rate] * duration + np.linspace(0, rate_increase, duration)
    rate = np.where(rate < 0, 0, rate).astype(int)
    
    if coarse:
        s = population_80_plus(data.loc[region, "age_distribution"]).iloc[1:]
        age_distribution = pd.Series(s.values.reshape((2, len(s) // 2)).sum(0), index = s.index[1::2])
        age_distribution.name = region
    else:
        age_distribution = population_80_plus(data.loc[region, "age_distribution"])

    plan = parse_plan(plan, age_distribution)
    events = execute_plan(plan, rate, age_distribution, delay=second_dose_delay)
    severe = hospitalization.loc[region].iloc[::-1]
    
    reduced = plot_vaccine_curves(events, severe, immunization_delay, efficiency=eff, duration=duration)
    vaccines = int(events['applied'].sum())
    reduction = 100 - reduced.min() * 100

    plt.show()
    
    start = vaccination_day(events, start=True, start_threshold=vaccine_start, delay=immunization_delay).iloc[:duration]
    end = vaccination_day(events, start=False, delay=immunization_delay).iloc[:duration]
    
