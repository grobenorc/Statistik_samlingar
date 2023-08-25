# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 02:09:02 2023

Det här är ett dokument med olika ega funktioner, främst
för statistik och anpassade efter vanligt förekommande
beräkningar.

@author: claes
"""
import os
os.chdir('C:\\Users\\claes\\OneDrive\\Python\\Statistik_funktioner_egna')

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import sympy as sp
import scipy
from scipy import*
import math
import random

from scipy.stats import norm, poisson, expon


def mean(values): 
    """
    Beräknar medelvärdet av en lista med värden.
    
    Argument:
      values (list): En lista med numeriska värden.
    
    Returnerar:
      Medelvärdet av de givna värdena.
    """
    mean = sum(values) / len(values)
    return mean

def weighted_mean(values, weights):
    """
    Beräknar viktat medelvärde för en lista med värden och vikter.
    
    Argument:
      values (list): En lista med numeriska värden.
      weights (list): En lista med vikter som motsvarar varje värde.
    
    Returnerar:
      Det viktade medelvärdet av de givna värdena.
    """
    
    if len(values) != len(weights):
        raise ValueError("Värden och vikter måste ha samma längd")
    
    weighted_mean = sum(s * w for s, w in zip(values, weights)) / sum(weights)
    return weighted_mean


def median(values):
    """
    Beräknar medianen av en lista med värden.
    
    Argument:
      values (list): En lista med numeriska värden.
    
    Returnerar:
      Medianvärdet av de givna värdena.
    """
    
    ordered = sorted(values)
    n = len(ordered)
    mid = int(n / 2) - 1 if n % 2 == 0 else int(n/2)
    
    if n % 2 == 0:
        return (ordered[mid] + ordered[mid+1]) / 2.0
    else:
        return ordered[mid]

def mode(values):
    """
    Beräknar lägenheten (modus) av en lista med värden.
    
    Argument:
      values (list): En lista med numeriska värden.
    
    Returnerar:
      En lista med mest förekomande tal-(en/ett) av de givna värdena (kan vara flera om flera värden upprepas lika ofta).
    """
    
    from collections import defaultdict
    
    counts = defaultdict(lambda: 0)
    
    for s in values:
        counts[s] += 1
        
    max_count = max(counts.values())
    modes = [v for v in set(values) if counts[v] == max_count]
    return modes


def stddev(values, is_sample: bool = True):
    """
    Beräknar standardavvikelsen av en lista med värden.
    
    Argument:
      values (list): En lista med numeriska värden.
      is_sample (bool): En boolsk parameter som anger om datan representerar urval (True) eller populationen (False).
    
    Returnerar:
      Standardavvikelsen av de givna värdena.
    """
    
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - (1 if is_sample else 0))
    stddev = variance**(1/2)
    return stddev


def variance(values, is_sample: bool = True):
    """
    Beräknar variansen av en lista med värden.
    
    Argument:
      values (list): En lista med numeriska värden.
      is_sample (bool): En boolsk parameter som anger om datan representerar urval (True) eller populationen (False).
    
    Returnerar:
      Variansen av de givna värdena.
    """
    
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - (1 if is_sample else 0))
    return variance

def standardise_value(x, mean, sd):
    """
    Standardiserar ett unikt x-värde.
    
    Argument:
      x: Ett unikt x-värde.
      mean: Medelvärde av stickprovet(delpopulation)
      sd: standardavvikelse av stickprovet (delpopulation)
      
    Returnerar:
      Det standardiserade z-värdet av en unik observation.
    """
    z_value = (x-mean)/sd
    return z_value

def standardise_list(values, is_sample: bool = True):
    """
    Standardiserar en lista med värden.
    
    Argument:
      values (list): En lista med numeriska värden.
      is_sample (bool): En boolsk parameter som anger om datan representerar urval (True) eller populationen (False).
    
    Returnerar:
      En lista med standardiserade värden baserat på medelvärde och standardavvikelse.
    """    
    mean = sum(values)/len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - (1 if is_sample else 0))
    z_value = [ ((value-mean)/variance**(1/2)) for value in values]
    return z_value


def unstandardise_value(z, mean, sd):
    """
    Standardiserar ett unikt x-värde.
    
    Argument:
      x: Ett unikt x-värde.
      mean: Medelvärde av stickprovet(delpopulation)
      sd: standardavvikelse av stickprovet (delpopulation)
      
    Returnerar:
      Det standardiserade z-värdet av en unik observation.
    """
    z_value = (z*sd) + mean
    return z_value


def unstandardise_list(z_values, is_sample: bool = True):
    """
    Standardiserar en lista med värden.
    
    Argument:
      values (list): En lista med numeriska värden.
      is_sample (bool): En boolsk parameter som anger om datan representerar urval (True) eller populationen (False).
    
    Returnerar:
      En lista med standardiserade värden baserat på medelvärde och standardavvikelse.
    """    
    mean = sum(z_values)/len(z_values)
    variance = sum((v - mean) ** 2 for v in z_values) / (len(z_values) - (1 if is_sample else 0))
    z_value = [ ((value-mean)/variance**(1/2)) for value in z_values]
    return z_value


def critical_z_value(p):
    norm_dist = scipy.norm(loc=0.0, scale=1.0)
    left_tail_area = (1.0 - p) / 2.0
    upper_area = 1.0 - ((1.0 - p) / 2.0)
    return norm_dist.ppf(left_tail_area), norm_dist.ppf(upper_area)


def confidence_interval(p, sample_mean, sample_std, n):
    # Stickprovsstorlek måste vara större än 30.
    lower, upper = critical_z_value(p)
    lower_ci = lower * (sample_std / n**(1/2))
    upper_ci = upper * (sample_std / n**(1/2))
    return sample_mean + lower_ci, sample_mean + upper_ci





def visualise_normal_probability(start, end, mean=0, std=1, x_range = (-3,3)):
    x = np.linspace(x_range[0], x_range[1], 1000)
    y = norm.pdf(x, mean, std)
    prob = norm.cdf(end,mean,std) - norm.cdf(start, mean, std)
    prob_round = round(prob, 4)
    
    fig, ax = plt.subplots()
    ax.plot(x, y, 'b', linewidth=2)
    ax.fill_between(x, y, where=[(val >= start and val <= end) for val in x], color='red', alpha=0.4)
    title = f'Normal Distribution: $Pr({sp.latex(start)} \leq X \leq {sp.latex(end)})={prob_round}$'
    ax.set_title(title, fontsize=14)
    plt.xlabel('$X$')
    plt.ylabel('Probability Density')
    plt.show()



def visualise_poisson_probability(start, end, lam=1, x_range=(0, 10)):
    x = np.arange(x_range[0], x_range[1] + 1)
    y = poisson.pmf(x, lam)
    prob = np.sum(y[start:end+1])
    prob_round = round(prob, 4)
    
    fig, ax = plt.subplots()
    ax.stem(x, y, basefmt='b-', linefmt='b-', markerfmt='bo')
    ax.fill_between(x, y, where=[(val >= start and val <= end) for val in x], color='red', alpha=0.4)
    title = f'Poisson Distribution: $Pr({sp.latex(start)} \leq X \leq {sp.latex(end)})={prob_round}$'
    ax.set_title(title, fontsize=14)
    plt.xlabel('$X$')
    plt.ylabel('Probability')
    plt.show()



def visualise_exponential_probability(start, end, scale=1, x_range=(0, 10)):
    x = np.linspace(x_range[0], x_range[1], 1000)
    y = expon.pdf(x, scale=scale)
    prob = expon.cdf(end, scale=scale) - expon.cdf(start, scale=scale)
    prob_round = round(prob, 4)
    
    fig, ax = plt.subplots()
    ax.plot(x, y, 'b', linewidth=2)
    ax.fill_between(x, y, where=[(val >= start and val <= end) for val in x], color='red', alpha=0.4)
    title = f'Exponential Distribution: $Pr({sp.latex(start)} \leq X \leq {sp.latex(end)})={prob_round}$'
    ax.set_title(title, fontsize=14)
    plt.xlabel('$X$')
    plt.ylabel('Probability Density')
    plt.show()


def exponential_cdf(start, end, scale=1):
    x = sp.symbols('x')
    cdf_expression = 1 - sp.exp(-x / scale)
    cdf_between_values = sp.integrate(cdf_expression, (x, start, end))
    return cdf_between_values




