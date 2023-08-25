# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 23:47:25 2023

@author: claes
"""

import math
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import scipy.special
import scipy.stats

class Distribution:
    def __init__(self, name, distribution_type, parameter_names, mean_params=None, variance_params=None, range_start=None, range_end=None):
        self.name = name
        self.distribution_type = distribution_type
        self.parameter_names = parameter_names
        self.mean_params = mean_params if mean_params is not None else parameter_names
        self.variance_params = variance_params if variance_params is not None else parameter_names
        self.parameters = []
        self.range_start = range_start
        self.range_end = range_end

    def set_parameters(self):
        for parameter_name in self.parameter_names:
            parameter = float(input(f'Ange {parameter_name}: '))
            self.parameters.append(parameter)

    def get_mean_params(self):
        return [self.parameters[self.parameter_names.index(param)] for param in self.mean_params]

    def get_variance_params(self):
        return [self.parameters[self.parameter_names.index(param)] for param in self.variance_params]

    def mean(self, *params):
        raise NotImplementedError("Subclasses must implement the mean method.")

    def variance(self, *params):
        raise NotImplementedError("Subclasses must implement the variance method.")

class PoissonFordelning(Distribution):
    
    def __init__(self):
        super().__init__('Poisson', 'Diskret', ['lambda'], mean_params=['lambda'], variance_params=['lambda'], range_start=0, range_end=float('inf'))

    def mean(self, lambda_):
        return lambda_

    def variance(self, lambda_):
        return lambda_
    
    # def pmf(self, lambda_, k):
    #     k = int(k)  # Convert k to an integer
    #     return (lambda_ ** k * math.exp(-lambda_)) / math.factorial(k)
    
    def pmf(self, lambda_, k):
        return (lambda_ ** k * math.exp(-lambda_)) / math.factorial(k)

    def cdf(self, lambda_, k):
        k = int(k)  # Convert k to an integer
        cdf_value = 0
        for i in range(k + 1):
            cdf_value += self.pmf(lambda_, i)
        return cdf_value


class BinomialFordelning(Distribution):
    def __init__(self):
        super().__init__('Binomial', 'Diskret', ['n', 'p'], mean_params=['n', 'p'], variance_params=['n', 'p'], range_start=0, range_end=float('inf'))
    
    def mean(self, n, p):
        mean = n*p
        return mean
    
    def variance(self, n, p):
        variance = n*p*(1-p)
        return variance

    def pmf(self, n, p, k):
        n = int(n)
        k = int(k)
        return math.comb(n, k)*p**k*(1-p)**(n-k)
    
    def cdf(self, n, p, k):
        k = int(k)
        cdf_value = 0
        for i in range(k + 1):
            cdf_value += self.pmf(n, p, i)
        return cdf_value
    
class GeometriskFordelning(Distribution):
    def __init__(self):
        super().__init__('Geometrisk', 'Diskret', ['p'], mean_params=['p'], variance_params=['p'], range_start=1, range_end=float('inf'))
    
    def mean(self, p):
        return 1 / p
    
    def variance(self, p):
        return (1 - p) / (p ** 2)

    def pmf(self, p, k):
        p = float(p)
        k = int(k)
        return (1 - p) ** (k - 1) * p
    
    def cdf(self, p, k):
        p = float(p)
        k = int(k)
        cdf_value = 0
        for i in range(1, k + 1):
            cdf_value += self.pmf(p, i)
        return cdf_value

        

class NormalFordelning(Distribution):
    def __init__(self):
        super().__init__('Normalfördelning', 'Kontinuerlig', ['mu', 'std'], mean_params=['mu'], variance_params=['std'], range_start=-float('inf'), range_end=float('inf'))


    def mean(self, mu):
        return mu
    
    def variance(self, std_dev):
        return std_dev ** 2

    def pdf(self, mu, std_dev, x):
        x = float(x)
        return (1 / (std_dev * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) / std_dev) ** 2)
    
    def cdf(self, mu, std_dev, x):
        x = float(x)
        return 0.5 * (1 + math.erf((x - mu) / (std_dev * math.sqrt(2))))
    
class ExponentialFordelning(Distribution):
    def __init__(self):
        super().__init__('Exponentialfördelning', 'Kontinuerlig', ['lambda'], mean_params=['lambda'], variance_params=['lambda'], range_start=float(0), range_end=float('inf'))

    def mean(self, lambda_):
        return 1 / lambda_
    
    def variance(self, lambda_):
        return 1 / (lambda_**2)
    
    def pdf(self, lambda_, x):
        x = float(x) 
        return lambda_*math.exp(-lambda_*x)
    
    def cdf(self, lambda_, x):
        x = float(x)
        return 1-math.exp(-lambda_*x)


class GammaFordelning(Distribution):
    def __init__(self):
        super().__init__('Gamma Fördelning', 'Kontinuerlig', ['k', 'theta'], mean_params=['k', 'theta'], variance_params=['k', 'theta'], range_start=0, range_end=float('inf'))

    def mean(self, k, theta):
        return k * theta

    def variance(self, k, theta):
        return k * theta ** 2

    def pdf(self, k, theta, x):
        k = float(k)
        theta = float(theta)
        x = float(x)
        return (1 / (math.gamma(k) * (theta ** k))) * (x ** (k - 1)) * math.exp(-x / theta)

    def cdf(self, k, theta, x):
        k = float(k)
        theta = float(theta)
        x = float(x)
        return scipy.special.gammainc(k, x / theta)


class ChiSquaredFordelning(Distribution):
    def __init__(self):
        super().__init__('Chi-squared Fördelning', 'Kontinuerlig', ['f'], mean_params=['f'], variance_params=['f'], range_start=0, range_end=float('inf'))

    def mean(self, f):
        return f

    def variance(self, f):
        return 2 * f

    def pdf(self, f, x):
        f = float(f)
        x = float(x)
        return (1 / (2 ** (f / 2) * math.gamma(f / 2))) * (x ** (f / 2 - 1)) * math.exp(-x / 2)

    def cdf(self, f, x):
        f = float(f)
        x = float(x)
        return scipy.special.gammainc(f / 2, x / 2)

# class FFordelning(Distribution):
#     def __init__(self):
#         super().__init__('F-Fördelning', 'Kontinuerlig', ['df1', 'df2'], mean_params=['df2','df2'], variance_params=['df2','df2'], range_start=0, range_end=float('inf'))

#     def mean(self, df1, df2):
#         return df2 / (df2 - 2) if df2 > 2 else 1

#     def variance(self, df1, df2):
#         return (2 * df2 ** 2 * (df1 + df2 - 2)) / (df1 * (df2 - 2) ** 2 * (df2 - 4))

#     def pdf(self, df1, df2, x):
#         x = float(x)
#         return scipy.stats.f.pdf(x, df1, df2)

#     def cdf(self, df1, df2, x):
#         x = float(x)
#         return scipy.stats.f.cdf(x, df1, df2)


def plot_distribution_curve(distribution_instance, parameters, dist_choice, observerat, px):
    if distribution_instance.distribution_type == 'Diskret':
        mean_value = distribution_instance.mean(*distribution_instance.get_mean_params())
        std_dev_value = math.sqrt(distribution_instance.variance(*distribution_instance.get_variance_params()))
        x_min = max(mean_value - 3 * std_dev_value, distribution_instance.range_start)
        x_max = min(mean_value + 3 * std_dev_value, distribution_instance.range_end)
        x_range = range(int(x_min),int(x_max))
    else:
        mean_value = distribution_instance.mean(*distribution_instance.get_mean_params())
        std_dev_value = math.sqrt(distribution_instance.variance(*distribution_instance.get_variance_params()))
        x_min = max(mean_value - 3 * std_dev_value, distribution_instance.range_start)
        x_max = min(mean_value + 3 * std_dev_value, distribution_instance.range_end)
        x_range = np.linspace(x_min, x_max, 1000)

        
    if dist_choice == 1:  # Plot PMF for discrete distributions
        if distribution_instance.distribution_type == 'Diskret':
            y_range = [distribution_instance.pmf(*distribution_instance.parameters, xi) for xi in x_range]
        else:
            y_range = [distribution_instance.pdf(*distribution_instance.parameters, xi) for xi in x_range]
        title = 'PMF' if distribution_instance.distribution_type == 'Diskret' else 'PDF'
    else:  # Plot CDF
        y_range = [distribution_instance.cdf(*distribution_instance.parameters, xi) for xi in x_range]
        title = 'CDF'

    
    plt.figure(figsize=(9, 6), dpi=150)
    # plt.plot(x_range, y_range, label=title)
    
    if distribution_instance.distribution_type == 'Diskret':
        plt.plot(x_range, y_range, label=title)
        for x_value, y_value in zip(x_range, y_range):
            plt.vlines(x_value, 0, y_value, colors='red', linestyles='solid')
            plt.scatter(x_value, y_value, color='red', marker='o', label=f'P({x_value}) = {y_value:.3f}')
    else:
        plt.plot(x_range, y_range, label=title)
        plt.scatter(observerat, px, color='red', marker='o', label=f'P({observerat}) = {px:.3f}')
    # plt.scatter(observerat, px, color='red', marker='o', label=f'P({observerat}) = {px:.3f}')
    if dist_choice == 2 and input("Fyll area under kurva? (ja/nej): ").lower() == 'ja' and distribution_instance.distribution_type == 'Kontinuerlig':  # Fill area under CDF
        plt.fill_between(x_range, 0, y_range, where=(x_range >= x_min) & (x_range <= observerat), alpha=0.3, label = "", color='red')
    
    plt.title(f'{distribution_instance.name} {title}')
    plt.xlabel(f'$x$ \nMedelvärde:  $\mu=${mean_value:.2f}\nVarians: $\sigma^2=${std_dev_value**2:.2f}')
    plt.ylabel('$f(x)$')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.8)) 
    
    # plt.text(mean_value, -0.2, f"Medelvärde:  $\mu=${mean_value:.2f}", ha='center')
    # plt.text(mean_value, -0.25, f"Varians: $\sigma^2=${std_dev_value**2:.2f}", ha='center')
    
    plt.show()
    


def main():
    distributions = {
        'poisson': PoissonFordelning(),
        'binomial': BinomialFordelning(),
        'geometrisk': GeometriskFordelning(),
        'normal': NormalFordelning(),
        'exponential': ExponentialFordelning(),
        'gamma': GammaFordelning(),
        'chi2': ChiSquaredFordelning(),
        # 'f': FFordelning(),
    }

    while True:  # Outer loop for repeating distribution selection
        dist_name = input("Ange fördelning: ").lower()
    
        if dist_name not in distributions:
            print("Fördelningen känns inte igen.")
            continue
    
        distribution_instance = distributions[dist_name]
        distribution_instance.set_parameters()
    
        while True:  # Inner loop for repeating distribution calculations
            dist_choice = int(input("1; PMF/PDF, 2; CDF (int): "))
            observerat = float(input("Observation x: "))
            plot_choice = input("Plotta fördelningen? (ja/nej): ")
    
            if dist_choice == 1:
                if distribution_instance.distribution_type == 'Diskret':
                    distribution_choice = distribution_instance.pmf
                else:
                    distribution_choice = distribution_instance.pdf
                px = distribution_choice(*distribution_instance.parameters, observerat)
            else:
                distribution_choice = distribution_instance.cdf
                px = distribution_choice(*distribution_instance.parameters, observerat)
    
            if plot_choice.lower() == 'ja':
                plot_distribution_curve(distribution_instance, distribution_instance.parameters, dist_choice, observerat, px)
    
            mean_value = distribution_instance.mean(*distribution_instance.get_mean_params())
            variance_value = distribution_instance.variance(*distribution_instance.get_variance_params())
    
            print("Mean Value:", mean_value)
            print("Variance Value:", variance_value)
    
            repeat = input("Vill du repetera? (Ja/Nej): ")
            if repeat.lower() != 'ja':  # Reset parameters for the next distribution instance
                break  # Exit the inner loop

        quit_choice = input("Vill du avsluta? (y/n): ")
        if quit_choice.lower() != 'n':
            break  # Exit the outer loop

# if __name__ == "__main__":
#     main()
