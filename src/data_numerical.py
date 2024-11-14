'''
The data_numerical.py file provides the solution of the equation by numerical methods \
and the simplest processing of the results obtained.
'''

import math
import mplcyberpunk
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

const_a, const_b = (3, -10)
y_0 =1/(3 ** (2/3) * math.gamma(2/3))
y_prime_0 = -1/(3 ** (1/3) * math.gamma(1/3))

def _steps_calculation(data_length):
    '''Calculates the step for the numerical method based on the required data size'''
 
    length = abs(const_b - const_a)
    return length / data_length

def _derivatives(x, y_state):
   '''The function that defines the equation'''

   y, y_prime = y_state
   y_second_prime = x * y
   return [y_prime, y_second_prime]

def _reverse_direction(x, y_state):
   '''Reflects the equation with respect to the x direction'''

   y_prime, y_second_prime = _derivatives(-x, y_state)
   return [-y_prime, -y_second_prime]

def get_numerical_airy(data_length=10_000):
   '''Solves the equation using the RK45 method and returns it as a pandas-dataframe'''

   initial_state = [y_0, y_prime_0]
   span_forward, span_back = (0, const_a), (0, -const_b)
   step = _steps_calculation(data_length)

   forward_solution =  solve_ivp(_derivatives, span_forward, initial_state,
                                 method='RK45', max_step=step)
   back_solution = solve_ivp(_reverse_direction, span_back, initial_state,
                             method='RK45', max_step=step)
   
   x_list = list([-t for t in back_solution.t][::-1]) + list(forward_solution.t)
   y_list = list(back_solution.y[0][::-1]) + list(forward_solution.y[0])

   return pd.DataFrame({'x_value': x_list, 'y_value': y_list})

def save_dataframe(dataframe, filename='numerical_dataframe'):
   '''Saves the dataframe in csv format'''

   dataframe.to_csv(f'data/numerical_airy/{filename}.csv', index=False)

def load_dataframe(filename):
   '''Loads a dataframe from the csv format'''

   return pd.read_csv(f'data/numerical_airy/{filename}.csv')

def draw_plot_airy(dataframe, filename='Plot of the Airy function', show=True):
   '''Draws a plot of the numerical solution and saves it'''

   x_list = list(dataframe['x_value'].values)
   y_list = list(dataframe['y_value'].values)

   plt.style.use("cyberpunk")
   plt.figure()
   plt.plot(x_list, y_list, label='The numerical method')
   plt.legend(loc='lower right')
   plt.title('Dependence of y on x', weight='bold')
   plt.xlabel('x-axis', weight='bold')
   plt.ylabel('y-axis', weight='bold')
   mplcyberpunk.add_glow_effects()
   plt.savefig(f'data/images/{filename}.png')
   if show: plt.show()