'''
The processing.py file provides data processing, quality metrics and plots
'''

import numpy as np
import mplcyberpunk
import matplotlib.pyplot as plt
from sklearn.metrics import\
mean_squared_error as mse, root_mean_squared_error as rmse,\
r2_score as r2, mean_absolute_error as mae, mean_absolute_percentage_error as mape

def wape(y_true, y_pred):
   '''Weighted Average Percentage Error'''
 
   y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
   return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

def print_all_metrics(y_true, y_pred, rounding=5):
   '''A function that outputs the results of metrics with rounding'''

   print(f'All metrics are rounded to {rounding} digits after the decimal point:')
   print(f'Mean Squared Error --> {round(mse(y_true, y_pred), rounding)}')
   print(f'Root Mean Squared Error --> {round(rmse(y_true, y_pred), rounding)}')
   print(f'Mean Absolute Error --> {round(mae(y_true, y_pred), rounding)}')
   print(f'Mean Absolute Percentage Error --> {round(mape(y_true, y_pred), rounding)}')
   print(f'Weighted Average Percentage Error --> {round(wape(y_true, y_pred), rounding)}')
   print(f'R2 Score --> {round(r2(y_true, y_pred), rounding)}')

def draw_plot_numerical(dataframe, filename='Plot of the Airy function', show=True):
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
   print('--> The plot of the numerical solution has been saved. \
The result can be seen in data/images/\n')
   if show: plt.show()

def draw_plot_compare(x_list, y_numerical, y_nn,
                      filename='Comparing PINN and numerical methods', show=True):
   '''Draws a plot comparing the numerical method and the neural network'''

   plt.style.use("cyberpunk")
   plt.figure()
   plt.plot(x_list, y_numerical, label='The numerical method')
   plt.plot(x_list, y_nn, label='Neural network')
   plt.legend(loc='lower right')
   plt.xlabel('x-axis', weight='bold')
   plt.ylabel('y-axis', weight='bold')
   plt.title('Dependence of y on x', weight='bold')
   mplcyberpunk.add_glow_effects()
   plt.savefig(f'data/images/{filename}.png')
   print('--> The plot of the comparing has been saved. \
The result can be seen in data/images/\n')
   if show: plt.show()