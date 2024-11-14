'''The main executable file that implements the neural network and \
its comparison with the numerical method'''

from processing import draw_plot_numerical, draw_plot_compare, print_all_metrics
from data_numerical import get_numerical_airy, save_dataframe
from neural_network import Sin, launch, save_checkpoint
from torch import nn
import torch

numerical_dataframe = get_numerical_airy(10_000)
draw_plot_numerical(numerical_dataframe, filename='example_numerical_plot')
save_dataframe(numerical_dataframe, filename='example_numerical_df')

layer_list = [
    nn.Linear(1, 128), nn.Tanh(),
    nn.Linear(128, 64), Sin(),
    nn.Linear(64, 32), nn.Tanh(),
    nn.Linear(32, 16), Sin(),
    nn.Linear(16, 1)
]

model = launch(layer_list, numerical_dataframe,
               epochs=250, batchsize=128, lr=0.005)
save_checkpoint(model, filename='example_checkpoint')

x_list = list(numerical_dataframe['x_value'].values)
y_numerical = list(numerical_dataframe['y_value'].values)
y_nn = list(float(model(torch.Tensor([x]))) for x in x_list)
draw_plot_compare(x_list, y_numerical, y_nn, filename='example_comparing_plot')
print_all_metrics(y_numerical, y_nn, rounding=7)