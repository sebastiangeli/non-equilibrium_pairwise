# -*- coding: utf-8 -*-


"""
Code for "Non-equilibrium whole-brain dynamics arise from pairwise interactions"
Sebastian M. Geli, Christopher W. Lynn, Morten L. Kringelbach, Gustavo Deco and
Yonatan Sanz Perl.


This script trains an autoencoder neural network on time series data.
It returns the latent space representation of the time series. The number of
latent units can be modified to set the dimension of the reduced representation.

In the training set the time series of all subjects during all conditions are
concatenated, with the tasks in the following order: REST, SOCIAL, RELATIONAL,
MOTOR, LANGUAGE, WM, GAMBLING and EMOTION. Then, the shape of the training
dataset is (TPOINTS*NSUB*NTASKS, NAREAS). Where:

TPOINTS = number of timepoints in a measurement
NSUB = number of subjects in the training dataset
NTASKS = number of tasks
NAREAS = number of regions in the parcellation


For queries or issues, please contact Sebastian Geli at sebastianmanuel.geli@upf.edu
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
import pandas as pd
import matplotlib.pyplot as plt
import random



#%% Random seed and set device functions

def set_seed(seed=None, seed_torch=True):
  """
  Function that controls randomness. NumPy and random modules must be imported.

  Args:
    seed : Integer
      A non-negative integer that defines the random state. Default is `None`.
    seed_torch : Boolean
      If `True` sets the random seed for pytorch tensors, so pytorch module
      must be imported. Default is `True`.

  Returns:
    Nothing.
  """
  if seed is None:
    seed = np.random.choice(2 ** 32)
  random.seed(seed)
  np.random.seed(seed)
  if seed_torch:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

  print(f'Random seed {seed} has been set.')


# In case that `DataLoader` is used
def seed_worker(worker_id):
  """
  DataLoader will reseed workers following randomness in
  multi-process data loading algorithm.

  Args:
    worker_id: integer
      ID of subprocess to seed. 0 means that
      the data will be loaded in the main process
      Refer: https://pytorch.org/docs/stable/data.html#data-loading-randomness for more details

  Returns:
    Nothing
  """
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)



def set_device():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  if device != "cuda":
      print("For this notebook to perform best, run in a GPU")
  else:
      print("GPU is enabled.")

  return device

#%% Define network and training functions

class ae_net(nn.Module):
  def __init__(self,latent_dim):
    """Intitalize neural net layers"""
    super(ae_net, self).__init__()

    self.fc1 = nn.Linear(in_features=NAREAS, out_features=128)
    self.fc2 = nn.Linear(in_features=128, out_features=64)
    self.fc3 = nn.Linear(in_features=64, out_features=32)
    self.fc4 = nn.Linear(in_features=32, out_features=16)
    self.fc5 = nn.Linear(in_features=16, out_features=latent_dim)
    self.fc6 = nn.Linear(in_features=latent_dim, out_features=16)
    self.fc7 = nn.Linear(in_features=16, out_features=32)
    self.fc8 = nn.Linear(in_features=32, out_features=64)
    self.fc9 = nn.Linear(in_features=64, out_features=128)
    self.fc10 = nn.Linear(in_features=128, out_features=NAREAS)

    self.batchnorm1 = nn.BatchNorm1d(num_features=128)
    self.batchnorm2 = nn.BatchNorm1d(num_features=64)
    self.batchnorm3 = nn.BatchNorm1d(num_features=32)
    self.batchnorm4 = nn.BatchNorm1d(num_features=16)
    self.batchnorm5 = nn.BatchNorm1d(num_features=16)
    self.batchnorm6 = nn.BatchNorm1d(num_features=32)
    self.batchnorm7 = nn.BatchNorm1d(num_features=64)
    self.batchnorm8 = nn.BatchNorm1d(num_features=128)

    # self.dropout = nn.Dropout(p=0.3, inplace=False)

  def forward(self, x):

    x = self.fc1(x)
    x = self.batchnorm1(x)
    x = F.relu(x)

    x = self.fc2(x)
    x = self.batchnorm2(x)
    x = F.relu(x)

    x = self.fc3(x)
    x = self.batchnorm3(x)
    x = F.relu(x)

    x = self.fc4(x)
    x = self.batchnorm4(x)
    x = F.relu(x)

    x = self.fc5(x)

    x = self.fc6(x)
    x = self.batchnorm5(x)
    x = F.relu(x)

    x = self.fc7(x)
    x = self.batchnorm6(x)
    x = F.relu(x)

    x = self.fc8(x)
    x = self.batchnorm7(x)
    x = F.relu(x)

    x = self.fc9(x)
    x = self.batchnorm8(x)
    x = F.relu(x)

    x = self.fc10(x)

    return x

  def encoder(self, x):

    x = self.fc1(x)
    x = self.batchnorm1(x)
    x = F.relu(x)

    x = self.fc2(x)
    x = self.batchnorm2(x)
    x = F.relu(x)

    x = self.fc3(x)
    x = self.batchnorm3(x)
    x = F.relu(x)

    x = self.fc4(x)
    x = self.batchnorm4(x)
    x = F.relu(x)

    x = self.fc5(x)

    return x

def train(model, device, train_loader, validation_loader, epochs,patience, learning_rate):

  """
  Train the autoencoder model.

  Args:
    model (nn.Module): Autoencoder model.
    device (str): Device to train the model on ('cpu' or 'cuda').
    train_loader (DataLoader): DataLoader for training data.
    validation_loader (DataLoader): DataLoader for validation data.
    epochs (int): Number of epochs for training.
    patience (int): Number of epochs to wait for improvement before early stopping.
    learning_rate (float): Learning rate.

  Returns:
    Tuple: Training loss, validation loss, best model, and the epochs after finding the best model.
  """

  t_i = time.time()
  criterion =  nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  train_loss, validation_loss = [], []

  # Keeps track of number of epochs during which the val_acc was less than best_acc
  wait = 0
  best_loss = 900**2
  best_epoch = 0


  for epoch in range(epochs):
      model.train()
      # keep track of the running loss
      running_loss = 0.


      for data in train_loader:
        # getting the training set
        data = data.to(device, dtype=torch.float)
        # Get the model output (call the model with the data from this batch)
        output = model(data)
        # Zero the gradients out
        optimizer.zero_grad()
        # Get the Loss
        loss  = criterion(output, data)
        # Calculate the gradients
        loss.backward()
        # Update the weights (using the training step of the optimizer)
        optimizer.step()

        running_loss += loss  # add the loss for this batch

      # append the loss for this epoch
      train_loss.append(running_loss.detach().cpu().item()/len(train_loader))

      # evaluate on validation data
      model.eval()
      running_loss = 0.

      for data in validation_loader:
        # getting the validation set
        data = data.to(device, dtype=torch.float)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        running_loss += loss.item()

      val_loss = running_loss/len(validation_loader)
      validation_loss.append(val_loss)

      if (val_loss < best_loss):
        best_loss = val_loss
        best_epoch = epoch
        best_model = copy.deepcopy(model)
        wait = 0
        print('Min val-loss. so far: ', np.round(val_loss,3), '. Epoch:', epoch)
      else:
        wait += 1

      # Early stopping
      if (wait > patience):
        print(f'Early stopped on epoch: {epoch}')
        break

  t_f = time.time()-t_i
  print(f'Training completed. Elapsed time {int(t_f)}s')

  return np.array(train_loss), np.array(validation_loss), best_model, best_epoch

def create_toy_data(num_regions, time_points, NSUB, phase_std = None):

  """
        Creates an array of standarized sine waves with random phases and noise.

        Parameters
        ----------
        num_regions : int
            The number of sine waves to create.

        time_points : int
            The number of time points for each sine wave.

        NSUB : int
            The number of repetitions.

        phase_std : float, optional
            The standard deviation of the phase of each sine wave. Default is
            None, which means that the phase is randomly chosen from a uniform
            distribution between 0 and 2*pi. If not None, the phase is drawn from
            a normal distribution with mean 0 and the given standard deviation.

        Returns
        -------
        standardized_sine_waves : ndarray (time_points * NSUB, num_regions)
            An array of standardized sine waves with random phases and noise. All
            the "subjects" are concatenated along the first dimension.

  """

  time = np.linspace(0, NSUB, NSUB * time_points)
  sine_waves = np.zeros((NSUB * time_points, num_regions))
  for i in range(num_regions):
    if phase_std == None:
      phase = np.random.uniform(0, 2 * np.pi)
    else:
      phase = np.random.normal(0, phase_std)

    # Random gaussian noise
    noise = np.random.normal(0, 0.2, NSUB * time_points)
    # By default there will be 4 complete oscillations in each measurement
    f = 4
    sine_waves[:, i] = np.sin(2 * np.pi * f * time + phase) + noise

  # Calculate the mean and standard deviation of each row
  mean = np.mean(sine_waves, axis=0, keepdims=True)
  std = np.std(sine_waves, axis=0, keepdims=True)

  # Standardize each row
  standardized_sine_waves = (sine_waves - mean) / std

  return standardized_sine_waves

#%%
tasks = ['REST1', 'SOCIAL', 'RELATIONAL', 'MOTOR', 'LANGUAGE', 'WM', 'GAMBLING', 'EMOTION']

# Set dimensions of the data
NSUB = 10         # Subjects in train datasat (originally 989)
TPOINTS = 176     #Timepoints of each measurement
NAREAS = 62       #Regions in the parcellation

# Define latent space dimension and an index identifying the model trained.
latent_dim = 7
index_save = 0  # [0,49] -> Repetition of training. For this work, each model was trained 50 times

# Use index as seed
SEED = index_save

# Define hyperparameters
learning_rate = 0.0004
epochs = 600     # Maximum number of training epochs
patience = 60    # Maximum number of epochs to wait before early stopping


#%% Load or generate data

# train_dataset = np.loadtxt(f"{path_load}train.csv", delimiter=",")
# val_dataset = np.loadtxt(f"{path_load}validation.csv", delimiter=",")

# Define file paths for loading and saving data
# path_load = ''
# path_save = ''

dataset = create_toy_data(num_regions=NAREAS,time_points=TPOINTS,NSUB=NSUB, phase_std = 0.15)

# Split train and validation sets
train_dataset = dataset[:int(NSUB*0.7)*176,:]  
val_dataset = dataset[int(NSUB*0.7)*176:,:]

# Create dataloaders
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128*2*len(tasks), shuffle=True, num_workers = 0,
                                worker_init_fn=seed_worker)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers = 0,
                                worker_init_fn=seed_worker)

#%% Train network

# Set random seed and device
set_seed(seed=SEED)
device = set_device()
# Initialize and move model to device
net = ae_net(latent_dim).to(device)

# Train the model
t_loss, v_loss, model, best_epoch = train(net, device, train_loader, val_loader, epochs=epochs,patience=patience,learning_rate=learning_rate)

# Obtain latent space representation of the whole training set
latent_train = model.encoder(torch.Tensor(train_dataset).to(device, dtype=torch.float)).detach().cpu()

#%% Save
# Save training and validation losses
df_learning = pd.DataFrame(np.array([t_loss,v_loss]).T,columns=['train_loss','validation_loss'])
# df_learning.to_csv(f'{path_save}learning_dim={latent_dim}_{index_save}.csv',index=False)

# Save latent space representations and labels
# labels_train = np.repeat(tasks,NSUB*TPOINTS)
# df_ls = pd.DataFrame(np.column_stack((np.array(latent_train),labels_train)),columns=['x'+str(i) for i in range(np.shape(latent_train)[1])]+['task'])
# df_ls.to_csv(f'{path_save}latspace_dim={latent_dim}_{index_save}.csv',index=False)

# Save the trained model
# torch.save(model.state_dict(), f'{path_save}model_dim={latent_dim}_{index_save}.pth')

plt.figure(figsize=(5, 3))
plt.plot(t_loss, label='Training Loss')
plt.plot(v_loss, label='Validation Loss')
plt.scatter(best_epoch, v_loss[best_epoch], color='red', label=f'Best Epoch ({best_epoch})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()