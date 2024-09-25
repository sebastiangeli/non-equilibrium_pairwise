# -*- coding: utf-8 -*-


"""
Code for "Non-equilibrium whole-brain dynamics arise from pairwise interactions"
Sebastian M. Geli, Christopher W. Lynn, Morten L. Kringelbach, Gustavo Deco and
Yonatan Sanz Perl.

This code computes the entropy production of pairs of regions, the S2(i,j) matrix.
It first discretizes the time series into a chosen number of states. Then, it
concatenates sets of the desired size of subject time series to accumulate enough
transitions to estimate the joint transition probabilities, and it computes the
entropy production of the region pairs for each "suprasubject". The final part
of the script trains a random forest classifier to predict the task based on the
S2(i,j) matrix.


For queries or issues, please contact Sebastian Geli at sebastianmanuel.geli@upf.edu
"""


import numpy as np
from itertools import combinations
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE





#%% Functions

def signal2discrete(signal,NCLUSTERS):

    """
      Transforms a signal to a discrete time series by thresholding.

          Parameters
          ----------
          signal : array
              Standarized time series.

          NCLUSTERS : int
              Number of desired microstates

          Returns
          -------
          points : array
              Discrete time series.

    """


    tresholds = np.linspace(-2,2,NCLUSTERS-1)
    conditions = np.array([signal > t for t in tresholds])
    discretized = np.sum(conditions,axis=0)

    return discretized.astype('int')



def transition_matrix(clusters):

    """
      Builds the matrix of observed transitions from a discrete 1D time series.

      Parameters
      ----------
      clusters : 1d-array
          Time series of clusters

      Returns
      -------
      transitions :  2 column-array
          Transition matrix in which each row represents a transition. The
          first column is the initial state, the second the final state.

    """

    transitions=np.zeros((len(clusters)-1,2))
    transitions[:,0]=clusters[:-1]
    transitions[:,1]=clusters[1:]

    return transitions.astype(int)


def global_transition_matrix(clusters,k):

    """
      Builds the matrix of observed transitions from a discrete ND time series.

          Parameters
          ----------
          clusters : 1d-array
              Time series of clusters

          k : int
            Number of states of each dimension

          Returns
          -------
          transitions :  2 column-array
              Transition matrix in which each row represents a global transition.
              The first column is the initial global state, the second the final
              global state.

    """

    dims = (np.ones_like(clusters[0])*k).astype(int)  #  Builds array like [k,...,k]
    clusters_tot = np.ravel_multi_index(clusters.T, dims)  #ind2sub
    transitions = transition_matrix(clusters_tot)

    return transitions


def compute_entropy_nomultip(transitions,NCLUSTERS):

    """
      Computes the entropy production from the transition matrix

      Parameters
      ----------
      transitions : 2 column-array
          Transition matrix from which the entropy will be computed.
          The first column is the initial state, the second the final state.

      NCLUSTERS : int
          Number of states that each individual element of the system can take.
          If the data comes from a discretized signal, it is the number of discrete
          states in each dimension. It should be the same for all dimensions.

      Returns
      -------
      H : float
          Entropy production. Infomation-theoretic measure of the
          irreversibility of the system, given by the Kullback-Leibler divergence
          between the forward and backward joint transition probability
          distributions.

      Notes
      -------
      It is not necessary to make the multipartite assumption here, as the
      intention is not to decompose entropy from this matrix.

    """

    n_states = NCLUSTERS**2

    # Count matrix (dim = K x K). Each cell Pij contains the counts of transitions from state i to j
    p = np.zeros((n_states,n_states), dtype=int);

    # To avoid dividing by 0 when calculating entropy, we add a pseudocount to all transitions
    p = p + 1


    for i in range(len(transitions)):
      p[(transitions[i,0]),(transitions[i,1])] = p[(transitions[i,0]),(transitions[i,1])] + 1

    # Number of possible global states. Used normalize the entropy production
    num_possible = (n_states)**2

    H = np.sum(p*np.log2(p/p.T)) * 1/(len(transitions)+num_possible)

    return H

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

# path_load_bold = ''
# path_save_i2 = ''

NCLUSTERS = 18         # Number of discrete states into which the signal is to be transformed
NSUPRASUB = 10         # Number of timepoints in a measurement
TPOINTS = 176          # Number of timepoints in a measurement
NAREAS = 4  #62        # Total number of regions in the parcellation
NSUB =  920            # Number of subjects

tasks = ['REST1', 'SOCIAL', 'RELATIONAL', 'MOTOR', 'LANGUAGE', 'WM', 'GAMBLING', 'EMOTION']
num_features = (NAREAS*(NAREAS-1))//2
column_names_df = [f'I2_{i}' for i in combinations(np.arange(NAREAS), 2)]+['task']
datasetI2 = pd.DataFrame(columns=column_names_df) # Create an empty DataFrame

for task_n, task in tqdm(enumerate(tasks)):
    bolds_task = create_toy_data(NAREAS, TPOINTS, NSUB, phase_std = task_n*0.1*np.pi)
    # bolds_task = np.load(f'{path_load}...npy')
    discrete_task = signal2discrete(bolds_task,NCLUSTERS)

    I2_task = np.zeros([len(discrete_task)//(TPOINTS*NSUPRASUB),num_features+1]) # I2 of Each combination of columns

    for f,cols in enumerate(combinations(np.arange(NAREAS), 2)):
        points_sub = [discrete_task[(NSUPRASUB*i*TPOINTS):(NSUPRASUB*i+NSUPRASUB)*TPOINTS ,cols] for i in range(len(discrete_task)//(NSUPRASUB*TPOINTS))]

        for i,sub in enumerate(points_sub):
            tms = [global_transition_matrix(sub[i*TPOINTS:(i+1)*TPOINTS],NCLUSTERS) for i in range(NSUPRASUB)]
            tm = np.concatenate(tms)
            I2_task[i,f] = compute_entropy_nomultip(tm,NCLUSTERS)


    I2_task[:,-1] = np.repeat(task_n,len(I2_task))
    df_task = pd.DataFrame(I2_task,columns=column_names_df) # Create a DataFrame for the current task
    datasetI2 = pd.concat([datasetI2, df_task], ignore_index=True) # Append the task DataFrame to the main DataFrame

# datasetI2.to_csv(f'{path_save}...csv',index=False)

dataset = datasetI2
# dataset = pd.read_csv(f'')

X = dataset.drop('task',axis='columns')
y = dataset.task

N_reps = 50
accuracies_decomposed = np.zeros((N_reps,9))  # This is the Recall per class

np.random.seed(0)

cm = np.zeros((8,8))

for i in tqdm(range(N_reps)):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
    model = RandomForestClassifier(n_estimators = 200)
    model.fit(X_train, y_train)

    y_predicted = model.predict(X_test)
    accuracies_decomposed[i,-1] = model.score(X_test, y_test)
    cm_i = confusion_matrix(y_test, y_predicted)
    cm = cm + cm_i
    accuracies_decomposed[i,:-1] = np.diag(cm_i/cm_i.sum(axis=1)[:,None])


# np.save(f'{path_save}CM_SUPRASUB={NSUPRASUB}_k={NCLUSTERS}.npy',cm)
# result = pd.DataFrame(accuracies_decomposed,columns=tasks+['TOTAL']) # I save the recalls and the total accuracy
# result.to_csv(f'{path_save}recalls_SUPRASUB={NSUPRASUB}_k={NCLUSTERS}.csv',index=False)

tasks_label = ['REST1', 'SOCIAL', 'RELATIONAL', 'MOTOR', 'LANGUAGE', 'WM', 'GAMBLING', 'EMOTION']

tsne = TSNE(n_components=2, verbose=1,perplexity=3)
z = tsne.fit_transform(X)

df_cm = pd.DataFrame(cm, index = tasks_label,
                  columns = tasks_label)


cmap = plt.get_cmap("tab10")
colordict_scatter = {}
for n in range(8):
    colordict_scatter[str(n)] = cmap(n-1)

colordict_scatter['0']='gray'

plt.figure(figsize=(6,2.8))
plt.subplots_adjust(wspace=0.2)

plt.subplot(121)

sn.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g',cbar=False)

plt.xticks()
ticks = plt.xticks()[0]
labels = plt.xticks()[1]
for i, (tick, label) in enumerate(zip(ticks, labels)):
    plt.text(tick, -0.4, label.get_text(), color=colordict_scatter[str(i)], horizontalalignment='left',rotation=90)
plt.xticks([])
plt.yticks()
ticks = plt.yticks()[0]
labels = plt.yticks()[1]
for i, (tick, label) in enumerate(zip(ticks, labels)):
    plt.text(-0.3, tick, label.get_text(), color=colordict_scatter[str(i)], horizontalalignment='right',rotation=0)
plt.yticks([])


plt.subplot(122)
for n in range(8):
    plt.plot(z[y==n,0],z[y==n,1],'o',color=colordict_scatter[str(n)],ms=3,label=tasks_label[n],alpha=1)

plt.xlabel('tSNE 1')
plt.ylabel('tSNE 2')
plt.xticks([])
plt.yticks([])
plt.show()