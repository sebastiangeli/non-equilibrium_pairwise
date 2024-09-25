# -*- coding: utf-8 -*-

"""
Code for "Non-equilibrium whole-brain dynamics arise from pairwise interactions"
Sebastian M. Geli, Christopher W. Lynn, Morten L. Kringelbach, Gustavo Deco and
Yonatan Sanz Perl.


This code is used to compute the joint transition matrices for decomposing the
entropy production. It samples random subsets of brain regions and transforms
the signals into binary time series. Then, it ensures that the time series satisfy
the multipartite assumption (which states that not two elements can change at once).
Finally, it returns the transition matrix of subsets of random regions for each task.


For queries or issues, please contact Sebastian Geli at sebastianmanuel.geli@upf.edu
"""

import numpy as np
from tqdm import tqdm
import random
from scipy import sparse



def intermediate_states(state1,state2):

    """

    Generates intermediate states between two states that differ
    in more than one element, randomly picking one element to change at once.

        Parameters
        ----------
        state1 : ndarray
            Initial state of the system. Shape: 1 x dims
        state2 : ndarray
            Final state of the system. Shape: 1 x dims

        Returns
        -------
        intermediates : ndarray
            Aditional intermediate states between state1 and state2 such that
            the transitions follow the multipartite assumption.

        Examples
        --------
         >>> state1 = np.array([0, 0, 0, 0])
         >>> state2 = np.array([1, 1, 1, 0])

         >>> intermediate_states(state1,state2)
         array([[1, 0, 0, 0],
               [1, 1, 0, 0]])


    """

    nodes2flip = np.random.permutation(np.where(state1-state2 != 0)[0])
    intermediates = [state1]

    for i in range(len(nodes2flip)-1):
        node = nodes2flip[i]
        new_state = intermediates[i].copy()
        new_state[node] = not(new_state[node])
        intermediates.append(new_state)

    intermediates = np.array(intermediates[1:])

    return intermediates


def correct_multipartite(states):

    """

        Corrects a time series of binary states to make it satisfy the multipartite
        assumption

        Parameters
        ----------
        states : ndarray
            The time series of the system. Shape: timepoints x dims

        Returns
        -------
        states_corrected : ndarray
            Corrected time series of the system in which a single random element was
            assigned to change at once.

        Examples
        --------
         >>> states = np.array([
                 [0, 0, 0, 0],
                 [1, 1, 1, 0],
                 [1, 1, 0, 1],
                 [1, 1, 1, 1],
                 [0, 0, 1, 1]])

         >>> correct_multipartite(states)
         array([[0, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 1, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 1],
                [1, 1, 0, 1],
                [1, 1, 1, 1],
                [1, 0, 1, 1],
                [0, 0, 1, 1]])


    """

    states = states.astype(np.int32)

    places = np.where(np.sum(np.abs(states[1:]-states[:-1]),axis=1)>1)[0] + 1
    corrections = [intermediate_states(states[places[i]-1],states[places[i]]) for i in range(len(places))]
    corrected_places = places + np.append(0,np.cumsum([len(i) for i in corrections[:-1]]))


    states_corrected = states.copy()

    for i in range(len(corrected_places)):
        states_corrected = np.insert(states_corrected,corrected_places[i].astype(np.int32),corrections[i],axis=0)

    return states_corrected


def transition_matrix(corrected_states):

    """

        Generates the transition matrix T, where T(i,j) is the number of times
        element j flipped when in state i.

        Parameters
        ----------
        corrected_states : ndarray
            The multipartite binary time series of the system (timepoints x dims)

        Returns
        -------
        T : ndarray
            Transition matrix of shape (2^dims, dims + 1). Each entry (i, j)
            indicates the number of times element j changed when the system was
            in state i. The values in the last column j indicate the number of
            times the system remained in state i.


    """
    # Number of dimensions in the time series
    dim = np.shape(corrected_states)[1]


    # Label each unique state with a global number (e.g [0,0,1] -> 1)
    global_states = np.dot(corrected_states,2**(np.arange(dim)[::-1]))[:-1]

    # Compute the absolute difference between consecutive timepoints for each dimension
    diffs = np.abs(corrected_states[1:]-corrected_states[:-1])

    # Determine which element (dimension) changed at each time step
    element_changing = np.dot(diffs,np.arange(1,dim+1))

    # Set the index to (dim + 1) for cases where no change occurred
    element_changing[element_changing == 0] = dim + 1
    element_changing -= 1       # Align with zero-based indexing

    # Create an array of [state, changed_element] pairs
    transitions = np.array([global_states, element_changing]).T

    # Get unique transitions and their counts
    transitions_unique = np.unique(transitions,axis=0,return_counts=True)

    # Construct the transition matrix
    T = sparse.csr_matrix((transitions_unique[1], (transitions_unique[0][:,0],transitions_unique[0][:,1])), shape=(2**dim, dim+1), dtype=np.int32).toarray()

    return T

def create_toy_data(num_regions, time_points, NSUB):

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

        Returns
        -------
        standardized_sine_waves : ndarray (time_points * NSUB, num_regions)

  """

  time = np.linspace(0, 8 * np.pi * NSUB, NSUB * time_points)
  sine_waves = np.zeros((NSUB * time_points, num_regions))
  for i in range(num_regions):
    phase = random.uniform(0, 2 * np.pi)
    sine_waves[:, i] = np.sin(time + phase) + np.random.normal(0,0.2, NSUB * time_points)

  # Calculate the mean and standard deviation of each row
  mean = np.mean(sine_waves, axis=0, keepdims=True)
  std = np.std(sine_waves, axis=0, keepdims=True)

  # Standardize each row
  standardized_sine_waves = (sine_waves - mean) / std

  return standardized_sine_waves

tasks = ['REST1', 'SOCIAL', 'RELATIONAL', 'MOTOR', 'LANGUAGE', 'WM', 'GAMBLING', 'EMOTION']

# Define file paths for loading and saving data
# path_save = ''
# path_load = ''

treshold = 1           # Treshold to binarize data (sigmas)
NSUB = 2               # Number of subjects
NREPS = 200            # Number of random samples
AREAS_SAMPLE = 5       # Number of regions to sample
NAREAS = 62            # Total number of regions in the parcellation
TPOINTS = 176          # Number of timepoints in a measurement

tasks = ['REST1','SOCIAL','RELATIONAL', 'MOTOR', 'LANGUAGE', 'WM', 'GAMBLING', 'EMOTION']

tmatrix_task_reps = {}


selected_areas = np.zeros((NREPS,AREAS_SAMPLE))

for nt, task in enumerate(tasks):

  tmatrix_task_reps[task] = np.zeros((NREPS, 2**AREAS_SAMPLE, AREAS_SAMPLE + 1))

  # timeseries = load_data(task)
  timeseries = create_toy_data(NAREAS, TPOINTS, NSUB)

  timeseries_point = timeseries > treshold
  states = np.array(timeseries_point,dtype=np.int32);
  del timeseries, timeseries_point

  for rep in tqdm(range(NREPS)):

    random_indices = np.random.choice(states.shape[1], size=AREAS_SAMPLE, replace=False)
    selected_areas[rep,:] = random_indices

    corrected_states = [correct_multipartite(states[i*TPOINTS:i+1*TPOINTS,random_indices]) for i in range(NSUB)]
    Ts = np.array([transition_matrix(i) for i in corrected_states])
    tmatrix_task_reps[task][rep, :, :] = np.sum(Ts,axis=0)

    # io.savemat(f'{path_save}source_sample_rep={rep}_condition={task}_Tmatrix.mat', {'T' : T_full.astype(np.double)})

  # np.save(f'{path_save}selected_areas',selected_areas.astype('int'))