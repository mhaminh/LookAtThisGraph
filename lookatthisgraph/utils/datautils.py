import logging
import copy
import numpy as np
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import torch
from torch_geometric.data import Data
from lookatthisgraph.utils.dataloader import get_pulses
from lookatthisgraph.utils.icecubeutils import get_dom_positions


def flatten(superlist):
    """ Flatten a list of lists
    Args:
        superlist (list of lists) List of lists

    Returns:
        Flattened list
    """
    return [x for sublist in superlist for x in sublist]


def filter_list(inlist, indices):
    """Select elements of list at indices
    Args:
        inlist (list): list to filter
        indices (list of int): indices to select
    Returns:
        Filtered list
    """
    if len(indices) > len(inlist):
        logging.warning('List of indices bigger than list to filter')
    return [inlist[i] for i in indices]


def get_file(file, labels):
    """ Try to get_pulses, pass else
    Args:
        file (str): Location of hdf5 file
        labels (list of str): Labels to load

    Returns:
        Pulses and truths
    """
    try:
        infile = get_pulses(file, labels=labels)
    except OSError:
        infile = None
    return infile


def get_input_data(fnames, labels, n_jobs=1):
    """ Load from multiple files
    Args:
        fnames (list of str): File locations
        labels (list of str): List of labels

    Returns:
        Pulses, Truths
    """
    file_input = Parallel(n_jobs=n_jobs)(delayed(get_file)(f, labels=labels)
            for f in tqdm(fnames, desc='Loading files'))
    file_input = [f for f in file_input if f is not None]
    x, y = map(list, zip(*file_input))
    x = flatten(x)
    y = flatten(y)
    return x, y


def filter_pulses(pulses, truths, max_pulses):
    """Filter events by number of pulses
    Args:
        pulses (list): List of raw pulses
        truths (list): List of raw truths
    Returns:
        Filtered pulses, filtered truths, indices
    """
    pulse_mask = [i for i, event in enumerate(pulses) if 0 < len(event) < max_pulses]
    x = filter_list(pulses, pulse_mask)
    y = filter_list(truths, pulse_mask)
    return x, y, pulse_mask


def time_to_position(pulses, col_t):
    """Convert time to spatial position
    Args:
        pulses (list): List of raw pulses
        col_t (int): Column of time in ns
    Returns:
        Position in meters
    """
    c = 299792458 # mps

    xt = np.copy(pulses)
    time_to_pos = [event[:,col_t]*1e-9*c for event in xt]
    time_to_pos = [time.reshape(-1, 1) for time in time_to_pos]
    return time_to_pos


def get_dom_xyz(pulses, col_DOM):
    """Convert DOM id to xyz position
    Args:
        pulses (list): Raw pulses
        col_DOM (int): Column of DOM id
    Returns:
        DOM positions in xyz
    """
    dom_positions = get_dom_positions()
    hit_doms = [dom_positions[event[:,col_DOM].astype(int)] for event in pulses]
    return hit_doms


def calc_distance_information(event):
    """ Calculate matrix of pulse distances
    Args:
	event: ctxyz information of pulses
    Returns:
	Matrix of distances
    """
    npulses = len(event)
    idx = np.asarray(list(np.ndindex((npulses, npulses))))
    calc_dist = lambda i, j: np.linalg.norm(event[i] - event[j]) \
                            * np.sign(event[j, 0] - event[i, 0])

    dist_matrix = np.vectorize(calc_dist)(idx[:, 0], idx[:, 1])
    idx = np.swapaxes(idx, 0, 1)
    return idx, dist_matrix


def get_edge_information(ctxyz, n_jobs=1, distance_scaler=1e-3):
    """Get edge indices and edge weights for all events
    Args:
        ctxyz: Time converted to distance and xyz of DOM position of pulses
        n_jobs: Number of jobs to initialize for calculation
        distance_scaler: Factor to mulitply d**2 value with
    Returns:
        Edge indices, edge weights
    """
    distcontainer = Parallel(n_jobs=n_jobs)(delayed(calc_distance_information)(event)
                                            for event in tqdm(ctxyz, desc="Calculating distance matrices"))
    edge_indices, dists = map(list, zip(*distcontainer))
    edge_weights = [1 / (1 + distance_scaler* np.abs(m)**2) * np.sign(m)
                    for m in tqdm(dists, desc="Calculating edge weights")]
    return edge_indices, edge_weights


def one_hot_encode(event, n_categories):
    """One-hot encode integer quanitity
    Args:
        event (int): feature quantity between {0..n_categories}
        n_categories (int): number of unique features
    Returns:
        One-hot encoded list of arrays of size (event x n_categories)

    """
    encoded = np.zeros((len(event), n_categories))
    for i, cat in enumerate(event):
        encoded[i, int(cat)] = 1
    return encoded


def one_hot_encode_omtypes(pulses, col_omtype):
    """One-hot encode OM type (IceCube/DeepCore, PDOM, mDOM, D-Egg)
    Args:
        pulses (list): Raw pulses
        col_omtype (int): Column at which OM type is stored
    Returns:
        list with arrays of size (n_pulses x n_omtypes): encoded OM types
    """
    omtypes = np.array([ev[:, col_omtype] for ev in pulses])
    omtypes_encoded = [one_hot_encode(event, 4)
                       for event in tqdm(omtypes, desc="Encoding OM types")]
    return omtypes_encoded


def one_hot_encode_pmts(pulses, col_omtype, col_pmt):
    """One-hot encode PMTs based on OM type and direction
    Args:
        pulses (list): Raw pulses
        col_omtype (int): Column at which OM type is stored
        col_pmt (int): Column at which PMT number is stored
    Returns:
        list with arrays of size (n_pulses x n_pmttypes): encoded PMTs
    """

    om_codes = []
    for event in tqdm(pulses):
        omtypes = event[:, col_omtype]
        pmts = event[:, col_pmt]

        oms = copy.deepcopy(pmts)
        oms[np.where(omtypes == 0)[0]] = 0 # ic
        oms[np.where(omtypes == 1)[0]] = 1 # pdom
        oms[np.where(omtypes == 2)[0]] += 2 # mdom
        oms[np.where(omtypes == 3)[0]] += 26 # degg

        om_codes.append(oms)
    om_matrix = [one_hot_encode(oms, 28) for oms in tqdm(om_codes)]
    return om_matrix


def build_data_list(normalized_features, edge_indices, edge_weights, y_transformed):
    data_list = []
    for features, i, w, truth in tqdm(zip(normalized_features, edge_indices, edge_weights, y_transformed),
                                      total=len(y_transformed)):
        dd = Data(x=torch.tensor(features, dtype=torch.float),
                  y=torch.tensor(truth, dtype=torch.float),
                  edge_index=torch.tensor(i, dtype=torch.long),
                  edge_attr=torch.tensor(w, dtype=torch.float),
                 )
        data_list.append(dd)
    return data_list
