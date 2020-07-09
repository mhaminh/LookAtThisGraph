import logging
import copy
import numpy as np
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from copy import deepcopy
import torch
from torch_geometric.data import Data
from lookatthisgraph.utils.icecubeutils import get_dom_positions


def flatten(superlist):
    """ Flatten a list of lists
    Args:
        superlist (list of lists) List of lists

    Returns:
        Flattened list
    """
    return [x for sublist in superlist for x in sublist]


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


def torch_to_numpy(x):
        return np.asarray(x.cpu().detach())


def process_charges(event, charge_col=4):
    new_event = deepcopy(event)
    log_charge = np.log10(new_event[:, charge_col])
    new_event[:, charge_col] = log_charge
    return new_event


def build_data_list(normalized_features, y_transformed, include_charge=True):
    data_list = []
    if include_charge:
        feature_cols = [0, 1, 2, 3, 4]
    else:
        feature_cols = [0, 1, 2, 3]
    for features, truth in tqdm(zip(normalized_features, y_transformed),
                                total=len(y_transformed),
                                desc='Filling data list'):
        dd = Data(
            x=torch.tensor(features[:, feature_cols], dtype=torch.float),
            y=torch.tensor(truth, dtype=torch.float),
            )
        data_list.append(dd)
    return data_list


def evaluate(model, loader, device):
    pred = []
    truth = []
    with torch.no_grad():
        model.eval()
        for batch in tqdm(loader):
            data = batch.to(device)
            pred.append(torch_to_numpy(model(data)))
            truth.append(torch_to_numpy(data.y))

    pred = np.concatenate(pred)
    truth = np.concatenate(truth)
    return pred, truth


def reconvert_zenith(arr):
    return np.arctan2(arr[:, 0], arr[:, 1])


def filter_dict(dictionary, filter_mask):
    try:
        dictionary = {key: item[filter_mask] for key, item in dictionary.items()}
    except:
        dictionary = {key: [item[i] for i in filter_mask] for key, item in dictionary.items()}

    return dictionary
