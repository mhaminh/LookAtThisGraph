import os
import logging
import h5py
import numpy as np
from tqdm.auto import tqdm
from lookatthisgraph.utils.datautils import filter_dict
from lookatthisgraph.utils.datautils import reconvert_zenith
from copy import deepcopy
from joblib import Parallel, delayed


def convert_to_pisa(outfile, sim_info, dataset_nue, dataset_numu, dataset_nutau, with_weighted_aeff=True, convert_results=True):
    with h5py.File(outfile, 'w') as f:
        for dset, flavor in zip([dataset_nue, dataset_numu, dataset_nutau], ['nue', 'numu', 'nutau']):
            nu_cc, nu_nc, nubar_cc, nubar_nc = get_keys(dset, sim_info, flavor, with_weighted_aeff, convert_results)
            for key, item in nu_cc.items():
                f.create_dataset(flavor + '_cc/' + key, data=item)
            for key, item in nu_nc.items():
                f.create_dataset(flavor + '_nc/' + key, data=item)
            for key, item in nubar_cc.items():
                f.create_dataset(flavor + 'bar_cc/' + key, data=item)
            for key, item in nubar_nc.items():
                f.create_dataset(flavor + 'bar_nc/' + key, data=item)

    logging.info('Dataset successfully converted and saved')


def get_keys(dataset, sim_info, flavor, with_weighted_aeff, convert_results):
    keys = load_keys(dataset.files[0])
    if len(dataset.files) > 1:
        logging.warning('Warning: Multiple input paths found in dataset, using first to extract simulation keys')
    keys = filter_dict(keys, dataset.non_empty_mask)
    dataset_filter = dataset.filter
    keys = filter_dict(keys, dataset_filter)

    if with_weighted_aeff:
        keys['weighted_aeff'] = calculate_weighted_aeff(
            keys['OneWeight'],
            dataset.filtered_truths['energy'],
            keys['MCInIcePrimary.pdg_encoding'],
            sim_info,
            flavor,
            dataset.n_files
        )

    keys['L7_reconstructed_coszen'] = dataset.results['zenith']
    keys['L7_reconstructed_total_energy'] = dataset.results['energy']
    keys['L7_PIDClassifier_ProbTrack'] = dataset.results['pid']

    if convert_results:
        keys['L7_reconstructed_coszen'] = np.cos(keys['L7_reconstructed_coszen'])
        keys['L7_PIDClassifier_ProbTrack'] = keys['L7_PIDClassifier_ProbTrack']

    # Split in CC / NC
    CC_mask = keys['I3MCWeightDict.InteractionType'] == 1
#     CC_keys = {key: item[CC_mask] for key, item in keys.items()}
#     NC_keys = {key: item[~CC_mask] for key, item in keys.items()}
    # Split in nu / nubar
    nu_mask = keys['MCInIcePrimary.pdg_encoding'] > 0
    nu_cc_mask = nu_mask & CC_mask
    nu_nc_mask = nu_mask & ~CC_mask
    nubar_cc_mask = ~nu_mask & CC_mask
    nubar_nc_mask = ~nu_mask & ~CC_mask
    nu_cc_keys = {key: item[nu_cc_mask] for key, item in keys.items()}
    nu_nc_keys = {key: item[nu_nc_mask] for key, item in keys.items()}
    nubar_cc_keys = {key: item[nubar_cc_mask] for key, item in keys.items()}
    nubar_nc_keys = {key: item[nubar_nc_mask] for key, item in keys.items()}

    # return CC_keys, NC_keys
    return nu_cc_keys, nu_nc_keys, nubar_cc_keys, nubar_nc_keys



def calculate_energy_factor(energies, sim_info, flavor):
    e_lower = sim_info[flavor]['lower_e_limits']
    e_upper = sim_info[flavor]['upper_e_limits']
    epf = sim_info[flavor]['events_per_file']
    total_epf = np.sum(epf)

    e_counter = 0
    energy_factors = np.empty(len(energies))
    energies = 10**np.asarray(energies)
    for low, high, events_per_file in zip(e_lower, e_upper, epf):
        mask = np.where(np.logical_and(low <= energies, energies < high))[0]
        energy_factors[mask] = events_per_file / total_epf
        e_counter += len(mask)
    if e_counter != len(energy_factors):
        raise IndexError('Error in assigning energy factors; %d events expected, %d received' % (len(energy_factors), e_counter))

    return energy_factors


def calculate_nu_nubar_factor(pdg_codes, sim_info):
    r_nu_nubar = sim_info['ratio_nu']
    nu_nubar_factor = np.empty(len(pdg_codes))
    mask = pdg_codes > 0
    nu_nubar_factor[mask] = r_nu_nubar
    nu_nubar_factor[~mask] = 1 - r_nu_nubar
    return nu_nubar_factor


def calculate_weighted_aeff(one_weight, energies, pdg_codes, sim_info, flavor, n_files):
    energy_factors = calculate_energy_factor(energies, sim_info, flavor)
    nu_nubar_factors = calculate_nu_nubar_factor(pdg_codes, sim_info)
    events_per_file = np.sum(sim_info[flavor]['events_per_file'])
    weighted_aeff = one_weight / (energy_factors * nu_nubar_factors * events_per_file * n_files) * 1e-4
    return weighted_aeff


def load_keys(location):
    load_key = lambda key: np.load(os.path.join(location, key + '/data.npy'))
    mcprimary = load_key('MCInIcePrimary')
    weightdict = load_key('I3MCWeightDict')
    keys = {
        'MCInIcePrimary.energy': mcprimary['energy'],
        'MCInIcePrimary.dir.coszen': np.cos(mcprimary['dir']['zenith']),
        'MCInIcePrimary.pdg_encoding': mcprimary['pdg_encoding'],
        'I3MCWeightDict.InteractionType': weightdict['InteractionType'],
        'OneWeight': weightdict['OneWeight'],
        'MCDeepCoreStartingEvent': load_key('MCDeepCoreStartingEvent'),
        'L7_MuonClassifier_ProbNu': load_key('L7_MuonClassifier_ProbNu'),
        'L7_CoincidentMuon_bool': load_key('L7_CoincidentMuon_bool'),
        'L7_oscNext_bool': load_key('L7_oscNext_bool').astype(bool)
    }
    return keys
