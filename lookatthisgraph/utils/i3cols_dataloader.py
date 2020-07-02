'''Module to read in i3cols format data
i3cols is available here: https://github.com/jllanfranchi/i3cols
'''
import os
import pkg_resources
import numpy as np
from tqdm.auto import tqdm


def get_energies(mcprimary, mctree, mctree_idx, dtype=np.float32):
    '''Get energies per event'''

    neutrino_energy = mcprimary['energy']
    track_energy = np.zeros_like(neutrino_energy, dtype=dtype)
    invisible_energy = np.zeros_like(neutrino_energy, dtype=dtype)

    for i in range(len(mctree_idx)):
        this_idx = mctree_idx[i]
        this_mctree = mctree[this_idx['start'] : this_idx['stop']]
        pdg = this_mctree['particle']['pdg_encoding']
        en = this_mctree['particle']['energy']

        muon_mask = np.abs(pdg) == 13
        if np.any(muon_mask):
            track_energy[i] = np.max(en[muon_mask])

        invisible_mask = (np.abs(pdg) == 12) | (np.abs(pdg) == 14) | (np.abs(pdg) == 16)
        # exclude primary:
        invisible_mask[0] = False
        if np.any(invisible_mask):
            # we'll make the bold assumptions that none of the neutrinos re-interact ;)
            invisible_energy[i] = np.sum(en[invisible_mask])

    cascade_energy = neutrino_energy - track_energy - invisible_energy
    return neutrino_energy, track_energy, cascade_energy


def get_params(labels, mcprimary, mctree, mctree_idx, dtype=np.float32):
    '''construct params array

    retruns:
    params : array shape (len(mcprimary), len(labels))
    '''

    neutrino_energy, track_energy, cascade_energy = get_energies(mcprimary, mctree, mctree_idx, dtype=dtype)

    params = np.empty(mcprimary.shape + (len(labels), ), dtype=dtype)

    for i, label in enumerate(labels):
        if label == 'x': params[:, i] = mcprimary['pos']['x']
        elif label == 'y': params[:, i] = mcprimary['pos']['y']
        elif label == 'z': params[:, i] = mcprimary['pos']['z']
        elif label == 'time': params[:, i] = mcprimary['time']
        elif label == 'azimuth': params[:, i] = mcprimary['dir']['azimuth']
        elif label == 'zenith': params[:, i] = mcprimary['dir']['zenith']
        elif label == 'neutrino_energy': params[:, i] = neutrino_energy
        elif label == 'energy': params[:, i] = track_energy + cascade_energy
        elif label == 'cascade_energy': params[:, i] = cascade_energy
        elif label == 'track_energy': params[:, i] = track_energy

    return params



def load_charges(dir='/home/iwsatlas1/peller/work/oscNext/level7_v01.04/140000_i3cols',
                 labels=['x', 'y', 'z', 'time', 'azimuth','zenith', 'cascade_energy', 'track_energy'],
                 dtype=np.float32):
    """
    Create training data for chargenet

    Returns:
    --------
    total_charge : ndarray
        shape (N_events, 2)
    params : ndarray
        shape (N_events, len(labels))
    labels
    """

    hits_idx = np.load(os.path.join(dir, 'SRTTWOfflinePulsesDC/index.npy'))
    hits = np.load(os.path.join(dir, 'SRTTWOfflinePulsesDC/data.npy'))
    mctree_idx = np.load(os.path.join(dir, 'I3MCTree/index.npy'))
    mctree = np.load(os.path.join(dir, 'I3MCTree/data.npy'))
    mcprimary = np.load(os.path.join(dir, 'MCInIcePrimary/data.npy'))

    # Get charge per event
    total_charge = np.zeros((hits_idx.shape[0], 2), dtype=dtype)
    for i in range(len(hits_idx)):
        this_idx = hits_idx[i]
        this_hits = hits[this_idx['start'] : this_idx['stop']]

        total_charge[i][0] = np.sum(this_hits['pulse']['charge'])
        total_charge[i][1] = len(np.unique(this_hits['key']))

    params = get_params(labels, mcprimary, mctree, mctree_idx)
    return total_charge, params, labels


def load_strings(dir='/home/iwsatlas1/peller/work/oscNext/level7_v01.04/140000_i3cols',
                 labels=['x', 'y', 'z', 'time', 'azimuth','zenith', 'cascade_energy', 'track_energy'],
                 # geo=pkg_resources.resource_filename('freedom', 'resources/geo_array.npy'),
                 geo = './resources/geo_array.npy',
                 dtype=np.float32):
    """
    Create training data for hitnet

    Returns:
    --------
    string_charges : ndarray
        shape (N_events*86, 5)
    repeated_params : ndarray
        shape (N_events*86, len(labels))
    labels
    """

    hits_idx = np.load(os.path.join(dir, 'SRTTWOfflinePulsesDC/index.npy'))
    hits = np.load(os.path.join(dir, 'SRTTWOfflinePulsesDC/data.npy'))
    mctree_idx = np.load(os.path.join(dir, 'I3MCTree/index.npy'))
    mctree = np.load(os.path.join(dir, 'I3MCTree/data.npy'))
    mcprimary = np.load(os.path.join(dir, 'MCInIcePrimary/data.npy'))

    geo = np.load(geo)

    # constrcut strings array
    # shape N x 86 x (x, y, min(z), q, string_idx)

    strings = np.zeros(hits_idx.shape + (86, 5,), dtype=dtype)
    strings[:, :, 0:3] = geo[np.newaxis, :, -1]
    strings[:, :, 4] = np.arange(86)[np.newaxis, :]

    # Get charge per event and string
    for i in range(len(hits_idx)):
        this_idx = hits_idx[i]
        this_hits = hits[this_idx['start'] : this_idx['stop']]
        for hit in this_hits:
            s_idx = hit['key']['string'] - 1
            strings[i, s_idx, 3] += hit['pulse']['charge']

    strings = strings.reshape(-1, 5)

    params = get_params(labels, mcprimary, mctree, mctree_idx)

    repeated_params = np.repeat(params, repeats=86, axis=0)

    return strings, repeated_params, labels


def load_hits(dir='/home/iwsatlas1/peller/work/oscNext/level7_v01.04/140000_i3cols',
              labels=['x', 'y', 'z', 'time', 'azimuth','zenith', 'cascade_energy', 'track_energy'],
              # geo=pkg_resources.resource_filename('freedom', 'resources/geo_array.npy'),
              geo = './resources/geo_array.npy',
              dtype=np.float32):
    """
    Create training data for hitnet

    Returns:
    --------
    single_hits : ndarray
        shape (N_hits, 9)
    repeated_params : ndarray
        shape (N_hits, len(labels))
    labels
    """

    hits_idx = np.load(os.path.join(dir, 'SRTTWOfflinePulsesDC/index.npy'))
    hits = np.load(os.path.join(dir, 'SRTTWOfflinePulsesDC/data.npy'))
    mctree_idx = np.load(os.path.join(dir, 'I3MCTree/index.npy'))
    mctree = np.load(os.path.join(dir, 'I3MCTree/data.npy'))
    mcprimary = np.load(os.path.join(dir, 'MCInIcePrimary/data.npy'))

    geo = np.load(geo)

    # constrcut hits array

    # shape N x (x, y, z, t, q, ...)
    single_hits = np.empty(hits.shape + (9,), dtype=dtype)
    string_idx = hits['key']['string'] - 1
    om_idx = hits['key']['om'] - 1

    single_hits[:, 0:3] = geo[string_idx, om_idx]
    single_hits[:, 3] = hits['pulse']['time']
    single_hits[:, 4] = hits['pulse']['charge']
    single_hits[:, 5] = hits['pulse']['flags'] & 1 # is LC or not?
    single_hits[:, 6] = (hits['pulse']['flags'] & 2) / 2 # has ATWD or not?
    single_hits[:, 7] = string_idx
    single_hits[:, 8] = om_idx

    params = get_params(labels, mcprimary, mctree, mctree_idx)

    # repeats = (hits_idx['stop'] - hits_idx['start']).astype(np.int64)
    # repeated_params = np.repeat(params, repeats=repeats, axis=0)


    return single_hits, params, labels


def load_events(dir='/home/iwsatlas1/peller/work/oscNext/level7_v01.04/140000_i3cols',
              labels=['x', 'y', 'z', 'time', 'azimuth','zenith', 'cascade_energy', 'track_energy', 'neutrino_energy'],
              geo='./resources/geo_array.npy',
              recos = {},
              dtype=np.float32):
    """
    Create event=by=event data for hit and charge net

    Returns:
    --------
    list of:
        single_hits : ndarray
            shape (N_hits, 9)
        total_charge : float
        params : ndarray
            shape (len(labels))
    labels
    """

    hits_idx = np.load(os.path.join(dir, 'SRTTWOfflinePulsesDC/index.npy'))

    single_hits, params, labels = load_hits(dir=dir, labels=labels, geo=geo, dtype=dtype)

    reco_params = {}
    for r,f in recos.items():
        reco = np.load(os.path.join(dir, f, 'data.npy'))
        reco_params[r] = np.zeros_like(params)
        for i, label in enumerate(labels):
            if label == 'x': reco_params[r][:, i] = reco['pos']['x']
            elif label == 'y': reco_params[r][:, i] = reco['pos']['y']
            elif label == 'z': reco_params[r][:, i] = reco['pos']['z']
            elif label == 'time': reco_params[r][:, i] = reco['time']
            elif label == 'azimuth': reco_params[r][:, i] = reco['dir']['azimuth']
            elif label == 'zenith': reco_params[r][:, i] = reco['dir']['zenith']
            elif label == 'energy': reco_params[r][:, i] = reco['energy']

        # for retro unfortunately these are in different keys...
        if 'track_energy' in labels and f.strip('/').endswith('__neutrino'):
            reco_track = np.load(os.path.join(dir, f.replace('__neutrino','__track'), 'data.npy'))
            idx = labels.index('track_energy')
            reco_params[r][:, idx] = reco_track['energy']

        if 'cascade_energy' in labels and f.strip('/').endswith('__neutrino'):
            reco_track = np.load(os.path.join(dir, f.replace('__neutrino','__cascade'), 'data.npy'))
            idx = labels.index('cascade_energy')
            reco_params[r][:, idx] = reco_track['energy']



    events = []

    for i, _ in enumerate(params):
        event = {}
        # event['total_charge'] = total_charge[i]
        event['hits'] = single_hits[hits_idx[i]['start'] : hits_idx[i]['stop']]
        event['params'] = params[i]
        # event['strings'] = string_charges[i]
        for r in recos.keys():
            event[r] = reco_params[r][i]
        events.append(event)
    return events, labels
