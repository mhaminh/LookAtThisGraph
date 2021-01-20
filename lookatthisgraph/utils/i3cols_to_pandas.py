import numpy as np
import pandas as pd
import os
import pickle
from importlib.resources import path
import lookatthisgraph.resources

def cart2polar(x, y, z):
    phi = np.arctan2(y, x)
    phi[phi < 0] += 2 * np.pi
    theta = np.arctan2(np.sqrt(x**2+y**2), z)
    return phi, theta


def polar2cart(phi, theta):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z


def key_to_xyz(df, gcd_path=None):
    """
    Converts OM key to xyz position

    Parameters:
    ----------
    df:
        Dataframe with pulses
    """
    if gcd_path is None:
        with path(lookatthisgraph.resources, 'geo_array.npy') as p:
            gcd_path = p

    # gcd = np.load('../lookatthisgraph/resources/geo_array.npy')
    gcd = np.load(gcd_path)
    codes = df[['string', 'om']] - 1 # IC naming starts at 1
    codes = codes.to_numpy()
    xyz = gcd[codes[:,0], codes[:,1]]
    return pd.DataFrame(xyz, columns=['x_om', 'y_om', 'z_om'])


def key_to_xyz_upgrade(df, gcd_path=None):
    # TODO: make gcd loading more flexible
    if gcd_path is None:
        with path(lookatthisgraph.resources, 'icu_gcd.p') as p:
            gcd_path = p
    gcd = pickle.load(open(gcd_path, 'rb'), encoding='latin1')
    codes = df[['string', 'om', 'pmt']].copy()
    codes[['string', 'om']] -= 1 # IC naming starts at 1
    codes = codes.to_numpy()
    get_keys = lambda key: gcd[key][codes[:,0], codes[:,1], codes[:,2]]
    xyz = get_keys('geo')
    dirs = get_keys('direction')
    omtype = get_keys('omtype').astype(str)
    cart_dirs = np.array(polar2cart(*dirs.T)).T
    df_xyz = pd.DataFrame(xyz, columns=['x_om', 'y_om', 'z_om'])
    df_dirs = pd.DataFrame(dirs, columns=['phi_om', 'theta_om'])
    df_cartdirs = pd.DataFrame(cart_dirs, columns=['xdir_om', 'ydir_om', 'zdir_om'])
    df_omtypes = pd.DataFrame(omtype, columns=['omtype'])

    # One-hot-encode OM types
    omdict = {"is_" + tlabel: (omtype == tlabel).astype(float) for tlabel in np.unique(omtype)}
    df_om_onehot = pd.DataFrame(omdict)

    return pd.concat([df_xyz, df_dirs, df_cartdirs, df_omtypes, df_om_onehot], axis=1)


def load_pulses(indir, pulse_frame='SRTTWOfflinePulsesDC'):
    """
    Load pulse frame from folder created by i3cols
    Parameters:
    ----------
    indir: str
        Path to i3cols output
    pulse_frame: str
        Name of pulse frame to convert
    """
    data = np.load(os.path.join(indir, pulse_frame,  'data.npy'))
    index = np.load(os.path.join(indir, pulse_frame, 'index.npy'))
    pulse_info = pd.DataFrame(data[:]['pulse'])
    key_info = pd.DataFrame(data[:]['key'])
    pulses = pd.concat([pulse_info, key_info], axis=1)
    lc = pd.Series(pulses['flags'].values & 1, name='is_LC')
    atwd = pd.Series((pulses['flags'].values & 2)/2, name='has_ATWD')
    pulses = pd.concat([pulses, lc, atwd], axis=1)
    n_pulses = index[:]['stop'] - index[:]['start']
    pulse_idx = np.repeat(np.arange(len(n_pulses)), n_pulses.astype(int))
    pulse_idx = pd.DataFrame({'event': pulse_idx})
    events = pd.concat([pulses, pulse_idx], axis=1)
    return events


def get_pulses(indir, upgrade=False):
    # TODO: Test if np.repeat works correctly if 0 times repeated
    """
    Load pulse information from input directory and provide xyz for OMs

    Parameters:
    ----------
    indir: str
        Input directory from i3cols output
    upgrade: bool
        Assume Upgrade architecture, returns also PMT direction and OM type
    """

    if not upgrade:
        pulse_frame = 'SRTTWOfflinePulsesDC'
    else:
        pulse_frame = 'SplitInIcePulsesTWSRT'

    events = load_pulses(indir, pulse_frame)

    if not upgrade:
        xyz = key_to_xyz(events)
    else:
        xyz = key_to_xyz_upgrade(events)
    return pd.concat([events, xyz], axis=1)


def get_energies(mctree, mctree_idx):
    n_events = len(mctree_idx)
    track_energies = np.zeros(n_events)
    track_lengths =  np.zeros(n_events) 
    invisible_energies = np.zeros(n_events)
    neutrino_energies = np.zeros(n_events)
    
    for i, (start, stop) in enumerate(mctree_idx):
        event = mctree[start:stop]
        pdg = event['particle']['pdg_encoding']
        energies = event['particle']['energy']
        lengths = event['particle']['length']
        track_mask = np.abs(pdg) == 13
        # Get maximum energy track-like particle, otherwise 0
        try:
            track_idx = np.argmax(energies[track_mask])
            track_energies[i] = energies[track_mask][track_idx]
            track_lengths[i] = lengths[track_mask][track_idx]
        except ValueError:
            pass

        invisible_mask = (np.abs(pdg) == 12) | (np.abs(pdg) == 14) | (np.abs(pdg) == 16)
        invisible_mask[0] = False # exclude primary particle
        invisible_energies[i] = np.sum(energies[invisible_mask])

        neutrino_energies[i] = event[0]['particle']['energy']

    cascade_energies = neutrino_energies - (track_energies + invisible_energies)
    df = pd.DataFrame({'shower_energy': cascade_energies, 'track_energy': track_energies, 'track_length': track_lengths})
    return df


def load_primary_information(mcprimary):
    # Have to split this in two parts because of hierarchical structure of input data
    # maybe there's a smarter way of doing this
    labels = ['pdg_encoding', 'time', 'energy']
    superlabels = ['pos', 'dir']
    df_0 = pd.DataFrame(mcprimary[labels])
    df_1 = pd.concat([pd.DataFrame(mcprimary[l]) for l in superlabels], axis=1)
    return pd.concat([df_0, df_1], axis=1)


def get_truths(indir, eps=1e-3):
    """
    Get all truth information necessary

    Parameters:
    ----------
    indir: str
        Input directory from i3cols output
    eps: float, optional
        Value to offset for 0-valued track energies
        Otherwise NaN with np.log
    """
    mctree_data = np.load(os.path.join(indir, 'I3MCTree', 'data.npy'))
    mctree_idx = np.load(os.path.join(indir, 'I3MCTree', 'index.npy'))

    primary_idx = mctree_idx['start']
    mcprimary = mctree_data[primary_idx]['particle']

    primaries = load_primary_information(mcprimary)
    energies = get_energies(mctree_data, mctree_idx)

    if len(primaries) != len(energies):
        raise IndexError('Indices of primary information and MCTree do not align')

    df = pd.concat([primaries, energies], axis=1)
    event_idx = pd.DataFrame(np.arange(len(df)), columns=['event'])

    # Get PID
    pid = pd.Series((df['track_energy'].values > 0).astype(float), name='PID')
    df = pd.concat([df, pid], axis=1)

    # Convert energies to log
    energies = df[['energy', 'shower_energy', 'track_energy']].copy()
    log_energies = np.log10(energies.values + eps)
    log_energy_names = ['log10('+ name + ')'  for name in energies.columns]
    log_energy_df = pd.DataFrame(log_energies, columns=log_energy_names)

    # cos(zenith), hacky
    cosz_df = pd.Series(np.cos(df['zenith']).values, name='cos(zenith)')
    # Convert azimuth and zenith values to cartesian
    polar_directions = np.array(polar2cart(df['azimuth'].values, df['zenith'].values))
    polar_df = pd.DataFrame(polar_directions.T, columns=['x_dir', 'y_dir', 'z_dir'])

    # Get PID information (0 = cascade, 1 = track)
    pid = pd.DataFrame((df['track_energy'].values > 0).astype(int), columns=['pid'])

    # Get interaction type (1 = NC, 2 = CC); not available for Upgrade data yet
    try:
        weight_dict = np.load(os.path.join(indir, 'I3MCWeightDict', 'data.npy'))
        interaction_df = pd.DataFrame(weight_dict['InteractionType'], columns=['interaction_type'])
        df = pd.concat([df, interaction_df], axis=1)
    except FileNotFoundError:
        pass

    return pd.concat([df, log_energy_df, polar_df, pid, event_idx, cosz_df], axis=1)


def dataframe_to_event_list(df, labels=None):
    """
    Convert pulse dataframe to list of np.array objects
    For fast handling

    Parameters:
    ----------
    df: pd.DataFrame
        DataFrame with pulse information
    labels: list, optional:
        list with columns of interest
    """
    if not df['event'].is_monotonic:
        raise IndexError('Dataframe is not sorted by pulse')

    if labels == None:
        labels = df.columns

    event_idx = df['event'].values
    df = df[labels]
    values = df.values
    names = df.columns
    ukeys, index = np.unique(event_idx, True)
    if 'event' in names:
        col_events = np.where(names=='event')
        values = np.delete(values, col_events, axis=1) # Remove event column
    arrays = np.split(values, index[1:])
    return arrays