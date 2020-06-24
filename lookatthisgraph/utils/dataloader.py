import numpy as np
from tqdm.auto import tqdm
import h5py

def get_pulses(
    fname,
    truth_i3key='MCInIcePrimary',
    pulses_i3key='SRTTWOfflinePulsesDC',
    features=['time', 'charge'],
    labels=['zenith', 'azimuth'],
    N_events=None,
    dtype=np.float32,
    ):


    h = h5py.File(fname, 'r')

    truth = np.array(h[truth_i3key])
    pulses = np.array(h[pulses_i3key])
    try:
        retro_zenith = np.array(h['L7_reconstructed_zenith']['value'])
        retro_energy= np.array(h['L7_reconstructed_total_energy']['value'])
        retro_x = np.array(h['L7_reconstructed_vertex_x']['value'])
        retro_y = np.array(h['L7_reconstructed_vertex_y']['value'])
        retro_z = np.array(h['L7_reconstructed_vertex_z']['value'])
    except:
        pass

    if N_events is None:
        N_events = truth.shape[0]

    x = []
    y = []

    data_idx = 0
    bincount = np.bincount(pulses['Event'])

    # fill array
    # for data_idx, event_idx in tqdm(enumerate(zip(non_empty, bincount[non_empty])), total=len(non_empty)):

    for event_idx, num_pulses in tqdm(enumerate(bincount), total=len(bincount)):
        if num_pulses == 0:
            continue

        p = pulses[pulses['Event'] == event_idx]
        hitlist = []
        for hit in p:

            hit_idx = hit['vector_index']
            string_idx = hit['string'] - 1
            dom_idx = hit['om'] - 1
            channel_idx = 60 * string_idx + dom_idx

            feature_vector = [channel_idx]
            for i, feature in enumerate(features):
                if (feature == 'time'):
                    time_axis = i+1
                feature_vector.append(hit[feature])

            hitlist.append(feature_vector)

        # Sort pulses by time for each event
        if ('time' in features):
            hitlist = np.asarray(hitlist)
            time_idx = np.argsort(hitlist[:, time_axis])
            hitlist = hitlist[time_idx]

        x.append(hitlist)

        l = truth[truth['Event']==event_idx]
        feature = np.asarray([l[label] for label in labels], dtype=dtype)

        y.append(feature.flatten())

        data_idx += 1
    try:
        return x, y, retro_zenith, retro_energy, retro_x, retro_y, retro_z
    except:
        return x, y

