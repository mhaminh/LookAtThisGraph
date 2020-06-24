import pickle
import pkg_resources
import numpy as np
from importlib.resources import path
from lookatthisgraph import resources

def get_doms_by_strings(fname=None):
    if fname is None:
        with path(resources,
                  'GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.pkl') as p:
            fname = p
    gcd_file = pickle.load(open(fname, 'rb'), encoding='latin1')

    dom_positions = gcd_file['geo']
    return dom_positions


def get_dom_positions():
    dom_positions = get_doms_by_strings()

    dom_pos_flat = []
    for string in dom_positions:
        for xyz in string:
            dom_pos_flat.append(xyz)
    dom_pos_flat = np.asarray(dom_pos_flat)
    return dom_pos_flat
