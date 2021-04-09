import torch



def weighted_MSE(source, target, weights=torch.tensor([1.])):
    squared_diff = (source - target)**2
    w = weights / weights.sum()
    return (squared_diff * w).flatten().mean()


def distance_loss(source, target, ndim=3):
    """
    Calculates the squared normalized distance between two 3D vectors
    Best to be used for directional vectors on a unit sphere
    """
    a = target.view(-1, ndim)
    b = source.view(-1, ndim)
    # inner_product = (a * b).sum(dim=1)
    a_norm = torch.norm(a, dim=1).view(-1, 1)
    b_norm = torch.norm(b, dim=1).view(-1, 1)

    a = a/a_norm
    b = b/b_norm

    diff = (a-b).pow(2)
    dists = (diff.sum(dim=1))
    return dists.mean()


def angular_loss_3d(source, target, ndim=3):
    """
    Calcuates the squared angular loss of two vectors
    Might be prone to nan errors
    """
    a = target.view(-1, ndim)
    b = source.view(-1, ndim)
    inner_product = (a * b).sum(dim=1)
    a_norm = torch.norm(a, dim=1)
    b_norm = torch.norm(b, dim=1)
    cos = inner_product / (a_norm * b_norm)
    angle = torch.acos(cos)
    return angle.pow(2).mean()


def arc_loss(source, target, ndim=3):
    """
    Alternative calcuation of angular loss
    Might be less nan error prone
    """
    dists = distance_loss(source, target, ndim)
    angles = 2*torch.asin(dists.pow(.2)/2)
    # w = weights / weights.sum()
    # return (angles.pow(2)*w).mean()
    return angles.pow(2).mean()
