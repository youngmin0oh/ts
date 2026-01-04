
import torch
import numpy as np

def move_axis_to_end(x, dim):
    #return np.rollaxis(array, axis, start=array.ndim)
    axis = list(range(len(x.shape)))
    del axis[dim]
    axis.append(dim)
    return x.permute(axis)

def argsort_indices(a, axis=-1):
    """Like argsort, but returns an index suitable for sorting the
    the original array even if that array is multidimensional
    """
    ind = list(np.ix_(*[np.arange(d) for d in a.shape]))
    ind[axis] = a.argsort(axis)
    return tuple(ind)

def _crps_ensemble_vectorized(observations, forecasts, weights=None):
    """
    An alternative but simpler implementation of CRPS for testing purposes

    This implementation is based on the identity:

    .. math::
        CRPS(F, x) = E_F|X - x| - 1/2 * E_F|X - X'|

    where X and X' denote independent random variables drawn from the forecast
    distribution F, and E_F denotes the expectation value under F.

    Hence it has runtime O(n^2) instead of O(n log(n)) where n is the number of
    ensemble members.
    """
    if observations.ndim == forecasts.ndim - 1:
        # sum over the last axis
        assert observations.shape == forecasts.shape[:-1]
        observations = observations.unsqueeze(-1)
        if weights is None:
            results = torch.abs(forecasts - observations)
            score = torch.nanmean(results, -1)
            # forecasts_diff = (forecasts.unsqueeze(-1) - forecasts.unsqueeze(-2))
            # score += -0.5 * torch.nanmean(torch.abs(forecasts_diff), dim=(-2, -1))
            forecasts_diff = (forecasts.unsqueeze(-1) - forecasts.unsqueeze(-2)).abs()
            score += -0.5 * torch.nanmean(forecasts_diff, dim=(-2, -1))
            return score.abs().mean()

        weights = torch.where(~torch.isnan(forecasts), weights, torch.nan)
        weights = weights / torch.nanmean(weights, dim=-1, keepdims=True)
        results = weights * torch.abs(forecasts - observations)
        score = torch.nanmean(results, -1)
        # insert new axes along last and second to last forecast dimensions so
        # forecasts_diff expands with the array broadcasting
        forecasts_diff = (forecasts.unsqueeze(-1) - forecasts.unsqueeze(-2))
        weights_matrix = (weights.unsqueeze(-1) * weights.unsqueeze(-2))
        weight_forecasts = weights_matrix * torch.abs(forecasts_diff)
        score += -0.5 * torch.nanmean(weight_forecasts, dim=(-2, -1))
        return score.abs().mean()
    elif observations.ndim == forecasts.ndim:
        # there is no 'realization' axis to sum over (this is a deterministic
        # forecast)
        return torch.abs(observations - forecasts).mean()


def crps_ensemble(observations, forecasts, weights=None, issorted=False,
                  axis=-1):
    """
    Calculate the continuous ranked probability score (CRPS) for a set of
    explicit forecast realizations.
    """
    if axis != -1:
        forecasts = move_axis_to_end(forecasts, axis)

    if weights is not None:
        weights = move_axis_to_end(weights, axis)
        if weights.shape != forecasts.shape:
            raise ValueError('forecasts and weights must have the same shape')

    if observations.shape not in [forecasts.shape, forecasts.shape[:-1]]:
        raise ValueError('observations and forecasts must have matching '
                         'shapes or matching shapes except along `axis=%s`'
                         % axis)

    if observations.shape == forecasts.shape:
        if weights is not None:
            raise ValueError('cannot supply weights unless you also supply '
                             'an ensemble forecast')
        return abs(observations - forecasts)

    if not issorted:
        if weights is None:
            forecasts, _ = forecasts.sort(axis=-1)
        else:
            idx = argsort_indices(forecasts, axis=-1)
            forecasts = forecasts[idx]
            weights = weights[idx]

    return _crps_ensemble_vectorized(observations, forecasts, weights)
