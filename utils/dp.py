import torch

def flatten_grads(grads: dict):
    """
    grads: dict
        key: parameter name
        value: per-example gradients from one batch
    Flatten the dict of per-example gradients into a list (length = batch size) of flat gradients.
    """
    _all_keys, _all_grads = [], []
    for _k in grads:
        _all_keys.append(_k)
        _all_grads.append(grads[_k])
    _bsz = _all_grads[0].size(0)
    return _all_keys, [torch.cat([torch.flatten(_gs[_i]) for _gs in _all_grads]) for _i in range(_bsz)]

def clip_grads(grads: dict, clip_norm):
    """
    grads: dict
        key: parameter name
        value: grad
    Clip each grad in grads to clip norm.
    Return a dict of grad for each parameter
    """
    _new_grads = {}
    for _k in grads:
        _norm = torch.linalg.vector_norm(grads[_k], ord=2)
        _new_grads[_k] = torch.div(
            grads[_k],
            torch.max(
                torch.tensor(1, device=grads[_k].device),
                torch.div(_norm, clip_norm)
            )
        )
    return _new_grads

def clip_grad(grad: torch.tensor, clip_norm):
    """
    Clip each flattened grad to clip norm.
    """
    _norm = torch.linalg.vector_norm(grad, ord=2)
    return torch.div(
        grad,
        torch.max(
            torch.tensor(1, device=grad.device),
            torch.div(_norm, clip_norm)
        )
    )

def get_shape(grads: dict):
    """
    grads: dict
        key: parameter name
        value: per-example gradients from one batch
    Return a dict of shapes (discard batch dimension) given a dict of tensors.
    """
    shapes = {_k: _v.shape[1:] for _k, _v in grads.items()}
    return shapes

def reconstruct_shape(grads, keys, shapes):
    """
    Reconstruct the original shapes of the tensors list.
    """
    _run = 0
    _recon_grad = {}
    for _k in keys:
        _n_elms = torch.prod(torch.tensor(shapes[_k])).item()
        _recon_grad[_k] = grads[_run:_run+_n_elms].reshape(shapes[_k])
        _run += _n_elms
    return _recon_grad

def add_noise(grads, batch_size, noise_multiplier, sensitivity):
    """
    grads: dict
        key: parameter name
        value: accumulated grad
    Add noise to grads.
    """
    for _k in grads:
        _sz = grads[_k].size()
        _mean = torch.zeros(_sz)
        _std = torch.mul(torch.ones(_sz), (sensitivity * noise_multiplier))
        _noise = torch.normal(mean=_mean, std=_std).to(grads[_k].device)
        grads[_k]= torch.div(torch.add(grads[_k], _noise), batch_size)
    return grads
