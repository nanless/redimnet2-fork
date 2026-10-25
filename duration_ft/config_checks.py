"""Dependency-free numeric guards shared by the maintained training entries."""
import math


def _number(value, name, minimum=0, strict=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('finite numeric value required: '+name)
    if value < minimum or (strict and value == minimum):
        raise ValueError('numeric range invalid: '+name)


def validate_numeric_hyperparameters(config):
    """Reject bool counts and NaN optimizer/head inputs before device setup."""
    if not isinstance(config, dict):
        raise ValueError('training configuration must be a dictionary')
    for name in ('seed', 'head_seed'):
        if type(config.get(name)) is not int or config[name] < 0:
            raise ValueError('nonnegative integer required: '+name)
    if type(config.get('amp')) is not bool:
        raise ValueError('amp must be boolean')
    opt=config.get('optimizer');head=config.get('head')
    if not isinstance(opt, dict) or not isinstance(head, dict):
        raise ValueError('optimizer and head must be dictionaries')
    _number(opt.get('lr'), 'optimizer.lr', strict=True)
    for name in ('momentum', 'weight_decay', 'dampening'):
        if name in opt:_number(opt[name], 'optimizer.'+name)
    if 'nesterov' in opt and type(opt['nesterov']) is not bool:
        raise ValueError('optimizer.nesterov must be boolean')
    if opt.get('nesterov') and (opt.get('momentum',0)<=0 or opt.get('dampening',0)!=0):
        raise ValueError('Nesterov requires positive momentum and zero dampening')
    for name in ('scale','t'):
        if name in head:_number(head[name], 'head.'+name, strict=True)
    for name in ('margin','lanbuda'):
        if name in head:_number(head[name], 'head.'+name)
    if 'dynamic_margin' in head and type(head['dynamic_margin']) is not bool:
        raise ValueError('head.dynamic_margin must be boolean')
