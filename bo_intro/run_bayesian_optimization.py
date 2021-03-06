import torch
import numpy as np
import sys
import warnings
import bo_intro.datasets
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.optim.fit import fit_gpytorch_scipy
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement, qExpectedImprovement, qUpperConfidenceBound
from botorch.acquisition.utils import is_nonnegative
from botorch.sampling.samplers import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.settings import suppress_botorch_warnings


def load_dataset(config):
    if config['dataset'].lower() == 'sine':
        return bo_intro.datasets.Sine(config)
    elif config['dataset'].lower() == 'branin':
        return bo_intro.datasets.Branin(config)
    else:
        raise ValueError('dataset {} does not exist'.format(config['dataset']))

def load_acquisition_function(config, model, dataset, y, seed):
    if config['acquisition_function'].lower() == 'ucb':
        beta = config.setdefault('beta', 3)
        return UpperConfidenceBound(model, beta=beta)
    elif config['acquisition_function'].lower() == 'ei':
        return ExpectedImprovement(model, best_f=y.max())
    config.setdefault('mc_samples', 500)
    sampler = SobolQMCNormalSampler(num_samples=config['mc_samples'], seed=seed)
    if config['acquisition_function'].lower() == 'qnei':
        qNEI = qNoisyExpectedImprovement(
            model=model, 
            X_baseline=dataset.x,
            sampler=sampler, 
        )
        return qNEI
    elif config['acquisition_function'].lower() == 'qucb':
        qUCB = qUpperConfidenceBound(
            model=model, 
            beta=config['beta'],
            sampler=sampler, 
        )
        return qUCB
    elif config['acquisition_function'].lower() == 'qei':
        qEI = qExpectedImprovement(
            model=model, 
            best_f = y.max(),
            sampler=sampler, 
        )
        return qEI
    else:
        raise ValueError('acquisition function {} does not exists'.format(config['acquisition_function']))

def fit_gp(config, x, y, state_dict):
    """
    fit gaussian process to x and y using the gp_optimizer and previous state in state_dict
    """
    gp = SingleTaskGP(x, y)
    if state_dict:
        gp.load_state_dict(state_dict)
    mll = ExactMarginalLogLikelihood(likelihood=gp.likelihood, model=gp)
    fit_gpytorch_model(mll, optimizer=fit_gpytorch_scipy)
    return gp

def bo_iteration(config, dataset, state_dict, iteration, seed):
    """
    fit a Gaussian process to dataset.x and a normalized version of dataset.y
    choose new points new_x to sample from by maximizing the acquisition function
    in case of a simulation, query the dataset in new_x
    in case of a real experiment, write the requested parameters to a file and exit
    """
    std = dataset.y.std() if dataset.y.std() > 0 else 1.0
    y = (dataset.y - dataset.y.mean())/std

    gp = fit_gp(config, dataset.x, y, state_dict)
      
    # keep track of whether acquisition function is unintentionally performing random search
    # because of bad initial conditions
    random_search = False

    if config['acquisition_function'].lower() == 'random':
        new_x = torch.rand(config['batch_size'], dataset.dim, dtype=torch.double)

    else:
        acquisition_function = load_acquisition_function(config, gp, dataset, y, seed)
        
        new_x, acq_values = optimize_acqf(acq_function=acquisition_function,
                                    bounds=torch.tensor([[0.0] * dataset.dim, [1.0] * dataset.dim],  dtype=torch.double),
                                    q=config['batch_size'],
                                    num_restarts=config['num_restarts'],
                                    raw_samples=config['raw_samples'],
                                    options = {'batch_limit':50, 'seed':seed})    
    new_x = new_x.detach()
    new_y = dataset.query(new_x)
    state_dict = gp.state_dict()
    return new_x, new_y, state_dict

def run_bo_experiment(config, seed=0, print_progress=False):
    """
    Run a Bayesian optimization experiment

    Args:
        config: a dictionary containing the configuration
        seed: int, random seed for random number generators
        print_progress: bool, indicates whether progress per iteration should be printed

    Returns:
        x, y values of queried points (numpy array, numpy array)
    """
    torch.manual_seed(seed)
    suppress_botorch_warnings(False)
    warnings.filterwarnings('ignore', 'Unknown solver options: seed')

    config.setdefault('acquisition_function', 'EI')
    config.setdefault('noise', 0)
    config.setdefault('num_restarts', 5)
    config.setdefault('raw_samples', 500)
    config.setdefault('batch_size', 1)

    dataset = load_dataset(config)
    state_dict = {}
    random_search = []

    for iteration in range(config['iterations']):
        new_x, new_y, state_dict = bo_iteration(config, dataset, state_dict, iteration, seed)
        dataset.add(new_x, new_y)
        if print_progress:
            print('Optimum found upto iteration {}: {}'.format(iteration, dataset.y.max().numpy()))
    
    return dataset.rescale(dataset.x).numpy(), dataset.y.numpy()

if __name__ == "__main__":
    config = {'iterations':200, 'initial_observations':1, 'dataset':'sine',}
    results = run_bo_experiment(config, print_progress=True)
    print(results)
