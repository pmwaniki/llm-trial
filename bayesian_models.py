from pyro.distributions import constraints


import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.reparam import LocScaleReparam
from pyro.infer import  Predictive, NUTS, MCMC, HMC


def logistic(x, y=None,device='cpu',prior_scale=1.0):
    P = x.shape[1]
    beta_plate = pyro.plate("beta_plate", P)
    bias = pyro.sample('bias', dist.Normal(torch.tensor(0., device=device),
                                           torch.tensor(2., device=device)).expand([1]).to_event(1))
    # betas=[]
    loc_beta = pyro.param("loc_beta", torch.zeros(P,device=device))
    scale_beta = pyro.param("scale_beta", torch.ones(P,device=device)*prior_scale, constraint=constraints.positive)

    with poutine.reparam(config={"betas": LocScaleReparam()}):
        with beta_plate:
            beta = pyro.sample(f'betas', dist.Normal(loc_beta, scale_beta))
    logits = bias + x @ beta
    mean = pyro.deterministic('mean', value=logits.reshape(-1, 1).sigmoid())
    with pyro.plate('data', x.shape[0]):
        pyro.sample("obs", dist.Bernoulli(mean).to_event(1), obs=y)
    return mean

def logistic_glmm(x,clinician,center, y=None,device='cpu',prior_scale=1.0):
    P = x.shape[1]
    n_clinicians=int(clinician.max())+1
    n_centers=int(center.max())+1
    beta_plate = pyro.plate("beta_plate", P)
    bias = pyro.sample('bias', dist.Normal(torch.tensor(0., device=device),
                                           torch.tensor(2., device=device)).expand([1]).to_event(1))
    # betas=[]
    loc_beta = pyro.param("loc_beta", torch.zeros(P,device=device))
    scale_beta = pyro.param("scale_beta", torch.ones(P,device=device)*prior_scale, constraint=constraints.positive)

    with poutine.reparam(config={"betas": LocScaleReparam()}):
        with beta_plate:
            beta = pyro.sample(f'betas', dist.Normal(loc_beta, scale_beta))

    # random effects for clinician
    sigma_clin = pyro.sample("sigma_clin", dist.HalfNormal(1.0))
    with pyro.plate("clinicians", n_clinicians):
        CLIN = pyro.sample("b_clin", dist.Normal(0.0, sigma_clin))

    # random effects for center
    sigma_center = pyro.sample("sigma_cen", dist.HalfNormal(1.0))
    with pyro.plate("centers", n_centers):
        CEN = pyro.sample("b_cen", dist.Normal(0.0, sigma_center))



    logits = bias + x @ beta + CLIN[clinician] + CEN[center]
    mean = pyro.deterministic('mean', value=logits.reshape(-1, 1).sigmoid())
    with pyro.plate('data', x.shape[0]):
        pyro.sample("obs", dist.Bernoulli(mean).to_event(1), obs=y)
    return mean


def logistic_glmm2(x,clinician,center,arm, y=None,device='cpu',prior_scale=1.0):
    P = x.shape[1]
    n_clinicians=int(clinician.max())+1
    n_centers=int(center.max())+1
    beta_plate = pyro.plate("beta_plate", P)
    bias = pyro.sample('bias', dist.Normal(torch.tensor(0., device=device),
                                           torch.tensor(2., device=device)).expand([1]).to_event(1))
    # betas=[]
    loc_beta = pyro.param("loc_beta", torch.zeros(P,device=device))
    scale_beta = pyro.param("scale_beta", torch.ones(P,device=device)*prior_scale, constraint=constraints.positive)

    with poutine.reparam(config={"betas": LocScaleReparam()}):
        with beta_plate:
            beta = pyro.sample(f'betas', dist.Normal(loc_beta, scale_beta))

    # random effects for clinician
    sigma_clin = pyro.sample("sigma_clin", dist.HalfNormal(1.0).expand([2,]))
    with pyro.plate("clinicians", n_clinicians):
        CLIN = pyro.sample("b_clin", dist.Normal(0.0, sigma_clin[arm]))

    # random effects for center
    sigma_center = pyro.sample("sigma_cen", dist.HalfNormal(1.0))
    with pyro.plate("centers", n_centers):
        CEN = pyro.sample("b_cen", dist.Normal(0.0, sigma_center))



    logits = bias + x @ beta + CLIN[clinician] + CEN[center]
    mean = pyro.deterministic('mean', value=logits.reshape(-1, 1).sigmoid())
    with pyro.plate('data', x.shape[0]):
        pyro.sample("obs", dist.Bernoulli(mean).to_event(1), obs=y)
    return mean


def logistic_glmm_meta(arm,clinician,center, y=None,device='cpu'):
    # P = arm.shape[1]
    n_clinicians=int(clinician.max())+1
    n_centers=int(center.max())+1
   
    bias = pyro.sample('bias', dist.Normal(torch.tensor(0., device=device),
                                           torch.tensor(2., device=device)).expand([1]).to_event(1))
    # overall effect of treatment
    overall_treat_effect = pyro.sample('overall_treat_effect', dist.Normal(torch.tensor(0., device=device),
                                           torch.tensor(2., device=device)))
    # random effects for center centered around overall treatment effect (normaly distributed with mean overall treat effect)
    sigma_center = pyro.sample("sigma_cen", dist.HalfNormal(1.0))
    # with loc scale reparam
    with poutine.reparam(config={"b_cen": LocScaleReparam()}):
        with pyro.plate("centers", n_centers):
            CEN = pyro.sample("b_cen", dist.Normal(overall_treat_effect, sigma_center))

    # random effects for clinician
    sigma_clin = pyro.sample("sigma_clin", dist.HalfNormal(1.0))
    with pyro.plate("clinicians", n_clinicians):
        CLIN = pyro.sample("b_clin", dist.Normal(0.0, sigma_clin))

   
    logits = bias  + CEN[center] * arm.reshape(-1) + CLIN[clinician]
    mean = pyro.deterministic('mean', value=logits.reshape(-1, 1).sigmoid())
    with pyro.plate('data', arm.shape[0]):
        pyro.sample("obs", dist.Bernoulli(mean).to_event(1), obs=y)
    return mean



def run_inference(model,x_data,y_data,weights=None,mcmc_burnin=100,mcmc_samples=100,full_mass=False,max_tree_depth=10,num_chains=1):
    ppg_kernel = NUTS(model, full_mass=full_mass, max_tree_depth=max_tree_depth)
    mcmc = MCMC(ppg_kernel, num_samples=mcmc_samples, warmup_steps=mcmc_burnin, num_chains=num_chains)
    mcmc.run(x_data, y_data.reshape(-1, 1),weights)
    return mcmc

def predict(model,x_data_test,posterior_samples,site='mean'):
    predictive = Predictive(model, posterior_samples)
    samples = predictive(x_data_test, None)
    return samples[site]