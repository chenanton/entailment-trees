import torch
import pickle
import pyro
from pyro.infer import MCMC, NUTS

from model import model

if __name__ == "__main__":
  # Load dataset
  dataset = "dev"
  processed_dataset_fp = f'data/processed/{dataset}.pkl'
  with open(processed_dataset_fp, 'rb') as file:
    trees = pickle.load(file)

  trees = trees[:10]

  # Run model
  nuts_kernel = NUTS(model)
  mcmc = MCMC(nuts_kernel, num_samples=100, warmup_steps=20)
  with pyro.validation_enabled():
    res = mcmc.run(trees)
  samples = mcmc.get_samples()

  print(samples)
