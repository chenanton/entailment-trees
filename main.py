import torch
import matplotlib.pyplot as plt
import pickle
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from model import model
from guide import guide

if __name__ == "__main__":
  # Load dataset
  dataset = "dev"
  processed_dataset_fp = f'data/processed/{dataset}.pkl'
  with open(processed_dataset_fp, 'rb') as file:
    trees = pickle.load(file)

  # Test example
  s_test = [trees[8].id_to_sentence['sent1'], trees[6].id_to_sentence['sent1']]
  c_test = torch.stack(
      [trees[8].id_to_embedding['sent1'], trees[6].id_to_embedding['sent1']],
      dim=0)
  print(s_test)

  pyro.clear_param_store()

  # These should be reset each training loop.
  adam = pyro.optim.Adam({"lr": 0.02})  # Consider decreasing learning rate.
  elbo = pyro.infer.Trace_ELBO()
  svi = pyro.infer.SVI(model, guide, adam, elbo)

  losses = []
  for step in range(500):  # Consider running for more steps.
    loss = svi.step(trees, c_test)
    losses.append(loss)
    if step % 50 == 0:
      print("ELBO loss: {}".format(loss))

  plt.figure(figsize=(5, 2))
  plt.plot(losses)
  plt.xlabel("SVI step")
  plt.ylabel("ELBO loss")
