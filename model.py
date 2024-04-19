import torch
import pyro
import pyro.distributions as dist


def model(Y, s0):
  """
  Probability model for entailments trees.

  Un-vectorized implementation as different trees 
  and different iterations (i, t) 
  will have different dimension operations.

  NOTATION:
  y:          input dataset of entailment trees.
  s0:         input set of premises to construct a tree off of.
  n:          number of samples in dataset.
  l (lambda): rate parameter for k_i.
  w:          parameter for linear model.
  c:          available premises.
  c_tilde:    retrieved premises.

  """
  n = len(Y)
  l = pyro.sample("lambda", dist.Exponential(torch.tensor([1])))
  w = pyro.sample("W", dist.Normal(torch.tensor([1])))
  s2 = pyro.sample("sigmasquare", dist.Exponential(torch.tensor([1])))

  for i in range(n):
    y = Y[i]

    # Fetch available, retrieved, and generated premises
    # for each iteration.
    c = y.to_embedding(y.available_premises())
    c_tilde = y.to_embedding(y.retrieved_premises())
    c_star = y.to_embedding([[id] for id in y.generated_premises()])

    # Sample for number of retrieved premises
    k = torch.tensor([c_tilde_t.size(0) for c_tilde_t in c_tilde])
    with pyro.plate("k", len(k)):
      pyro.sample("k_t", dist.Poisson(l), obs=k - 1)

    # Sample for retrieved premises
