import torch
import pyro
import pyro.distributions as dist

from model import sample_model_params, compute_theta
from embed import d


def guide(Y, c_test):
  """
  Guide function to implement SVI.

  :param Y: trees (UNUSED).
  :param c_test: m_test x d tensor of initial premises.

  :return c_tilde_test: retrieved premise.
  :return c_star_test: generated premise.
  """
  p, wr, wg, s2 = sample_model_params(d)

  # Sample new premise
  m_test = c_test.shape[0]
  k_test = 2  # TEMPORARY

  # Retrieve samples
  theta_test = compute_theta([c_test], wr, torch.tensor([k_test]))[0].squeeze()
  j_test = []
  for j in range(k_test):
    # Sample then update theta
    idx = pyro.sample(f"j_test,{j}^(0)", dist.Categorical(theta_test))
    j_test.append(idx)
    theta_test[idx] = 0.0

  c_tilde_test = c_test[j_test, :]

  # Sample new premise
  mean_c_test_tilde = torch.mean(c_tilde_test, dim=0, keepdim=True)
  mu_t = torch.matmul(wg, mean_c_test_tilde.T).squeeze()
  Sigma = torch.eye(d) * s2
  c_star_test = pyro.sample(
      f"c_star_test",
      dist.MultivariateNormal(mu_t, Sigma),
  )

  return c_star_test, c_tilde_test
