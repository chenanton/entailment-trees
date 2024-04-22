import torch
import pyro
import pyro.distributions as dist

from embed import d


def model(Y, c_test):
  """
  Probability model for entailments trees.

  Inputs:
  Y:       input dataset of entailment trees.
  c_test:  input set of premises to sample a new premise from.

  Parameters:
  p:       p; binom parameter for k_i.
  wr:      parameters for linear premise retrieval.
  wg:      parameters for linear premise generation.
  c:       available premises.
  s2:      sigma squared.

  Transformed:
  c:       available premises.
  c_tilde: retrieved premises.
  c_star:  generated premises.
  n:       number of samples in dataset.

  Returns:
  c_tilde_test: retrieved premise.
  c_star_test: generated premise.

  Future work: vectorize implementation with pyro.plate.
  """
  n = len(Y)

  # Model parameter priors
  p, wr, wg, s2 = sample_model_params(d)

  for i in range(n):
    y = Y[i]

    # 1. Process data

    # Fetch available premises per iteration.
    c = y.to_embedding(y.available_premises())
    m = torch.tensor([c_t.size(0) for c_t in c])

    # Fetch retrieved premises per iteration.
    c_tilde = y.to_embedding(y.retrieved_premises())
    c_tilde_indices = y.get_indices_of_retrieved_premises()
    k = torch.tensor([c_tilde_t.size(0) for c_tilde_t in c_tilde])

    # Fetch generated premises per iteration.
    c_star = y.to_embedding([[id] for id in y.generated_premises()])

    # 2. Sample data

    # Sample for number of retrieved premises
    with pyro.plate(f"k_{i}", len(m)):
      pyro.sample(
          f"k_{i}^t",
          dist.Binomial(m - 1, p),
          obs=k - 1,
      )

    # Sample for retrieved premises
    theta = compute_theta(c, wr, k)

    # Construct theta for distinct sampling
    theta = [theta[t].repeat(1, k[t]) for t in range(len(theta))]
    for t in range(len(k)):  # iteration
      for j in range(k[t] - 1):  # sample number
        theta[t][c_tilde_indices[t][j], j + 1:] = 0

    for t in range(len(k)):  # iteration
      for j in range(k[t]):  # sample number
        # Sample then update theta
        pyro.sample(
            f"j_{i},{j}^({t})",
            dist.Categorical(theta[t][:, j]),
            obs=c_tilde_indices[t][j],
        )

    # Sample for generated premises
    Sigma = torch.eye(d) * s2
    for t in range(len(c_tilde)):  # iteration
      # Compute normal model parameters
      mean_c_tilde_t = torch.mean(c_tilde[t], dim=0, keepdim=True)
      mu_t = torch.matmul(wg, mean_c_tilde_t.T).squeeze()
      c_star_t = c_star[t].squeeze()

      pyro.sample(f"c_star,{i}^({t})",
                  dist.MultivariateNormal(mu_t, Sigma),
                  obs=c_star_t)

  # 3. Sample new premise
  m_test = c_test.shape[0]

  # Number of samples to consider
  # k_test = pyro.sample(
  #     "k_test",
  #     dist.Binomial(torch.tensor([m_test]), torch.tensor([p])),
  # ) + 1
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
  c_star_test = pyro.sample(
      f"c_star_test",
      dist.MultivariateNormal(mu_t, Sigma),
  )

  return c_star_test, c_tilde_test


def sample_model_params(d):
  """
  Samples model parameters from corresponding priors.

  :param d: embedding dimension.

  :return p: binomial parameter.
  :return wr: normal parameter.
  :return wg: normal parameter.
  :return s2: normal parameter.
  """
  p = pyro.sample(
      "p",
      dist.Uniform(torch.tensor([0.0]), torch.tensor([1.0])),
  )

  with pyro.plate("wr", d * d):
    wr = pyro.sample("wr_entries",
                     dist.Normal(torch.tensor([0.0]), torch.tensor([1.0])))
  wr = wr.reshape(d, d)

  with pyro.plate("wg", d * d):
    wg = pyro.sample("wg_entries",
                     dist.Normal(torch.tensor([0.0]), torch.tensor([1.0])))
  wg = wg.reshape(d, d)

  s2 = pyro.sample(
      "sigmasquare",
      dist.Exponential(torch.tensor([1.0])),
  )

  return p, wr, wg, s2


def compute_theta(c, wr, k):
  """
  Construct evolved categorical distributions
  of retrieval probability for available premises.
  
  :param c: available premises, m x d tensor.
  :param wr: model parameters for retrieval, d x d tensor.
  :param k: number of distinct premises to sample.
  :param c_tilde_indices: indices of retrieved premises w.r.t. 
  the available premises.

  :return theta: list of m_t x k_t tensors.
  """
  # Construct distribution on retrieved premises
  psis = [psi(c_t, wr) for c_t in c]
  psi_exp = [torch.exp(p) for p in psis]
  psi_exp_sum = [torch.sum(p_exp) for p_exp in psi_exp]

  theta = [psi_exp[t] / psi_exp_sum[t] for t in range(len(psis))]
  return theta


def psi(c, wr):
  """
  Scoring function to measure similarity of each 
  available premise to mean available premise.

  :param c: available premises, m x d tensor.
  :param wr: model parameters for retrieval, d x d tensor.
  """
  mean_c = torch.mean(c, dim=0, keepdim=True)
  cTwr = torch.matmul(c, wr)
  scores = torch.matmul(cTwr, mean_c.T)

  return scores
