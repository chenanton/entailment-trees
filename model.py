import torch
import pyro
import pyro.distributions as dist


def model(Y):
  """
  Probability model for entailments trees.

  Inputs:
  Y:       input dataset of entailment trees.
  c0:      input set of premises to construct a tree off of (latent).

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

  Future work: vectorize implementation with pyro.plate.
  """
  n = len(Y)
  d = 384

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
    pyro.sample(
        f"k_{i}",
        dist.Binomial(torch.tensor(m) - 1,
                      torch.ones_like(k) * p),
        obs=k - 1,
    )

    # Sample for retrieved premises
    theta = compute_theta(c, wr, k, c_tilde_indices)
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
      mu_t = torch.matmul(wg, mean_c_tilde_t.T)

      pyro.sample(f"c_star,{i}^({t})",
                  dist.MultivariateNormal(mu_t, Sigma),
                  obs=c_star[t])


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

  wr = pyro.sample(
      "W_retrieved",
      dist.Normal(torch.zeros(d, d), torch.ones(d, d)),
  )
  wg = pyro.sample(
      "W_generated",
      dist.Normal(torch.zeros(d, d), torch.ones(d, d)),
  )
  s2 = pyro.sample("sigmasquare", dist.Exponential(torch.tensor([1.0])))

  return p, wr, wg, s2


def compute_theta(c, wr, k, c_tilde_indices):
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

  # Construct evolved distributions
  theta = [theta[t].repeat(1, k[t]) for t in range(len(theta))]

  for t in range(len(k)):  # iteration
    for j in range(k[t] - 1):  # sample number
      theta[t][c_tilde_indices[t][j], j + 1:] = 0

  return theta


def psi(c, wr):
  """
  Scoring function to measure similarity of each 
  available premise to mean available premise.

  :param c: available premises, m x d tensor.
  :param wr: model parameters for retrieval, d x d tensor.
  """
  mean_c = torch.mean(c, dim=0, keepdim=True)
  scores = torch.matmul(torch.matmul(c, wr), mean_c.T)

  return scores
