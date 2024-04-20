import torch
import pyro
import pyro.distributions as dist


def model(Y, c0):
  """
  Probability model for entailments trees.

  Inputs:
  y:       input dataset of entailment trees.
  c0:      input set of premises to construct a tree off of (latent).

  Parameters:
  l:       lambda; rate parameter for k_i.
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
  d = 768

  # Model parameter priors
  l, wr, wg, s2 = sample_model_params(d)

  for i in range(n):
    y = Y[i]

    # Fetch available, retrieved, and generated premises per iteration.
    c = y.to_embedding(y.available_premises())
    m = torch.tensor([c_t.size(0) for c_t in c])
    T = len(m)

    c_tilde = y.to_embedding(y.retrieved_premises())
    k = torch.tensor([c_tilde_t.size(0) for c_tilde_t in c_tilde])

    c_star = y.to_embedding([[id] for id in y.generated_premises()])

    # Sample for number of retrieved premises
    with pyro.plate("k", len(k)):
      pyro.sample("k_t", dist.Poisson(l), obs=k - 1)

    # Sample for retrieved premises
    theta = compute_theta(c, wr)
    sample_retrieved(theta, k, c_tilde)

    # Sample for generated premises
    sample_generated(wg, s2, c_tilde, c_star)


def sample_model_params(d):
  """
  Samples model parameters from corresponding priors.

  :param d: embedding dimension.

  :return l: lambda
  :return wr: W_retrieved
  :return wg: W_generated
  :return s2: sigma^2
  """
  l = pyro.sample("lambda", dist.Exponential(torch.tensor([1])))
  wr = pyro.sample(
      "W_retrieved",
      dist.Normal(torch.zeros(d, d), torch.ones(d, d)),
  )
  wg = pyro.sample(
      "W_generated",
      dist.Normal(torch.zeros(d, d), torch.ones(d, d)),
  )
  s2 = pyro.sample("sigmasquare", dist.Exponential(torch.tensor([1])))

  return l, wr, wg, s2


def compute_theta(c, wr)
  """
  Construct categorical distribution of retrieval probability
  for available premises.
  
  :param c: available premises, m x d tensor.
  :param wr: model parameters for retrieval, d x d tensor.

  :return theta: list of m_t x 1 tensors.
  """
  # Construct distribution on retrieved premises
  psis = [psi(c_t, Wr) for c_t in c]
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
  # Calculate the mean row of C
  mean_c = torch.mean(c, dim=0, keepdim=True)

  # Compute the scores for each row of C
  scores = torch.matmul(torch.matmul(c, Wr), mean_c.T)

  return scores


def sample_retrieved(theta, k, c_tilde):
  """
  
  :param theta: initial categorical distribution parameter.
  :param k: total number of distinct samples.
  :c_tilde:
  """
