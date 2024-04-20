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
        dist.Binomial(
            torch.tensor(m),
            torch.tensor(torch.ones_like(k) * l),
        ),
        obs=k - 1,
    )

    # Sample for retrieved premises
    theta = compute_theta(c, wr)
    sample_retrieved(theta, k, c_tilde_indices, i)

    # Sample for generated premises
    sample_generated(wg, s2, d, c_tilde, c_star, i)


def sample_model_params(d):
  """
  Samples model parameters from corresponding priors.

  :param d: embedding dimension.

  :return l: lambda
  :return wr: W_retrieved
  :return wg: W_generated
  :return s2: sigma^2
  """
  l = pyro.sample("lambda", dist.Exponential(torch.tensor([1.0])))
  wr = pyro.sample(
      "W_retrieved",
      dist.Normal(torch.zeros(d, d), torch.ones(d, d)),
  )
  wg = pyro.sample(
      "W_generated",
      dist.Normal(torch.zeros(d, d), torch.ones(d, d)),
  )
  s2 = pyro.sample("sigmasquare", dist.Exponential(torch.tensor([1.0])))

  return l, wr, wg, s2


def compute_theta(c, wr):
  """
  Construct categorical distribution of retrieval probability
  for available premises.
  
  :param c: available premises, m x d tensor.
  :param wr: model parameters for retrieval, d x d tensor.

  :return theta: list of m_t x 1 tensors.
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
  # Calculate the mean row of C
  mean_c = torch.mean(c, dim=0, keepdim=True)

  # Compute the scores for each row of C
  scores = torch.matmul(torch.matmul(c, wr), mean_c.T)

  return scores


def sample_retrieved(theta, k, c_tilde_indices, i):
  """
  Sample k distinct premises from categorical distribution
  parameterized by theta.

  :param theta: parameters for initial categorical distribution.
    List of m_t * 1 tensors.
  :param k: number of distinct premises to sample.
  :param c_tilde_indices: indices of retrieved premises w.r.t. 
  the available premises.
  :i: current sample index.
  """
  for t in range(len(k)):  # iteration
    for j in range(k[t]):  # sample number
      # Sample then update theta
      pyro.sample(
          f"j_{i},{j}^({t})",
          dist.Categorical(theta[t].T),  # Ensure 1-dim
          obs=c_tilde_indices[t][j],
      )
      theta[t][c_tilde_indices[t][j]] = 0


def sample_generated(wg, s2, d, c_tilde, c_star, i):
  """
  Samples the generated premises given the retrieved premises.
  Uses linear model defined by wg and s2.
  
  :param wg: model parameter.
  :param s2: model parameter.
  :param d: embedding dimension.
  :c_tilde: retrieved premises.
  :c_star: generated premises.
  :i: current sample index.
  """
  # Covariance matrix
  Sigma = torch.eye(d) * s2

  for t in range(len(c_tilde)):  # iteration
    # Compute normal model parameters
    mean_c_tilde_t = torch.mean(c_tilde[t], dim=0, keepdim=True)
    mu_t = torch.matmul(wg, mean_c_tilde_t.T)

    pyro.sample(f"c_star,{i}^({t})", dist.Normal(mu_t, Sigma), obs=c_star[t])
