import pickle
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from model import model

if __name__ == "__main__":
  # Load dataset
  dataset = "dev"
  processed_dataset_fp = f'data/processed/{dataset}.pkl'

  print("loading dataset")
  with open(processed_dataset_fp, 'rb') as file:
    trees = pickle.load(file)
  print("dataset loaded")

  trees = trees[:10]

  pyro.render_model(
      model,
      model_args=(trees,),
      filename="model.png",
      # render_distributions=True,
  )

#   # Run model
#   print("running model")
#   pyro.clear_param_store()
#   svi = SVI(model, None, Adam({"lr": 0.01}), loss=Trace_ELBO())

#   # Run inference
#   num_iterations = 1000
#   for i in range(num_iterations):
#     loss = svi.step(trees)
#     if i % 10 == 0:
#       print(f"Iteration {i}, Loss = {loss}")

#   print("model ran")

#   # Get the learned parameters
#   posterior = pyro.infer.Predictive(model, guide=None, num_samples=1000)
#   posterior_samples = posterior(trees)

#   # Extracting posterior distributions of parameters
#   posterior_weight = posterior_samples['weight'].mean().item()
#   posterior_bias = posterior_samples['bias'].mean().item()
