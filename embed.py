from sentence_transformers import SentenceTransformer


def sentence_to_vec(sentence, model_name='paraphrase-distilroberta-base-v1'):
  model = SentenceTransformer(model_name)
  embeddings = model.encode(sentence, convert_to_tensor=True)

  return embeddings


if __name__ == "__main__":
  sentence = "This is a sample sentence."
  vector = sentence_to_vec(sentence)

  # Print the shape of the vector
  print("Shape of the vector:", vector.shape)
  print("Vector:", vector)