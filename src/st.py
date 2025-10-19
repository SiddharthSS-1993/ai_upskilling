from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import mlflow, mlflow.sklearn
import time


sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')

sentence_1 = "Data Science measures uncertainty"
sentence_2 = "Machine Learning reduces uncertainty"
sentence_3 = "Bananas are yellow"

start = time.time()
sentence_embedding_1 = sentence_transformer_model.encode(sentence_1, convert_to_tensor=True)
sentence_embedding_2 = sentence_transformer_model.encode(sentence_2, convert_to_tensor=True)
sentence_embedding_3 = sentence_transformer_model.encode(sentence_3, convert_to_tensor=True)
latency = time.time() - start

print(f"Embedding_1 Shape: {sentence_embedding_1.shape}")
print(f"Embedding_2 Shape: {sentence_embedding_2.shape}")
print(f"Embedding_3 Shape: {sentence_embedding_3.shape}")
print(f"Latency: {latency:.3f} seconds")

similarity_sentence_1_2 = util.cos_sim(sentence_embedding_1, sentence_embedding_2)
similarity_sentence_1_3 = util.cos_sim(sentence_embedding_1, sentence_embedding_3)

print("Similarity between Sentence 1 and Sentence 2:", similarity_sentence_1_2.item())
print("Similarity between Sentence 1 and Sentence 3:", similarity_sentence_1_3.item())

embeddings = [sentence_embedding_1, sentence_embedding_2, sentence_embedding_3]
labels = [sentence_1, sentence_2, sentence_3]

pca = PCA(n_components=2)

reduced = pca.fit_transform([e.cpu().numpy() for e in embeddings])
plt.scatter(reduced[:,0],reduced[:,1])

for i, label in enumerate(labels):
    plt.annotate(label, (reduced[i,0], reduced[i,1]))
plt.title("Sentence Embedding Space (PCA Projection)")
plt.savefig("Sentence Embedding Space (PCA Projection).png")
plt.show()

# Log with MLFLOW
mlflow.set_tracking_uri("file:///C:/MachineLearningCourse/AI Upskilling/ai upskilling/mlruns/llm_feature_results_tracking")
mlflow.set_experiment("Sentence Transformers Tracking and Evaluation")

with mlflow.start_run(run_name="Sentence Transformers Evaluation"):
    mlflow.log_param("model", "all-MiniLM-L6-v2")
    mlflow.log_metric("Latency in seconds", latency)
    mlflow.log_metric("Similarity Average", (similarity_sentence_1_2 + similarity_sentence_1_3) / 2)
    mlflow.sklearn.log_model(sentence_transformer_model, "sentence transformer model")
    mlflow.sklearn.log_model(pca, "Principal Component Analysis model")

    print("Logged to MLFlow - Open UI with:")
    print("mlflow ui --backend-store-uri file:///C:/MachineLearningCourse/AI Upskilling/ai upskilling/mlruns/llm_feature_results_tracking")
    
    # Basic Explanation
    query = "What does Data Science measure?"
    query_embedding = sentence_transformer_model.encode(query, convert_to_tensor=True)

    similarity_query_sentence_1 = util.cos_sim(query_embedding, sentence_embedding_1)
    similarity_query_sentence_2 = util.cos_sim(query_embedding, sentence_embedding_2)
    similarity_query_sentence_3 = util.cos_sim(query_embedding, sentence_embedding_3)

    print(f"Similarity between Query and Sentence 1: {similarity_query_sentence_1.item():.3f} -> {sentence_1}")
    print(f"Similarity between Query and Sentence 2: {similarity_query_sentence_2.item():.3f} -> {sentence_2}")
    print(f"Similarity between Query and Sentence 3: {similarity_query_sentence_2.item():.3f} -> {sentence_3}") 
    mlflow.log_metric("Max Similarity Score between Query and Sentences",
                      np.max([similarity_query_sentence_1.item(),
                              similarity_query_sentence_2.item(),
                              similarity_query_sentence_3.item()]))
    mlflow.end_run()





