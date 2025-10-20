from sentence_transformers import SentenceTransformer, util
import  numpy as np
import optuna
import mlflow
import time

def evaluate(top_k: int, model_name: str) -> float:
    model = SentenceTransformer(model_name)
    document_embeddings = model.encode(documents, normalize_embeddings=True)
    scores = []
    for query, correct_index in queries:
        query_embedding = model.encode([query], normalize_embeddings=True)
        similarities = util.cos_sim(query_embedding, document_embeddings)[0].cpu().numpy()
        top_index = similarities.argsort()[::-1][:top_k]
        # Precision@k: 1 if correct doc is in top_k else 0
        hit = 1.0 if correct_index in top_index else 0.0
        scores.append(hit)
    # Average precision@k across queries
    return float(np.mean(scores))

def objective_lm(trial: optuna.Trial) -> float:
    model_name = trial.suggest_categorical("embed_model", ["all-MiniLM-L6-v2",
                                                           "paraphrase-MiniLM-L6-v2"])
    top_k = trial.suggest_int("top_k", 1, 3)
    # 3 We can also tune bigger chunks if docs are big
    return evaluate(top_k=top_k, model_name=model_name)

documents = ["The Titanic sank in 1912 after striking asn Iceberg.",
      "Random forest in an ensemble of decision trees.",
      "Passengers boarded at Southhampton before departure.",
      "StandardScalar normalizes numeric features.",
      "One hot encoding converts categories into binary columns."]

queries = [("Where did the passengers board?", 2), # Expects doc index 2
           ("Why did the titanic sink?", 0), # Expects doc index 0
           ("What is a Random Forest?", 1) # Expects doc index 1
           ]

llm_study = optuna.create_study(direction="maximize", study_name="retrieval_at_top_k")
mlflow.set_tracking_uri("file:///C:/MachineLearningCourse/AI Upskilling/ai upskilling/mlruns/result_tracking_for_titanic")
mlflow.set_experiment("LLM Pipeline For Titanic Data")
with mlflow.start_run(run_name="optuna_retrieval") as run:
    llm_study.optimize(objective_lm, n_trials=12, show_progress_bar=False)
    best = llm_study.best_trial
    
    mlflow.log_metric("best_precision_at_k", best.value)
    mlflow.log_params(best.params)

    # Latency capture with the best configurations
    best_model = SentenceTransformer(best.params["embed_model"])
    t0 = time.time()
    _ = best_model.encode(["latency_check"], normalize_embeddings=True)
    latency_ms = (time.time() - t0) * 1000

    mlflow.log_metric("emed_latency_in_millisecond", latency_ms)
    print("Best precision@k:", llm_study.best_value)
    print("Best Parameters:", best.params)




