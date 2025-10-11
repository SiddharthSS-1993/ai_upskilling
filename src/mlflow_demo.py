from transformers import pipeline
import time
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')
start_time = time.time()
result=generator('whats the weather', max_new_tokens=30)
end_time = time.time()
print("latency is:", end_time - start_time)
print("generated_text is:" +result[0]["generated_text"])