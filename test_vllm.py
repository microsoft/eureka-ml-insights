from eureka_ml_insights.models.models import vLLMModel

vllm_model = vLLMModel(model_name="gpt2", max_tokens=10, gpu_memory_utilization=0.1)
print(vllm_model.generate("Hello"))

