#!/bin/bash

export PYTHONPATH="$(pwd):$PYTHONPATH"
model_name="microsoft/phi-4"
current_datetime=$(date +"%Y-%m-%d %H:%M:%S")
log_dir="logs/$(current_datetime)/deploy_and_run"

mkdir -p $log_dir

# vllm args
num_servers=4
tensor_parallel_size=1
pipeline_parallel_size=1
gpu_skip=$((tensor_parallel_size * pipeline_parallel_size))
base_port=8000
VLLM_ARGS="\
    --tensor-parallel-size=${tensor_parallel_size} \
    --pipeline-parallel-size=${pipeline_parallel_size} \
    --gpu-memory-utilization=0.9 \
"
echo "Spinning up servers..."
for (( i = 0; i < $num_servers; i++ )) do
    port=$((base_port + i))
    first_gpu=$((i * gpu_skip))
    last_gpu=$((first_gpu + gpu_skip - 1))
    devices=$(seq -s, $first_gpu $last_gpu)
    CUDA_VISIBLE_DEVICES=${devices} vllm serve ${model_name} "$@" --port ${port} ${VLLM_ARGS}  >> $log_dir/${port}.log 2>&1 &
done

# Health check to see when servers come online
url="http://0.0.0.0:"${base_port}"/health"

while true; do
  # Send the GET request and store the response
  response=$(curl -s -o /dev/null -w "%{http_code}" "$url")

  if [ "$response" -eq 200 ]; then
    echo "Servers online..."
    break
  else
    echo "Waiting for servers to come online..."
  fi

  sleep 10
done

sleep 10

# Now call eureka to initiate evals.
ports=$(seq -s ' ' $base_port $((base_port + num_servers - 1)))
EUREKA_ARGS="\
    --model_config=${model_name} \
    --exp_config="IFEval_PIPELINE" \
    --local_vllm \
    --ports ${ports} \
"
echo "Starting evals..."
python main.py ${EUREKA_ARGS} >> $log_dir/phi4.log 2>&1

echo "Shutting down vllm servers..."
pgrep -f "vllm serve" | xargs kill -INT