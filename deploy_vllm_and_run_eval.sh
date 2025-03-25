#!/bin/bash

export PYTHONPATH="$(pwd):$PYTHONPATH"
model_name="microsoft/phi-4"
exp_config="IFEval_PIPELINE"
current_datetime=$(date +"%Y-%m-%d-%H:%M:%S")
log_dir="logs/deploy_vllm_and_run_eval/$current_datetime"
mkdir -p $log_dir

# vLLM args
num_servers=4
tensor_parallel_size=1 
pipeline_parallel_size=1
base_port=8000
gpus_per_port=$((tensor_parallel_size * pipeline_parallel_size))

# Add any additional args accepted by vLLM serve here
VLLM_ARGS="\
    --tensor-parallel-size=${tensor_parallel_size} \
    --pipeline-parallel-size=${pipeline_parallel_size} \
    --gpu-memory-utilization=0.9 \
"

# Start servers
echo "Spinning up servers..."
for (( i = 0; i < $num_servers; i++ )) do
    port=$((base_port + i))
    first_gpu=$((i * gpus_per_port))
    last_gpu=$((first_gpu + gpus_per_port - 1))
    devices=$(seq -s, $first_gpu $last_gpu)
    CUDA_VISIBLE_DEVICES=${devices} vllm serve ${model_name} "$@" --port ${port} ${VLLM_ARGS} >> $log_dir/${port}.log 2>&1 &
done

# Wait for servers to come online
while true; do

    servers_online=0
    for (( i = 0; i < $num_servers; i++ )) do
        port=$((base_port + i))
        url="http://0.0.0.0:${port}/health"
        response=$(curl -s -o /dev/null -w "%{http_code}" "$url")

        if [ "$response" -eq 200 ]; then
            servers_online=$((servers_online + 1))
        fi
    done

    if [ $servers_online -eq $num_servers ]; then
        echo "All servers are online."
        break
    else
        echo "Waiting for $((num_servers - servers_online)) more servers to come online..."
    fi

    sleep 10
done

# Call Eureka to initiate evals
ports=$(seq -s ' ' $base_port $((base_port + num_servers - 1)))
EUREKA_ARGS="\
    --model_config=${model_name} \
    --exp_config=${exp_config} \
    --local_vllm \
    --ports ${ports} \
"
echo "Starting evals..."
python main.py ${EUREKA_ARGS} >> $log_dir/out.log 2>&1

# Shut down servers
echo "Shutting down vLLM servers..."
for (( i = 0; i < $num_servers; i++ )) do
    port=$((base_port + i))
    logfile="$log_dir/${port}.log"
    pid=$(grep "Started server process" $logfile | grep -o '[0-9]\+')
    echo "Shutting down server on port ${port} (PID ${pid})"
    kill -INT $pid
done