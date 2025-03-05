#!/bin/bash
set -e -x

for (( i = 0; i < $NUM_SERVERS; i++ )) do
    port=$((8000 + i))
    # Here GPU_SKIP is set as tensor_parallel_size*pipeline_parallel_size
    first_gpu=$((i * GPU_SKIP))
    last_gpu=$((first_gpu + GPU_SKIP - 1))
    devices=$(seq -s, $first_gpu $last_gpu)
    CUDA_VISIBLE_DEVICES=${devices} "$CURRENT_PYTHON_EXEC" -m vllm.entrypoints.openai.api_server "$@" --port ${port} >> ${LOCAL_VLLM_LOG_DIR}/${port}.log 2>&1 &
done