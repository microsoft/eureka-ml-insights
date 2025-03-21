## source .eurekaenv/bin/activate


################################


# ### gpt4o test

# python main.py --exp_config NPHARD_TSP_PIPELINE_MULTIPLE_RUNS --model_config OAI_GPT4O_2024_11_20_CONFIG --exp_logdir nphard_tsp_level_gpt4o_2024-11-20 --resume_from /home/vivineet/projects/evaluation/NPHardEval/TSP_02-07-2025/eureka-ml-insights/logs/NPHARD_TSP_PIPELINE_MULTIPLE_RUNS/nphard_tsp_level_gpt4o_2024-11-20/2025-02-10-22-57-48.817329/inference_result/inference_result.jsonl

################################


############################################

# python main.py --exp_config NPHARD_TSP_PIPELINE_MULTIPLE_RUNS --model_config CLAUDE_3_7_SONNET_THINKING_CONFIG --exp_logdir nphard_tsp_CLAUDE_3_7_SONNET_THINKING 
# --resume_from /home/vivineet/projects/evaluation/NPHardEval/TSP_03_12_2025_claude_3_7/eureka-ml-insights/logs/NPHARD_TSP_PIPELINE_MULTIPLE_RUNS/nphard_tsp_CLAUDE_3_7_SONNET_THINKING/2025-03-17-22-47-57.710788/inference_result/inference_result.jsonl

############################################


# python main.py --exp_config NPHARD_TSP_SEQ_PIPELINE --model_config CLAUDE_3_5_SONNET_CONFIG --exp_logdir nphard_tsp_level_gpt4o_2024-08-06 --n_iter 3

python main.py --exp_config NPHARD_TSP_SEQ_PIPELINE --model_config GATEWAY_GPT_4O_CONFIG --exp_logdir nphard_tsp_level_gpt4o_2024-08-06

# python main.py --exp_config NPHARD_TSP_SEQ_PIPELINE --model_config TRAPI_GPT4O_2024_08_06_CONFIG --exp_logdir nphard_tsp_level_gpt4o_2024-08-06