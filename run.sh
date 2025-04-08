## source .eurekaenv/bin/activate



# python main.py --exp_config NPHARD_TSP_PIPELINE_Runs --model_config OAI_O1_PREVIEW_CONFIG --exp_logdir nphard_tsp_level_test

# python main.py --exp_config NPHARD_TSP_PIPELINE_Runs --model_config OAI_GPT4O_2024_05_13_CONFIG --exp_logdir nphard_tsp_level_test

## TRAPI model

# python main.py --exp_config NPHARD_TSP_PIPELINE_multipleRuns --model_config TRAPI_GPT4O_2024_05_13_CONFIG --exp_logdir nphard_tsp_level_test

# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config PHI4HF_CONFIG --exp_logdir nphard_sat_level_phi4





# ### claude
# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config GEMINI_EXP_1206_CONFIG --exp_logdir nphard_sat_level_claude-3-5-sonnet-20241022 --resume_from /home/vivineet/projects/evaluation/NPHardEval/SAT_02-04-2025/eureka-ml-insights/logs/NPHARD_SAT_PIPELINE_MULTIPLE_RUNS/nphard_sat_level_claude-3-5-sonnet-20241022/2025-01-31-01-31-46.679463/inference_result/inference_result.jsonl



# # ### gemini
# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config GEMINI_V2_FLASH_THINKING_EXP_0121_CONFIG --exp_logdir nphard_sat_level_gemini-2.0-flash-thinking-exp-01-21 --resume_from /home/vivineet/projects/evaluation/NPHardEval/SAT_02-04-2025/eureka-ml-insights/logs/NPHARD_SAT_PIPELINE_MULTIPLE_RUNS/nphard_sat_level_gemini-2.0-flash-thinking-exp-01-21/2025-02-13-09-36-32.777977/inference_result/inference_result.jsonl


# # ### gemini
# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config GEMINI_EXP_1206_CONFIG --exp_logdir nphard_sat_level_gemini-exp-1206 --resume_from /home/vivineet/projects/evaluation/NPHardEval/SAT_02-04-2025/eureka-ml-insights/logs/NPHARD_SAT_PIPELINE_MULTIPLE_RUNS/nphard_sat_level_gemini-exp-1206/2025-02-09-20-12-47.286010/inference_result/inference_result.jsonl


# ### gpt4o
# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config OAI_GPT4O_2024_11_20_CONFIG --exp_logdir nphard_sat_level_gpt-4o_2024-11-20 --resume_from /home/vivineet/projects/evaluation/NPHardEval/SAT_02-04-2025/eureka-ml-insights/logs/NPHARD_SAT_PIPELINE_MULTIPLE_RUNS/nphard_sat_level_gpt-4o_2024-11-20/2025-01-31-01-38-28.899516/inference_result/inference_result.jsonl


# ### o1
# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config OAI_O1_20241217_CONFIG --exp_logdir nphard_sat_level_o1_2024-12-17 --resume_from /home/vivineet/projects/evaluation/NPHardEval/SAT_02-04-2025/eureka-ml-insights/logs/NPHARD_SAT_PIPELINE_MULTIPLE_RUNS/nphard_sat_level_o1_2024-12-17/2025-02-10-10-12-05.370511/inference_result/inference_result.jsonl


# # # ### llama_3_1
# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config AIF_NT_LLAMA3_1_405B_INSTRUCT_EASTUS_OSS_CONFIG_2 --exp_logdir nphard_sat_level_llama3_1_405b --resume_from /home/vivineet/projects/evaluation/NPHardEval/SAT_02-04-2025/eureka-ml-insights/logs/NPHARD_SAT_PIPELINE_MULTIPLE_RUNS/nphard_sat_level_llama3_1_405b/2025-01-31-16-35-24.497990/inference_result/inference_result.jsonl



# # # # # ### phi4 ### not ready
# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config PHI4HF_CONFIG --exp_logdir nphard_sat_level_phi4


# # # # # ### o3_mini_high trapi
# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config TRAPI_AIF_O3_MINI_CONFIG --exp_logdir nphard_sat_level_o3_mini_high

# # # # # ### o3_mini_high msr_lit
# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config TRAPI_AIF_O3_MINI_CONFIG --exp_logdir nphard_sat_level_o3_mini_high --resume_from /home/vivineet/projects/evaluation/NPHardEval/SAT_02-04-2025/eureka-ml-insights/logs/NPHARD_SAT_PIPELINE_MULTIPLE_RUNS/nphard_sat_level_o3_mini_high/2025-03-10-11-16-43.083154/inference_result/inference_result.jsonl

# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config AIF_NT_LLAMA3_1_405B_INSTRUCT_EASTUS_OSS_CONFIG_2 --exp_logdir nphard_sat_level_llama3_1_405b 

# --resume_from /home/vivineet/projects/evaluation/NPHardEval/SAT_02-04-2025/eureka-ml-insights/logs/NPHARD_SAT_PIPELINE_MULTIPLE_RUNS/nphard_sat_level_llama3_1_405b/2025-01-31-23-31-01.301467/inference_result/inference_result.jsonl


# # # # # ### o3_mini_high trapi
python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config MSR_LIT_O1_reasoning_1_CONFIG --exp_logdir nphard_sat_level_o3_mini_high
