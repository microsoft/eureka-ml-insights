## source .eurekaenv/bin/activate

# # # # # ### o3_mini_high msr_lit
# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config TRAPI_AIF_O3_MINI_CONFIG --exp_logdir nphard_sat_level_o3_mini_high
# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config OAI_O3_MINI_HIGH_CONFIG --exp_logdir nphard_sat_level_o3_mini_high_direct_oai --resume_from /home/vivineet/projects/evaluation/NPHardEval/SAT_02-04-2025/eureka-ml-insights_sat_o3_mini_high/logs/NPHARD_SAT_PIPELINE_MULTIPLE_RUNS/nphard_sat_level_o3_mini_high_direct_oai/2025-03-12-23-13-30.487726/inference_result/inference_result.jsonl
# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config TRAPI_AIF_O3_MINI_CONFIG --exp_logdir nphard_sat_level_o3_mini_high_direct_oai --resume_from /home/vivineet/projects/evaluation/NPHardEval/SAT_02-04-2025/eureka-ml-insights_sat_o3_mini_high/logs/NPHARD_SAT_PIPELINE_MULTIPLE_RUNS/nphard_sat_level_o3_mini_high_direct_oai/2025-03-14-00-42-51.448876/inference_result/inference_result.jsonl
# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config MSR_LIT_O3_mini_reasoning_1_CONFIG --exp_logdir nphard_sat_level_o3_mini_high_direct_oai --resume_from /home/vivineet/projects/evaluation/NPHardEval/SAT_02-04-2025/eureka-ml-insights_sat_o3_mini_high/logs/NPHARD_SAT_PIPELINE_MULTIPLE_RUNS/nphard_sat_level_o3_mini_high_direct_oai/2025-03-14-00-42-51.448876/inference_result/inference_result.jsonl


# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config TOGETHER_DEEPSEEK_R1_CONFIG --exp_logdir nphard_sat_level_o1_2024-12-17

# --resume_from /home/vivineet/projects/evaluation/NPHardEval/SAT_02-04-2025/eureka-ml-insights_sat_o3_mini_high/logs/NPHARD_SAT_PIPELINE_MULTIPLE_RUNS/nphard_sat_level_o3_mini_high_direct_oai/2025-03-14-00-42-51.448876/inference_result/inference_result.jsonl 


python main.py --exp_config ToxiGen_Generative_PIPELINE --model_config OAI_GPT4_1106_PREVIEW_CONFIG --exp_logdir phi_4_reasoning_toxigen_gen --resume_from /home/vivineet/projects/evaluation/NPHardEval/toxigen_generative_phi_reasoning_eval/eureka-ml-insights/logs/ToxiGen_Generative_PIPELINE/phi_4_reasoning_toxigen_gen/2025-04-09-08-26-46.270552/inference_result/inference_result.jsonl
