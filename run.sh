## source .eurekaenv/bin/activate



# # # # # ### GATEWAY_PHI_4_CONFIG

## phi4-reasoning -- small drop (~7%)

# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config GATEWAY_PHI_4_CONFIG --exp_logdir Phi-4_reasoning --resume_from /home/vivineet/projects/evaluation/NPHardEval/launch_aml/launch_aml_04-07-2025_sat/eureka-ml-insights_sat_remove_dummy/logs/NPHARD_SAT_PIPELINE_MULTIPLE_RUNS/Phi-4_reasoning/2025-04-11-16-24-34.324761/inference_result/inference_result.jsonl


## o3 - no drop

# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config GATEWAY_PHI_4_CONFIG --exp_logdir nphard_sat1_o3_mini_reasoning_10_high_1_all_runs 

#--resume_from /home/vivineet/projects/evaluation/NPHardEval/launch_aml/launch_aml_04-07-2025_sat/eureka-ml-insights_sat_remove_dummy/logs/NPHARD_SAT_PIPELINE_MULTIPLE_RUNS/nphard_sat1_o3_mini_reasoning_10_high_1_all_runs/2025-03-14-15-00-24.312011/inference_result/inference_result.jsonl

## o1 -- no drop

# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config GATEWAY_PHI_4_CONFIG --exp_logdir nphard_sat_level_o1_2024-12-17 --resume_from /home/vivineet/projects/evaluation/NPHardEval/launch_aml/launch_aml_04-07-2025_sat/eureka-ml-insights_sat_remove_dummy/logs/NPHARD_SAT_PIPELINE_MULTIPLE_RUNS/nphard_sat_level_o1_2024-12-17/2025-02-17-19-45-54.233604/inference_result/inference_result.jsonl

## gpt4o -- no drop

# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config GATEWAY_PHI_4_CONFIG --exp_logdir nphard_sat_gpt4o_2024_08_06 --resume_from /home/vivineet/projects/evaluation/NPHardEval/launch_aml/launch_aml_04-07-2025_sat/eureka-ml-insights_sat_remove_dummy/logs/NPHARD_SAT_PIPELINE_MULTIPLE_RUNS/nphard_sat_gpt4o_2024_08_06/2025-03-13-16-57-07.160657/inference_result/inference_result.jsonl

## deepseek r1 -- drop of 0.5%

# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config GATEWAY_PHI_4_CONFIG --exp_logdir nphard_sat1_deepseek_together --resume_from /home/vivineet/projects/evaluation/NPHardEval/launch_aml/launch_aml_04-07-2025_sat/eureka-ml-insights_sat_remove_dummy/logs/NPHARD_SAT_PIPELINE_MULTIPLE_RUNS/nphard_sat1_deepseek_together/2025-03-13-16-16-47.179595/inference_result/inference_result.jsonl


# ## claude 3.7 thinking -- drop of 0.5

# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config GATEWAY_PHI_4_CONFIG --exp_logdir nphard_sat_CLAUDE_3_7_SONNET_THINKING --resume_from /home/vivineet/projects/evaluation/NPHardEval/launch_aml/launch_aml_04-07-2025_sat/eureka-ml-insights_sat_remove_dummy/logs/NPHARD_SAT_PIPELINE_MULTIPLE_RUNS/nphard_sat_CLAUDE_3_7_SONNET_THINKING/2025-03-21-16-33-50.561466/inference_result/inference_result.jsonl


# ## claude 3.5 -- no drop

# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config GATEWAY_PHI_4_CONFIG --exp_logdir nphard_sat_level_claude-3-5-sonnet-20241022 --resume_from /home/vivineet/projects/evaluation/NPHardEval/launch_aml/launch_aml_04-07-2025_sat/eureka-ml-insights_sat_remove_dummy/logs/NPHARD_SAT_PIPELINE_MULTIPLE_RUNS/nphard_sat_level_claude-3-5-sonnet-20241022/2025-02-17-19-26-05.135180/inference_result/inference_result.jsonl


# # ## gemini 2.0 pro -- no drop (0.2%)

# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config GATEWAY_PHI_4_CONFIG --exp_logdir nphard_sat_gemini-2.0-pro --resume_from /home/vivineet/projects/evaluation/NPHardEval/launch_aml/launch_aml_04-07-2025_sat/eureka-ml-insights_sat_remove_dummy/logs/NPHARD_SAT_PIPELINE_MULTIPLE_RUNS/nphard_sat_gemini-2.0-pro/2025-03-05-22-50-27.830655/inference_result/inference_result.jsonl

# # ## nphard_sat_level_gemini-2.0-flash-thinking-exp-01-21_2_no_code -- no drop (0.2%)

# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config GATEWAY_PHI_4_CONFIG --exp_logdir nphard_sat_level_gemini-2.0-flash-thinking-exp-01-21_2_no_code --resume_from /home/vivineet/projects/evaluation/NPHardEval/launch_aml/launch_aml_04-07-2025_sat/eureka-ml-insights_sat_remove_dummy/logs/NPHARD_SAT_PIPELINE_MULTIPLE_RUNS/nphard_sat_level_gemini-2.0-flash-thinking-exp-01-21_2_no_code/2025-03-18-12-31-47.382927/inference_result/inference_result.jsonl


# # ## llama 405b -- large drop (14%)

# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config GATEWAY_PHI_4_CONFIG --exp_logdir nphard_sat_level_llama3_1_405b_2 --resume_from /home/vivineet/projects/evaluation/NPHardEval/launch_aml/launch_aml_04-07-2025_sat/eureka-ml-insights_sat_remove_dummy/logs/NPHARD_SAT_PIPELINE_MULTIPLE_RUNS/nphard_sat_level_llama3_1_405b_2/2025-03-17-18-33-31.635028/inference_result/inference_result.jsonl

###############################################


# # # # # ### GATEWAY_PHI_4_CONFIG

## phi4-reasoning -- small drop (~7%)

# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config GATEWAY_PHI_4_CONFIG --exp_logdir Phi-4_reasoning --resume_from /home/vivineet/projects/evaluation/NPHardEval/launch_aml/launch_aml_04-07-2025_sat/eureka-ml-insights_sat_remove_dummy/logs/NPHARD_SAT_PIPELINE_MULTIPLE_RUNS/Phi-4_reasoning/2025-04-11-16-24-34.324761/inference_result/inference_result.jsonl


##########################################################


# # # # # ### GATEWAY_PHI_4_CONFIG

## phi4-reasoning -- small drop (~7%)

# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config GATEWAY_PHI_4_CONFIG --exp_logdir Phi-4_reasoning --resume_from /home/vivineet/projects/evaluation/NPHardEval/launch_aml/launch_aml_04-07-2025_sat/eureka-ml-insights_sat_remove_dummy/logs/NPHARD_SAT_PIPELINE_MULTIPLE_RUNS/Phi-4_reasoning/2025-04-11-16-24-34.324761/inference_result/inference_result.jsonl


## o3 - no drop

# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config GATEWAY_PHI_4_CONFIG --exp_logdir nphard_sat1_o3_mini_reasoning_10_high_1_all_runs 

#--resume_from /home/vivineet/projects/evaluation/NPHardEval/launch_aml/launch_aml_04-07-2025_sat/eureka-ml-insights_sat_remove_dummy/logs/NPHARD_SAT_PIPELINE_MULTIPLE_RUNS/nphard_sat1_o3_mini_reasoning_10_high_1_all_runs/2025-03-14-15-00-24.312011/inference_result/inference_result.jsonl

## o1 -- no drop

# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config GATEWAY_PHI_4_CONFIG --exp_logdir nphard_sat_level_o1_2024-12-17 --resume_from /home/vivineet/projects/evaluation/NPHardEval/launch_aml/launch_aml_04-07-2025_sat/eureka-ml-insights_sat_remove_dummy/logs/NPHARD_SAT_PIPELINE_MULTIPLE_RUNS/nphard_sat_level_o1_2024-12-17/2025-02-17-19-45-54.233604/inference_result/inference_result.jsonl

## gpt4o -- no drop

# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config GATEWAY_PHI_4_CONFIG --exp_logdir nphard_sat_gpt4o_2024_08_06 --resume_from /home/vivineet/projects/evaluation/NPHardEval/launch_aml/launch_aml_04-07-2025_sat/eureka-ml-insights_sat_remove_dummy/logs/NPHARD_SAT_PIPELINE_MULTIPLE_RUNS/nphard_sat_gpt4o_2024_08_06/2025-03-13-16-57-07.160657/inference_result/inference_result.jsonl

## deepseek r1 -- drop of 0.5%

# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config GATEWAY_PHI_4_CONFIG --exp_logdir nphard_sat1_deepseek_together --resume_from /home/vivineet/projects/evaluation/NPHardEval/launch_aml/launch_aml_04-07-2025_sat/eureka-ml-insights_sat_remove_dummy/logs/NPHARD_SAT_PIPELINE_MULTIPLE_RUNS/nphard_sat1_deepseek_together/2025-03-13-16-16-47.179595/inference_result/inference_result.jsonl


## deepseek distilled llama -- 

# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config GATEWAY_PHI_4_CONFIG --exp_logdir nphard_sat_level_deepseek_distill_llama --resume_from /home/vivineet/projects/evaluation/NPHardEval/launch_aml/launch_aml_04-07-2025_sat/eureka-ml-insights_sat_remove_dummy/logs/NPHARD_SAT_PIPELINE_MULTIPLE_RUNS/nphard_sat_level_deepseek_distill_llama/2025-04-11-01-43-52.244610/inference_result/inference_result.jsonl


# ## claude 3.7 thinking -- drop of 0.5

# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config GATEWAY_PHI_4_CONFIG --exp_logdir nphard_sat_CLAUDE_3_7_SONNET_THINKING 

#--resume_from /home/vivineet/projects/evaluation/NPHardEval/launch_aml/launch_aml_04-07-2025_sat/eureka-ml-insights_sat_remove_dummy/logs/NPHARD_SAT_PIPELINE_MULTIPLE_RUNS/nphard_sat_CLAUDE_3_7_SONNET_THINKING/2025-03-21-16-33-50.561466/inference_result/inference_result.jsonl


# ## claude 3.5 -- no drop

# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config GATEWAY_PHI_4_CONFIG --exp_logdir nphard_sat_level_claude-3-5-sonnet-20241022 --resume_from /home/vivineet/projects/evaluation/NPHardEval/launch_aml/launch_aml_04-07-2025_sat/eureka-ml-insights_sat_remove_dummy/logs/NPHARD_SAT_PIPELINE_MULTIPLE_RUNS/nphard_sat_level_claude-3-5-sonnet-20241022/2025-02-17-19-26-05.135180/inference_result/inference_result.jsonl


# # ## gemini 2.0 pro -- no drop (0.2%)

# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config GATEWAY_PHI_4_CONFIG --exp_logdir nphard_sat_gemini-2.0-pro --resume_from /home/vivineet/projects/evaluation/NPHardEval/launch_aml/launch_aml_04-07-2025_sat/eureka-ml-insights_sat_remove_dummy/logs/NPHARD_SAT_PIPELINE_MULTIPLE_RUNS/nphard_sat_gemini-2.0-pro/2025-03-05-22-50-27.830655/inference_result/inference_result.jsonl

# # ## nphard_sat_level_gemini-2.0-flash-thinking-exp-01-21_2_no_code -- no drop (0.2%)

# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config GATEWAY_PHI_4_CONFIG --exp_logdir nphard_sat_level_gemini-2.0-flash-thinking-exp-01-21_2_no_code --resume_from /home/vivineet/projects/evaluation/NPHardEval/launch_aml/launch_aml_04-07-2025_sat/eureka-ml-insights_sat_remove_dummy/logs/NPHARD_SAT_PIPELINE_MULTIPLE_RUNS/nphard_sat_level_gemini-2.0-flash-thinking-exp-01-21_2_no_code/2025-03-18-12-31-47.382927/inference_result/inference_result.jsonl


# # ## llama 405b -- large drop (14%)

# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config GATEWAY_PHI_4_CONFIG --exp_logdir nphard_sat_level_llama3_1_405b_2 --resume_from /home/vivineet/projects/evaluation/NPHardEval/launch_aml/launch_aml_04-07-2025_sat/eureka-ml-insights_sat_remove_dummy/logs/NPHARD_SAT_PIPELINE_MULTIPLE_RUNS/nphard_sat_level_llama3_1_405b_2/2025-03-17-18-33-31.635028/inference_result/inference_result.jsonl


# # ## phi4 -- large drop (14%)

# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config GATEWAY_PHI_4_CONFIG --exp_logdir nphard_sat_level_phi4_0-8 --resume_from /home/vivineet/projects/evaluation/NPHardEval/launch_aml/launch_aml_04-07-2025_sat/eureka-ml-insights_sat_remove_dummy/logs/NPHARD_SAT_PIPELINE_MULTIPLE_RUNS/nphard_sat_level_phi4_0-8/2025-04-10-17-41-47.352197/inference_result/inference_result.jsonl


###############################################


# # # # # ### GATEWAY_PHI_4_CONFIG

## phi4-reasoning -- small drop (~7%)

# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config GATEWAY_PHI_4_CONFIG --exp_logdir Phi-4_reasoning --resume_from /home/vivineet/projects/evaluation/NPHardEval/launch_aml/launch_aml_04-07-2025_sat/eureka-ml-insights_sat_remove_dummy/logs/NPHARD_SAT_PIPELINE_MULTIPLE_RUNS/Phi-4_reasoning/2025-04-11-16-24-34.324761/inference_result/inference_result.jsonl


########################

# ## phi4-reasoning medium -- small drop (~7%)

# python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config GATEWAY_PHI_4_CONFIG --exp_logdir PHI_4_REASONING_phi-16b --resume_from /home/vivineet/projects/evaluation/NPHardEval/launch_aml/launch_aml_04-07-2025_sat/eureka-ml-insights_sat_remove_dummy/logs/NPHARD_SAT_PIPELINE_MULTIPLE_RUNS/PHI_4_REASONING_phi-16b/2025-04-17-01-31-27.066175/inference_result/inference_result.jsonl


## phi4-reasoning high -- small drop (~7%)

python main.py --exp_config NPHARD_SAT_PIPELINE_MULTIPLE_RUNS --model_config GATEWAY_PHI_4_CONFIG --exp_logdir PHI_4_REASONING_phi-15b-rl --resume_from /home/vivineet/projects/evaluation/NPHardEval/launch_aml/launch_aml_04-07-2025_sat/eureka-ml-insights_sat_remove_dummy/logs/NPHARD_SAT_PIPELINE_MULTIPLE_RUNS/PHI_4_REASONING_phi-15b-rl/2025-04-17-02-00-17.184237/inference_result/inference_result.jsonl
