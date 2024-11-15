# if your data has any of these columns, they may be removed or overwritten by Eureka
INFERENCE_RESERVED_NAMES = ["model_output", "is_valid", "response_time", "n_output_tokens"]
PROMPT_PROC_RESERVED_NAMES = ["prompt_hash", "prompt", "uid", "data_point_id", "data_repeat_id", "__hf_task", "__hf_split"]
