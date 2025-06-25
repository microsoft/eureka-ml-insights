"""This module defines reserved column names used by Eureka for inference and prompt processing.

The variables in this module list the columns that may be removed or overwritten by Eureka if present
in the data.
"""

# if your data has any of these columns, they may be removed or overwritten by Eureka
INFERENCE_RESERVED_NAMES = ["model_output", "is_valid", "response_time", "n_output_tokens"]
"""list of str: Reserved column names used for inference outputs and status."""

PROMPT_PROC_RESERVED_NAMES = ["prompt_hash", "prompt", "uid", "data_point_id", "data_repeat_id", "__hf_task", "__hf_split"]
"""list of str: Reserved column names used for prompt processing."""