"""Defines the pipeline for running LiveCodeBench.

See https://livecodebench.github.io/ for details.

Minimal usage example (from the project's root directory):
    $ python main.py \
        --exp_config "LIVE_CODE_BENCH_CODEGEN_PIPELINE" \
        --model_config "GATEWAY_GPT4O_CONFIG"

Other command line arguments can be provided as needed.
Example:
    $ python main.py \
        --exp_config "LIVE_CODE_BENCH_CODEGEN_PIPELINE" \
        --model_config "GATEWAY_GPT4O_CONFIG" \
        --lcb_release_version "release_v5" \
        --lcb_start_datetime "2024-08-01T00:00:00" \
        --lcb_end_datetime "2025-01-01T23:59:59"

See the arguments of the `configure_pipeline()` method below for available
parameters.
"""

import datetime
import pathlib
import textwrap
import sys

from typing import Any

from eureka_ml_insights import configs, core, data_utils
from eureka_ml_insights.configs import config
from eureka_ml_insights.core import eval_reporting
from eureka_ml_insights.data_utils.live_code_bench import (
    code_extraction_transform,
    decode_test_cases_transform,
)
from eureka_ml_insights.metrics.live_code_bench import (
    codegen_test_case_results_metric,
    pass_at_k_aggregator,
    sandbox_config,
)


# Default additional imports to include in the code being tested.
_DEFAULT_ADDITIONAL_PYTHON_IMPORTS: str = textwrap.dedent("""
    from string import *
    from re import *
    from datetime import *
    from collections import *
    from heapq import *
    from bisect import *
    from copy import *
    from math import *
    from random import *
    from statistics import *
    from itertools import *
    from functools import *
    from operator import *
    from io import *
    from sys import *
    from json import *
    from builtins import *
    from typing import *
    import string
    import re
    import datetime
    import collections
    import heapq
    import bisect
    import copy
    import math
    import random
    import statistics
    import itertools
    import functools
    import operator
    import io
    import sys
    import json
    sys.setrecursionlimit(50000)
""")

# Maximum memory limit for each test code execution.
_DEFAULT_MAX_MEMORY_BYTES: int = 4 * 1024**3  # 4 GB

_PLATFORM: str = sys.platform

if _PLATFORM.startswith("linux"):
    # Some dangerous syscalls
    # The code under test cannot call these syscalls.
    _DEFAULT_BLOCKED_SYSCALLS = frozenset({
        # File system
        "unlink", "rename", "mkdir", "rmdir", "chmod", "chown",
        # Process control
        "fork", "vfork", "clone", "kill", "setuid", "setgid",
        # Network
        "socket", "connect", "bind", "listen", "accept",
    })
else:
    # Unsupported platforms just get an empty set; no syscalls are blocked
    _DEFAULT_BLOCKED_SYSCALLS = frozenset()


def _get_output_file_path(output_dir: str, filename: str) -> str:
    """Constructs a file path within an output directory.

        Args:
            output_dir: The output directory path.
            filename: The filename to append.

        Returns:
            String representation of the full file path.
        """
    return str(pathlib.Path(output_dir) / filename)


class LIVE_CODE_BENCH_CODEGEN_PIPELINE(configs.ExperimentConfig):
    """Defines the pipeline for running the code generation benchmark.

    Pipeline stages:
        1. Prompt Creation: Loads dataset and creates prompts from template.
        2. Response Generation: Generates code responses using the model.
        3. Code Extraction: Extracts and processes code from model outputs.
        4. Code Evaluation: Executes code against test cases and computes
            metrics.
    """

    # Dataset configuration
    _HF_LCB_DATASET_NAME: str = "livecodebench/code_generation_lite"
    _HF_LCB_DATASET_SPLIT: str = "test"
    _HF_TRUST_REMOTE_CODE: bool = True

    _PROMPT_TEMPLATE_PATH: pathlib.Path = (
        pathlib.Path(__file__).parent.parent /
        "prompt_templates/live_code_bench_templates/codegen.jinja").resolve()

    # Standard file names
    _TRANSFORMED_DATA_FILE_NAME: str = "transformed_data.jsonl"
    _INFERENCE_RESULT_FILE_NAME: str = "inference_result.jsonl"
    _JSONL_FILE_FORMAT: str = ".jsonl"

    # Column names
    _CONTEST_DATE_COLUMN_NAME: str = "contest_date"
    _MODEL_OUTPUT_COLUMN_NAME: str = "model_output"
    _EXTRACTED_CODE_COLUMN_NAME: str = "extracted_code"
    _PUBLIC_TEST_CASES_COLUMN_NAME: str = "public_test_cases"
    _PRIVATE_TEST_CASES_COLUMN_NAME: str = "private_test_cases"
    _METADATA_COLUMN_NAME: str = "metadata"
    _ALL_TEST_CASES_COMBINED_COLUMN_NAME: str = "all_test_cases_combined"
    _DATAPOINT_ID_COLUMN_NAME: str = "data_point_id"

    # In the parameters below, we accept strings as well as the actual
    # types since the command line arguments are not parsed by main.py
    # and instead are passed as strings. The arguments are converted to the
    # appropriate types in the method body.
    def configure_pipeline(self,
                           model_config: configs.ModelConfig | None = None,
                           lcb_release_version: str = "release_latest",
                           lcb_start_datetime: str | None = None,
                           lcb_end_datetime: str | None = None,
                           sampler_random_seed: int | str = 42,
                           sample_count: int | str | None = None,
                           num_generated_responses_per_prompt: int | str = 5,
                           max_concurrent_inference_requests: int | str = 5,
                           closing_think_token: str = "",
                           code_evaluation_timeout_seconds: float | str = 20.0,
                           max_parallel_code_executions_per_attempt: int | str = 16,
                           max_memory_bytes: int | str = _DEFAULT_MAX_MEMORY_BYTES,
                           blocked_syscalls: str | frozenset[str] = _DEFAULT_BLOCKED_SYSCALLS,
                           additional_imports: str = _DEFAULT_ADDITIONAL_PYTHON_IMPORTS,
                           resume_from: str | None = None,
                           **kwargs: Any) -> configs.PipelineConfig:
        """Configures the steps of the pipeline.
        
        Args:
            model_config: The model configuration to use for generating code
                responses.
            lcb_release_version: The LiveCodeBench dataset release version to
                use. See the available versions at
                https://huggingface.co/datasets/livecodebench/code_generation_lite.
            lcb_start_datetime: The start datetime for filtering the
                LiveCodeBench dataset. Only include data points with
                contest_date >= lcb_start_datetime. Should be in the ISO 8601
                format, e.g., "2024-08-01T00:00:00". If None, do not apply a
                start date filter.
            lcb_end_datetime: The end datetime for filtering the LiveCodeBench
                dataset. Only include data points with contest_date <=
                lcb_end_datetime. Should be in the ISO 8601 format, e.g.,
                "2025-01-01T23:59:59". If None, do not apply an end date filter.
            sampler_random_seed: The random seed to use for sampling data points
                from the dataset for reproducibility.
            sample_count: The number of data points to sample from the dataset.
                This can be used for testing the pipeline with a smaller subset
                of the data.
            num_generated_responses_per_prompt: The number of code responses to
                generate per question. Higher numbers provide a better estimate
                of Pass@K metrics but increase computation cost.
            max_concurrent_inference_requests: The maximum number of concurrent
                inference requests to send to the model.
            closing_think_token: The token indicating the end of the model's
                reasoning process in the generated output. For example,
                Phi4-reasoning uses "</think>" as the closing think token. If
                set, only consider code snippets that appear
                after this token in the model output. Otherwise, look for the
                last code snippet in the entire model output.
            code_evaluation_timeout_seconds: The timeout in seconds for
                executing each test case.
            max_parallel_code_executions_per_attempt: The maximum number of code
                executions to run in parallel per code generation attempt.
            max_memory_bytes: The maximum memory in bytes that each code
                execution is allowed to use.
            blocked_syscalls: A set of system calls to block during code
                execution. If string, should be a comma-separated list of
                syscall names.
            additional_imports: Additional Python import statements to include
                at the start of the code being tested. This can be used to
                provide access to standard libraries that the code under test
                may require.
            resume_from: Path to the file where previous inference results are
                stored
        
        Returns:
            A PipelineConfig object defining the pipeline steps.
        """
        if model_config is None:
            raise ValueError("model_config must be provided.")
        
        num_generated_responses_per_prompt = int(
            num_generated_responses_per_prompt)
        if num_generated_responses_per_prompt < 1:
            raise ValueError(
                "num_generated_responses_per_prompt must be at least 1."
                f" Got {num_generated_responses_per_prompt}.")
        
        max_concurrent_inference_requests = int(
            max_concurrent_inference_requests)
        if max_concurrent_inference_requests <= 0:
            raise ValueError(
                "max_concurrent_inference_requests must be positive. "
                f"Got {max_concurrent_inference_requests}.")

        code_evaluation_timeout_seconds = float(
            code_evaluation_timeout_seconds)
        if code_evaluation_timeout_seconds <= 0:
            raise ValueError(
                "code_evaluation_timeout_seconds must be positive. "
                f"Got {code_evaluation_timeout_seconds}.")

        max_parallel_code_executions_per_attempt = int(
            max_parallel_code_executions_per_attempt)
        if max_parallel_code_executions_per_attempt <= 0:
            raise ValueError(
                "max_parallel_code_executions_per_attempt must be positive. "
                f"Got {max_parallel_code_executions_per_attempt}.")

        lcb_start_datetime_parsed: datetime.datetime | None = (
            datetime.datetime.fromisoformat(lcb_start_datetime)
            if lcb_start_datetime is not None else None)

        lcb_end_datetime_parsed: datetime.datetime | None = (
            datetime.datetime.fromisoformat(lcb_end_datetime)
            if lcb_end_datetime is not None else None)

        if lcb_start_datetime_parsed and lcb_end_datetime_parsed:
            if lcb_start_datetime_parsed > lcb_end_datetime_parsed:
                raise ValueError(
                    "lcb_start_datetime must be earlier than or equal to "
                    "lcb_end_datetime."
                    f" Got start: {lcb_start_datetime_parsed}, "
                    f"end: {lcb_end_datetime_parsed}.")

        sampler_random_seed = int(sampler_random_seed)

        if sample_count is not None:
            sample_count = int(sample_count)
            if sample_count <= 0:
                raise ValueError(
                    "sample_count must be positive. "
                    f"Got {sample_count}.")

        max_memory_bytes_int: int = int(max_memory_bytes)
        if max_memory_bytes_int <= 0:
            raise ValueError(
                "max_memory_bytes must be positive. "
                f"Got {max_memory_bytes_int}.")

        blocked_syscalls_set: frozenset[str] = frozenset(
            syscall.strip() for syscall in blocked_syscalls.split(",")
        ) if isinstance(blocked_syscalls, str) else blocked_syscalls

        self._prompt_creation = self._create_prompt_processing_config(
            lcb_release_version=lcb_release_version,
            lcb_start_datetime=lcb_start_datetime_parsed,
            lcb_end_datetime=lcb_end_datetime_parsed,
            sampler_random_seed=sampler_random_seed,
            sample_count=sample_count,
            num_generated_responses_per_prompt=(
                num_generated_responses_per_prompt),
        )

        self._response_generation = self._create_inference_config(
            prompts_filepath=_get_output_file_path(
                self._prompt_creation.output_dir,
                self._TRANSFORMED_DATA_FILE_NAME,
            ),
            model_config=model_config,
            max_concurrent_inference_requests=max_concurrent_inference_requests,
            resume_from=resume_from,
        )

        self._code_extraction = self._create_code_extraction_config(
            model_responses_filepath=_get_output_file_path(
                self._response_generation.output_dir,
                self._INFERENCE_RESULT_FILE_NAME,
            ),
            closing_think_token=closing_think_token,
        )

        self._code_evaluation = self._create_code_evaluation_config(
            extracted_code_filepath=_get_output_file_path(
                self._code_extraction.output_dir,
                self._TRANSFORMED_DATA_FILE_NAME,
            ),
            code_evaluation_timeout_seconds=code_evaluation_timeout_seconds,
            max_parallel_code_executions_per_attempt=(
                max_parallel_code_executions_per_attempt),
            max_memory_bytes=max_memory_bytes_int,
            blocked_syscalls=blocked_syscalls_set,
            additional_imports=additional_imports,
        )

        return configs.PipelineConfig(
            component_configs=[
                self._prompt_creation,
                self._response_generation,
                self._code_extraction,
                self._code_evaluation,
            ],
            log_dir=self.log_dir,
        )

    def _create_prompt_processing_config(
        self,
        lcb_release_version: str,
        lcb_start_datetime: datetime.datetime | None,
        lcb_end_datetime: datetime.datetime | None,
        sampler_random_seed: int,
        sample_count: int | None,
        num_generated_responses_per_prompt: int,
    ) -> configs.PromptProcessingConfig:
        """Creates the prompt processing configuration.

        Args:
            lcb_release_version: LiveCodeBench release version.
            lcb_start_date: Start date for filtering the LiveCodeBench dataset.
                Inclusive.
            lcb_end_date: End date for filtering the LiveCodeBench dataset.
                Inclusive.
            sampler_random_seed: The random seed to use for sampling data points
                from the dataset for reproducibility.
            sample_count: The number of data points to sample from the dataset.
                This can be used for testing the pipeline with a smaller subset
                of the data.
            num_generated_responses_per_prompt: Number of responses per prompt.

        Returns:
            PromptProcessingConfig for the prompt creation stage.
        """
        transforms: list[data_utils.DFTransformBase] = [
            data_utils.ApplyFunctionToColumnTransform(
                src_column_name=self._CONTEST_DATE_COLUMN_NAME,
                dst_column_name=self._CONTEST_DATE_COLUMN_NAME,
                function=lambda x: datetime.datetime.fromisoformat(x),
            ),
            data_utils.FilterColumnToRangeTransform(
                column_name=self._CONTEST_DATE_COLUMN_NAME,
                start=lcb_start_datetime,
                end=lcb_end_datetime,
            ),
            # Have to convert back to a string since the output of the step must
            # be JSON serializable and a timestamp is not.
            data_utils.ApplyFunctionToColumnTransform(
                src_column_name=self._CONTEST_DATE_COLUMN_NAME,
                dst_column_name=self._CONTEST_DATE_COLUMN_NAME,
                function=lambda x: x.isoformat(),
            ),
        ]

        if sample_count is not None:
            transforms.append(
                data_utils.SamplerTransform(
                    random_seed=sampler_random_seed,
                    sample_count=sample_count,
                )
            )

        transforms.extend([
            decode_test_cases_transform.DecodeTestCasesTransform(
                encoded_test_cases_column_name=(
                    self._PRIVATE_TEST_CASES_COLUMN_NAME),
                decoded_test_cases_column_name=(
                    self._PRIVATE_TEST_CASES_COLUMN_NAME)),
            data_utils.ConvertStrColumnToJsonTransform(
                # private_test_cases_column_name is already
                # decoded by DecodeTestCasesTransform into a
                # JSON object, so we only need to convert the other
                # columns.
                columns=[
                    self._METADATA_COLUMN_NAME,
                    self._PUBLIC_TEST_CASES_COLUMN_NAME
                ]),
            data_utils.ConcatColumnsToSingleColumnTransform(
                # Combines the public and private test cases into
                # a single column as code evaluation does not
                # distinguish between them.
                columns=[
                    self._PUBLIC_TEST_CASES_COLUMN_NAME,
                    self._PRIVATE_TEST_CASES_COLUMN_NAME
                ],
                new_column=self._ALL_TEST_CASES_COMBINED_COLUMN_NAME
            ),
            data_utils.MultiplyTransform(
                # This is to generate multiple responses per prompt.
                n_repeats=num_generated_responses_per_prompt,
            ),
        ])

        return configs.PromptProcessingConfig(
            component_type=core.PromptProcessing,
            prompt_template_path=str(self._PROMPT_TEMPLATE_PATH),
            data_reader_config=configs.DataSetConfig(
                class_name=data_utils.HFDataReader,
                init_args={
                    "path": self._HF_LCB_DATASET_NAME,
                    "split": self._HF_LCB_DATASET_SPLIT,
                    "release_version": lcb_release_version,
                    "trust_remote_code": self._HF_TRUST_REMOTE_CODE,
                    "transform": data_utils.SequenceTransform(transforms)
                }),
            output_dir=self._construct_output_dir_path("prompts"),
        )

    def _create_inference_config(
            self,
            prompts_filepath: str,
            model_config: configs.ModelConfig,
            max_concurrent_inference_requests: int = 5,
            resume_from: str | None = None) -> configs.InferenceConfig:
        """Constructs the inference configuration.

        Args:
            prompts_filepath: The file path to the prompts.
            model_config: The model configuration to use.
            resume_from: Path to the file where previous inference results are
                stored

        Returns:
            InferenceConfig for the response generation stage.
        """
        return configs.InferenceConfig(
            component_type=core.Inference,
            model_config=model_config,
            data_loader_config=configs.DataSetConfig(
                class_name=data_utils.DataLoader,
                init_args={
                    "path": prompts_filepath,
                }),
            resume_from=resume_from,  # type: ignore
            max_concurrent=max_concurrent_inference_requests,
            output_dir=self._construct_output_dir_path("responses"),
        )

    def _create_code_extraction_config(
            self,
            model_responses_filepath: str,
            closing_think_token: str) -> configs.DataProcessingConfig:
        """Creates the code extraction configuration.

        Args:
            model_responses_filepath: The file path to the model responses.
            closing_think_token: The token indicating the end of the model's
                reasoning process in the generated output. If empty,
                looks for the last code snippet in the entire model output.

        Returns:
            DataProcessingConfig for the code extraction stage.
        """
        return configs.DataProcessingConfig(
            component_type=core.DataProcessing,
            data_reader_config=configs.DataSetConfig(
                class_name=data_utils.DataReader,
                init_args={
                    "path": model_responses_filepath,
                    "format": self._JSONL_FILE_FORMAT,
                    "transform": data_utils.SequenceTransform([
                        code_extraction_transform.CodeExtractionTransform(
                            model_output_column=self._MODEL_OUTPUT_COLUMN_NAME,
                            code_column=self._EXTRACTED_CODE_COLUMN_NAME,
                            closing_think_token=closing_think_token,
                        ),
                    ])
                }),
            output_dir=self._construct_output_dir_path("extracted_code"))

    def _create_code_evaluation_config(
        self,
        extracted_code_filepath: str,
        code_evaluation_timeout_seconds: float,
        max_parallel_code_executions_per_attempt: int,
        max_memory_bytes: int,
        blocked_syscalls: frozenset[str],
        additional_imports: str,
    ) -> configs.EvalReportingConfig:
        """Creates the code evaluation configuration.

        Args:
            extracted_code_filepath: The file path to the extracted code.
            code_evaluation_timeout_seconds: The timeout in seconds for
                executing each test case.
            max_parallel_code_executions_per_attempt: The maximum number of code
                executions to run in parallel per code generation attempt.
            max_memory_bytes: The maximum memory in bytes that each code
                execution is allowed to use.
            blocked_syscalls: A set of system calls to block during code
                evaluation.
            additional_imports: Additional Python import statements to include
                at the start of the code being tested.

        Returns:
            EvalReportingConfig for the code evaluation stage.
        """
        return configs.EvalReportingConfig(
            component_type=eval_reporting.EvalReporting,
            data_reader_config=configs.DataSetConfig(
                class_name=data_utils.DataReader,
                init_args={
                    "path": extracted_code_filepath,
                    "format": self._JSONL_FILE_FORMAT,
                }),
            metric_config=config.MetricConfig(
                class_name=codegen_test_case_results_metric.
                CodegenTestCaseResultsMetric,
                init_args={
                    "code_column_name": self._EXTRACTED_CODE_COLUMN_NAME,
                    "test_cases_column_name": (
                        self._ALL_TEST_CASES_COMBINED_COLUMN_NAME),
                    "metadata_column_name": self._METADATA_COLUMN_NAME,
                    "timeout": datetime.timedelta(
                        seconds=code_evaluation_timeout_seconds),
                    "max_workers": max_parallel_code_executions_per_attempt,
                    "sandbox_cfg": sandbox_config.SandboxConfig(
                        max_memory_bytes=max_memory_bytes,
                        blocked_syscalls=blocked_syscalls,
                    ),
                    "additional_imports": additional_imports,
                }),
            aggregator_configs=[
                config.AggregatorConfig(
                    class_name=pass_at_k_aggregator.PassAtKAggregator,
                    init_args={
                        "passed_column_name": (
                            "CodegenTestCaseResultsMetric_all_passed"),
                        "k": 1,
                        "group_by": self._DATAPOINT_ID_COLUMN_NAME,
                        "filename_base": "Pass@1_by_question",
                    }),
            ],
            output_dir=self._construct_output_dir_path("test_case_results"),
        )

    def _construct_output_dir_path(self, *parts: str) -> str:
        """Constructs a path relative to the log directory.

        Args:
            parts: Path components to append to the log directory.

        Returns:
            String representation of the constructed path.
        """
        return str(pathlib.Path(self.log_dir).joinpath(*parts))
