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
import dataclasses
import pathlib
import textwrap

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
)
from eureka_ml_insights.core.job_runner.sandboxing import preexec_fn_sandboxing
from eureka_ml_insights.core.job_runner.command_runners import base as command_runners_base
from eureka_ml_insights.core.job_runner.command_runners import subprocess_runner
from eureka_ml_insights.metrics import reports


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


@dataclasses.dataclass(frozen=True)
class PipelineParams:
    """Parameters for configuring the LiveCodeBench code generation pipeline."""
    n_repeats: int
    code_evaluation_runner: command_runners_base.CommandRunner
    max_concurrent: int
    code_eval_timeout: float
    max_parallel_executions: int
    sampler_seed: int
    sample_count: int | None
    start_dt: datetime.datetime | None
    end_dt: datetime.datetime | None

    @classmethod
    def from_args(
        cls,
        *,
        n_repeats: int | str,
        code_evaluation_runner_name: str,
        max_concurrent: int | str,
        code_evaluation_timeout_seconds: float | str,
        max_parallel_code_executions_per_attempt: int | str,
        sampler_random_seed: int | str,
        sample_count: int | str | None,
        lcb_start_datetime: str | None,
        lcb_end_datetime: str | None,
    ):
        """Creates PipelineParams from command line arguments and validates them."""
        return cls(
            n_repeats=_validate_positive_int(n_repeats,
                                             "n_repeats"),
            code_evaluation_runner=_construct_runner(
                code_evaluation_runner_name
            ),
            max_concurrent=_validate_positive_int(max_concurrent,
                                                  "max_concurrent"),
            code_eval_timeout=_validate_positive_float(
                code_evaluation_timeout_seconds,
                "code_evaluation_timeout_seconds"
            ),
            max_parallel_executions=_validate_positive_int(
                max_parallel_code_executions_per_attempt,
                "max_parallel_code_executions_per_attempt",
            ),
            sampler_seed=int(sampler_random_seed),
            sample_count=(
                None
                if sample_count is None
                else _validate_positive_int(sample_count, "sample_count")
            ),
            start_dt=_parse_iso_datetime(lcb_start_datetime,
                                         "lcb_start_datetime"),
            end_dt=_parse_iso_datetime(lcb_end_datetime,
                                       "lcb_end_datetime"),
        )


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
    _JSONL_FILE_EXT: str = ".jsonl"

    # Column names
    _CONTEST_DATE_COLUMN_NAME: str = "contest_date"
    _MODEL_OUTPUT_COLUMN_NAME: str = "model_output"
    _EXTRACTED_CODE_COLUMN_NAME: str = "extracted_code"
    _PUBLIC_TEST_CASES_COLUMN_NAME: str = "public_test_cases"
    _PRIVATE_TEST_CASES_COLUMN_NAME: str = "private_test_cases"
    _METADATA_COLUMN_NAME: str = "metadata"
    _ALL_TEST_CASES_COMBINED_COLUMN_NAME: str = "all_test_cases_combined"
    _DIFFICULTY_LEVEL_COLUMN_NAME: str = "difficulty"
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
                           sampler_seed: int | str = 42,
                           sample_count: int | str | None = None,
                           n_repeats: int | str = 5,
                           max_concurrent: int | str = 5,
                           closing_think_token: str = "",
                           code_evaluation_runner_name: str = "safe_subprocess",
                           code_evaluation_timeout_seconds: float | str = 20.0,
                           max_parallel_code_executions_per_attempt: int | str = 16,
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
            sampler_seed: The random seed to use for sampling data points
                from the dataset for reproducibility.
            sample_count: The number of data points to sample from the dataset.
                This can be used for testing the pipeline with a smaller subset
                of the data.
            n_repeats: The number of code responses to
                generate per question. Higher numbers provide a better estimate
                of Pass@K metrics but increase computation cost.
            max_concurrent: The maximum number of concurrent
                inference requests to send to the model.
            closing_think_token: The token indicating the end of the model's
                reasoning process in the generated output. For example,
                Phi4-reasoning uses "</think>" as the closing think token. If
                set, only consider code snippets that appear
                after this token in the model output. Otherwise, look for the
                last code snippet in the entire model output.
            code_evaluation_runner_name: The name of the command runner to use
                for executing the generated code during evaluation. For options,
                see the `_construct_runner()` function.
            code_evaluation_timeout_seconds: The timeout in seconds for
                executing each test case.
            max_parallel_code_executions_per_attempt: The maximum number of code
                executions to run in parallel per code generation attempt.
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
        
        params = PipelineParams.from_args(
            n_repeats=n_repeats,
            code_evaluation_runner_name=code_evaluation_runner_name,
            max_concurrent=max_concurrent,
            code_evaluation_timeout_seconds=(
                code_evaluation_timeout_seconds),
            max_parallel_code_executions_per_attempt=(
                max_parallel_code_executions_per_attempt),
            sampler_random_seed=sampler_seed,
            sample_count=sample_count,
            lcb_start_datetime=lcb_start_datetime,
            lcb_end_datetime=lcb_end_datetime,
        )

        _validate_datetime_range(
            params.start_dt,
            params.end_dt,
            "lcb_start_datetime",
            "lcb_end_datetime",
        )

        self._prompt_creation = self._create_prompt_processing_config(
            lcb_release_version=lcb_release_version,
            lcb_start_datetime=params.start_dt,
            lcb_end_datetime=params.end_dt,
            sampler_random_seed=params.sampler_seed,
            sample_count=params.sample_count,
            n_repeats=params.n_repeats,
        )

        self._response_generation = self._create_inference_config(
            prompts_filepath=_get_output_file_path(
                self._prompt_creation.output_dir,
                self._TRANSFORMED_DATA_FILE_NAME,
            ),
            model_config=model_config,
            max_concurrent_inference_requests=params.max_concurrent,
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
            code_evaluation_runner=params.code_evaluation_runner,
            code_evaluation_timeout_seconds=params.code_eval_timeout,
            max_parallel_code_executions_per_attempt=(
                params.max_parallel_executions),
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
        n_repeats: int,
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
            n_repeats: Number of responses per prompt.

        Returns:
            PromptProcessingConfig for the prompt creation stage.
        """
        transforms: list[data_utils.DFTransformBase] = [
            data_utils.RunPythonTransform(
                python_code=textwrap.dedent(f"""
                   df["{self._CONTEST_DATE_COLUMN_NAME}"] = pd.to_datetime(
                       df["{self._CONTEST_DATE_COLUMN_NAME}"])
                """)
            ),
            data_utils.FilterColumnToRangeTransform(
                column_name=self._CONTEST_DATE_COLUMN_NAME,
                start=lcb_start_datetime,
                end=lcb_end_datetime,
            ),
            # Have to convert back to a string since the output of the step must
            # be JSON serializable and a timestamp is not.
            data_utils.RunPythonTransform(
                python_code=textwrap.dedent(f"""
                   df["{self._CONTEST_DATE_COLUMN_NAME}"] = (
                       df["{self._CONTEST_DATE_COLUMN_NAME}"]
                       .dt.strftime("%Y-%m-%dT%H:%M:%S"))
                """)
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
            data_utils.RunPythonTransform(
                python_code=textwrap.dedent(f"""
                    df[[
                        "{self._METADATA_COLUMN_NAME}",
                        "{self._PUBLIC_TEST_CASES_COLUMN_NAME}"]] = (
                        df[[
                            "{self._METADATA_COLUMN_NAME}",
                            "{self._PUBLIC_TEST_CASES_COLUMN_NAME}"]]
                        .map(json.loads))
                """)
            ),
            data_utils.RunPythonTransform(
                # Combines the public and private test cases into
                # a single column as code evaluation does not
                # distinguish between them.
                python_code=textwrap.dedent(f"""
                    df["{self._ALL_TEST_CASES_COMBINED_COLUMN_NAME}"] = (
                        df["{self._PUBLIC_TEST_CASES_COLUMN_NAME}"] +
                        df["{self._PRIVATE_TEST_CASES_COLUMN_NAME}"]
                    )
                """)
            ),
            data_utils.DropColumnsTransform(
                # Remove columns that are not needed for prompt generation
                # or later stages. This helps reduce data size and I/O
                # time.
                columns=[
                    self._PUBLIC_TEST_CASES_COLUMN_NAME,
                    self._PRIVATE_TEST_CASES_COLUMN_NAME,
                ]
            ),
            data_utils.MultiplyTransform(
                # This is to generate multiple responses per prompt.
                n_repeats=n_repeats,
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
            output_dir=self._construct_output_dir_path(
                "data_processing_output"),
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
            output_dir=self._construct_output_dir_path("inference_result"),
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
                    "format": self._JSONL_FILE_EXT,
                    "transform": data_utils.SequenceTransform([
                        code_extraction_transform.CodeExtractionTransform(
                            model_output_column=self._MODEL_OUTPUT_COLUMN_NAME,
                            code_column=self._EXTRACTED_CODE_COLUMN_NAME,
                            closing_think_token=closing_think_token,
                        ),
                    ])
                }),
            output_dir=self._construct_output_dir_path(
                "code_extraction_processing_output"))

    def _create_code_evaluation_config(
        self,
        extracted_code_filepath: str,
        code_evaluation_runner: command_runners_base.CommandRunner,
        code_evaluation_timeout_seconds: float,
        max_parallel_code_executions_per_attempt: int,
        additional_imports: str,
    ) -> configs.EvalReportingConfig:
        """Creates the code evaluation configuration.

        Args:
            extracted_code_filepath: The file path to the extracted code.
            code_evaluation_timeout_seconds: The timeout in seconds for
                executing each test case.
            max_parallel_code_executions_per_attempt: The maximum number of code
                executions to run in parallel per code generation attempt.
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
                    "format": self._JSONL_FILE_EXT,
                }),
            metric_config=config.MetricConfig(
                class_name=codegen_test_case_results_metric.
                CodegenTestCaseResultsMetric,
                init_args={
                    "code_column_name": self._EXTRACTED_CODE_COLUMN_NAME,
                    "test_cases_column_name": (
                        self._ALL_TEST_CASES_COMBINED_COLUMN_NAME),
                    "metadata_column_name": self._METADATA_COLUMN_NAME,
                    "runner": code_evaluation_runner,
                    "timeout": datetime.timedelta(
                        seconds=code_evaluation_timeout_seconds),
                    "max_workers": max_parallel_code_executions_per_attempt,
                    "additional_imports": additional_imports,
                }),
            aggregator_configs=[
                # Calculates the pass@1 by question across all attempts.
                config.AggregatorConfig(
                    class_name=reports.AverageAggregator,
                    init_args={
                        "column_names": [
                            "CodegenTestCaseResultsMetric_all_passed",
                        ],
                        "group_by": self._DATAPOINT_ID_COLUMN_NAME,
                        "filename_base": "Pass@1_ByQuestion",
                    }
                ),
                # Calculates the pass@1 by question across all attempts and then
                # averages them to get the overall pass@1.
                config.AggregatorConfig(
                    class_name=reports.BiLevelAggregator,
                    init_args={
                        "column_names": [
                            "CodegenTestCaseResultsMetric_all_passed",
                        ],
                        "first_groupby": self._DATAPOINT_ID_COLUMN_NAME,
                        "agg_fn": "mean",
                        "filename_base": "Pass@1_Overall",
                    }
                ),
                # Calculates the pass@1 by question across all attempts and then
                # averages them by difficulty level to get pass@1 by difficulty.
                config.AggregatorConfig(
                    class_name=reports.BiLevelAggregator,
                    init_args={
                        "column_names": [
                            "CodegenTestCaseResultsMetric_all_passed",
                        ],
                        "first_groupby": self._DATAPOINT_ID_COLUMN_NAME,
                        "second_groupby": self._DIFFICULTY_LEVEL_COLUMN_NAME,
                        "agg_fn": "mean",
                        "filename_base": "Pass@1_ByDifficulty",
                    }
                ),
            ],
            output_dir=self._construct_output_dir_path("eval_report"),
        )

    def _construct_output_dir_path(self, *parts: str) -> str:
        """Constructs a path relative to the log directory.

        Args:
            parts: Path components to append to the log directory.

        Returns:
            String representation of the constructed path.
        """
        return str(pathlib.Path(self.log_dir).joinpath(*parts))


def _get_output_file_path(output_dir: str, filename: str) -> str:
    """Constructs a file path within an output directory.

        Args:
            output_dir: The output directory path.
            filename: The filename to append.

        Returns:
            String representation of the full file path.
        """
    return str(pathlib.Path(output_dir) / filename)


def _validate_positive_int(value: int | str, name: str) -> int:
    """Converts to int and ensures the value is positive."""
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer. Got {value!r}.")
    if value <= 0:
        raise ValueError(f"{name} must be positive. Got {value}.")
    return value


def _validate_positive_float(value: float | str, name: str) -> float:
    """Converts to float and ensures the value is positive."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a float. Got {value!r}.")
    if value <= 0:
        raise ValueError(f"{name} must be positive. Got {value}.")
    return value


def _parse_iso_datetime(
        dt_str: str | None, name: str) -> datetime.datetime | None:
    """Parses an ISO 8601 datetime string if provided."""
    if dt_str is None:
        return None
    try:
        return datetime.datetime.fromisoformat(dt_str)
    except ValueError:
        raise ValueError(
            f"{name} must be a valid ISO 8601 datetime string. Got {dt_str!r}.")


def _validate_datetime_range(
    start: datetime.datetime | None,
    end: datetime.datetime | None,
    start_name: str,
    end_name: str
) -> None:
    """Ensures start <= end if both are provided."""
    if start and end and start > end:
        raise ValueError(
            f"{start_name} must be earlier than or equal to {end_name}. "
            f"Got start={start}, end={end}."
        )


def _construct_runner(runner_name: str,) -> command_runners_base.CommandRunner:
    """Constructs a code evaluation runner with optional sandboxing.

    Args:
        runner_name: The name of the runner to construct. Options are:
            - "safe_subprocess": Uses a subprocess runner with basic syscall
              sandboxing (Linux only). The processes also runs slower due to the
              sandboxing.
            - "unsafe": Uses a standard subprocess runner without sandboxing.

    Returns:
        A configured CommandRunner.
    """
    if runner_name == "safe_subprocess":
        # NOTE: This only works on linux systems and is not strict security.
        # The code evaluation also runs slower when this is used. The ideal
        # set up is running the entire pipeline within a container and using
        # the "unsafe" runner.
        sandbox_config = preexec_fn_sandboxing.SandboxConfig(
            max_memory_bytes=4 * 1024**3,  # 4 GB
            blocked_syscalls={"fork", "vfork", "clone", "kill",
                              "unlink", "socket", "connect",
                              "bind", "listen"}
        )
        return subprocess_runner.SubprocessCommandRunner(
            preexec_fn=sandbox_config.to_preexec_fn()
        )
    elif runner_name == "unsafe":
        return subprocess_runner.SubprocessCommandRunner()
    else:
        raise ValueError(f"Unknown runner name: {runner_name}")
