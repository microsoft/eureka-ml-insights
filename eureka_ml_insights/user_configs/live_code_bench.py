"""Defines the pipeline for running LiveCodeBench.

See https://livecodebench.github.io/ for details.

Typical usage example (from the project's root directory):
    $ python main.py \
        --exp_config="LIVE_CODE_BENCH_CODEGEN_PIPELINE" \
        --model_config="GATEWAY_GPT4O_CONFIG"
"""

import pathlib
import datetime

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
)


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

    _PROMPT_TEMPLATE_PATH: pathlib.Path = (
        pathlib.Path(__file__).parent.parent /
        "prompt_templates/live_code_bench_templates/codegen.jinja").resolve()

    # Standard file names
    _TRANSFORMED_DATA_FILE_NAME: str = "transformed_data.jsonl"
    _INFERENCE_RESULT_FILE_NAME: str = "inference_result.jsonl"
    _JSONL_FILE_FORMAT: str = ".jsonl"

    # Column names
    _MODEL_OUTPUT_COLUMN_NAME: str = "model_output"
    _EXTRACTED_CODE_COLUMN_NAME: str = "extracted_code"
    _PUBLIC_TEST_CASES_COLUMN_NAME: str = "public_test_cases"
    _PRIVATE_TEST_CASES_COLUMN_NAME: str = "private_test_cases"
    _METADATA_COLUMN_NAME: str = "metadata"
    _ALL_TEST_CASES_COMBINED_COLUMN_NAME: str = "all_test_cases_combined"
    _DATAPOINT_ID_COLUMN_NAME: str = "data_point_id"

    def configure_pipeline(self,
                           model_config: configs.ModelConfig | None = None,
                           lcb_release_version: str = "release_latest",
                           lcb_start_datetime: datetime.datetime | None = None,
                           lcb_end_datetime: datetime.datetime | None = None,
                           num_generated_responses_per_prompt: int = 5,
                           closing_think_token: str = "",
                           code_evaluation_timeout_seconds: float = 20.0,
                           max_parallel_code_executions_per_attempt: int = 16,
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
                contest_date >= lcb_start_datetime. If None, do not apply a
                start date filter.
            lcb_end_datetime: The end datetime for filtering the LiveCodeBench
                dataset. Only include data points with contest_date <=
                lcb_end_datetime. If None, do not apply an end date filter
            num_generated_responses_per_prompt: The number of code responses to
                generate per question. Higher numbers provide a better estimate
                of Pass@K metrics but increase computation cost.
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
        
        Returns:
            A PipelineConfig object defining the pipeline steps.
        """
        if model_config is None:
            raise ValueError("model_config must be provided.")

        if num_generated_responses_per_prompt < 1:
            raise ValueError(
                "num_generated_responses_per_prompt must be at least 1."
                f" Got {num_generated_responses_per_prompt}.")

        if code_evaluation_timeout_seconds <= 0:
            raise ValueError(
                "code_evaluation_timeout_seconds must be positive. "
                f"Got {code_evaluation_timeout_seconds}.")

        if max_parallel_code_executions_per_attempt <= 0:
            raise ValueError(
                "max_parallel_code_executions_per_attempt must be positive. "
                f"Got {max_parallel_code_executions_per_attempt}.")

        self._prompt_creation = self._create_prompt_processing_config(
            lcb_release_version=lcb_release_version,
            lcb_start_datetime=lcb_start_datetime,
            lcb_end_datetime=lcb_end_datetime,
            num_generated_responses_per_prompt=(
                num_generated_responses_per_prompt),
        )

        self._response_generation = self._create_inference_config(
            model_config=model_config)

        self._code_extraction = self._create_code_extraction_config(
            closing_think_token=closing_think_token)

        self._code_evaluation = self._create_code_evaluation_config(
            code_evaluation_timeout_seconds=code_evaluation_timeout_seconds,
            max_parallel_code_executions_per_attempt=(
                max_parallel_code_executions_per_attempt),
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
        num_generated_responses_per_prompt: int,
    ) -> configs.PromptProcessingConfig:
        """Creates the prompt processing configuration.

        Args:
            lcb_release_version: LiveCodeBench release version.
            lcb_start_date: Start date for filtering the LiveCodeBench dataset.
                Inclusive.
            lcb_end_date: End date for filtering the LiveCodeBench dataset.
                Inclusive.
            num_generated_responses_per_prompt: Number of responses per prompt.

        Returns:
            PromptProcessingConfig for the prompt creation stage.
        """
        return configs.PromptProcessingConfig(
            component_type=core.PromptProcessing,
            prompt_template_path=str(self._PROMPT_TEMPLATE_PATH),
            data_reader_config=configs.DataSetConfig(
                class_name=data_utils.HFDataReader,
                init_args={
                    "path": self._HF_LCB_DATASET_NAME,
                    "split": self._HF_LCB_DATASET_SPLIT,
                    "release_version": lcb_release_version,
                    "transform": data_utils.SequenceTransform([
                        data_utils.FilterDatetimeColumnToRangeTransform(
                            column="contest_date",
                            start_datetime=lcb_start_datetime,
                            end_datetime=lcb_end_datetime,
                        ),
                        # TODO: Remove SamplerTransform when testing is done.
                        data_utils.SamplerTransform(sample_count=2,
                                                    random_seed=42),
                        decode_test_cases_transform.DecodeTestCasesTransform(
                            encoded_test_cases_column_name=(
                                self._PRIVATE_TEST_CASES_COLUMN_NAME),
                            decoded_test_cases_column_name=(
                                self._PRIVATE_TEST_CASES_COLUMN_NAME)),
                        data_utils.StrToJsonTransform(
                            # private_test_cases_column_name is already
                            # decoded by DecodeTestCasesTransform into a
                            # JSON object, so we only need to convert the other
                            # columns.
                            columns=[
                                self._METADATA_COLUMN_NAME,
                                self._PUBLIC_TEST_CASES_COLUMN_NAME
                            ]),
                        data_utils.AddColumnValuesTransform(
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
                }),
            output_dir=self._construct_output_dir_path("prompts"),
        )

    def _create_inference_config(
            self,
            model_config: configs.ModelConfig) -> configs.InferenceConfig:
        """Constructs the inference configuration.

        Args:
            model_config: The model configuration to use.

        Returns:
            InferenceConfig for the response generation stage.
        """
        return configs.InferenceConfig(
            component_type=core.Inference,
            model_config=model_config,
            data_loader_config=configs.DataSetConfig(
                class_name=data_utils.DataLoader,
                init_args={
                    "path": _get_output_file_path(
                        self._prompt_creation.output_dir,
                        self._TRANSFORMED_DATA_FILE_NAME,
                    ),
                }),
            output_dir=self._construct_output_dir_path("responses"),
        )

    def _create_code_extraction_config(
            self, closing_think_token: str) -> configs.DataProcessingConfig:
        """Creates the code extraction configuration.

        Args:
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
                    "path": _get_output_file_path(
                        self._response_generation.output_dir,
                        self._INFERENCE_RESULT_FILE_NAME,
                    ),
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
        self, code_evaluation_timeout_seconds: float,
        max_parallel_code_executions_per_attempt: int
    ) -> configs.EvalReportingConfig:
        """Creates the code evaluation configuration.

        Args:
            code_evaluation_timeout_seconds: The timeout in seconds for
                executing each test case.
            max_parallel_code_executions_per_attempt: The maximum number of code
                executions to run in parallel per code generation attempt.

        Returns:
            EvalReportingConfig for the code evaluation stage.
        """
        return configs.EvalReportingConfig(
            component_type=eval_reporting.EvalReporting,
            data_reader_config=configs.DataSetConfig(
                class_name=data_utils.DataReader,
                init_args={
                    "path": _get_output_file_path(
                        self._code_extraction.output_dir,
                        self._TRANSFORMED_DATA_FILE_NAME,
                    ),
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
