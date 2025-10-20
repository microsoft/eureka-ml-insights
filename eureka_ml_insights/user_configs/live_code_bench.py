"""Defines the pipeline for running LiveCodeBench.

See https://livecodebench.github.io/ for details.

Typical usage example (from the project's root directory):
    $ python main.py \
        --exp_config="LIVE_CODE_BENCH_CODEGEN_PIPELINE" \
        --model_config="GATEWAY_GPT4O_CONFIG"
"""

import pathlib
import datetime

from eureka_ml_insights import configs, core, data_utils
from eureka_ml_insights.configs import config
from eureka_ml_insights.core import eval_reporting
from eureka_ml_insights.metrics import reports
from eureka_ml_insights.data_utils.live_code_bench import (
    code_extraction_transform,
    decode_test_cases_transform,
)
from eureka_ml_insights.metrics.live_code_bench import (
    codegen_test_case_results_metric,
    pass_at_k_aggregator,
)
from typing import Any


class LIVE_CODE_BENCH_CODEGEN_PIPELINE(configs.ExperimentConfig):
    """Defines the pipeline for running the code generation benchmark."""

    _HF_LCB_DATASET_NAME: str = "livecodebench/code_generation_lite"
    _HF_LCB_DATASET_SPLIT: str = "test"
    _HF_LCB_RELEASE_VERSION: str = "release_v5"
    _PROMPT_TEMPLATE_PATH: pathlib.Path = (
        pathlib.Path(__file__).parent /
        "../prompt_templates/live_code_bench_templates/codegen.jinja"
    ).resolve()

    def configure_pipeline(self,
                           model_config: configs.ModelConfig | None = None,
                           **kwargs: dict[str, Any]) -> configs.PipelineConfig:
        """Configures the steps of the pipeline.
        
        Args:
            TODO(luizdovalle): Add args.
        
        Returns:
            A PipelineConfig object defining the pipeline steps.
        """
        self._prompt_creation = configs.PromptProcessingConfig(
            component_type=core.PromptProcessing,
            prompt_template_path=str(self._PROMPT_TEMPLATE_PATH),
            data_reader_config=configs.DataSetConfig(
                class_name=data_utils.HFDataReader,
                init_args={
                    "path": self._HF_LCB_DATASET_NAME,
                    "split": self._HF_LCB_DATASET_SPLIT,
                    "release_version": self._HF_LCB_RELEASE_VERSION,
                    "transform": data_utils.SequenceTransform([
                        data_utils.SamplerTransform(
                            sample_count=2,
                            random_seed=42
                        ),
                        data_utils.MultiplyTransform(
                            n_repeats=3,
                        ),
                    ])
                }),
            output_dir=str(pathlib.Path(self.log_dir) / "prompts"),
        )

        self._response_generation = configs.InferenceConfig(
            component_type=core.Inference,
            model_config=model_config,
            data_loader_config=configs.DataSetConfig(
                class_name=data_utils.DataLoader,
                init_args={
                    "path": str(
                        pathlib.Path(self._prompt_creation.output_dir) /
                        "transformed_data.jsonl"
                    ),
                }
            ),
            output_dir=str(pathlib.Path(self.log_dir) / "responses"),
        )

        self._code_extraction = configs.DataProcessingConfig(
            component_type=core.DataProcessing,
            data_reader_config=configs.DataSetConfig(
                class_name=data_utils.DataReader,
                init_args={
                    "path": str(
                        pathlib.Path(self._response_generation.output_dir) /
                        "inference_result.jsonl"
                    ),
                    "format": ".jsonl",
                    "transform": data_utils.SequenceTransform([
                        code_extraction_transform.CodeExtractionTransform(
                            model_output_column="model_output",
                            code_column="extracted_code",
                            closing_think_token="",
                        ),
                        decode_test_cases_transform.DecodeTestCasesTransform(
                            encoded_test_cases_column_name="private_test_cases",
                            decoded_test_cases_column_name=(
                                "private_test_cases"
                            )
                        ),
                        data_utils.StrToJsonTransform(
                            # private_test_cases_column_name is already
                            # decoded by DecodeTestCasesTransform into a
                            # JSON object, so we only need to convert the other
                            # columns.
                            columns=["metadata", "public_test_cases"]
                        ),
                        data_utils.AddColumnValuesTransform(
                            columns=["public_test_cases", "private_test_cases"],
                            new_column="all_test_cases_combined"
                        )
                    ])
                }
            ),
            output_dir=str(pathlib.Path(self.log_dir) / "extracted_code")
        )

        self._grade_code = configs.EvalReportingConfig(
            component_type=eval_reporting.EvalReporting,
            data_reader_config=configs.DataSetConfig(
                class_name=data_utils.DataReader,
                init_args={
                    "path": str(
                        pathlib.Path(self._code_extraction.output_dir) /
                        "transformed_data.jsonl"
                    ),
                    "format": ".jsonl",
                }
            ),
            metric_config=config.MetricConfig(
                class_name=codegen_test_case_results_metric.CodegenTestCaseResultsMetric,
                init_args={
                    "code_column_name": "extracted_code",
                    "test_cases_column_name": "all_test_cases_combined",
                    "metadata_column_name": "metadata",
                    "timeout": datetime.timedelta(seconds=20),
                    "max_workers": 16,
                }
            ),
            aggregator_configs=[
               config.AggregatorConfig(
                   class_name=pass_at_k_aggregator.PassAtKAggregator,
                   init_args={
                       "passed_column_name": (
                           "CodegenTestCaseResultsMetric_all_passed"),
                       "k": 1,
                       "group_by": "data_point_id",
                       "filename_base": "Pass@1_by_question",
                   }
               ),
            ],
            output_dir=str(pathlib.Path(self.log_dir) / "test_case_results"),
        )

        return configs.PipelineConfig(
            component_configs=[
                self._prompt_creation,
                self._response_generation,
                self._code_extraction,
                self._grade_code,
            ],
            log_dir=self.log_dir,
        )
