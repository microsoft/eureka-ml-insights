import os
from typing import Any

from eureka_ml_insights.core import (
    DataProcessing,
    EvalReporting,
    Inference,
    PromptProcessing,
)
from eureka_ml_insights.data_utils import (
    AddColumn,
    ColumnRename,
    DataReader,
    HFDataReader,
    MMDataLoader,
    SamplerTransform,
    SequenceTransform,
)
from eureka_ml_insights.data_utils.toxigen_utils import (
    GPT4ToxiGenRegex,
    GPT4ToxiGenRegexGenerative,
)
from eureka_ml_insights.metrics import (
    AverageAggregator,
    CountAggregator,
    ExactMatch,
)

from eureka_ml_insights.configs import(
    AggregatorConfig,
    DataProcessingConfig,
    DataSetConfig,
    EvalReportingConfig,
    InferenceConfig,
    MetricConfig,
    ModelConfig,
    PipelineConfig,
    PromptProcessingConfig,
)
from eureka_ml_insights.configs import ExperimentConfig
from eureka_ml_insights.configs.model_configs import OAI_GPT4_1106_PREVIEW_CONFIG

"""This class specifies the config for running Toxigen discriminative benchmark."""


class ToxiGen_Discriminative_PIPELINE(ExperimentConfig):
    def configure_pipeline(self, model_config):
        # Configure the data pre processing component.
        self.data_pre_processing = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "toxigen/toxigen-data",
                    "tasks": ["annotated"],
                    # Note that "split" can be set to either ["test"] which will perform evaluation on 940 samples or
                    # ["train"] which will perform evaluation on 8,960 samples
                    "split": ["train"],
                    "transform": SequenceTransform(
                        [
                            ColumnRename(
                                name_mapping={
                                    "text": "statement",
                                    "target_group": "category",
                                    "toxicity_human": "ground_truth",
                                }
                            ),
                        ]
                    ),
                },
            ),
            prompt_template_path=os.path.join(
                os.path.dirname(__file__), "../prompt_templates/toxigen_templates/discriminative_eval.jinja"
            ),
            output_dir=os.path.join(self.log_dir, "data_pre_processing_output"),
        )

        # Inference component
        self.inference_comp = InferenceConfig(
            component_type=Inference,
            model_config=model_config,
            data_loader_config=DataSetConfig(
                MMDataLoader,
                {"path": os.path.join(self.data_pre_processing.output_dir, "transformed_data.jsonl")},
            ),
            output_dir=os.path.join(self.log_dir, "inference_result"),
        )
        # Configure the data post processing component.
        self.data_post_processing = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            AddColumn("merged_group"),
                            GPT4ToxiGenRegex("model_output", "ground_truth", "category", "merged_group"),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_post_processing_output"),
        )

        # Configure the evaluation and reporting component.
        self.evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.data_post_processing.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                },
            ),
            metric_config=MetricConfig(ExactMatch),
            aggregator_configs=[
                AggregatorConfig(CountAggregator, {"column_names": ["ExactMatch_result"], "normalize": True}),
                AggregatorConfig(
                    CountAggregator, {"column_names": ["ExactMatch_result"], "group_by": "category", "normalize": True}
                ),
                AggregatorConfig(
                    CountAggregator,
                    {"column_names": ["ExactMatch_result"], "group_by": "ground_truth", "normalize": True},
                ),
                AggregatorConfig(
                    CountAggregator,
                    {"column_names": ["ExactMatch_result"], "group_by": "merged_group", "normalize": True},
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )

        return PipelineConfig(
            [self.data_pre_processing, self.inference_comp, self.data_post_processing, self.evalreporting_comp],
            self.log_dir,
        )


"""This class specifies the config for running Toxigen generative benchmark."""


class ToxiGen_Generative_PIPELINE(ExperimentConfig):
    # def configure_pipeline(self, model_config):

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        
        # Data pre processing component.
        self.data_pre_processing = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "toxigen/toxigen-data",
                    "tasks": "prompts",
                    # Note that "split" can be set to one of the following 31 categories each one having 1000 samples:
                    #
                    "split": [
                        "hate_trans_1k",
                        "neutral_black_1k",
                        "hate_native_american_1k",
                        "neutral_immigrant_1k",
                        "hate_middle_east_1k",
                        "neutral_lgbtq_1k",
                        "neutral_women_1k",
                        "neutral_chinese_1k",
                        "hate_latino_1k",
                        "hate_bisexual_1k",
                        "hate_mexican_1k",
                        "hate_asian_1k",
                        "neutral_mental_disability_1k",
                        "neutral_mexican_1k",
                        "hate_mental_disability_1k",
                        "neutral_bisexual_1k",
                        "neutral_latino_1k",
                        "hate_chinese_1k",
                        "neutral_jewish_1k",
                        "hate_muslim_1k",
                        "neutral_asian_1k",
                        "hate_physical_disability_1k",
                        "hate_jewish_1k",
                        "neutral_muslim_1k",
                        "hate_immigrant_1k",
                        "hate_black_1k",
                        "hate_lgbtq_1k",
                        "hate_women_1k",
                        "neutral_middle_east_1k",
                        "neutral_native_american_1k",
                        "neutral_physical_disability_1k",
                    ],
                    "transform": SequenceTransform(
                        [
                            SamplerTransform(sample_count=1550, random_seed=1234),
                            ColumnRename(
                                name_mapping={
                                    "text": "prompt",
                                }
                            ),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_pre_processing_output"),
        )

        # Inference component
        self.inference_comp = InferenceConfig(
            component_type=Inference,
            model_config=model_config,
            data_loader_config=DataSetConfig(
                MMDataLoader,
                {"path": os.path.join(self.data_pre_processing.output_dir, "transformed_data.jsonl")},
            ),
            output_dir=os.path.join(self.log_dir, "inference_result"),
            resume_from=resume_from,
            max_concurrent=5,
        )

        # Eval data pre processing component.
        self.eval_data_pre_processing = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform([ColumnRename(name_mapping={"model_output": "statement"})]),
                },
            ),
            prompt_template_path=os.path.join(
                os.path.dirname(__file__), "../prompt_templates/toxigen_templates/discriminative_eval.jinja"
            ),
            output_dir=os.path.join(self.log_dir, "eval_data_pre_processing_output"),
        )

        # Eval Inference component
        self.eval_inference_comp = InferenceConfig(
            component_type=Inference,
            model_config=OAI_GPT4_1106_PREVIEW_CONFIG,
            data_loader_config=DataSetConfig(
                MMDataLoader,
                {"path": os.path.join(self.eval_data_pre_processing.output_dir, "transformed_data.jsonl")},
            ),
            output_dir=os.path.join(self.log_dir, "eval_inference_result"),
        )

        # Eval data post processing component.
        self.eval_data_post_processing = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.eval_inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            AddColumn("ground_truth"),
                            AddColumn("category"),
                            AddColumn("merged_group"),
                            GPT4ToxiGenRegexGenerative(
                                "model_output",
                                "ground_truth",
                                "category",
                                "merged_group",
                            ),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "eval_data_post_processing_output"),
        )

        # Configure the evaluation and reporting component.
        self.evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.eval_data_post_processing.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                },
            ),
            aggregator_configs=[
                AggregatorConfig(AverageAggregator, {"column_names": ["model_output"], "ignore_non_numeric": True}),
                AggregatorConfig(
                    AverageAggregator,
                    {"column_names": ["model_output"], "group_by": "ground_truth", "ignore_non_numeric": True},
                ),
                AggregatorConfig(
                    AverageAggregator,
                    {"column_names": ["model_output"], "group_by": "category", "ignore_non_numeric": True},
                ),
                AggregatorConfig(
                    AverageAggregator,
                    {"column_names": ["model_output"], "group_by": "merged_group", "ignore_non_numeric": True},
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )

        # Configure the pipeline
        return PipelineConfig(
            [
                self.data_pre_processing,
                self.inference_comp,
                self.eval_data_pre_processing,
                self.eval_inference_comp,
                self.eval_data_post_processing,
                self.evalreporting_comp,
            ],
            self.log_dir,
        )
