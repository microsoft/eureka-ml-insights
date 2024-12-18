import os
from typing import Any

from eureka_ml_insights.configs import (
    AggregatorConfig,
    DataProcessingConfig,
    DataSetConfig,
    EvalReportingConfig,
    ExperimentConfig,
    InferenceConfig,
    MetricConfig,
    ModelConfig,
    PipelineConfig,
    PromptProcessingConfig,
)
from eureka_ml_insights.core import DataProcessing, Inference, PromptProcessing
from eureka_ml_insights.core.eval_reporting import EvalReporting
from eureka_ml_insights.data_utils import (
    AddColumn,
    ColumnRename,
    DataReader,
    HFDataReader,
    MajorityVoteTransform,
    MultiplyTransform,
    SequenceTransform,
    SamplerTransform,
)
from eureka_ml_insights.data_utils.aime_utils import AIMEExtractAnswer
from eureka_ml_insights.data_utils.data import DataLoader
from eureka_ml_insights.metrics.metrics_base import ExactMatch
from eureka_ml_insights.metrics.aime_metrics import NumericMatch

from eureka_ml_insights.metrics.reports import (
    BiLevelCountAggregator,
    CountAggregator,
)

# from eureka_ml_insights.data_utils.transform import MajorityVoteTransform


class AIME_PIPELINE(ExperimentConfig):
    """This class specifies the config for running AIME benchmark on any model"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:

        # data preprocessing
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "qq8933/AIME_1983_2024",
                    "split": "train",
                    "transform": SequenceTransform(
                        [
                            ColumnRename(
                                name_mapping={
                                    "Question": "prompt",
                                    "Answer": "ground_truth",
                                }
                            ),
                        ],
                    ),
                },
            ),
            prompt_template_path=os.path.join(
                os.path.dirname(__file__), "../prompt_templates/aime_templates/Template_1clean.jinja"
            ),
            output_dir=os.path.join(self.log_dir, "data_processing_output"),
        )

        # inference component
        self.inference_comp = InferenceConfig(
            component_type=Inference,
            model_config=model_config,
            data_loader_config=DataSetConfig(
                DataLoader,
                {"path": os.path.join(self.data_processing_comp.output_dir, "transformed_data.jsonl")},
            ),
            output_dir=os.path.join(self.log_dir, "inference_result"),
            resume_from=resume_from,
            max_concurrent=10,
        )
        # post process the response to extract the answer
        self.data_post_processing = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            ColumnRename(
                                name_mapping={
                                    "model_output": "raw_output",
                                }
                            ),
                            AddColumn("model_output"),
                            AIMEExtractAnswer("raw_output", "model_output"),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_post_processing_output"),
        )

        # Configure the evaluation and reporting component for evaluation and dataset level aggregation
        self.evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.data_post_processing.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                },
            ),
            metric_config=MetricConfig(NumericMatch),
            aggregator_configs=[
                AggregatorConfig(
                    CountAggregator,
                    {
                        "column_names": [
                            "NumericMatch_result",
                        ],
                        "group_by": "Year",
                        "filename_base": "NumericMatch_GroupBy",
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )

        # Aggregate the results by a majority vote
        # First, let us perform majority_vote
        self.data_post_processing_addmv = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            ColumnRename(
                                name_mapping={
                                    "model_output": "raw_output",
                                }
                            ),
                            AddColumn("model_output"),
                            AIMEExtractAnswer("raw_output", "model_output"),
                            MajorityVoteTransform(id_col="ID"),
                            ColumnRename(
                                name_mapping={
                                    "model_output": "model_output_onerun",
                                    "majority_vote": "model_output",
                                }
                            ),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_addmv_output"),
        )
        # Second, compute eaxct match
        self.postevalprocess_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.data_post_processing_addmv.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                },
            ),
            metric_config=MetricConfig(NumericMatch),
            aggregator_configs=[
                AggregatorConfig(
                    BiLevelCountAggregator,
                    {
                        "column_names": [
                            "NumericMatch_result",
                        ],
                        "first_groupby": "ID",
                        "filename_base": "MajorityVote",
                        "normalize": True,
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report_majorityVote"),
        )

        # Configure the pipeline
        return PipelineConfig(
            [
                self.data_processing_comp,
                self.inference_comp,
                self.data_post_processing,
                self.evalreporting_comp,
                self.data_post_processing_addmv,
                self.postevalprocess_comp,
            ],
            self.log_dir,
        )


class AIME_PIPELINE5Run(AIME_PIPELINE):
    """This class specifies the config for running AIME benchmark 5 repeated times"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # data preprocessing
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms.append(
            MultiplyTransform(n_repeats=5)
        )
        return pipeline


class AIME_PIPELINE16Run(AIME_PIPELINE):
    """This class specifies the config for running AIME benchmark 5 repeated times"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # data preprocessing
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms.append(
            MultiplyTransform(n_repeats=16)
        )
        return pipeline


class AIME_PIPELINE32Run(AIME_PIPELINE):
    """This class specifies the config for running AIME benchmark 5 repeated times"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # data preprocessing
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms.append(
            MultiplyTransform(n_repeats=32)
        )
        return pipeline


class AIME_PIPELINE64Run(AIME_PIPELINE):
    """This class specifies the config for running AIME benchmark 5 repeated times"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # data preprocessing
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms.append(
            MultiplyTransform(n_repeats=64)
        )
        return pipeline


class AIME_PIPELINE128Run(AIME_PIPELINE):
    """This class specifies the config for running AIME benchmark 5 repeated times"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # data preprocessing
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms.append(
            MultiplyTransform(n_repeats=128)
        )
        return pipeline


class AIME_PIPELINE256Run(AIME_PIPELINE):
    """This class specifies the config for running AIME benchmark 5 repeated times"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # data preprocessing
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms.append(
            MultiplyTransform(n_repeats=256)
        )
        return pipeline


class AIME_PIPELINE512Run(AIME_PIPELINE):
    """This class specifies the config for running AIME benchmark 5 repeated times"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # data preprocessing
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms.append(
            MultiplyTransform(n_repeats=512)
        )
        return pipeline


class AIME_PIPELINE1024Run(AIME_PIPELINE):
    """This class specifies the config for running AIME benchmark 5 repeated times"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # data preprocessing
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms.append(
            MultiplyTransform(n_repeats=1024)
        )
        return pipeline


class AIME_PIPELINETag(AIME_PIPELINE):
    """This class specifies the config for running AIME benchmark 5 repeated times"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        '''
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms.append(
            SamplerTransform(random_seed=0,
                             sample_count=10,
                              )
        )
        '''
        # data preprocessing
        self.data_processing_comp.prompt_template_path=os.path.join(
                os.path.dirname(__file__), "../prompt_templates/aime_templates/Template_tag1.jinja"
            )
        return pipeline