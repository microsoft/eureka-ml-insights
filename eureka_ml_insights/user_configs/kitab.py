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
    CopyColumn,
    ColumnRename,
    DataReader,
    HFDataReader,
    MMDataLoader,
    SequenceTransform,
    RunPythonTransform,
    SamplerTransform
)
from eureka_ml_insights.data_utils.kitab_utils import (
    KitabExtractBooks,
    KitabExtractBooksAddMarker,
    PrepareContext,
)
from eureka_ml_insights.metrics import AverageAggregator
from eureka_ml_insights.metrics.kitab_metrics import KitabMetric

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

# Example template for an Azure Language Service config
# required for running entity recognition for evaluating human and city name
# working config for Azure Language Service needed for Kitab evaluation, non tenant
AZURE_LANG_SERVICE_CONFIG = {
    "url": "your/azure_lang_service_endpoint/url",
    "secret_key_params": {
        "key_name": "your_azure_lang_service_secret_key_name",
        "local_keys_path": "keys/keys.json",
        "key_vault_url": None,
    },
}

class KITAB_ONE_BOOK_CONSTRAINT_PIPELINE(ExperimentConfig):
    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
            
        # Configure the data pre processing component.
        self.data_pre_processing = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "microsoft/kitab",
                    "tasks": ["one-book-constraints"],
                    "split": "test",
                    "transform": SequenceTransform(
                        [
                            # SamplerTransform(random_seed=99, sample_count=5),
                            ColumnRename(
                                name_mapping={
                                    "Author": "author",
                                    "Birth Year": "birth_year",
                                }
                            ),
                        ]
                    ),
                },
            ),
            prompt_template_path=os.path.join(
                os.path.dirname(__file__), "../prompt_templates/kitab_templates/Template_2a.jinja"
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
        )
        # Configure the data post processing component.
        # For kitab this entails extracting all books and CoT reasons and removing the rest of the text
        self.data_post_processing = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [AddColumn("model_books"), KitabExtractBooks("model_output", "model_books")]
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
            metric_config=MetricConfig(
                KitabMetric,
                {
                    "temp_path_names": os.path.join(self.log_dir, "gpt_4_name_data_processed.csv"),
                    "azure_lang_service_config": AZURE_LANG_SERVICE_CONFIG,
                },
            ),
            aggregator_configs=[
                AggregatorConfig(
                    AverageAggregator,
                    {
                        "column_names": [
                            "KitabMetric_satisfied_rate",
                            "KitabMetric_unsatisfied_rate",
                            "KitabMetric_not_from_author_rate",
                            "KitabMetric_completeness",
                            "KitabMetric_all_correct",
                        ],
                        "filename_base": "AllKitabMetrics",
                    },
                ),
                AggregatorConfig(
                    AverageAggregator,
                    {
                        "column_names": [
                            "KitabMetric_satisfied_rate",
                            "KitabMetric_unsatisfied_rate",
                            "KitabMetric_not_from_author_rate",
                            "KitabMetric_completeness",
                            "KitabMetric_all_correct",
                        ],
                        "group_by": "constraint_type",
                        "filename_base": "AllKitabMetrics_GroupBy",
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )

        return PipelineConfig(
            # [self.data_pre_processing, self.inference_comp, self.data_post_processing, self.evalreporting_comp],
            [self.data_pre_processing, self.inference_comp, self.data_post_processing],
            self.log_dir,
        )


class GPT35_KITAB_ONE_BOOK_CONSTRAINT_PIPELINE(KITAB_ONE_BOOK_CONSTRAINT_PIPELINE):
    def configure_pipeline(self, model_config=None, resume_from=None, **kwargs):
        config = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # Change the extraction of books for the pipeline
        self.data_post_processing.data_reader_config.init_args["transform"] = SequenceTransform(
            [
                AddColumn("model_books"), 
                KitabExtractBooksAddMarker("model_output", "model_books")
            ]
        )
        return config

class Phi_KITAB_ONE_BOOK_CONSTRAINT_PIPELINE(KITAB_ONE_BOOK_CONSTRAINT_PIPELINE):
    def configure_pipeline(self, model_config=None, resume_from=None, **kwargs):
        config = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # Change the extraction of books for the pipeline
        self.data_post_processing.data_reader_config.init_args["transform"] = SequenceTransform(
            [
                AddColumn("model_books"), 
                CopyColumn("model_output", "post_cot_model_output"),
                RunPythonTransform("df['post_cot_model_output'] = df['post_cot_model_output'].apply(lambda x: x.split('<|dummy_87|>')[-1] if '<|dummy_87|>' in x else x)"),
                KitabExtractBooksAddMarker("post_cot_model_output", "model_books")
            ]
        )
        return config


class KITAB_ONE_BOOK_CONSTRAINT_PIPELINE_WITH_CONTEXT(KITAB_ONE_BOOK_CONSTRAINT_PIPELINE):
    def configure_pipeline(self, model_config=None, resume_from=None, **kwargs):
        config = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        self.data_pre_processing.data_reader_config = DataSetConfig(
            HFDataReader,
            {
                "path": "microsoft/kitab",
                "tasks": ["one-book-constraints"],
                "split": "test",
                "transform": SequenceTransform(
                    [
                        ColumnRename(
                            name_mapping={
                                "Author": "author",
                                "Birth Year": "birth_year",
                            }
                        ),
                        AddColumn("all_books_context"),
                        PrepareContext("all_books", "all_books_context"),
                    ]
                ),
            },
        )
        self.data_pre_processing.prompt_template_path = os.path.join(
            os.path.dirname(__file__), "../prompt_templates/kitab_templates/Template_2b.jinja"
        )
        return config

class Phi_KITAB_ONE_BOOK_CONSTRAINT_PIPELINE_WITH_CONTEXT(KITAB_ONE_BOOK_CONSTRAINT_PIPELINE_WITH_CONTEXT):
    def configure_pipeline(self, model_config=None, resume_from=None, **kwargs):
        config = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # Change the extraction of books for the pipeline
        self.data_post_processing.data_reader_config.init_args["transform"] = SequenceTransform(
            [
                AddColumn("model_books"), 
                CopyColumn("model_output", "post_cot_model_output"),
                RunPythonTransform("df['post_cot_model_output'] = df['post_cot_model_output'].apply(lambda x: x.split('<|dummy_87|>')[-1] if '<|dummy_87|>' in x else x)"),
                KitabExtractBooksAddMarker("post_cot_model_output", "model_books")
            ]
        )
        return config


class KITAB_ONE_BOOK_CONSTRAINT_PIPELINE_SELF_CONTEXT(KITAB_ONE_BOOK_CONSTRAINT_PIPELINE):
    def configure_pipeline(self, model_config=None, resume_from=None, **kwargs):
        config = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        self.data_pre_processing.prompt_template_path = os.path.join(
            os.path.dirname(__file__), "../prompt_templates/kitab_templates/Template_3.jinja"
        )
        return config


class KITAB_TWO_BOOK_CONSTRAINT_PIPELINE(KITAB_ONE_BOOK_CONSTRAINT_PIPELINE):
    def configure_pipeline(self, model_config=None, resume_from=None, **kwargs):
        """
        The main differences between the evaluation of one-book-constraints and two-book-constraints are the following:
        - The HF task is two-book-constraints
        - The evaluation metric evaluations satisfaction rate for both constraints individually
        """
        config = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        self.data_pre_processing.data_reader_config.init_args["tasks"] = ["two-book-constraints"]
        self.data_pre_processing.data_reader_config.init_args["transform"] = SequenceTransform(
            [
                ColumnRename(
                    name_mapping={
                        "Author": "author",
                        "Birth Year": "birth_year",
                    }
                ),
            ]
        )
        return config

class Phi_KITAB_TWO_BOOK_CONSTRAINT_PIPELINE(KITAB_TWO_BOOK_CONSTRAINT_PIPELINE):
    def configure_pipeline(self, model_config=None, resume_from=None, **kwargs):
        config = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # Change the extraction of books for the pipeline
        self.data_post_processing.data_reader_config.init_args["transform"] = SequenceTransform(
            [
                AddColumn("model_books"), 
                CopyColumn("model_output", "post_cot_model_output"),
                RunPythonTransform("df['post_cot_model_output'] = df['post_cot_model_output'].apply(lambda x: x.split('<|dummy_87|>')[-1] if '<|dummy_87|>' in x else x)"),
                KitabExtractBooksAddMarker("post_cot_model_output", "model_books")
            ]
        )
        return config


class KITAB_TWO_BOOK_CONSTRAINT_PIPELINE_WITH_CONTEXT(KITAB_ONE_BOOK_CONSTRAINT_PIPELINE_WITH_CONTEXT):
    def configure_pipeline(self, model_config=None, resume_from=None, **kwargs):
        """
        The main differences between the evaluation of one-book-constraints and two-book-constraints are the following:
        - The HF task is two-book-constraints
        - The evaluation metric evaluations satisfaction rate for both constraints individually
        """
        config = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        self.data_pre_processing.data_reader_config.init_args["tasks"] = ["two-book-constraints"]
        self.data_pre_processing.data_reader_config.init_args["transform"] = SequenceTransform(
            [
                ColumnRename(
                    name_mapping={
                        "Author": "author",
                        "Birth Year": "birth_year",
                    }
                ),
                AddColumn("all_books_context"),
                PrepareContext("all_books", "all_books_context"),
            ]
        )
        return config

class Phi_KITAB_TWO_BOOK_CONSTRAINT_PIPELINE_WITH_CONTEXT(KITAB_TWO_BOOK_CONSTRAINT_PIPELINE_WITH_CONTEXT):
    def configure_pipeline(self, model_config=None, resume_from=None, **kwargs):
        config = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # Change the extraction of books for the pipeline
        self.data_post_processing.data_reader_config.init_args["transform"] = SequenceTransform(
            [
                AddColumn("model_books"), 
                CopyColumn("model_output", "post_cot_model_output"),
                RunPythonTransform("df['post_cot_model_output'] = df['post_cot_model_output'].apply(lambda x: x.split('<|dummy_87|>')[-1] if '<|dummy_87|>' in x else x)"),
                KitabExtractBooksAddMarker("post_cot_model_output", "model_books")
            ]
        )
        return config