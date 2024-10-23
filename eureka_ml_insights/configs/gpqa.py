import os
from typing import Any
from eureka_ml_insights.core import EvalReporting, Inference, PromptProcessing
from eureka_ml_insights.data_utils import (
    ColumnRename,
    DataReader,
    HFDataReader,
    MMDataLoader,
    SequenceTransform,
    RunPythonTransform
)
from eureka_ml_insights.metrics import CountAggregator, GeoMCQMetric
from .config import (
    AggregatorConfig,
    DataSetConfig,
    EvalReportingConfig,
    InferenceConfig,
    MetricConfig,
    ModelConfig,
    PipelineConfig,
    PromptProcessingConfig,
)
from .experiment_config import ExperimentConfig
from eureka_ml_insights.data_utils.gpqa_utils import CreateGPQAPrompt

"""This file contains user defined configuration classes for the geometric reasoning task on geometer dataset.
"""
class GPQA_Experiment_Pipeline(ExperimentConfig):
    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        # Configure the data processing component.
        # mc_transform = """
        #     # List of the column names to shuffle
        #     columns_to_shuffle = ['Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3']

        #     # Shuffle the column names
        #     shuffled_columns = random.sample(columns_to_shuffle, len(columns_to_shuffle))

        #     # Map shuffled columns to A, B, C, D and store the mapping
        #     mapping = {
        #         'A': shuffled_columns[0],
        #         'B': shuffled_columns[1],
        #         'C': shuffled_columns[2],
        #         'D': shuffled_columns[3]
        #     }

        #     # Map shuffled columns to A, B, C, D
        #     df['A'] = df[shuffled_columns[0]]
        #     df['B'] = df[shuffled_columns[1]]
        #     df['C'] = df[shuffled_columns[2]]
        #     df['D'] = df[shuffled_columns[3]]

        #     # Create the 'ground_truth' column by checking which column contains the "Correct Answer"
        #     df['ground_truth'] = df.apply(lambda row: next(key for key, value in mapping.items() if row[value] == row['Correct Answer']), axis=1)
        #     """
        
        # data_processing_comp = PromptProcessingConfig(
        #     component_type=PromptProcessing,
        #     data_reader_config=DataSetConfig(
        #         HFDataReader,
        #         {
        #             "path": "Idavidrein/gpqa",
        #             "tasks": "gpqa_diamond",
        #             "split": "train",
        #             "transform": SequenceTransform(
        #                 [
        #                     CreateGPQAPrompt(),
        #                 ]
        #             ),
        #         },
        #     ),
        #     output_dir=os.path.join(self.log_dir, "data_processing_output"),
        # )
        # # Configure the inference component
        # inference_comp = InferenceConfig(
        #     component_type=Inference,
        #     model_config=model_config,
        #     data_loader_config=DataSetConfig(
        #         MMDataLoader,
        #         {"path": os.path.join(data_processing_comp.output_dir, "transformed_data.jsonl")},
        #     ),
        #     output_dir=os.path.join(self.log_dir, "inference_result"),
        #     resume_from=resume_from,
        # )

        # # Configure the evaluation and reporting component.
        evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": "C:/Users/shrivastavav/OneDrive - Microsoft/Desktop/Code/eureka-ml-insights/logs/GPQA_Experiment_Pipeline/gpt4o_0513/2024-10-22-23-02-53.193972/inference_result/inference_result.jsonl",
                    "format": ".jsonl",
                },
            ),
            metric_config=MetricConfig(GeoMCQMetric),
            aggregator_configs=[
                AggregatorConfig(CountAggregator, {"column_names": ["GeoMCQMetric_result"], "normalize": True}),
                AggregatorConfig(
                    CountAggregator,
                    {"column_names": ["GeoMCQMetric_result"], "group_by": "category"},
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )
        # # Configure the pipeline
        return PipelineConfig(
            [
                # data_processing_comp,
                # inference_comp,
                evalreporting_comp,
            ],
            self.log_dir,
        )