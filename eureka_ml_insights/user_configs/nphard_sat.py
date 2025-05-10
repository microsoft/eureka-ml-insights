import os
from typing import Any

from eureka_ml_insights.configs import (
    DataSetConfig,
    ExperimentConfig,
    InferenceConfig,
    ModelConfig,
    PipelineConfig,
    PromptProcessingConfig,
)
from eureka_ml_insights.core import (
    Inference,
    PromptProcessing,
)
from eureka_ml_insights.data_utils import (    
    ColumnRename,    
    HFDataReader,    
    MMDataLoader,
    MultiplyTransform,
    SequenceTransform,
)


"""This file contains user defined configuration classes for the 3-SAT problem.
"""


class NPHARD_SAT_PIPELINE(ExperimentConfig):
    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        # Configure the data processing component.
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "GeoMeterData/nphard_sat2",
                    "split": "train",
                    "transform": SequenceTransform(
                        [
                            ColumnRename(name_mapping={"query_text": "prompt", "target_text": "ground_truth"}),
                        ]
                    ),
                },
            ),
            prompt_template_path=os.path.join(
                os.path.dirname(__file__), "../prompt_templates/nphard_sat_templates/Template_sat_cot.jinja"
                # os.path.dirname(__file__), "../prompt_templates/nphard_sat_templates/Template_sat_o1.jinja"
            ),
            output_dir=os.path.join(self.log_dir, "data_processing_output"),
        )

        # Configure the inference component
        self.inference_comp = InferenceConfig(
            component_type=Inference,
            model_config=model_config,
            data_loader_config=DataSetConfig(
                MMDataLoader,
                {"path": os.path.join(self.data_processing_comp.output_dir, "transformed_data.jsonl")},
            ),
            output_dir=os.path.join(self.log_dir, "inference_result"), 
            resume_from=resume_from,
            max_concurrent=20,
        )

###############################################

        # Configure the pipeline
        return PipelineConfig(
            [
                self.data_processing_comp,
                self.inference_comp  
            ],
            self.log_dir,
        )
####################################

class NPHARD_SAT_PIPELINE_MULTIPLE_RUNS(NPHARD_SAT_PIPELINE):
    """This class specifies the config for running SAT benchmark n repeated times"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # data preprocessing
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms.append(
            MultiplyTransform(n_repeats=1)
        )
        return pipeline


##################


        # # Configure the evaluation and reporting component.
        # self.evalreporting_comp = EvalReportingConfig(
        #     component_type=EvalReporting,
        #     data_reader_config=DataSetConfig(
        #         DataReader,
        #         {
        #             "path": os.path.join(self.data_post_processing.output_dir, "transformed_data.jsonl"),                    
        #             "format": ".jsonl",
        #         },
        #     ),
        #     metric_config=MetricConfig(NPHardSATMetric),
        #     aggregator_configs=[
        #         AggregatorConfig(CountAggregator, {"column_names": ["NPHardSATMetric_result"], "normalize": True}),
        #     ],
        #     output_dir=os.path.join(self.log_dir, "eval_report"),
        # )

        # # Aggregate the results by a majority vote.
        # self.postevalprocess_comp = EvalReportingConfig(
        #     component_type=EvalReporting,
        #     data_reader_config=DataSetConfig(
        #         DataReader,
        #         {
        #             "path": os.path.join(self.data_post_processing.output_dir, "transformed_data.jsonl"),
        #             "format": ".jsonl",
        #             "transform": SequenceTransform(
        #                 [
        #                     MajorityVoteTransform(id_col="data_point_id"),
        #                     ColumnRename(
        #                         name_mapping={
        #                             "model_output": "model_output_onerun",
        #                             "majority_vote": "model_output",
        #                         }
        #                     ),
        #                 ]
        #             ),
        #         },
        #     ),
        #     metric_config=MetricConfig(NPHardSATMetric),
        #     aggregator_configs=[
        #         AggregatorConfig(CountAggregator, {"column_names": ["NPHardSATMetric_result"], "normalize": True}),
        #     ],
        #     output_dir=os.path.join(self.log_dir, "eval_report_majorityVote"),
        # )