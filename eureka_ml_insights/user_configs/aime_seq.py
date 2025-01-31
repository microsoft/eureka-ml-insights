import os
from typing import Any
from .aime import AIME_PIPELINE
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
    AddColumnAndData,
    ColumnRename,
    CopyColumn,
    DataReader,
    HFDataReader,
    MajorityVoteTransform,
    MultiplyTransform,
    RunPythonTransform,
    SamplerTransform,
    SequenceTransform,
)
from eureka_ml_insights.data_utils.aime_utils import AIMEExtractAnswer
from eureka_ml_insights.data_utils.data import DataLoader
from eureka_ml_insights.metrics.metrics_base import ExactMatch, MetricBasedVerifier
from eureka_ml_insights.metrics.reports import (
    BiLevelCountAggregator,
    CountAggregator,
)

# from eureka_ml_insights.data_utils.transform import MajorityVoteTransform


class AIME_SEQ_PIPELINE(AIME_PIPELINE):
    """This class specifies the config for running AIME benchmark on any model"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:

        # this call to super will configure the initial prompt processing and final eval reporting comps that can be reused.
        super().configure_pipeline(model_config, resume_from, **kwargs)
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms.append(SamplerTransform(random_seed=42, sample_count=2))
        component_configs = [self.data_processing_comp]

        for i in range(1, 5):
            # Student inference component
            self.student_inference_comp = InferenceConfig(
                component_type=Inference,
                model_config=model_config,
                data_loader_config=DataSetConfig(
                    DataLoader,
                    {"path": os.path.join(self.data_processing_comp.output_dir, "transformed_data.jsonl")},
                ),
                output_dir=os.path.join(self.log_dir, f"student_inference_result_{i}"),
                resume_from=resume_from,
                chat_mode=True,
            )
            component_configs.append(self.student_inference_comp)


            # Metric based verification and filtering out the correct answers
            self.verificaiton_comp = DataProcessingConfig(
                component_type=DataProcessing,
                data_reader_config=DataSetConfig(
                    DataReader,
                    {
                        "path": os.path.join(self.student_inference_comp.output_dir, "inference_result.jsonl"),
                        "format": ".jsonl",
                        "transform": SequenceTransform(
                            [
                                # extract and verify the student answer
                                AIMEExtractAnswer(f"model_output", f"student_extracted_answer_{i}"),
                                MetricBasedVerifier(ExactMatch, f"student_extracted_answer_{i}"),
                                # drop rows where verification_result is True
                                RunPythonTransform(python_code="df = df[df['verification_result'] == 'correct']"),
                                ColumnRename(
                                    name_mapping={
                                        "verification_result": f"verification_result_{i}",
                                        "model_output": f"student_output_{i}",
                                    }
                                ),
                            ]
                        ),
                    },
                ),
                output_dir=os.path.join(self.log_dir, f"verification_{i}"),
            )
            component_configs.append(self.verificaiton_comp)


            # Create a new prompt with ground truth hinting
            self.hint_processing_comp = PromptProcessingConfig(
                component_type=PromptProcessing,
                data_reader_config=DataSetConfig(
                    DataReader,
                    {
                        "path": os.path.join(self.verificaiton_comp.output_dir, "transformed_data.jsonl"),
                        "format": ".jsonl",   
                    }
                ),
                prompt_template_path=os.path.join(
                    os.path.dirname(__file__), "../prompt_templates/aime_templates/hint_creation.jinja"
                ),
                output_dir=os.path.join(self.log_dir, f"hint_processing_output_{i}"),
            )
            component_configs.append(self.hint_processing_comp)


            # Inference component for the teacher model to provide hints
            self.teacher_inference_comp = InferenceConfig(
                component_type=Inference,
                model_config=model_config,
                data_loader_config=DataSetConfig(
                    DataLoader,
                    {"path": os.path.join(self.hint_processing_comp.output_dir, "transformed_data.jsonl")},
                ),
                output_dir=os.path.join(self.log_dir, f"teacher_inference_result_{i}"),
                resume_from=resume_from,
                max_concurrent=10,
                chat_mode=True,
            )
            component_configs.append(self.teacher_inference_comp)


            # Prompt processing for the stundent to try again
            self.hint_prompt_processing = PromptProcessingConfig(
                component_type=PromptProcessing,
                data_reader_config=DataSetConfig(
                    DataReader,
                    {
                        "path": os.path.join(self.teacher_inference_comp.output_dir, "inference_result.jsonl"),
                        "format": ".jsonl",
                        "transform": SequenceTransform(
                            [
                                ColumnRename(name_mapping={"model_output": "teacher_hint"}),
                                AddColumnAndData("attempt_id", i),
                            ]
                        ),
                    },
                ),
                prompt_template_path=os.path.join(os.path.dirname(__file__), "../prompt_templates/aime_templates/prompt_w_hint.jinja"),
                output_dir=os.path.join(self.log_dir, f"teacher_hint_prompt_{i}"),
            )
            component_configs.append(self.hint_prompt_processing)

        component_configs.append(self.evalreporting_comp)


        # Configure the pipeline
        return PipelineConfig(
            component_configs,
            self.log_dir,
        )
