"""
This module defines the AIME_SEQ_PIPELINE class and configures the iterative pipeline used in the AIME benchmark.
"""

import os
from typing import Any

from eureka_ml_insights.configs import (
    DataProcessingConfig,
    DataSetConfig,
    DataUnionConfig,
    InferenceConfig,
    ModelConfig,
    PipelineConfig,
    PromptProcessingConfig,
)
from eureka_ml_insights.core import (
    DataProcessing,
    DataUnion,
    Inference,
    PromptProcessing,
)
from eureka_ml_insights.data_utils import (
    AddColumnAndData,
    ColumnRename,
    CopyColumn,
    DataReader,
    RunPythonTransform,
    SequenceTransform,
)
from eureka_ml_insights.data_utils.aime_utils import AIMEExtractAnswer
from eureka_ml_insights.data_utils.data import MMDataLoader
from eureka_ml_insights.metrics.metrics_base import (
    ExactMatch,
    MetricBasedVerifier,
)

from .aime import AIME_PIPELINE

DEFAULT_N_ITER = 3
RESULT_COLS = [
    "attempt_id",
    "model_output",
    "uid",
    "prompt",
    "ground_truth",
    "Year",
    "Part",
    "ID",
    "extracted_answer",
    "verification_result",
    "usage",
]
resume_from_dict = {}


class AIME_SEQ_PIPELINE(AIME_PIPELINE):
    """
    Specifies the configuration for running the AIME benchmark on any model.

    This class inherits from AIME_PIPELINE and configures multiple inference steps, verification,
    hint generation, and other steps used in the AIME benchmark pipeline.
    """

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        """
        Configures the iterative pipeline for the AIME benchmark.

        This method sets up multiple inference steps, verification processes, and hint generation
        to iteratively refine the solution to the AIME question.

        Args:
            model_config (ModelConfig): The configuration for the model.
            resume_from (str, optional): The checkpoint or directory to resume from. Defaults to None.
            **kwargs (dict[str, Any]): Additional arguments to pass to the pipeline configuration.

        Returns:
            PipelineConfig: The complete configuration for the AIME pipeline.
        """

        # this call to super will configure the initial prompt processing and final eval reporting comps that can be reused.
        super().configure_pipeline(model_config, resume_from, **kwargs)

        n_iter = kwargs.get("n_iter", DEFAULT_N_ITER)
        # Uncomment if you want to sample a subset of the data for debugging
        # self.data_processing_comp.data_reader_config.init_args["transform"].transforms.append(
        #    SamplerTransform(sample_count=2, random_seed=42)
        # )
        component_configs = [self.data_processing_comp]
        for i in range(1, n_iter + 1):
            # Student inference component, reads prompts from the last prompt processing component
            last_prompt_proc_comp = component_configs[-1]
            self.student_inference_comp = InferenceConfig(
                component_type=Inference,
                model_config=model_config,
                data_loader_config=DataSetConfig(
                    MMDataLoader,
                    {
                        "path": os.path.join(last_prompt_proc_comp.output_dir, "transformed_data.jsonl"),
                        # if this is not the first iteration, we need to add the previous messages to the data loader config
                        "misc_columns": ["previous_messages"] if i > 1 else None,
                    },
                ),
                output_dir=os.path.join(self.log_dir, f"student_inference_result_{i}"),
                resume_from=resume_from_dict.get(i, None),
                chat_mode=True,
            )

            component_configs.append(self.student_inference_comp)

            # Answer extraction and metric-based verification
            self.verification_comp = DataProcessingConfig(
                component_type=DataProcessing,
                data_reader_config=DataSetConfig(
                    DataReader,
                    {
                        "path": os.path.join(self.student_inference_comp.output_dir, "inference_result.jsonl"),
                        "format": ".jsonl",
                        "transform": SequenceTransform(
                            [
                                # extract and verify the student answer
                                AIMEExtractAnswer(f"model_output", f"extracted_answer"),
                                MetricBasedVerifier(ExactMatch, f"extracted_answer"),
                                AddColumnAndData("attempt_id", i),
                                CopyColumn(column_name_src="model_output", column_name_dst=f"student_output"),
                            ]
                        ),
                    },
                ),
                output_dir=os.path.join(self.log_dir, f"verification_{i}"),
            )
            component_configs.append(self.verification_comp)

            # Variable maintaining link to the most recent inference result results to be used for evaluation
            # This will be updated to point to the concatenation of results from all iterations

            if i > 1:
                self.last_inference_result_join_comp = DataUnionConfig(
                    component_type=DataUnion,
                    data_reader_config=DataSetConfig(
                        DataReader,
                        {
                            "path": os.path.join(self.verification_comp.output_dir, "transformed_data.jsonl"),
                            "format": ".jsonl",
                        },
                    ),
                    other_data_reader_config=DataSetConfig(
                        DataReader,
                        {
                            "path": os.path.join(last_agg_dir, "transformed_data.jsonl"),
                            "format": ".jsonl",
                        },
                    ),
                    output_data_columns=RESULT_COLS,
                    dedupe_cols=["ID", "attempt_id"],
                    output_dir=os.path.join(self.log_dir, f"last_inference_result_join_{i}"),
                )
                last_agg_dir = self.last_inference_result_join_comp.output_dir
                component_configs.append(self.last_inference_result_join_comp)
            else:
                last_agg_dir = self.verification_comp.output_dir

            # Filtering out the rows with correct answer
            self.filtering_comp = DataProcessingConfig(
                component_type=DataProcessing,
                data_reader_config=DataSetConfig(
                    DataReader,
                    {
                        "path": os.path.join(self.verification_comp.output_dir, "transformed_data.jsonl"),
                        "format": ".jsonl",
                        "transform": RunPythonTransform(python_code="df = df[df['verification_result'] != 'correct']"),
                    },
                ),
                output_dir=os.path.join(self.log_dir, f"filtering_{i}"),
            )
            component_configs.append(self.filtering_comp)

            # Create a new prompt to ask the teacher model to provide hints.
            self.hint_processing_comp = PromptProcessingConfig(
                component_type=PromptProcessing,
                data_reader_config=DataSetConfig(
                    DataReader,
                    {
                        "path": os.path.join(self.filtering_comp.output_dir, "transformed_data.jsonl"),
                        "format": ".jsonl",
                    },
                ),
                prompt_template_path=os.path.join(
                    os.path.dirname(__file__), "../prompt_templates/aime_templates/hint_creation.jinja"
                ),
                output_dir=os.path.join(self.log_dir, f"hint_processing_output_{i}"),
            )
            component_configs.append(self.hint_processing_comp)

            # Inference component to ask teacher model to provide hints
            self.teacher_inference_comp = InferenceConfig(
                component_type=Inference,
                model_config=model_config,
                data_loader_config=DataSetConfig(
                    MMDataLoader,
                    {
                        "path": os.path.join(self.hint_processing_comp.output_dir, "transformed_data.jsonl"),
                        "misc_columns": ["previous_messages"],
                    },
                ),
                output_dir=os.path.join(self.log_dir, f"teacher_inference_result_{i}"),
                max_concurrent=10,
                chat_mode=False,
            )
            component_configs.append(self.teacher_inference_comp)

            # Prompt processing to ask the stundent to try again
            self.prompt_processing_with_hint = PromptProcessingConfig(
                component_type=PromptProcessing,
                data_reader_config=DataSetConfig(
                    DataReader,
                    {
                        "path": os.path.join(self.teacher_inference_comp.output_dir, "inference_result.jsonl"),
                        "format": ".jsonl",
                        "transform": ColumnRename(name_mapping={"model_output": "teacher_hint"}),
                    },
                ),
                prompt_template_path=os.path.join(
                    os.path.dirname(__file__), "../prompt_templates/aime_templates/prompt_w_hint.jinja"
                ),
                output_dir=os.path.join(self.log_dir, f"teacher_hint_prompt_{i}"),
            )
            component_configs.append(self.prompt_processing_with_hint)

        # Pass the combined results from all iterations to the eval reporting component
        self.final_preeval_data_processing.data_reader_config.init_args["path"] = os.path.join(
            last_agg_dir, "transformed_data.jsonl"
        )

        component_configs.extend(
            [
                self.final_preeval_data_processing,
                self.evalreporting_comp,
                self.data_post_processing_addmv,
                self.mv_evalreporting_comp,
                self.posteval_data_post_processing_comp,
                self.bon_evalreporting_comp,
                self.won_evalreporting_comp,
            ]
        )

        # Configure the pipeline
        return PipelineConfig(
            component_configs,
            self.log_dir,
        )
