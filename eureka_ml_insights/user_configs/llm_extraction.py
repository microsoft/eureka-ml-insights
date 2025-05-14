import os

"""This file contains user defined configuration classes for the GPQA dataset."""


from eureka_ml_insights.configs import (
    DataJoinConfig,
    DataProcessingConfig,
    DataSetConfig,
    InferenceConfig,
    ModelConfig,
    PromptProcessingConfig,
)
from eureka_ml_insights.core import (
    Component,
    DataJoin,
    DataProcessing,
    Inference,
    PromptProcessing,
)
from eureka_ml_insights.data_utils import (
    ColumnRename,
    DataReader,
    MMDataLoader,
    RunPythonTransform,
    SequenceTransform,
)


class LLM_EXTRACTION_SUBPIPELINE_MIXIN:
    """This class specifies the config for running the LLM extraction subpipeline"""

    def configure_subpipeline(
        self,
        extraction_attempt_component: Component,
        extracted_answer_col: str,
        llm_extraction_promp_template: str,
        llm_extractor_model_config: ModelConfig,
        llm_extractor_max_concurrent: int = 1,
        llm_extractor_answer_transforms: list = None,
    ):

        self.filter_empty_answer = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(extraction_attempt_component.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            RunPythonTransform(f"df = df[df['{extracted_answer_col}'] == '']"),
                            ColumnRename(name_mapping={"prompt": "original_prompt"}),
                        ]
                    ),
                },
            ),
            prompt_template_path=llm_extraction_promp_template,
            output_dir=os.path.join(self.log_dir, "filter_empty_answer"),
        )

        self.inference_llm_answer_extract = InferenceConfig(
            component_type=Inference,
            model_config=llm_extractor_model_config,
            data_loader_config=DataSetConfig(
                MMDataLoader,
                {"path": os.path.join(self.filter_empty_answer.output_dir, "transformed_data.jsonl")},
            ),
            output_dir=os.path.join(self.log_dir, "llm_answer_extract_inference_result"),
            max_concurrent=llm_extractor_max_concurrent,
        )

        self.data_join = DataJoinConfig(
            component_type=DataJoin,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(extraction_attempt_component.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                },
            ),
            other_data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference_llm_answer_extract.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            # drop all columns except the uid and model_output
                            RunPythonTransform(
                                "df = df[[col for col in ['data_repeat_id','data_point_id', 'model_output'] if col in df.columns]]"
                            ),
                        ]
                        + llm_extractor_answer_transforms
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_join_output"),
            pandas_merge_args={"on": ["data_repeat_id", "data_point_id"], "how": "left"},
        )

        self.post_join_consolidation_component = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.data_join.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            # consolidate model_output_y to replace the original model_output whenever empty
                            # the initial if statement checks whether there has been a join beforehand
                            RunPythonTransform(
                                "df['model_output'] = df.apply(lambda row: row['model_output'] if 'model_output_x' not in row else row['model_output_y'] if row['model_output_x'] == '' else row['model_output_x'], axis=1)"
                            ),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "post_join_consolidation_output"),
        )
        return [
            self.filter_empty_answer,
            self.inference_llm_answer_extract,
            self.data_join,
            self.post_join_consolidation_component,
        ]
