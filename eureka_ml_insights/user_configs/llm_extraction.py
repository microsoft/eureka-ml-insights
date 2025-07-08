import os

"""
Module for configuring the LLM extraction subpipeline.

This module contains the LLM_EXTRACTION_SUBPIPELINE_MIXIN class, which is used to
configure the LLM extraction subpipeline. Use it after the extraction attempt
component in the main pipeline, which should be a DataProcessing component passed
to the configure_subpipeline method.

The transformed_data.jsonl file from the extraction attempt component is used as
input to the LLM extraction subpipeline, and must contain the following columns:
- prompt
- uid
- data_point_id (optional) is added in this subpipeline if not present
- data_repeat_id (optional) is added in this subpipeline if not present
- {extracted_answer_col}

The output of the LLM extraction subpipeline is a transformed_data.jsonl file with
the following new/updated columns:
- prompt (the prompt used for the LLM extraction)
- original_prompt (the original prompt present in the input data to this subpipeline)
- data_repeat_id (if not already present in the input data)
- data_point_id (if not already present in the input data)
- {extracted_answer_col} column containing the LLM-extracted answer where the
  original extracted_answer_col was empty.
"""

from eureka_ml_insights.configs import (
    DataJoinConfig,
    DataProcessingConfig,
    DataSetConfig,
    InferenceConfig,
    ModelConfig,
    PromptProcessingConfig,
)
from eureka_ml_insights.core import (
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
    """
    Configures the LLM extraction subpipeline.

    This mixin is used to define the steps needed for extracting answers
    using a large language model (LLM) in a subpipeline.
    """

    def configure_subpipeline(
        self,
        extraction_attempt_component: DataProcessing,
        extracted_answer_col: str,
        llm_extraction_prompt_template: str,
        llm_extractor_model_config: ModelConfig,
        log_dir: str,
        llm_extractor_max_concurrent: int = 1,
        llm_extractor_answer_transforms: list = [],
        not_extracted_answer_value: str = "",
    ):
        """
        Configures and returns the LLM extraction subpipeline components.

        Args:
            extraction_attempt_component (DataProcessing): DataProcessing component that
                was used to extract the answer.
            extracted_answer_col (str): Name of the column that contains the extracted
                answer in the output of the extraction_attempt_component.
            llm_extraction_prompt_template (str): Path to the prompt template file for
                the LLM extraction.
            llm_extractor_model_config (ModelConfig): Config for the LLM model to be
                used for extraction.
            log_dir (str): Directory corresponding to the output directory of the
                calling pipeline.
            llm_extractor_max_concurrent (int, optional): max_concurrent parameter for
                the inference component used for LLM extraction. Defaults to 1.
            llm_extractor_answer_transforms (list, optional): List of transforms to be
                applied to the model output after the LLM extraction. Defaults to [].
            not_extracted_answer_value (str, optional): Placeholder that signals no valid
                answer yet in the extracted_answer_col. Defaults to "".

        Returns:
            list: A list of components that constitute the LLM extraction subpipeline.
        """

        self.filter_empty_answer = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(extraction_attempt_component.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            # if data_repeat_id/data_point_id is not present, create it just so the subpipeline can run both with and without MultiplyTransform applied to the original data
                            RunPythonTransform(
                                "df['data_repeat_id'] = df.get('data_repeat_id', pd.Series('repeat_0', index=df.index)); df['data_point_id'] = df.get('data_point_id', df.get('uid'))"
                            ),
                            RunPythonTransform(
                                f"df = df[df['{extracted_answer_col}'] == '{not_extracted_answer_value}']"
                            ),
                            ColumnRename(name_mapping={"prompt": "original_prompt"}),
                        ]
                    ),
                },
            ),
            prompt_template_path=llm_extraction_prompt_template,
            output_dir=os.path.join(log_dir, "filter_empty_answer"),
        )

        self.inference_llm_answer_extract = InferenceConfig(
            component_type=Inference,
            model_config=llm_extractor_model_config,
            data_loader_config=DataSetConfig(
                MMDataLoader,
                {"path": os.path.join(self.filter_empty_answer.output_dir, "transformed_data.jsonl")},
            ),
            output_dir=os.path.join(log_dir, "llm_answer_extract_inference_result"),
            max_concurrent=llm_extractor_max_concurrent,
        )

        self.data_join = DataJoinConfig(
            component_type=DataJoin,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(extraction_attempt_component.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                    # if data_repeat_id/data_point_id is not present, create it just so the subpipeline can run both with and without MultiplyTransform applied to the original data
                    "transform": RunPythonTransform(
                        "df['data_repeat_id'] = df.get('data_repeat_id', pd.Series('repeat_0', index=df.index)); df['data_point_id'] = df.get('data_point_id', df.get('uid'))"
                    ),
                },
            ),
            other_data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference_llm_answer_extract.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            # drop all columns except the uid and extracted_answer_col
                            RunPythonTransform(
                                f"df = df[[col for col in ['data_repeat_id','data_point_id', 'model_output'] if col in df.columns]]"
                            ),
                            # rename model_output to extracted_answer_col
                            ColumnRename(name_mapping={"model_output": extracted_answer_col}),
                        ]
                        + llm_extractor_answer_transforms
                    ),
                },
            ),
            output_dir=os.path.join(log_dir, "data_join_output"),
            pandas_merge_args={"on": ["data_repeat_id", "data_point_id"], "how": "left"},
        )

        self.post_join_consolidation_component = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.data_join.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                    # consolidate new {extracted_answer_col} to replace the original {extracted_answer_col} whenever the latter is empty
                    # the initial if statement checks whether there has been a join beforehand
                    "transform": RunPythonTransform(
                        f"df['{extracted_answer_col}'] = df.apply(lambda row: row['{extracted_answer_col}'] if '{extracted_answer_col}_x' not in row else row['{extracted_answer_col}_y'] if row['{extracted_answer_col}_x'] == '' else row['{extracted_answer_col}_x'], axis=1)"
                    ),
                },
            ),
            output_dir=os.path.join(log_dir, "post_join_consolidation_output"),
        )
        return [
            self.filter_empty_answer,
            self.inference_llm_answer_extract,
            self.data_join,
            self.post_join_consolidation_component,
        ]
