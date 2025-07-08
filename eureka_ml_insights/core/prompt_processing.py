"""This module provides functionalities for prompt processing, including a function to compute an MD5 hash and
a class that extends DataProcessing to handle prompt generation workflows.
"""

import logging
import os
from hashlib import md5
from typing import List, Optional

from eureka_ml_insights.data_utils import JinjaPromptTemplate

from .data_processing import DataProcessing
from .reserved_names import INFERENCE_RESERVED_NAMES


def compute_hash(val: str) -> str:
    """Compute the MD5 hash of a given string.

    Args:
        val (str): The value to be hashed.

    Returns:
        str: The MD5 hash of the given value.
    """
    return md5(val.encode("utf-8")).hexdigest()


class PromptProcessing(DataProcessing):
    """Handles the prompt generation workflow by extending DataProcessing."""

    @classmethod
    def from_config(cls, config):
        """Create a PromptProcessing instance from a configuration object.

        Args:
            config: The configuration object.

        Returns:
            PromptProcessing: A new PromptProcessing instance.
        """
        return cls(
            config.data_reader_config,
            config.output_dir,
            config.output_data_columns,
            config.prompt_template_path,
            config.ignore_failure,
        )

    def __init__(
        self,
        data_reader_config,
        output_dir: str,
        output_data_columns: Optional[List[str]] = None,
        prompt_template_path: Optional[str] = None,
        ignore_failure: bool = False,
    ) -> None:
        """Initialize the PromptProcessing object.

        Args:
            data_reader_config: DataReaderConfig object that specifies the data reading configuration.
            output_dir (str): Directory to save the output files of this component.
            output_data_columns (Optional[List[str]]): A list of columns (subset of input columns) to keep in
                the transformed data output file.
            prompt_template_path (Optional[str]): Path to the prompt template .jinja file.
            ignore_failure (bool): Whether to ignore failure in prompt generation or not.
        """
        super().__init__(data_reader_config, output_dir, output_data_columns)
        self.ignore_failure = ignore_failure
        if prompt_template_path is None:
            self.prompt_data_processor = None
            logging.info('Prompt template path is not provided, will use "prompt" column in the input data as prompt.')
        else:
            self.prompt_data_processor = JinjaPromptTemplate(prompt_template_path)

    def run(self) -> None:
        """Execute the prompt processing workflow.

        Loads input data, optionally uses a Jinja template to create prompts, and writes
        output data with generated prompts and their hashes to the output directory.

        Raises:
            Exception: If an error occurs during prompt creation and ignore_failure is False.
        """
        # data reader loads data into a pandas dataframe and applies any transformations
        input_df = self.data_reader.load_dataset()
        logging.info(f"input has: {len(input_df)} rows, and the columns are: {input_df.columns}.")
        prompt_output_file = os.path.join(self.output_dir, "processed_prompts.jsonl")
        success_indexes = []
        prompt_hashes = []
        prompts = []

        # if prompt data processor is not provided, use the prompt column in the input data as prompt
        if self.prompt_data_processor is None:
            prompts = input_df["prompt"].tolist()
            prompt_hashes = [compute_hash(prompt) for prompt in prompts]
        # otherwise, use the prompt data processor to generate prompts and save in the "prompt" column
        else:
            with open(prompt_output_file, "w", encoding="utf-8") as writer:
                for i, row in input_df.iterrows():

                    placeholders = row.to_dict()
                    try:
                        prompt = self.prompt_data_processor.create(placeholders)
                        success_indexes.append(i)
                        prompt_hashes.append(compute_hash(prompt))
                        prompts.append(prompt)
                        writer.write(prompt + "\n")
                    except Exception as e:
                        if self.ignore_failure:
                            prompt_hashes.append("")
                            prompts.append("")
                            continue
                        else:
                            raise e

        input_df = self.get_desired_columns(input_df)
        # Remove `model_output`, `is_valid`, `response_time`, `n_output_tokens` columns if they exists
        # in the data because these names are reserved for the inference component's use.
        for col in INFERENCE_RESERVED_NAMES:
            if col in self.output_data_columns:
                self.output_data_columns.remove(col)
                logging.warning(
                    f"Removed '{col}' column from transformed data columns because it is reserved for the inference component."
                )

        input_df = input_df[self.output_data_columns]
        input_df["prompt_hash"] = prompt_hashes
        input_df["prompt"] = prompts
        input_df["uid"] = input_df.index

        # filter to only valid rows
        if self.prompt_data_processor is not None:
            logging.info(f"There are {len(success_indexes)} out of {len(input_df)} prompts generated.")
            input_df = input_df.iloc[success_indexes]

        self.write_output(input_df)
