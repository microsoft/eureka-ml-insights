import logging
import os
import statistics
from hashlib import md5
from typing import List, Optional

from transformers import GPT2TokenizerFast

from eureka_ml_insights.data_utils import JinjaPromptTemplate

from .data_processing import DataProcessing
from .reserved_names import INFERENCE_RESERVED_NAMES


def compute_hash(val: str) -> str:
    """
    Hashes the provided value using MD5.
    """
    return md5(val.encode("utf-8")).hexdigest()


class PromptProcessing(DataProcessing):
    @classmethod
    def from_config(cls, config):
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
        """
        args:
            data_reader_config: DataReaderConfig
            prompt_template_path: str path to the prompt template .jinja file.
            output_dir: str directory to save the output files of this component.
            output_data_columns: Optional[List[str]] list of columns (subset of input columns)
                                      to keep in the transformed data output file.
            ignore_failure: bool whether to ignore failure in prompt generation or not.
        """
        super().__init__(data_reader_config, output_dir, output_data_columns)
        self.ignore_failure = ignore_failure
        if prompt_template_path is None:
            self.prompt_data_processor = None
            logging.info('Prompt template path is not provided, will use "prompt" column in the input data as prompt.')
        else:
            self.prompt_data_processor = JinjaPromptTemplate(prompt_template_path)

    def run(self) -> None:
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
            prompt_num_tokens = []
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            with open(prompt_output_file, "w", encoding="utf-8") as writer:
                for i, row in input_df.iterrows():

                    placeholders = row.to_dict()
                    try:
                        prompt = self.prompt_data_processor.create(placeholders)
                        success_indexes.append(i)
                        prompt_num_tokens.append(len(tokenizer.tokenize(prompt)))
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

            logging.info(f"Average prompt num tokens: {statistics.fmean(prompt_num_tokens)}.")

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
