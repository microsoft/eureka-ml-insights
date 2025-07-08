"""This module provides the Inference class, which extends the Component class to handle inference
processes for a given dataset and model configuration. It supports resuming from existing inference
results, applying rate limiting, and running parallel inferences.
"""

import logging
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from eureka_ml_insights.configs.config import DataSetConfig, ModelConfig
from eureka_ml_insights.data_utils.data import DataReader, JsonLinesWriter
from eureka_ml_insights.models.models import Model

from .pipeline import Component
from .reserved_names import INFERENCE_RESERVED_NAMES

MINUTE = 60


class Inference(Component):
    """Handles inference processes for a given dataset and model configuration."""

    def __init__(
        self,
        model_config: ModelConfig,
        data_config: DataSetConfig,
        output_dir,
        resume_from=None,
        new_columns=None,
        requests_per_minute=None,
        max_concurrent=1,
        chat_mode=False,
    ):
        """Initializes the Inference component.

        Args:
            model_config (ModelConfig): Object specifying the model class and initialization arguments.
            data_config (DataSetConfig): Object specifying the data loader class and initialization arguments.
            output_dir (str): Directory to save the inference results.
            resume_from (Optional[str]): Path to a file containing previous inference results to resume from.
            new_columns (Optional[list]): List of new columns to match the current inference response when resuming from old results.
            requests_per_minute (Optional[int]): Number of inference requests per minute for rate limiting. If None, no rate limiting is applied.
            max_concurrent (int): Maximum number of concurrent inferences. Defaults to 1.
            chat_mode (bool): If True, runs in chat mode with a maintained history of messages in the "previous_messages" column.

        Raises:
            FileNotFoundError: If the provided resume_from file is not found.
        """
        super().__init__(output_dir)
        self.model: Model = model_config.class_name(**model_config.init_args)
        self.data_loader = data_config.class_name(**data_config.init_args)
        self.appender = JsonLinesWriter(os.path.join(output_dir, "inference_result.jsonl"), mode="a")

        self.resume_from = resume_from
        if resume_from and not os.path.exists(resume_from):
            raise FileNotFoundError(f"File {resume_from} not found.")
        self.new_columns = new_columns
        self.pre_inf_results_df = None
        self.last_uid = None

        # rate limiting parameters
        self.requests_per_minute = requests_per_minute
        self.request_times = deque()
        self.period = MINUTE

        # parallel inference parameters
        self.max_concurrent = max_concurrent
        self.chat_mode = chat_mode
        self.model.chat_mode = self.chat_mode
        self.output_dir = output_dir
        self.writer_lock = threading.Lock()

    @classmethod
    def from_config(cls, config):
        """Creates an Inference instance from a provided configuration object.

        Args:
            config: A configuration object containing attributes:
                model_config, data_loader_config, output_dir, resume_from,
                new_columns, requests_per_minute, max_concurrent, and chat_mode.

        Returns:
            Inference: An instance of the Inference class.
        """
        return cls(
            config.model_config,
            config.data_loader_config,
            config.output_dir,
            resume_from=config.resume_from,
            new_columns=config.new_columns,
            requests_per_minute=config.requests_per_minute,
            max_concurrent=config.max_concurrent,
            chat_mode=config.chat_mode,
        )

    def fetch_previous_inference_results(self):
        """Loads the contents from the resume_from file and validates alignment with the current
        model configuration.

        Returns:
            Tuple[pd.DataFrame, int]: A tuple containing:
                - The DataFrame with previous inference results.
                - The highest uid value that was processed.

        Raises:
            ValueError: If the 'model_output' or 'is_valid' columns are not found in the resume_from file.
            ValueError: If the sample inference call returns invalid results.
            ValueError: If there is a mismatch between the columns in the resume_from file and the model's expected keys.
        """
        logging.info(f"Resuming inference from {self.resume_from}")
        pre_inf_results_df = DataReader(self.resume_from, format=".jsonl").load_dataset()

        if self.new_columns:
            for col in self.new_columns:
                if col not in pre_inf_results_df.columns:
                    pre_inf_results_df[col] = None

        with self.data_loader as loader:
            _, sample_model_input, sample_model_kwargs = loader.get_sample_model_input()
            sample_data_keys = loader.reader.read().keys()

            if "model_output" not in pre_inf_results_df.columns or "is_valid" not in pre_inf_results_df.columns:
                raise ValueError("Columns 'model_output' and 'is_valid' are required in the resume_from file.")

            sample_response_dict = self.model.generate(*sample_model_input, **sample_model_kwargs)
            if not sample_response_dict["is_valid"]:
                raise ValueError(
                    "Sample inference call for resume_from returned invalid results, please check the model configuration."
                )
            eventual_keys = set(sample_response_dict.keys()) | set(sample_data_keys)
            match_keys = set(pre_inf_results_df.columns) | set(INFERENCE_RESERVED_NAMES)

            if set(eventual_keys) != match_keys:
                diff = set(eventual_keys) ^ set(match_keys)
                raise ValueError(
                    f"Columns in resume_from file do not match the current input data and inference response. "
                    f"Problemtaic columns: {diff}"
                )

        last_uid = pre_inf_results_df["uid"].astype(int).max()
        logging.info(f"Last uid inferenced: {last_uid}")
        return pre_inf_results_df, last_uid

    def validate_response_dict(self, response_dict):
        """Validates that the response dictionary contains the mandatory fields 'model_output'
        and 'is_valid'.

        Args:
            response_dict (dict): The response dictionary returned by the model.

        Raises:
            ValueError: If 'model_output' or 'is_valid' is missing.
        """
        if "model_output" not in response_dict or "is_valid" not in response_dict:
            raise ValueError("Response dictionary must contain 'model_output' and 'is_valid' keys.")

    def retrieve_exisiting_result(self, data, pre_inf_results_df):
        """Retrieves a valid previous inference result for a given data record.

        Args:
            data (dict): The data record to be inferenced.
            pre_inf_results_df (pd.DataFrame): The DataFrame containing previous inference results.

        Returns:
            dict or None: The updated data record with previous model outputs if valid, otherwise None.
        """
        prev_results = pre_inf_results_df[pre_inf_results_df.uid == data["uid"]]
        if prev_results.empty:
            return None
        prev_result_is_valid = bool(prev_results["is_valid"].values[0])
        prev_model_output = prev_results["model_output"].values[0]

        if prev_result_is_valid:
            logging.info(f"Skipping inference for uid: {data['uid']}. Using previous results.")
            try:
                prev_model_tokens = prev_results["n_output_tokens"].values[0]
            except KeyError:
                logging.warn(
                    "Previous results do not contain 'n_output_tokens' column, setting to None for this data point."
                )
                prev_model_tokens = None
            try:
                prev_model_time = prev_results["response_time"].values[0]
            except KeyError:
                logging.warn(
                    "Previous results do not contain 'response_time' column, setting to None for this data point."
                )
                prev_model_time = None

            data["model_output"], data["is_valid"], data["n_output_tokens"], data["response_time"] = (
                prev_model_output,
                prev_result_is_valid,
                prev_model_tokens,
                prev_model_time,
            )
            for col in pre_inf_results_df.columns:
                if col not in data or col == "previous_messages":
                    data[col] = prev_results[col].values[0]

            return data

    def run(self):
        """Executes the inference process, optionally resuming from previous results, and writes
        the final results to the output file.

        This method manages parallel execution with a ThreadPoolExecutor, performs rate limiting
        if specified, and tracks progress with a tqdm progress bar.
        """
        with self.appender as appender:
            pass
        if self.resume_from:
            self.pre_inf_results_df, self.last_uid = self.fetch_previous_inference_results()
        with self.data_loader as loader, ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            futures = [executor.submit(self._run_single, record) for record in loader]
            for future in tqdm(as_completed(futures), total=len(loader), mininterval=2.0, desc="Inference Progress: "):
                result = future.result()
                if result:
                    self._append_threadsafe(result)

    def _append_threadsafe(self, data):
        """Appends inference results to the output file in a thread-safe manner.

        Args:
            data (dict): The inference result data to be appended.
        """
        with self.writer_lock:
            with self.appender as appender:
                appender.write(data)

    def _run_single(self, record: tuple[dict, tuple, dict]):
        """Runs the model's generate method for a single data record from the data loader.

        Args:
            record (Tuple[dict, tuple, dict]): A tuple containing:
                - The data record (dict).
                - The model's positional arguments (tuple).
                - The model's keyword arguments (dict).

        Returns:
            dict or None: The updated data record with inference results, or None if inference was
            skipped or the record is invalid.
        """
        data, model_args, model_kwargs = record
        if self.chat_mode and data.get("is_valid", True) is False:
            return None
        if self.resume_from and (data["uid"] <= self.last_uid):
            prev_result = self.retrieve_exisiting_result(data, self.pre_inf_results_df)
            if prev_result:
                return prev_result

        if self.requests_per_minute and self.max_concurrent == 1:
            while len(self.request_times) >= self.requests_per_minute:
                if time.time() - self.request_times[0] > self.period:
                    self.request_times.popleft()
                else:
                    time.sleep(1)
            self.request_times.append(time.time())

        response_dict = self.model.generate(*model_args, **model_kwargs)
        self.validate_response_dict(response_dict)
        data.update(response_dict)
        return data
