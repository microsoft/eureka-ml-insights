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
        """
        Initialize the Inference component.
        args:
            model_config (dict): ModelConfig object.
            data_config (dict): DataSetConfig object.
            output_dir (str): Directory to save the inference results.
            resume_from (str): optional. Path to the file where previous inference results are stored.
            new_columns (list): optional. List of new columns to be added to resume_from data to match the current inference response.
            requests_per_minute (int): optional. Number of inference requests to be made per minute, used for rate limiting. If not provided, rate limiting will not be applied.
            max_concurrent (int): optional. Maximum number of concurrent inferences to run. Default is 1.
            chat_mode (bool): optional. If True, the model will be used in chat mode, where a history of messages will be maintained in "previous_messages" column.
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
        if self.model.max_concurrent and hasattr(self.model, "handler"):
            self.max_concurrent = self.model.max_concurrent * len(self.model.handler.clients)
        elif self.model.max_concurrent:
            self.max_concurrent = self.model.max_concurrent
        else:
            self.max_concurrent = max_concurrent
        self.chat_mode = chat_mode
        self.model.chat_mode = self.chat_mode
        self.output_dir = output_dir
        self.writer_lock = threading.Lock()

    @classmethod
    def from_config(cls, config):
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
        """This method loads the contents of the resume_from file and validates if it
        contains the required columns and keys in alignment with the current model configuration."""

        logging.info(f"Resuming inference from {self.resume_from}")
        # fetch previous results from the provided resume_from file.
        pre_inf_results_df = DataReader(self.resume_from, format=".jsonl").load_dataset()

        # add new columns listed by the user to the previous inference results, in case we know that the current model will
        # generate new columns that were not present in the previous results
        if self.new_columns:
            for col in self.new_columns:
                if col not in pre_inf_results_df.columns:
                    pre_inf_results_df[col] = None

        # validate the resume_from contents both stand-alone and against the current model response keys
        with self.data_loader as loader:
            _, sample_model_input, sample_model_kwargs = loader.get_sample_model_input()
            sample_data_keys = loader.reader.read().keys()

            # verify that "model_output" and "is_valid" columns are present
            if "model_output" not in pre_inf_results_df.columns or "is_valid" not in pre_inf_results_df.columns:
                raise ValueError("Columns 'model_output' and 'is_valid' are required in the resume_from file.")

            # perform a sample inference call to get the model output keys and validate the resume_from contents
            sample_response_dict = self.model.generate(*sample_model_input, **sample_model_kwargs)
            if not sample_response_dict["is_valid"]:
                raise ValueError(
                    "Sample inference call for resume_from returned invalid results, please check the model configuration."
                )
            # check if the inference response dictionary contains the same keys as the resume_from file
            eventual_keys = set(sample_response_dict.keys()) | set(sample_data_keys)

            # in case of resuming from a file that was generated by an older version of the model,
            # we let the discrepancy in the reserved keys slide and later set the missing keys to None
            match_keys = set(pre_inf_results_df.columns) | set(INFERENCE_RESERVED_NAMES)

            if set(eventual_keys) != match_keys:
                diff = set(eventual_keys) ^ set(match_keys)
                raise ValueError(
                    f"Columns in resume_from file do not match the current input data and inference response. "
                    f"Problemtaic columns: {diff}"
                )

        # find the last uid that was inferenced
        last_uid = pre_inf_results_df["uid"].astype(int).max()
        logging.info(f"Last uid inferenced: {last_uid}")
        return pre_inf_results_df, last_uid

    def validate_response_dict(self, response_dict):
        # Validate that the response dictionary contains the required fields
        # "model_output" and "is_valid" are mandatory fields to be returned by any model
        if "model_output" not in response_dict or "is_valid" not in response_dict:
            raise ValueError("Response dictionary must contain 'model_output' and 'is_valid' keys.")

    def retrieve_exisiting_result(self, data, pre_inf_results_df):
        """Finds the previous result for the given data point from the pre_inf_results_df and returns it if it is valid
        data: dict, data point to be inferenced
        pre_inf_results_df: pd.DataFrame, previous inference results
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
            # add remaining pre_inf_results_df columns to the data point, making sure to update the previous_messages column
            for col in pre_inf_results_df.columns:
                if col not in data or col == "previous_messages":
                    data[col] = prev_results[col].values[0]

            return data

    def run(self):
        with self.appender as appender:
            pass  # create the output file. This will guarantee that an empty inference_result.jsonl file is created even if no inference is run.
        if self.resume_from:
            self.pre_inf_results_df, self.last_uid = self.fetch_previous_inference_results()

        # If batch_size > 1 and the model supports batch inference, opt for that instead of single inference.
        if hasattr(self.model, 'batch_size') and self.model.batch_size > 1:
            if hasattr(self.model, 'batch_generate') and callable(self.model.batch_generate):
                self._run_batch_inference()

        else:
            with self.data_loader as loader, ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
                futures = [executor.submit(self._run_single, record) for record in loader]
                for future in tqdm(as_completed(futures), total=len(loader), mininterval=2.0, desc="Inference Progress: "):
                    result = future.result()
                    if result:
                        self._append_threadsafe(result)

    def _append_threadsafe(self, data):
        with self.writer_lock:
            with self.appender as appender:
                appender.write(data)

    def _run_batch_inference(self):

        # Check if the model requires equal prompt length for batch inference
        # and if the get_prompt_length method is defined and callable.
        eq_prompts = hasattr(self.model, 'requires_equal_prompt_length') and self.model.requires_equal_prompt_length
        if eq_prompts and (not hasattr(self.model, 'get_prompt_length') or not callable(self.model.get_prompt_length)):
            raise ValueError("Model requires equal prompt length for batch inference, but get_prompt_length method is not defined or callable.")
        
        batches = {}
        # Organize all records from the data loader into batches (based on existence of image and prompt length, if required).
        with self.data_loader as loader:
            for record in loader:
                data, model_args, model_kwargs = record
                if self.resume_from and (data["uid"] <= self.last_uid):
                    prev_result = self.retrieve_exisiting_result(data, self.pre_inf_results_df)
                    if prev_result:
                        self._append_threadsafe(prev_result)
                else:
                    num_imgs = len(model_args[1]) if (len(model_args) > 1 and model_args[1] is not None and isinstance(model_args[1], list)) else 0
                    prompt_length = self.model.get_prompt_length(model_args[0], num_images=num_imgs) if eq_prompts else -1
                    key = (num_imgs, prompt_length)
                    if key not in batches:
                        batches[key] = [[]]
                    cur_batch = batches[key][-1]
                    if len(cur_batch) < self.model.batch_size:
                        cur_batch.append(record)
                    else:
                        batches[key].append([record])
            all_batches = [batch for batch_list in batches.values() for batch in batch_list if len(batch) > 0]

        # Determine the number of servers if remote-hosted, and kick off the batch inference using ThreadPoolExecutor.
        if hasattr(self.model, "ports") and self.model.ports is not None and isinstance(self.model.ports, list):
            num_threads = len(self.model.ports)
        else:
            num_threads = 1
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(self._run_batch, *(batch, i)) for i, batch in enumerate(all_batches)]
            for future in tqdm(as_completed(futures), total=len(all_batches), mininterval=2.0, desc="Batch Inference Progress: "):
                future.result()

    def _run_batch(self, records: list[tuple[dict, tuple, dict]], index: int):
        """Runs model.batch_generate() with respect to a batch of elements from the dataloader."""

        prompts = []
        query_images = []
        for record in records:
            _, model_args, _ = record
            prompts.append(model_args[0])
            if len(model_args) > 1:
                if model_args[1] is None:
                    query_images = None
                elif isinstance(model_args[1], list) and len(model_args[1]) == 0:
                    query_images = None
                elif isinstance(model_args[1], list):
                    query_images.append(model_args[1])
                else:
                    query_images.append([model_args[1]])
        response_dicts = self.model.batch_generate(prompts, query_images=query_images, index=index)
        for record, response_dict in zip(records, response_dicts):
            self.validate_response_dict(response_dict)
            data, _, _ = record
            data.update(response_dict)
            self._append_threadsafe(data)

    def _run_single(self, record: tuple[dict, tuple, dict]):
        """Runs model.generate() with respect to a single element of the dataloader."""

        data, model_args, model_kwargs = record
        if self.chat_mode and data.get("is_valid", True) is False:
            return None
        if self.resume_from and (data["uid"] <= self.last_uid):
            prev_result = self.retrieve_exisiting_result(data, self.pre_inf_results_df)
            if prev_result:
                return prev_result

        # Rate limiter -- only for sequential inference
        if self.requests_per_minute and self.max_concurrent == 1:
            while len(self.request_times) >= self.requests_per_minute:
                # remove the oldest request time if it is older than the rate limit period
                if time.time() - self.request_times[0] > self.period:
                    self.request_times.popleft()
                else:
                    # rate limit is reached, wait for a second
                    time.sleep(1)
            self.request_times.append(time.time())

        response_dict = self.model.generate(*model_args, **model_kwargs)
        self.validate_response_dict(response_dict)
        data.update(response_dict)
        return data
