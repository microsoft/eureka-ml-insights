import asyncio
import logging
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from eureka_ml_insights.data_utils.data import DataReader, JsonLinesWriter

from .pipeline import Component

MINUTE = 60


class Inference(Component):
    def __init__(self, model_config, data_config, output_dir, resume_from=None, n_calls_per_min=None, max_concurrent=1):
        """
        Initialize the Inference component.
        args:
            model_config (dict): ModelConfig object.
            data_config (dict): DataSetConfig object.
            output_dir (str): Directory to save the inference results.
            resume_from (str): optional. Path to the file where previous inference results are stored.
            n_calls_per_min (int): optional. Number of calls to be made per minute, used for rate limiting. If not provided, rate limiting will not be applied.
            max_concurrent (int): optional. Maximum number of concurrent inferences to run. Default is 1.
        """
        super().__init__(output_dir)
        self.model = model_config.class_name(**model_config.init_args)
        self.data_loader = data_config.class_name(**data_config.init_args)
        self.writer = JsonLinesWriter(os.path.join(output_dir, "inference_result.jsonl"))

        self.resume_from = resume_from
        if resume_from and not os.path.exists(resume_from):
            raise FileNotFoundError(f"File {resume_from} not found.")

        # rate limiting parameters
        self.n_calls_per_min = n_calls_per_min
        self.call_times = deque()
        self.period = MINUTE

        # parallel inference parameters
        self.max_concurrent = max_concurrent

    @classmethod
    def from_config(cls, config):
        return cls(
            config.model_config,
            config.data_loader_config,
            config.output_dir,
            resume_from=config.resume_from,
            n_calls_per_min=config.n_calls_per_min,
            max_concurrent=config.max_concurrent,
        )

    def fetch_previous_inference_results(self):
        # fetch previous results from the provided resume_from file
        logging.info(f"Resuming inference from {self.resume_from}")
        pre_inf_results_df = DataReader(self.resume_from, format=".jsonl").load_dataset()

        # validate the resume_from contents
        with self.data_loader as loader:
            sample_data = loader.reader.read()
            sample_data_keys = sample_data.keys()

            # verify that "model_output" and "is_valid" columns are present
            if "model_output" not in pre_inf_results_df.columns or "is_valid" not in pre_inf_results_df.columns:
                raise ValueError("Columns 'model_output' and 'is_valid' are required in the resume_from file.")

            # check if remaining columns match those in current data loader
            pre_inf_results_keys = pre_inf_results_df.columns.drop(["model_output", "is_valid"])
            if set(sample_data_keys) != set(pre_inf_results_keys):
                raise ValueError(
                    f"Columns in resume_from do not match the columns in the current data loader."
                    f"Current data loader columns: {sample_data_keys}. "
                    f"Provided inference results columns: {pre_inf_results_keys}."
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
        prev_result_is_valid = bool(prev_results["is_valid"].values[0])
        prev_model_output = prev_results["model_output"].values[0]
        if prev_result_is_valid:
            logging.info(f"Skipping inference for uid: {data['uid']}. Using previous results.")
            data["model_output"], data["is_valid"] = prev_model_output, prev_result_is_valid
            return data

    def run(self):
        if self.max_concurrent > 1:
            asyncio.run(self._run_par())
        else:
            self._run()

    def _run(self):
        """sequential inference"""
        if self.resume_from:
            pre_inf_results_df, last_uid = self.fetch_previous_inference_results()
        with self.data_loader as loader:
            with self.writer as writer:
                for data, model_inputs in tqdm(loader, desc="Inference Progress:"):

                    if self.resume_from and (data["uid"] <= last_uid):
                        prev_result = self.retrieve_exisiting_result(data, pre_inf_results_df)
                        if prev_result:
                            writer.write(prev_result)
                            continue

                    # generate text from model (optionally at a limited rate)
                    if self.n_calls_per_min:
                        while len(self.call_times) >= self.n_calls_per_min:
                            # remove the oldest call time if it is older than the rate limit period
                            if time.time() - self.call_times[0] > self.period:
                                self.call_times.popleft()
                            else:
                                # rate limit is reached, wait for a second
                                time.sleep(1)
                        self.call_times.append(time.time())
                    response_dict = self.model.generate(*model_inputs)
                    self.validate_response_dict(response_dict)
                    # write results
                    data.update(response_dict)
                    writer.write(data)

    async def run_in_excutor(self, model_inputs, executor):
        """Run model.generate in a ThreadPoolExecutor.
        args:
            model_inputs (tuple): inputs to the model.generate function.
            executor (ThreadPoolExecutor): ThreadPoolExecutor instance.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, self.model.generate, *model_inputs)

    async def _run_par(self):
        """parallel inference"""
        concurrent_inputs = []
        concurrent_metadata = []
        if self.resume_from:
            pre_inf_results_df, last_uid = self.fetch_previous_inference_results()
        with self.data_loader as loader:
            with self.writer as writer:
                for data, model_inputs in tqdm(loader, desc="Inference Progress:"):
                    if self.resume_from and (data["uid"] <= last_uid):
                        prev_result = self.retrieve_exisiting_result(data, pre_inf_results_df)
                        if prev_result:
                            writer.write(prev_result)
                            continue

                    # if batch is ready for concurrent inference
                    elif len(concurrent_inputs) >= self.max_concurrent:
                        with ThreadPoolExecutor() as executor:
                            await self.run_batch(concurrent_inputs, concurrent_metadata, writer, executor)
                        concurrent_inputs = []
                        concurrent_metadata = []
                    else:
                        # add data to batch for concurrent inference
                        concurrent_inputs.append(model_inputs)
                        concurrent_metadata.append(data)
                # if data loader is exhausted but there are remaining data points that did not form a full batch
                if concurrent_inputs:
                    with ThreadPoolExecutor() as executor:
                        await self.run_batch(concurrent_inputs, concurrent_metadata, writer, executor)

    async def run_batch(self, concurrent_inputs, concurrent_metadata, writer, executor):
        """Run a batch of inferences concurrently using ThreadPoolExecutor.
        args:
            concurrent_inputs (list): list of inputs to the model.generate function.
            concurrent_metadata (list): list of metadata corresponding to the inputs.
            writer (JsonLinesWriter): JsonLinesWriter instance to write the results.
            executor (ThreadPoolExecutor): ThreadPoolExecutor instance.
        """
        tasks = [asyncio.create_task(self.run_in_excutor(input_data, executor)) for input_data in concurrent_inputs]
        results = await asyncio.gather(*tasks)
        for i in range(len(concurrent_inputs)):
            data, response_dict = concurrent_metadata[i], results[i]
            self.validate_response_dict(response_dict)
            # prepare results for writing
            data.update(response_dict)
            writer.write(data)
