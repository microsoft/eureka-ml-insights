import logging
import os

from tqdm import tqdm

from eureka_ml_insights.data_utils.data import DataReader, JsonLinesWriter

from .pipeline import Component


class Inference(Component):
    def __init__(self, model_config, data_config, output_dir, resume_from=None):
        """
        Initialize the Inference component.
        args:
            model_config (dict): ModelConfig object.
            data_config (dict): DataSetConfig object.
            output_dir (str): Directory to save the inference results.
            resume_from (str): optional. Path to the file where previous inference results are stored.
        """
        super().__init__(output_dir)
        self.model = model_config.class_name(**model_config.init_args)
        self.data_loader = data_config.class_name(**data_config.init_args)
        self.writer = JsonLinesWriter(os.path.join(output_dir, "inference_result.jsonl"))
        self.resume_from = resume_from
        if resume_from and not os.path.exists(resume_from):
            raise FileNotFoundError(f"File {resume_from} not found.")

    @classmethod
    def from_config(cls, config):
        return cls(config.model_config, config.data_loader_config, config.output_dir, config.resume_from)

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

    def run(self):
        if self.resume_from:
            pre_inf_results_df, last_uid = self.fetch_previous_inference_results()
        with self.data_loader as loader:
            with self.writer as writer:
                for data, model_inputs in tqdm(loader, desc="Inference Progress:"):
                    # if resume_from file is provided and valid inference results
                    # for the current data point are present in it, use them.
                    if self.resume_from and (data["uid"] <= last_uid):
                        prev_results = pre_inf_results_df[pre_inf_results_df.uid == data["uid"]]
                        prev_result_is_valid = bool(prev_results["is_valid"].values[0])
                        prev_model_output = prev_results["model_output"].values[0]
                        if prev_result_is_valid:
                            logging.info(f"Skipping inference for uid: {data['uid']}. Using previous results.")
                            data["model_output"], data["is_valid"] = prev_model_output, prev_result_is_valid
                            writer.write(data)
                            continue
                    # generate text from model
                    response_dict = self.model.generate(*model_inputs)
                    # "model_output" and "is_valid" are mandatory fields by any inference component
                    if "model_output" not in response_dict or "is_valid" not in response_dict:
                        raise ValueError("Response dictionary must contain 'model_output' and 'is_valid' keys.")
                    # write results
                    data.update(response_dict)
                    writer.write(data)
