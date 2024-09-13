import os

from eureka_ml_insights.data_utils import DataReader, MMDataLoader

from ..config import DataSetConfig


class LOCAL_DATA_PIPELINE:
    def configure_pipeline(self, model_config, resume_from, local_path):
        config = super().configure_pipeline(model_config, resume_from)

        json_file = self.data_processing_comp.data_reader_config.init_args["blob_name"]
        json_file_local = os.path.join(local_path, json_file)

        self.data_processing_comp.data_reader_config = DataSetConfig(
            DataReader,
            {"path": json_file_local, "transform": self.data_processing_comp.data_reader_config.init_args["transform"]},
        )
        self.inference_comp.data_config = DataSetConfig(
            MMDataLoader,
            {
                "path": os.path.join(self.data_processing_comp.output_dir, "transformed_data.jsonl"),
                "mm_data_path_prefix": local_path,
                "image_column_names": self.inference_comp.data_config.init_args["image_column_names"],
            },
        )
        return config
