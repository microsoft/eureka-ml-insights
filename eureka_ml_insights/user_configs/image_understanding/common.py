"""This module provides the LOCAL_DATA_PIPELINE class, which extends the pipeline configuration
to handle local data paths for both data reading and inference components.
"""

import os

from eureka_ml_insights.configs import DataSetConfig
from eureka_ml_insights.data_utils import DataReader, MMDataLoader


class LOCAL_DATA_PIPELINE:
    """Pipeline class that configures data paths for local storage.

    This class overrides the configure_pipeline method to replace data
    and inference component configurations with local paths.
    """

    def configure_pipeline(self, model_config, resume_from, local_path):
        """Configures the pipeline to use local data paths.

        Args:
            model_config: The configuration object for the model.
            resume_from: The checkpoint or state to resume from.
            local_path (str): The path to local data directory.

        Returns:
            The updated pipeline configuration.
        """
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
