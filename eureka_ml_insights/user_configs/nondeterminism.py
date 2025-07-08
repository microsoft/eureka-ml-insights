"""This module provides classes that configure pipeline transformations with non-deterministic operations 
such as sampling and repeated prompt multiplication for various pipelines.
"""

from eureka_ml_insights.data_utils import (
    ColumnRename,
    MultiplyTransform,
    SamplerTransform,
    SequenceTransform,
)

from .geometer import GEOMETER_PIPELINE
from .ifeval import IFEval_PIPELINE
from .kitab import KITAB_ONE_BOOK_CONSTRAINT_PIPELINE
from .mmmu import MMMU_BASELINE_PIPELINE


class IFEval_Nondeterminism(IFEval_PIPELINE):
    """Configures an IFEval pipeline with non-deterministic data processing.

    Inherits from IFEval_PIPELINE and modifies the data transforms to include
    downsampling and repeated prompts.
    """

    def configure_pipeline(self, **kwargs):
        """Configures and returns the pipeline for non-deterministic usage.

        Downsamples the data and repeats each prompt three times. This is done by
        setting a SequenceTransform that includes a SamplerTransform and a
        MultiplyTransform.

        Args:
            **kwargs: Additional configuration parameters for the base pipeline.

        Returns:
            dict: The updated configuration dictionary after applying
            non-deterministic data processing.
        """
        config = super().configure_pipeline(**kwargs)
        # Downsample the data and repeat each prompt 3 time
        self.data_processing_comp.data_reader_config.init_args["transform"] = SequenceTransform(
            [SamplerTransform(random_seed=99, sample_count=150), MultiplyTransform(n_repeats=3)]
        )
        return config


class Geo_Nondeterminism(GEOMETER_PIPELINE):
    """Configures a GEOMETER pipeline with non-deterministic data processing.

    Inherits from GEOMETER_PIPELINE and modifies the data transforms to include
    downsampling by category and repeated prompts.
    """

    def configure_pipeline(self, **kwargs):
        """Configures and returns the pipeline for non-deterministic usage.

        Downsamples the data by category and repeats each prompt three times. This is achieved
        by extending the existing SequenceTransform with a SamplerTransform and a MultiplyTransform.

        Args:
            **kwargs: Additional configuration parameters for the base pipeline.

        Returns:
            dict: The updated configuration dictionary after applying
            non-deterministic data processing.
        """
        config = super().configure_pipeline(**kwargs)
        # Downsample the data and repeat each prompt 3 time
        config.component_configs[0].data_reader_config.init_args["transform"].transforms.extend(
            [SamplerTransform(random_seed=42, sample_count=75, stratify_by="category"), MultiplyTransform(n_repeats=3)]
        )
        return config


class Kitab_Nondeterminism(KITAB_ONE_BOOK_CONSTRAINT_PIPELINE):
    """Configures a KITAB pipeline with non-deterministic data processing.

    Inherits from KITAB_ONE_BOOK_CONSTRAINT_PIPELINE and modifies the data
    transforms to include renaming columns, downsampling by constraint type,
    and repeating each prompt.
    """

    def configure_pipeline(self, **kwargs):
        """Configures and returns the pipeline for non-deterministic usage.

        Includes column renaming, downsampling (with stratification by constraint type),
        and repeating prompts via a SequenceTransform that contains a ColumnRename,
        SamplerTransform, and MultiplyTransform.

        Args:
            **kwargs: Additional configuration parameters for the base pipeline.

        Returns:
            dict: The updated configuration dictionary after applying
            non-deterministic data processing.
        """
        config = super().configure_pipeline(**kwargs)
        # Downsample the data and repeat each prompt 3 time
        config.component_configs[0].data_reader_config.init_args["transform"] = SequenceTransform(
            [
                ColumnRename(name_mapping={"Author": "author", "Birth Year": "birth_year"}),
                SamplerTransform(random_seed=99, sample_count=20, stratify_by="constraint_type"),
                MultiplyTransform(n_repeats=3),
            ]
        )
        return config


class MMMU_Nondeterminism(MMMU_BASELINE_PIPELINE):
    """Configures an MMMU pipeline with non-deterministic data processing.

    Inherits from MMMU_BASELINE_PIPELINE and modifies the data transforms to include
    downsampling by a specified task key and repeating prompts.
    """

    def configure_pipeline(self, **kwargs):
        """Configures and returns the pipeline for non-deterministic usage.

        Downsamples the data by a specified task key (__hf_task) and repeats each prompt
        three times. This is done by extending the existing transforms with a
        SamplerTransform and a MultiplyTransform.

        Args:
            **kwargs: Additional configuration parameters for the base pipeline.

        Returns:
            dict: The updated configuration dictionary after applying
            non-deterministic data processing.
        """
        config = super().configure_pipeline(**kwargs)
        # Downsample the data and repeat each prompt 3 time
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms.extend(
            [SamplerTransform(random_seed=42, sample_count=5, stratify_by="__hf_task"), MultiplyTransform(n_repeats=3)]
        )
        return config
