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
    def configure_pipeline(self, **kwargs):
        config = super().configure_pipeline(**kwargs)
        # Downsample the data and repeat each prompt 3 time
        self.data_processing_comp.data_reader_config.init_args["transform"] = SequenceTransform(
            [SamplerTransform(random_seed=99, sample_count=150), MultiplyTransform(n_repeats=3)]
        )
        return config


class Geo_Nondeterminism(GEOMETER_PIPELINE):
    def configure_pipeline(self, **kwargs):
        config = super().configure_pipeline(**kwargs)
        # Downsample the data and repeat each prompt 3 time
        config.component_configs[0].data_reader_config.init_args["transform"].transforms.extend(
            [SamplerTransform(random_seed=42, sample_count=75, stratify_by="category"), MultiplyTransform(n_repeats=3)]
        )
        return config


class Kitab_Nondeterminism(KITAB_ONE_BOOK_CONSTRAINT_PIPELINE):
    def configure_pipeline(self, **kwargs):
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
    def configure_pipeline(self, **kwargs):
        config = super().configure_pipeline(**kwargs)
        # Downsample the data and repeat each prompt 3 time
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms.extend(
            [SamplerTransform(random_seed=42, sample_count=5, stratify_by="task"), MultiplyTransform(n_repeats=3)]
        )
        return config
