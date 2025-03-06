from eureka_ml_insights.configs import DataSetConfig
from eureka_ml_insights.data_utils import (
    ColumnRename,
    HFDataReader,
    MultiplyTransform,
    SequenceTransform,
)
from eureka_ml_insights.user_configs import GSM8K_PIPELINE


class GSMSYMBOLIC_PIPELINE(GSM8K_PIPELINE):
    """This class specifies the config for running GSM8K benchmark on any model"""

    def configure_pipeline(self, model_config, resume_from=None, n_repeats=1, **kwargs):
        pipeline = super().configure_pipeline(
            model_config,
            resume_from,
            n_repeats,
            **kwargs,
        )

        self.preprocessing_comp.data_reader_config = DataSetConfig(
            HFDataReader,
            {
                "path": "apple/GSM-Symbolic",
                "split": "test",
                "tasks": "main",
                "transform": SequenceTransform(
                    [
                        ColumnRename(
                            name_mapping={
                                "question": "prompt",
                            }
                        ),
                        # SamplerTransform(sample_count=5, random_seed=99),
                        MultiplyTransform(n_repeats=int(n_repeats)),
                    ],
                ),
            },
        )

        return pipeline
