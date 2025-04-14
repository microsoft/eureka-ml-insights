import os
from typing import Any

from eureka_ml_insights.core import Inference, PromptProcessing
from eureka_ml_insights.core.data_processing import DataProcessing
from eureka_ml_insights.core.eval_reporting import EvalReporting
from eureka_ml_insights.data_utils.ba_calendar_utils import (
    BA_Calendar_ExtractAnswer,
)
from eureka_ml_insights.data_utils.data import (
    DataLoader,
    DataReader,
    HFDataReader,
)
from eureka_ml_insights.metrics.ba_calendar_metrics import BACalendarMetric
from eureka_ml_insights.metrics.reports import (
    AverageAggregator,
    BiLevelCountAggregator,
    BiLevelAggregator,
    CountAggregator
)

from eureka_ml_insights.data_utils.transform import (
    AddColumn,
    AddColumnAndData,
    ColumnRename,
    CopyColumn,
    ExtractUsageTransform,
    MajorityVoteTransform,
    MultiplyTransform,
    RunPythonTransform,
    SamplerTransform,
    SequenceTransform,
)
from eureka_ml_insights.metrics.ba_calendar_metrics import BACalendarMetric

from ..configs.config import (
    AggregatorConfig,
    DataProcessingConfig,
    DataSetConfig,
    EvalReportingConfig,
    InferenceConfig,
    MetricConfig,
    ModelConfig,
    PipelineConfig,
    PromptProcessingConfig,
)
from ..configs.experiment_config import ExperimentConfig


class ARC_AGI_v1_PIPELINE(ExperimentConfig):
    """This class specifies the config for running any benchmark on any model"""

    def configure_pipeline(self, model_config=None, resume_from=None, resume_logdir=None, **kwargs) -> PipelineConfig:
        # data preprocessing
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            prompt_template_path=os.path.join(
                os.path.dirname(__file__), "../prompt_templates/arc_agi_templates/arc_agi_v1_basic.jinja"
            ),
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                   "path": "pxferna/ARC-AGI-v1",
                   "split": "test",
                }
            ),
            output_dir=os.path.join(self.log_dir, "data_processing_output"),
        )

        # inference component
        self.inference_comp = InferenceConfig(
            component_type=Inference,
            model_config=model_config,
            data_loader_config=DataSetConfig(
                DataLoader,
                {"path": os.path.join(self.data_processing_comp.output_dir, "transformed_data.jsonl")},
            ),
            output_dir=os.path.join(self.log_dir, "inference_result"),
            resume_from=resume_from,
            max_concurrent=1,
        )

        if resume_logdir:
            self.log_dir = resume_from.split("/")[0:len(resume_from.split("/")) - 1]

        # Configure the pipeline
        return PipelineConfig(
            [
                self.data_processing_comp,
                self.inference_comp,
            ],
            self.log_dir,
        )
