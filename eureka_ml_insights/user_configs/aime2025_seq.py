import os
from typing import Any

from eureka_ml_insights.configs import (
    DataProcessingConfig,
    DataSetConfig,
    DataUnionConfig,
    InferenceConfig,
    ModelConfig,
    PipelineConfig,
    PromptProcessingConfig,
)
from eureka_ml_insights.core import (
    DataProcessing,
    DataUnion,
    Inference,
    PromptProcessing,
)
from eureka_ml_insights.data_utils import (
    AddColumnAndData,
    ColumnRename,
    DataReader,
    HFDataReader,
    RunPythonTransform,
    SamplerTransform,
    SequenceTransform,
)
from eureka_ml_insights.data_utils.aime_utils import AIMEExtractAnswer
from eureka_ml_insights.data_utils.data import MMDataLoader
from eureka_ml_insights.metrics.metrics_base import (
    ExactMatch,
    MetricBasedVerifier,
)


from eureka_ml_insights.configs import (
    AggregatorConfig,
    DataProcessingConfig,
    DataSetConfig,
    EvalReportingConfig,
    ExperimentConfig,
    InferenceConfig,
    MetricConfig,
    ModelConfig,
    PipelineConfig,
    PromptProcessingConfig,
)
from eureka_ml_insights.core import DataProcessing, Inference, PromptProcessing
from eureka_ml_insights.core.eval_reporting import EvalReporting
from eureka_ml_insights.data_utils import (
    AddColumn,
    ColumnRename,
    DataReader,
    HFDataReader,
    MajorityVoteTransform,
    MultiplyTransform,
    SequenceTransform,
    SamplerTransform,
       CopyColumn,
       ReplaceStringsTransform,
       ExtractUsageTransform,
       RunPythonTransform

)
from eureka_ml_insights.data_utils.aime_utils import AIMEExtractAnswer
from eureka_ml_insights.data_utils.data import DataLoader, MMDataLoader
# from eureka_ml_insights.metrics.aime_metrics import NumericMatch

from eureka_ml_insights.metrics.reports import (
    BiLevelCountAggregator,
    CountAggregator,
    BiLevelAggregator,
)

from .aime import AIME_PIPELINE
from .aime_seq import AIME_SEQ_PIPELINE

DEFAULT_N_ITER = 5

resume_from_dict = {
    1: None,
    2: None,
    3: None,
    4: None,
    5: None
}

class AIME2025_SEQ_PIPELINE(AIME_SEQ_PIPELINE):
    """This class specifies the config for running AIME benchmark on any model"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:

        # this call to super will configure the initial prompt processing and final eval reporting comps that can be reused.
        pipeline = super().configure_pipeline(model_config, resume_from, **kwargs)
        
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "lchen001/AIME2025",
                    "split": "train",
                    "transform": SequenceTransform(
                        [
                            ColumnRename(
                                name_mapping={
                                    "Question": "prompt",
                                    "Answer": "ground_truth",
                                }
                            ),
                            #SamplerTransform( random_seed=0,sample_count=2),
                        ],
                    ),
                },
            ),
            prompt_template_path=os.path.join(
                os.path.dirname(__file__), "../prompt_templates/aime_templates/Template_1clean.jinja"
            ),
            output_dir=os.path.join(self.log_dir, "data_processing_output"),
        )
        
                # post process the response to extract the answer
        self.data_post_processing = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.last_agg_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            ColumnRename(
                                name_mapping={
                                    "model_output":"raw_output",
#                                    "student_extracted_answer": "model_output",
                                }
                            ),
                            AddColumn("model_output"),
                            AIMEExtractAnswer("raw_output", "model_output"),
                            ExtractUsageTransform(model_config),                        
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_post_processing_output"),
        )


        self.component_configs[0] = self.data_processing_comp
       
        self.component_configs.append(self.data_post_processing)
        self.component_configs.append(self.evalreporting_comp)
        self.component_configs.append(self.posteval_data_post_processing_comp)
        self.component_configs.append(self.bon_evalreporting_comp)
        return pipeline
