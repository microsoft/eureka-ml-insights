import os
from typing import Any

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
from eureka_ml_insights.core import (
    DataProcessing,
    EvalReporting,
    Inference,
    PromptProcessing,
)
from eureka_ml_insights.data_utils import (
    ColumnRename,
    CopyColumn,
    DataReader,
    ExtractUsageTransform,
    ImputeNA,
    MajorityVoteTransform,
    MMDataLoader,
    MultiplyTransform,
    RegexTransform,
    ReplaceStringsTransform,
    RunPythonTransform,
    SamplerTransform,
    SequenceTransform,
)
from eureka_ml_insights.metrics import (
    BiLevelAggregator,
    BiLevelCountAggregator,
    CountAggregator,
    ExactMatch,
)

# Custom metric for tutoring evaluation with format tracking
class TutoringMistakeDetection:
    """Custom metric that detects mistake={yes/no} pattern and tracks format compliance"""
    
    def __init__(self, ground_truth_col: str = "ground_truth", model_output_col: str = "model_output"):
        self.ground_truth_col = ground_truth_col
        self.model_output_col = model_output_col
    
    def evaluate(self, data):
        accuracy_results = []
        format_results = []
        
        for _, row in data.iterrows():
            expected_result = str(row[self.ground_truth_col]).strip()
            model_output = str(row[self.model_output_col]).strip().lower()
            
            # Extract mistake detection from model output
            if "\\mistake={yes}" in model_output or "mistake={yes}" in model_output:
                predicted_mistake = "yes"
                format_followed = "format_followed"
            elif "\\mistake={no}" in model_output or "mistake={no}" in model_output:
                predicted_mistake = "no"
                format_followed = "format_followed"
            else:
                predicted_mistake = "unknown"
                format_followed = "format_not_followed"
            
            # Only evaluate accuracy if format was followed
            if format_followed == "format_followed":
                # Map expected result to mistake expectation
                if expected_result == "Answer Accepted":
                    expected_mistake = "no"  # No mistake expected
                elif expected_result == "Answer Not Accepted":
                    expected_mistake = "yes"  # Mistake expected
                else:
                    expected_mistake = "unknown"
                
                # Compare predicted vs expected
                is_correct = predicted_mistake == expected_mistake
                accuracy_results.append("correct" if is_correct else "incorrect")
            else:
                accuracy_results.append("format_not_followed")
            
            format_results.append(format_followed)
        
        # Add both accuracy and format tracking columns
        data[f"{self.__class__.__name__}_result"] = accuracy_results
        data[f"{self.__class__.__name__}_format"] = format_results
        return data

# Custom transform to download and load JSON data
class DownloadAndLoadJSON:
    
    def __init__(self, url: str):
        self.url = url
    
    def transform(self, data):
        import requests
        import json
        import pandas as pd
        import tempfile
        import os
        
        # Download JSON data
        print(f"Downloading data from {self.url}...")
        response = requests.get(self.url)
        response.raise_for_status()
        json_data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(json_data)
        print(f"Loaded {len(df)} examples from JSON")
        
        return df

# Custom transform to format conversation data
class ConversationFormatter:
    
    def transform(self, data):
        formatted_conversations = []
        final_user_messages = []
        
        for _, row in data.iterrows():
            conversation_data = row['data']
            
            # Format the conversation into a single string
            conversation_text = ""
            final_user_message = ""
            
            for message in conversation_data:
                role = message['role']
                content = message['content']
                if role == "user":
                    conversation_text += f"User: {content}\n\n"
                    final_user_message = content  # Keep updating to get the last one
                elif role == "assistant":
                    conversation_text += f"Tutor: {content}\n\n"
            
            formatted_conversations.append(conversation_text.strip())
            final_user_messages.append(final_user_message)
        
        data['formatted_conversation'] = formatted_conversations
        data['final_user_message'] = final_user_messages
        return data

class TUTORING_Experiment_Pipeline(ExperimentConfig):
    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        
        import tempfile
        import os
        
        temp_dir = tempfile.mkdtemp()
        temp_csv = os.path.join(temp_dir, "temp.csv")
        with open(temp_csv, 'w') as f:
            f.write("dummy\n1\n")  # Minimal CSV content
        
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": temp_csv,
                    "format": "csv",
                    "transform": SequenceTransform(
                        [
                            # Download and load the actual JSON data (replaces the temp CSV)
                            DownloadAndLoadJSON("https://raw.githubusercontent.com/Khan/tutoring-accuracy-dataset/main/CoMTA_dataset.json"),
                            # Sample only first 5 examples for debugging
                            SamplerTransform(sample_count=50, random_seed=1234),
                            # Format the conversation data
                            ConversationFormatter(),
                            # Copy expected_result to ground_truth for evaluation
                            CopyColumn(column_name_src="expected_result", column_name_dst="ground_truth"),
                            # Copy math_level for potential analysis
                            CopyColumn(column_name_src="math_level", column_name_dst="math_level"),
                            MultiplyTransform(n_repeats=1),
                        ]
                    ),
                },
            ),
            prompt_template_path=os.path.join(
                os.path.dirname(__file__),
                "../prompt_templates/tutoring_templates/tutoring_evaluation.jinja",
            ),
            output_dir=os.path.join(self.log_dir, "data_processing_output"),
        )
        
        # Configure the inference component
        self.inference_comp = InferenceConfig(
            component_type=Inference,
            model_config=model_config,
            data_loader_config=DataSetConfig(
                MMDataLoader,
                {
                    "path": os.path.join(self.data_processing_comp.output_dir, "transformed_data.jsonl"),
                    "load_images": False,  # Text-only evaluation
                },
            ),
            output_dir=os.path.join(self.log_dir, "inference_result"),
            resume_from=resume_from,
        )

        # Pre-evaluation data processing
        self.preeval_data_post_processing_comp = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            # Extract token usage information
                            ExtractUsageTransform(model_config),
                            # Keep raw model output for analysis
                            CopyColumn(
                                column_name_src="model_output",
                                column_name_dst="raw_model_output",
                            ),
                            # Handle any missing outputs
                            ImputeNA(columns="model_output", value=""),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "preeval_data_post_processing_output"),
        )

        # Configure evaluation with format tracking
        self.evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.preeval_data_post_processing_comp.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                },
            ),
            metric_config=MetricConfig(TutoringMistakeDetection),
            aggregator_configs=[
                # Overall accuracy including format violations
                AggregatorConfig(
                    CountAggregator,
                    {
                        "column_names": ["TutoringMistakeDetection_result"],
                        "filename_base": "Overall_Results_Including_Format_Violations",
                        "normalize": True,
                    },
                ),
                # Accuracy by expected result including format violations
                AggregatorConfig(
                    CountAggregator,
                    {
                        "column_names": ["TutoringMistakeDetection_result"],
                        "group_by": ["expected_result"],
                        "filename_base": "Results_By_Expected_Result_Including_Format",
                        "normalize": True,
                    },
                ),
                # Format compliance tracking
                AggregatorConfig(
                    CountAggregator,
                    {
                        "column_names": ["TutoringMistakeDetection_format"],
                        "filename_base": "Format_Compliance",
                        "normalize": True,
                    },
                ),
                # Format compliance by expected result
                AggregatorConfig(
                    CountAggregator,
                    {
                        "column_names": ["TutoringMistakeDetection_format"],
                        "group_by": ["expected_result"],
                        "filename_base": "Format_Compliance_By_Expected_Result",
                        "normalize": True,
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )

        # Post-evaluation data processing for numerical analysis
        self.posteval_data_post_processing_comp = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.evalreporting_comp.output_dir, "metric_results.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            CopyColumn(
                                column_name_src="TutoringMistakeDetection_result",
                                column_name_dst="TutoringMistakeDetection_result_numeric",
                            ),
                            ReplaceStringsTransform(
                                columns=["TutoringMistakeDetection_result_numeric"],
                                mapping={"incorrect": "0", "correct": "1", "none": "NaN"},
                                case=False,
                            ),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "posteval_data_post_processing_output"),
        )

        # Best of N evaluation
        self.bon_evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.posteval_data_post_processing_comp.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                },
            ),
            aggregator_configs=[
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": ["TutoringMistakeDetection_result_numeric"],
                        "first_groupby": "data_point_id",
                        "filename_base": "TutoringAccuracy_BestOfN",
                        "agg_fn": "max",
                    },
                ),
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": ["usage_completion"],
                        "first_groupby": "data_point_id",
                        "filename_base": "UsageCompletion_BestOfN",
                        "agg_fn": "sum",
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "bestofn_eval_report"),
        )

        # Majority vote processing
        self.data_post_processing_mv = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.evalreporting_comp.output_dir, "metric_results.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            MajorityVoteTransform(model_output_col="model_output"),
                            ColumnRename(
                                name_mapping={
                                    "model_output": "model_output_onerun",
                                    "majority_vote": "model_output",
                                }
                            ),
                            RunPythonTransform("df = df[df['data_repeat_id'] == 'repeat_0']"),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_post_processing_mv"),
        )

        # Majority vote evaluation
        self.mv_evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.data_post_processing_mv.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                },
            ),
            metric_config=MetricConfig(TutoringMistakeDetection),
            aggregator_configs=[
                AggregatorConfig(
                    CountAggregator,
                    {"column_names": ["TutoringMistakeDetection_result"], "filename_base": "MajorityVote", "normalize": True},
                ),
            ],
            output_dir=os.path.join(self.log_dir, "majorityvote_eval_report"),
        )

        # Configure the pipeline - simplified version
        return PipelineConfig(
            [
                self.data_processing_comp,
                self.inference_comp,
                self.preeval_data_post_processing_comp,
                self.evalreporting_comp,  # Just the core evaluation
            ],
            self.log_dir,
        )


class TUTORING_PIPELINE_5Run(TUTORING_Experiment_Pipeline):
    """This class specifies the config for running the tutoring benchmark 5 repeated times"""

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        pipeline = super().configure_pipeline(model_config=model_config, resume_from=resume_from)
        # Add 5 repeats to the data preprocessing
        self.data_processing_comp.data_reader_config.init_args["transform"].transforms.append(
            MultiplyTransform(n_repeats=5)
        )
        return pipeline