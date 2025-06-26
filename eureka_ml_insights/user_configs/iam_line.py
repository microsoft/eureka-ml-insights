import os
import base64
import io
import pandas as pd
from typing import Any
from PIL import Image
from datasets import load_dataset

from eureka_ml_insights.configs.experiment_config import ExperimentConfig
from eureka_ml_insights.core import EvalReporting, Inference, PromptProcessing
from eureka_ml_insights.data_utils import (
    ColumnRename,
    CopyColumn,
    DataReader,
    HFDataReader,
    MMDataLoader,
    SequenceTransform,
)
from eureka_ml_insights.metrics import CountAggregator

from eureka_ml_insights.configs import (
    AggregatorConfig,
    DataSetConfig,
    EvalReportingConfig,
    InferenceConfig,
    MetricConfig,
    ModelConfig,
    PipelineConfig,
    PromptProcessingConfig,
)


class CustomIAMLineDataReader(DataReader):
    """
    Custom data reader for IAM-line dataset that properly handles PIL images.
    Uses DataReader instead of HFDataReader since HFDataReader is MMMU-specific.
    """
    
    def __init__(self, path: str, split: str = "validation", image_cache_dir: str = None, **kwargs):
        self.hf_path = path
        self.split = split
        self.image_cache_dir = image_cache_dir
        # Pass a dummy path to the parent class since we're handling loading differently
        super().__init__(path="dummy", **kwargs)
    
    def _load_dataset(self) -> pd.DataFrame:
        """Load IAM-line dataset and convert to proper format."""
        # Load the dataset directly
        dataset = load_dataset(self.hf_path, split=self.split)
        
        # Take only first 100 samples for testing
        dataset = dataset.select(range(min(100, len(dataset))))
        
        # Convert to pandas DataFrame
        df = dataset.to_pandas()
        
        # Use provided cache directory or create temporary one
        if self.image_cache_dir is None:
            import tempfile
            self.image_cache_dir = tempfile.mkdtemp(prefix="iam_line_images_")
        
        # Ensure cache directory exists
        os.makedirs(self.image_cache_dir, exist_ok=True)
        print(f"Cache directory created: {self.image_cache_dir}")
        
        # Convert PIL images to file paths and save them
        saved_images = []
        for idx, img in enumerate(df['image']):
            # Debug first few images to understand the format
            if idx < 3:
                print(f"Image {idx}: type={type(img)}")
                if isinstance(img, dict):
                    print(f"Image {idx} dict keys: {list(img.keys())}")
                    for key, value in img.items():
                        print(f"  {key}: {type(value)} - {str(value)[:100]}")
            
            if hasattr(img, 'save'):  # PIL Image
                # Save image to file
                image_filename = f"image_{idx}.jpg"
                image_path = os.path.join(self.image_cache_dir, image_filename)
                img.save(image_path, format='JPEG')
                saved_images.append(image_filename)
                print(f"Successfully saved PIL: {image_path}")
                
            elif isinstance(img, dict):
                # Handle dictionary format - try to extract PIL image or bytes
                image_filename = f"image_{idx}.jpg"
                image_path = os.path.join(self.image_cache_dir, image_filename)
                
                saved = False
                
                # Check if there's a PIL image in the dict
                for key, value in img.items():
                    if hasattr(value, 'save'):  # Found PIL image
                        value.save(image_path, format='JPEG')
                        print(f"Successfully saved from dict[{key}]: {image_path}")
                        saved = True
                        break
                
                # If no PIL image found, check for bytes
                if not saved and 'bytes' in img:
                    try:
                        # Try different approaches for bytes data
                        bytes_data = img['bytes']
                        
                        if isinstance(bytes_data, str):
                            # Assume it's base64 encoded
                            import base64
                            image_data = base64.b64decode(bytes_data)
                        elif isinstance(bytes_data, bytes):
                            # Already bytes, use directly
                            image_data = bytes_data
                        else:
                            # Try to convert to bytes
                            image_data = bytes(bytes_data)
                        
                        # Verify it's valid image data by trying to open it with PIL first
                        from io import BytesIO
                        test_img = Image.open(BytesIO(image_data))
                        test_img.verify()  # Check if it's a valid image
                        
                        # If verification passed, save as JPEG
                        real_img = Image.open(BytesIO(image_data))
                        real_img.save(image_path, format='JPEG')
                        print(f"Successfully saved from bytes: {image_path}")
                        saved = True
                        
                    except Exception as e:
                        print(f"Failed to save from bytes for image {idx}: {e}")
                        # Try to save raw bytes and see what we get
                        try:
                            with open(image_path, 'wb') as f:
                                f.write(image_data)
                            print(f"Saved raw bytes: {image_path}")
                            saved = True
                        except Exception as e2:
                            print(f"Failed to save raw bytes: {e2}")
                
                if saved:
                    saved_images.append(image_filename)
                else:
                    saved_images.append(f"image_{idx}.jpg")
                    print(f"Could not extract image from dict for {idx}")
                    
            else:
                # Fallback - create a placeholder
                saved_images.append(f"image_{idx}.jpg")
                print(f"Using placeholder for image {idx}: {type(img)}")
        
        # Replace the image column with filenames
        df['image'] = saved_images
        
        print(f"Sample image values after conversion: {df['image'].head().tolist()}")
        print(f"Image column dtype: {df['image'].dtype}")
        print(f"Image cache directory: {self.image_cache_dir}")
        
        # List files in cache directory to verify
        if os.path.exists(self.image_cache_dir):
            cache_files = os.listdir(self.image_cache_dir)
            print(f"Files in cache directory: {cache_files[:10]}")  # Show first 10
        
        return df


class MeanAggregator:
    """Custom aggregator to calculate mean of numerical values."""
    
    def __init__(self, column_names, output_dir=None, **kwargs):
        self.column_names = column_names
        self.output_dir = output_dir
        self.results = {}
    
    def aggregate(self, df):
        for col in self.column_names:
            if col in df.columns:
                mean_val = df[col].mean()
                self.results[f"{col}_mean"] = mean_val
                print(f"Mean {col}: {mean_val:.4f}")
        return self.results
    
    def write_results(self):
        """Required method for framework compatibility."""
        if self.output_dir and self.results:
            import json
            import os
            output_file = os.path.join(self.output_dir, "mean_results.json")
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2)
        # Do nothing if no output_dir or results


class CharacterErrorRateMetric:
    """
    Standard Character Error Rate (CER) metric for OCR evaluation.
    CER = (Substitutions + Insertions + Deletions) / Total_Characters_in_Reference
    """
    
    def __init__(self):
        try:
            import jiwer
            self.jiwer = jiwer
        except ImportError:
            # Fallback to manual implementation if jiwer not available
            self.jiwer = None
            print("jiwer not available, using manual CER implementation")
    
    def _manual_cer(self, reference: str, hypothesis: str) -> float:
        """Manual implementation of CER using Levenshtein distance."""
        if not reference:
            return 1.0 if hypothesis else 0.0
        if not hypothesis:
            return 1.0
            
        # Calculate Levenshtein distance (edit distance)
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)

            if len(s2) == 0:
                return len(s1)

            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    # Cost of insertions, deletions, substitutions
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        edit_distance = levenshtein_distance(reference, hypothesis)
        total_chars = len(reference)
        cer = edit_distance / total_chars
        
        return min(cer, 1.0)  # Cap at 1.0
    
    def __call__(self, ground_truth: str, prediction: str) -> float:
        """
        Calculate Character Error Rate.
        
        Args:
            ground_truth: The expected text transcription
            prediction: The model's predicted transcription
            
        Returns:
            float: CER value (0.0 = perfect match, 1.0 = completely wrong)
        """
        if not ground_truth and not prediction:
            return 0.0
        
        if self.jiwer:
            try:
                return self.jiwer.cer(ground_truth, prediction)
            except Exception as e:
                return self._manual_cer(ground_truth, prediction)
        else:
            return self._manual_cer(ground_truth, prediction)
    
    def evaluate(self, df):
        """
        Evaluate CER on a dataframe.
        
        Args:
            df: DataFrame with 'ground_truth' and 'response' columns
            
        Returns:
            DataFrame with CER results
        """
        # Find the response column
        response_col = None
        for col in ['response', 'model_output', 'prediction', 'output']:
            if col in df.columns:
                response_col = col
                break
        
        if response_col is None:
            # No response column found, return all 1.0s
            df['CharacterErrorRateMetric_result'] = 1.0
            return df
        
        # Apply CER to each row
        df['CharacterErrorRateMetric_result'] = df.apply(
            lambda row: self(row.get('ground_truth', ''), row.get(response_col, '')), 
            axis=1
        )
        return df


class ConvertPILToBase64Transform:
    """
    Transform to convert PIL Image objects to base64 format expected by the framework.
    """
    
    def __call__(self, sample: dict) -> dict:
        """
        Convert PIL Image to base64 format with a generated filename.
        
        Args:
            sample: Dictionary containing PIL Image in 'image' key
            
        Returns:
            dict: Sample with image converted to base64 format
        """
        if 'image' in sample:
            img = sample['image']
            
            # Debug: print the type and structure of the image
            print(f"Image type: {type(img)}")
            print(f"Image content: {img}")
            
            # Handle different possible image formats
            if hasattr(img, 'save'):  # PIL Image
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG')
                img_bytes = buffer.getvalue()
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                
                sample['image'] = {
                    'bytes': img_base64,
                    'path': f"image_{abs(hash(img_base64)) % 100000}.jpg"
                }
            elif isinstance(img, dict):
                # If already in dict format, ensure it has the right structure
                if 'path' not in img or img['path'] is None:
                    # Generate a path if missing
                    img_id = abs(hash(str(img))) % 100000
                    img['path'] = f"image_{img_id}.jpg"
                sample['image'] = img
            else:
                # For other formats, try to create a minimal valid structure
                print(f"Unexpected image format: {type(img)}")
                sample['image'] = {
                    'bytes': '',
                    'path': f"unknown_{abs(hash(str(img))) % 100000}.jpg"
                }
        
        return sample


class IAM_LINE_BASELINE_PIPELINE(ExperimentConfig):
    """
    This defines an ExperimentConfig pipeline for the IAM-line handwriting OCR dataset.
    There is no model_config by default and the model config must be passed in via command line.
    """

    def configure_pipeline(self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]) -> PipelineConfig:
        
        # Create a shared image cache directory
        self.image_cache_dir = os.path.join(self.log_dir, "image_cache")
        os.makedirs(self.image_cache_dir, exist_ok=True)
        
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            prompt_template_path=os.path.join(
                os.path.dirname(__file__),
                "../prompt_templates/iam_line_templates/basic.jinja",
            ),
            data_reader_config=DataSetConfig(
                CustomIAMLineDataReader,
                {
                    "path": "Teklia/IAM-line",
                    "split": "validation",
                    "image_cache_dir": self.image_cache_dir,
                    "transform": SequenceTransform(
                        [
                            ColumnRename(name_mapping={"text": "ground_truth"}),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_processing_output"),
            ignore_failure=False,
        )

        # Configure the inference component
        self.inference_comp = InferenceConfig(
            component_type=Inference,
            model_config=model_config,
            data_loader_config=DataSetConfig(
                MMDataLoader,
                {
                    "path": os.path.join(self.data_processing_comp.output_dir, "transformed_data.jsonl"),
                    "mm_data_path_prefix": self.image_cache_dir,
                },
            ),
            output_dir=os.path.join(self.log_dir, "inference_result"),
            resume_from=resume_from,
        )

        # Configure the evaluation and reporting component
        self.evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                },
            ),
            metric_config=MetricConfig(CharacterErrorRateMetric),
            aggregator_configs=[
                AggregatorConfig(
                    MeanAggregator, 
                    {"column_names": ["CharacterErrorRateMetric_result"]}
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )

        # Configure the pipeline
        return PipelineConfig([self.data_processing_comp, self.inference_comp, self.evalreporting_comp], self.log_dir)
