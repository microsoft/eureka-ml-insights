"""Defines the pipeline for running LiveCodeBench.

See https://livecodebench.github.io/ for details.

Typical usage example (from the project's root directory):
    $ python main.py \
        --exp_config="LIVE_CODE_BENCH_CODEGEN_PIPELINE" \
        --model_config="OAI_O1_CONFIG"
"""

import pathlib

from eureka_ml_insights import configs, core, data_utils
from typing import Any


class LIVE_CODE_BENCH_CODEGEN_PIPELINE(configs.ExperimentConfig):
    """Defines the pipeline for running the code generation benchmark."""

    _HF_LCB_DATASET_NAME: str = "livecodebench/code_generation_lite"
    _HF_LCB_DATASET_SPLIT: str = "test"
    _PROMPT_TEMPLATE_PATH: pathlib.Path = (
        pathlib.Path(__file__).parent /
        "../prompt_templates/live_code_bench_templates/codegen.jinja"
    ).resolve()

    def configure_pipeline(self,
                           **kwargs: dict[str, Any]) -> configs.PipelineConfig:
        """Configures the steps of the pipeline.
        
        Args:
            TODO(luizdovalle): Add args.
        
        Returns:
            A PipelineConfig object defining the pipeline steps.
        """
        self._prompt_creation = configs.PromptProcessingConfig(
            component_type=core.PromptProcessing,
            prompt_template_path=str(self._PROMPT_TEMPLATE_PATH),
            data_reader_config=configs.DataSetConfig(
                class_name=data_utils.HFDataReader,
                init_args={
                    "path": self._HF_LCB_DATASET_NAME,
                    "split": self._HF_LCB_DATASET_SPLIT,
                    "transform": data_utils.SequenceTransform([
                        data_utils.SamplerTransform(
                            sample_count=10,
                            random_seed=42
                        ),
                    ])
                }),
            output_dir=str(pathlib.Path(self.log_dir) / "prompts")
        )

        return configs.PipelineConfig(
            component_configs=[
                self._prompt_creation,
            ],
            log_dir=self.log_dir,
        )
