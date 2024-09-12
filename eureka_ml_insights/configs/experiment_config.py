import datetime
import os
from abc import ABC, abstractmethod
from typing import Optional

from .config import PipelineConfig


def create_logdir(exp_dir: str, exp_subdir: Optional[str] = None):
    """
    Create a unique log directory for the experiment.
    Args:
        exp_dir: The name of the experiment directory directly under /logs.
        exp_subdir: The name of the experiment subdirectory under the experiment directory.
    Returns:
        log_dir: The unique log directory for the experiment.
    """
    # generate a unique log dir based on time and date
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
    if exp_subdir is None:
        log_dir = os.path.join("logs", f"{exp_dir}", f"{date}")
    else:
        log_dir = os.path.join("logs", f"{exp_dir}", f"{exp_subdir}", f"{date}")
    os.makedirs(log_dir)
    return log_dir


class ExperimentConfig(ABC):
    """
    Abstract class for the experiment piplien configuration class.
    Child classes should implement the configure_pipeline method.
    """
    def __init__(self, exp_logdir: Optional[str] = None, **kwargs):

        dir_name = self.__class__.__name__
        self.log_dir = create_logdir(dir_name, exp_logdir)
        self.pipeline_config = self.configure_pipeline(**kwargs)

        with open(os.path.join(self.log_dir, "config.txt"), "w") as f:
            f.write(str(self.pipeline_config))

    @abstractmethod
    def configure_pipeline(self, **kwargs) -> PipelineConfig:
        raise NotImplementedError("configure_pipeline method must be implemented in the subclass")
