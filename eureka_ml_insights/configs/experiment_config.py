import datetime
import os
from abc import ABC, abstractmethod
from typing import Optional

from .config import PipelineConfig


def create_logdir(exp_dir: str, exp_subdir: Optional[str] = None):
    """Creates a unique log directory for the experiment.

    Args:
        exp_dir (str): The name of the experiment directory directly under "/logs".
        exp_subdir (Optional[str], optional): The name of the experiment subdirectory under
            the experiment directory. Defaults to None.

    Returns:
        str: The unique log directory for the experiment.
    """
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
    if exp_subdir is None:
        log_dir = os.path.join("logs", f"{exp_dir}", f"{date}")
    else:
        log_dir = os.path.join("logs", f"{exp_dir}", f"{exp_subdir}", f"{date}")
    os.makedirs(log_dir)
    return log_dir


class ExperimentConfig(ABC):
    """Abstract class for the experiment pipeline configuration.

    Child classes should implement the configure_pipeline method.
    """

    def __init__(self, exp_logdir: Optional[str] = None, **kwargs):
        """Initializes the ExperimentConfig.

        Args:
            exp_logdir (Optional[str], optional): The experiment log directory. Defaults to None.
            **kwargs: Additional keyword arguments to configure the pipeline.
        """
        dir_name = self.__class__.__name__
        self.log_dir = create_logdir(dir_name, exp_logdir)
        self.pipeline_config = self.configure_pipeline(**kwargs)

        with open(os.path.join(self.log_dir, "config.txt"), "w") as f:
            f.write(str(self.pipeline_config))

    @abstractmethod
    def configure_pipeline(self, **kwargs) -> PipelineConfig:
        """Configures and returns the pipeline.

        Args:
            **kwargs: Keyword arguments for pipeline configuration.

        Returns:
            PipelineConfig: The configured pipeline.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("configure_pipeline method must be implemented in the subclass")
