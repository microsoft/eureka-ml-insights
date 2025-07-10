"""This module defines an abstract base class for pipeline components and a pipeline class
that orchestrates the execution of these components sequentially.
"""

import os
import pathlib
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type, TypeVar

from eureka_ml_insights.configs.config import ComponentConfig

PathLike = TypeVar("PathLike", str, pathlib.Path)
T = TypeVar("T", bound="Component")


class Component(ABC):
    """Abstract base class for pipeline components.

    This class provides the basic interface for all components in the pipeline,
    ensuring they define a 'run' method and can be instantiated from a config.
    """

    def __init__(self, output_dir: PathLike, **_: Dict[str, Any]):
        """Initializes the component.

        Args:
            output_dir (PathLike): The output directory for the component.
            **_ (Dict[str, Any]): Additional parameters.

        Raises:
            FileExistsError: If the output directory already exists.
        """
        self.output_dir = output_dir
        # create the component output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        else:
            raise FileExistsError(f"Output directory {self.output_dir} already exists.")

    @abstractmethod
    def run(self) -> None:
        """Executes the component's main functionality.

        Raises:
            NotImplementedError: This method must be implemented in subclasses.
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls: Type[T], config: ComponentConfig) -> T:
        """Creates an instance of the component from the provided configuration.

        Args:
            config (ComponentConfig): The configuration for the component.

        Returns:
            T: An instance of the component.

        Raises:
            NotImplementedError: This method must be implemented in subclasses.
        """
        raise NotImplementedError


class Pipeline:
    """A pipeline that executes a list of components in sequence."""

    def __init__(self, component_configs: List[ComponentConfig], log_dir: str):
        """Initializes the pipeline with the given component configurations and log directory.

        Args:
            component_configs (List[ComponentConfig]): A list of configurations for pipeline components.
            log_dir (str): The directory where pipeline logs will be stored.
        """
        self.log_dir = log_dir
        self.components: List[Component] = []
        for config in component_configs:
            self.components.append(config.component_type.from_config(config))

    def run(self) -> None:
        """Runs all the components in the pipeline in order."""
        for component in self.components:
            component.run()
