import os
import pathlib
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type, TypeVar

from eureka_ml_insights.configs.config import ComponentConfig

PathLike = TypeVar("PathLike", str, pathlib.Path)
T = TypeVar("T", bound="Component")


class Component(ABC):
    def __init__(self, output_dir: PathLike, **_: Dict[str, Any]):
        self.output_dir = output_dir
        # create the component output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        else:
            raise FileExistsError(f"Output directory {self.output_dir} already exists.")

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError

    @classmethod
    def from_config(cls: Type[T], config: ComponentConfig) -> T:
        raise NotImplementedError


class Pipeline:
    def __init__(self, component_configs: List[ComponentConfig], log_dir: str):
        self.log_dir = log_dir
        self.components: List[Component] = []
        for config in component_configs:
            self.components.append(config.component_type.from_config(config))

    def run(self) -> None:
        for component in self.components:
            component.run()
