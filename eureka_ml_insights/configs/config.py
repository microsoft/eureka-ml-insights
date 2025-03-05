import copy
from dataclasses import dataclass, field
from typing import Any, List, Type, TypeVar

UtilityClassConfigType = TypeVar("UtilityClassConfigType", bound=Type["UtilityClassConfig"])
ComponentConfigType = TypeVar("ComponentConfigType", bound=Type["ComponentConfig"])

""" Config classes for utility classes """


@dataclass
class UtilityClassConfig:
    """Base class for all utility class configs
    Args:
        class_name (Any): The utility class to be used with this config
        init_args (dict): The arguments to be passed to the utility class constructor
    """

    class_name: Any = None
    init_args: dict = field(default_factory=dict)

    def __repr__(self):
        # remove "secret_key_params" from the init_args representation for security reasons
        # but keep it in the actual config
        init_args_copy = copy.deepcopy(self.init_args)
        init_args_copy.pop("secret_key_params", None)
        init_args_copy.pop("api_key", None)
        return f"{self.__class__.__name__}(class_name={self.class_name.__name__}, init_args={init_args_copy})"


@dataclass(repr=False)
class DataSetConfig(UtilityClassConfig):
    pass


@dataclass
class ModelConfig(UtilityClassConfig):
    pass


@dataclass(repr=False)
class MetricConfig(UtilityClassConfig):
    pass


@dataclass(repr=False)
class AggregatorConfig(UtilityClassConfig):
    pass


""" Config classes for component classes """


@dataclass
class ComponentConfig:
    """Base class for all component configs
    Args:
        component_type (Any): The component class to be used with this config
        output_dir (str): The directory to save the output files of this component
    """

    component_type: Any = None
    output_dir: str = None


@dataclass
class DataProcessingConfig(ComponentConfig):
    """Config class for the data processing component
    Args:
        data_reader_config (UtilityClassConfig): The data reader config to be used with this component
        output_data_columns (list): List of columns (subset of input columns) to keep in the transformed data output file
    """

    data_reader_config: UtilityClassConfigType = None
    output_data_columns: List[str] = None


@dataclass
class PromptProcessingConfig(DataProcessingConfig):
    """Config class for the prompt processing component
    Args:
        prompt_template_path (str): The path to the prompt template jinja file
        ignore_failure (bool): Whether to ignore the failures in the prompt processing and move on
    """

    prompt_template_path: str = None
    ignore_failure: bool = False


@dataclass
class DataJoinConfig(DataProcessingConfig):
    """Config class for the data join component
    Args:
        other_data_reader_config (UtilityClassConfig): The data reader config for the dataset to be joined with the main dataset
        pandas_merge_args (dict): Arguments to be passed to pandas merge function
    """

    other_data_reader_config: UtilityClassConfigType = None
    pandas_merge_args: dict = None


@dataclass
class DataUnionConfig(DataProcessingConfig):
    """Config class for the data union component
    Args:
        other_data_reader_config (UtilityClassConfig): The data reader config for the dataset to be joined with the main dataset
        output_data_columns (list): List of columns (subset of input columns) to keep in the transformed data output file
    """

    other_data_reader_config: UtilityClassConfigType = None
    output_data_columns: List[str] = None


@dataclass
class InferenceConfig(ComponentConfig):
    """Config class for the inference component
    Args:
        data_loader_config (UtilityClassConfig): The data loader config to be used with this component
        model_config (UtilityClassConfig): The model config to be used with this component
        resume_from (str): Optional. Path to the file where previous inference results are stored
    """

    data_loader_config: UtilityClassConfigType = None
    model_config: UtilityClassConfigType = None
    resume_from: str = None
    new_columns: List[str] = None
    requests_per_minute: int = None
    max_concurrent: int = 1
    chat_mode: bool = False


@dataclass
class EvalReportingConfig(ComponentConfig):
    """Config class for the evaluation reporting component
    Args:
        data_reader_config (UtilityClassConfig): The data reader config to configure the data reader for this component
        metric_config (UtilityClassConfig): The metric config
        aggregator_configs (list): List of aggregator configs
        visualizer_configs (list): List of visualizer configs
    """

    data_reader_config: UtilityClassConfigType = None
    metric_config: UtilityClassConfigType = None
    aggregator_configs: List[UtilityClassConfigType] = field(default_factory=list)
    visualizer_configs: List[UtilityClassConfigType] = field(default_factory=list)


""" Config class for the pipeline class """


@dataclass
class PipelineConfig:
    """Config class for the pipeline class
    Args:
        component_configs (list): List of ComponentConfigs
        log_dir (str): The directory to save the logs of the pipeline
    """

    component_configs: list[ComponentConfigType] = field(default_factory=list)
    log_dir: str = None

    def __repr__(self):
        res = ""
        for comp in self.component_configs:
            res += str(comp) + "\n"
        return res
