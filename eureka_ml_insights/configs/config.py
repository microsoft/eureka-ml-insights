import copy
from dataclasses import dataclass, field
from typing import Any, List, Type, TypeVar

UtilityClassConfigType = TypeVar("UtilityClassConfigType", bound=Type["UtilityClassConfig"])
ComponentConfigType = TypeVar("ComponentConfigType", bound=Type["ComponentConfig"])

"""Provides configuration classes for utility classes.

These classes define configurations for various utility classes utilized in the pipeline.
"""


@dataclass
class UtilityClassConfig:
    """
    Base class for all utility class configs.

    Attributes:
        class_name (Any): The utility class to be used with this config.
        init_args (dict): Arguments to be passed to the utility class constructor.
    """

    class_name: Any = None
    init_args: dict = field(default_factory=dict)

    def __repr__(self):
        """
        Return a string representation of the configuration.

        Omits 'secret_key_params' and 'api_key' for security reasons.

        Returns:
            str: A string representation of the configuration.
        """
        init_args_copy = copy.deepcopy(self.init_args)
        init_args_copy.pop("secret_key_params", None)
        init_args_copy.pop("api_key", None)
        return f"{self.__class__.__name__}(class_name={self.class_name.__name__}, init_args={init_args_copy})"


@dataclass(repr=False)
class DataSetConfig(UtilityClassConfig):
    """
    Configuration class for a dataset.

    Inherits from:
        UtilityClassConfig

    Attributes:
        class_name (Any): The utility class to be used with this config.
        init_args (dict): Arguments to be passed to the utility class constructor.
    """
    pass


@dataclass(repr=False)
class ModelConfig(UtilityClassConfig):
    """
    Configuration class for a model.

    Inherits from:
        UtilityClassConfig

    Attributes:
        class_name (Any): The utility class to be used with this config.
        init_args (dict): Arguments to be passed to the utility class constructor.
    """
    pass


@dataclass(repr=False)
class MetricConfig(UtilityClassConfig):
    """
    Configuration class for a metric.

    Inherits from:
        UtilityClassConfig

    Attributes:
        class_name (Any): The utility class to be used with this config.
        init_args (dict): Arguments to be passed to the utility class constructor.
    """
    pass


@dataclass(repr=False)
class AggregatorConfig(UtilityClassConfig):
    """
    Configuration class for an aggregator.

    Inherits from:
        UtilityClassConfig

    Attributes:
        class_name (Any): The utility class to be used with this config.
        init_args (dict): Arguments to be passed to the utility class constructor.
    """
    pass


"""Provides configuration classes for component classes.

This section includes classes that define various component configurations used in the pipeline.
"""


@dataclass
class ComponentConfig:
    """
    Base class for all component configurations.

    Attributes:
        component_type (Any): The component class to be used with this config.
        output_dir (str): The directory to save the output files of this component.
    """

    component_type: Any = None
    output_dir: str = None


@dataclass
class DataProcessingConfig(ComponentConfig):
    """
    Config class for the data processing component.

    Inherits from:
        ComponentConfig

    Attributes:
        component_type (Any): The component class to be used with this config.
        output_dir (str): The directory to save the output files of this component.
        data_reader_config (UtilityClassConfigType): The data reader config to be used with this component.
        output_data_columns (List[str]): List of columns (subset of input columns) to keep in the transformed data output file.
    """

    data_reader_config: UtilityClassConfigType = None
    output_data_columns: List[str] = None


@dataclass
class PromptProcessingConfig(DataProcessingConfig):
    """
    Config class for the prompt processing component.

    Inherits from:
        DataProcessingConfig

    Attributes:
        component_type (Any): The component class to be used with this config.
        output_dir (str): Directory where output files are saved.
        data_reader_config (UtilityClassConfigType): The data reader config used with this component.
        output_data_columns (List[str]): List of columns to keep in the transformed data output file.
        prompt_template_path (str): The path to the prompt template jinja file.
        ignore_failure (bool): Whether to ignore failures in prompt processing and move on.
    """

    prompt_template_path: str = None
    ignore_failure: bool = False


@dataclass
class DataJoinConfig(DataProcessingConfig):
    """
    Config class for the data join component.

    Inherits from:
        DataProcessingConfig

    Attributes:
        component_type (Any): The component class to be used with this config.
        output_dir (str): Directory where output files are saved.
        data_reader_config (UtilityClassConfigType): The data reader config used with this component.
        output_data_columns (List[str]): List of columns to keep in the transformed data output file.
        other_data_reader_config (UtilityClassConfigType): The data reader config for the dataset to be joined with the main dataset.
        pandas_merge_args (dict): Arguments passed to pandas merge function.
    """

    other_data_reader_config: UtilityClassConfigType = None
    pandas_merge_args: dict = None


@dataclass
class DataUnionConfig(DataProcessingConfig):
    """
    Config class for the data union component.

    Inherits from:
        DataProcessingConfig

    Attributes:
        component_type (Any): The component class to be used with this config.
        output_dir (str): Directory where output files are saved.
        data_reader_config (UtilityClassConfigType): The data reader config used with this component.
        output_data_columns (List[str]): List of columns (subset of input columns) to keep in the transformed data output file.
        other_data_reader_config (UtilityClassConfigType): The data reader config for the dataset to be joined with the main dataset.
        dedupe_cols (List[str]): Columns used to deduplicate the concatenated data frame.
    """

    other_data_reader_config: UtilityClassConfigType = None
    dedupe_cols: List[str] = None


@dataclass
class InferenceConfig(ComponentConfig):
    """
    Config class for the inference component.

    Inherits from:
        ComponentConfig

    Attributes:
        component_type (Any): The component class to be used with this config.
        output_dir (str): Directory where output files are saved.
        data_loader_config (UtilityClassConfigType): The data loader config used with this component.
        model_config (UtilityClassConfigType): The model config to be used with this component.
        resume_from (str): Path to the file where previous inference results are stored.
        new_columns (List[str]): New columns generated by this component that are not present in the resume_from file.
        requests_per_minute (int): Number of requests allowed per minute.
        max_concurrent (int): Maximum number of concurrent requests. Defaults to 1.
        chat_mode (bool): Whether the inference is in chat mode.
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
    """
    Config class for the evaluation reporting component.

    Inherits from:
        ComponentConfig

    Attributes:
        component_type (Any): The component class to be used with this config.
        output_dir (str): Directory where output files are saved.
        data_reader_config (UtilityClassConfigType): The data reader config used for this component.
        metric_config (UtilityClassConfigType): The metric config used for evaluation.
        aggregator_configs (List[UtilityClassConfigType]): Aggregator configurations.
        visualizer_configs (List[UtilityClassConfigType]): Visualizer configurations.
    """

    data_reader_config: UtilityClassConfigType = None
    metric_config: UtilityClassConfigType = None
    aggregator_configs: List[UtilityClassConfigType] = field(default_factory=list)
    visualizer_configs: List[UtilityClassConfigType] = field(default_factory=list)


"""Provides a configuration class for the pipeline.

This section includes a configuration class for the pipeline.
"""


@dataclass
class PipelineConfig:
    """
    Configuration class for the pipeline.

    Attributes:
        component_configs (list[ComponentConfigType]): List of component configurations.
        log_dir (str): Directory to save the pipeline logs.
    """

    component_configs: list[ComponentConfigType] = field(default_factory=list)
    log_dir: str = None

    def __repr__(self):
        """
        Return a string representation of the pipeline configuration.

        Returns:
            str: A string containing all component configurations.
        """
        res = ""
        for comp in self.component_configs:
            res += str(comp) + "\n"
        return res