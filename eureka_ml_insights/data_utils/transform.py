"""A set of classes and transformations for DataFrame modifications.

This module includes a base class (DFTransformBase), along with a variety of transformations
for renaming columns, running Python code, sampling data, repeating rows, adding columns, 
and more.
"""

import ast
import logging
import re
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
import tiktoken

from eureka_ml_insights.configs.config import ModelConfig
from eureka_ml_insights.models import (
    AzureOpenAIModel,
    AzureOpenAIOModel,
    ClaudeModel,
    ClaudeReasoningModel,
    DeepseekR1ServerlessAzureRestEndpointModel,
    DirectOpenAIModel,
    DirectOpenAIOModel,
    GeminiModel,
    LlamaServerlessAzureRestEndpointModel,
    MistralServerlessAzureRestEndpointModel,
    TogetherModel,
)


@dataclass
class DFTransformBase:
    """
    Base class for DataFrame transformations.

    This dataclass does not define any attributes. Subclasses must implement
    the transform() method.
    """

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms a pandas DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        ...


@dataclass
class SequenceTransform(DFTransformBase):
    """
    Chains multiple DFTransformBase transformations in sequence.

    Attributes:
        transforms (List[DFTransformBase]): The list of transformations to be applied in sequence.
    """

    transforms: List[DFTransformBase]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies each transformation in sequence to the given DataFrame.

        Args:
            df (pd.DataFrame): The initial DataFrame to transform.

        Returns:
            pd.DataFrame: The transformed DataFrame after all sequential transforms.
        """
        for transform in self.transforms:
            df = transform.transform(df)
        return df


@dataclass
class ColumnRename(DFTransformBase):
    """
    Renames columns in a DataFrame based on a provided name mapping.

    Attributes:
        name_mapping (Dict[str, str]): The dictionary mapping old column names to new column names.
    """

    name_mapping: Dict[str, str]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Renames columns according to name_mapping.

        Args:
            df (pd.DataFrame): The input DataFrame whose columns will be renamed.

        Returns:
            pd.DataFrame: DataFrame with renamed columns.
        """
        return df.rename(columns=self.name_mapping)


@dataclass
class RunPythonTransform(DFTransformBase):
    """
    Runs arbitrary Python code on the DataFrame.

    This transform executes Python code in a controlled environment, allowing only
    certain statements and imports for security reasons.

    Attributes:
        python_code (str): The Python code to run on the DataFrame.
        global_imports (list): A list of modules to import in the global scope, if needed.
    """

    python_code: str
    global_imports: list = field(default_factory=list)

    def __post_init__(self):
        """
        Validates that only permitted operations are allowed in python_code.
        """
        self.allowed_statement_prefixes = ["df = ", "df[", "import "]
        self.allowed_imports = ["ast", "math", "numpy"]
        self.validate()

    def validate(self):
        """
        Checks if the provided Python code has allowed statements and imports.

        Raises:
            ValueError: If any statement does not start with an allowed prefix or if
                attempting to import a disallowed module.
        """
        statements = [s.strip() for s in self.python_code.split(";")]
        for statement in statements:
            if not any(statement.startswith(prefix) for prefix in self.allowed_statement_prefixes):
                raise ValueError("For security reasons, only imports and operations on the data frame are allowed.")
        if self.global_imports:
            for module_name in self.global_imports:
                if module_name not in self.allowed_imports:
                    raise ValueError(f"Importing {module_name} in RunPythonTransform is not allowed.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the stored Python code on the input DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame on which to run the Python code.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        exec_globals = {"df": df}
        exec_globals.update(globals())

        for module_name in self.global_imports:
            exec_globals[module_name] = __import__(module_name)

        exec(self.python_code, exec_globals)

        return exec_globals["df"]


@dataclass
class SamplerTransform(DFTransformBase):
    """
    Samples rows from the DataFrame, either randomly or by stratification.

    Attributes:
        random_seed (int): The random seed for reproducibility.
        sample_count (int): The number of rows to sample.
        stratify_by (List[str]): Optional columns to stratify upon when sampling.
    """

    random_seed: int 
    sample_count: int
    stratify_by: List[str] = None

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Samples rows from the given DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame to sample from.

        Returns:
            pd.DataFrame: A sampled subset of the DataFrame.
        """
        if self.stratify_by:
            return df.groupby(self.stratify_by, group_keys=False).apply(
                lambda x: x.sample(n=self.sample_count, random_state=self.random_seed)
            )
        else:
            return df.sample(n=self.sample_count, random_state=self.random_seed)


@dataclass
class MultiplyTransform(DFTransformBase):
    """
    Repeats each row n times and adds a column to indicate the repeat number and data point ID.

    Attributes:
        n_repeats (int): The number of times to repeat each row.
    """

    n_repeats: int

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Repeats each row in the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: A new DataFrame with repeated rows, including columns for repeat ID and data point ID.
        """
        dfs = []
        for i in range(self.n_repeats):
            df_copy = df.copy()
            df_copy["data_repeat_id"] = f"repeat_{i}"
            df_copy["data_point_id"] = df_copy.index
            dfs.append(df_copy)
        return pd.concat(dfs, ignore_index=True)


@dataclass
class AddColumn(DFTransformBase):
    """
    Adds a new column to the DataFrame with empty strings.

    Attributes:
        column_name (str): The name of the new column to add.
    """

    column_name: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds the specified column to the input DataFrame, initializing it with empty strings.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with the new column added.
        """
        df[self.column_name] = ""
        return df


@dataclass
class AddColumnAndData(DFTransformBase):
    """
    Adds a new column to the DataFrame with a specified value.

    Attributes:
        column_name (str): The name of the new column to add.
        data (str): The value used to populate the new column.
    """

    column_name: str
    data: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds the specified column to the input DataFrame, initializing it with the provided value.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with the new column and data added.
        """
        df[self.column_name] = str(self.data)
        return df


@dataclass
class CopyColumn(DFTransformBase):
    """
    Copies data from a source column to a destination column.

    Attributes:
        column_name_src (str): The name of the source column.
        column_name_dst (str): The name of the destination column.
    """

    column_name_src: str
    column_name_dst: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Copies the data from the source column to the destination column.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with data copied to the new or existing destination column.
        """
        df[self.column_name_dst] = df[self.column_name_src]
        return df


@dataclass
class MultiColumnTransform(DFTransformBase):
    """
    Base class for applying a transformation function to specified columns.

    This class is meant to be subclassed, and the subclass should implement
    the _transform method.

    Attributes:
        columns (List[str] | str): The columns to which transformations will be applied.
    """

    columns: List[str] | str

    def validate(self, df: pd.DataFrame):
        """
        Checks that all columns to be transformed are present in the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Raises:
            ValueError: If any specified columns are not in the DataFrame.
        """
        if not isinstance(self.columns, list):
            self.columns = [self.columns]
        extra_columns = set(self.columns) - set(df.columns)
        if extra_columns:
            msg = ", ".join(sorted(extra_columns))
            raise ValueError(f"The following columns are not present in the data frame: {msg}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the _transform method to each specified column.

        Args:
            df (pd.DataFrame): The DataFrame to transform.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        if df.empty:
            logging.warn("The input dataframe is empty, no transformation was applied.")
            return df
        self.validate(df)
        for column in self.columns:
            df[column] = df[column].apply(self._transform)
        return df

    def _transform(self, value):
        """
        Defines the transformation for a single cell value.

        Args:
            value: The cell value to transform.

        Returns:
            Any: The transformed cell value.
        """
        raise NotImplementedError


@dataclass
class ShuffleColumnsTransform(MultiColumnTransform):
    """
    Shuffles values across specified columns on a per-row basis.

    This transform is often used in MCQ-like tasks where answer choices
    need to be shuffled.

    Attributes:
        columns (List[str]): The list of columns to shuffle.
        rng (np.random.Generator): The random number generator used for the shuffling.
    """

    rng: np.random.Generator = np.random.default_rng(0)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Shuffles values across the specified columns on a per-row basis.

        Args:
            df (pd.DataFrame): The DataFrame whose columns will be shuffled.

        Returns:
            pd.DataFrame: A DataFrame with shuffled columns for each row.
        """
        self.validate(df)

        def shuffle_row(row):
            row[self.columns] = self.rng.permutation(row[self.columns].values)
            return row

        df = df.apply(shuffle_row, axis=1)
        return df


@dataclass
class ColumnMatchMapTransform(DFTransformBase):
    """
    Creates a new column that indicates which column matches a key column's value in each row.

    Attributes:
        key_col (str): The column whose value we look to match across other columns.
        new_col (str): The name of the resulting column that holds the matching column name.
        columns (List[str]): The list of columns to search for a match.
    """

    key_col: str
    new_col: str
    columns: List[str]

    def _find_matching_column(self, row):
        """
        Finds the name of the column matching the key column's value in a given row.

        Args:
            row (pd.Series): A row from the DataFrame.

        Returns:
            str or None: The name of the matching column, or None if none match.
        """
        for col in self.columns:
            if row[col] == row[self.key_col]:
                return col
        return None

    def validate(self, df: pd.DataFrame):
        """
        Validates that the key column and specified columns exist in the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Raises:
            ValueError: If any required columns are not in the DataFrame.
        """
        extra_columns = set(self.columns + [self.key_col]) - set(df.columns)
        if extra_columns:
            msg = ", ".join(sorted(extra_columns))
            raise ValueError(f"The following columns are not present in the data frame: {msg}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a new column indicating which column matches the key column's value for each row.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: A DataFrame with the new column holding the matching column name.
        """
        self.validate(df)
        df[self.new_col] = df.apply(self._find_matching_column, axis=1)
        return df


@dataclass
class ImputeNA(MultiColumnTransform):
    """
    Imputes missing values in selected columns with a specified value.

    Attributes:
        columns (List[str] | str): The column(s) to impute.
        value (str): The value to use for imputation.
    """

    value: str

    def _transform(self, value):
        """
        Replaces missing values with the specified value.

        Args:
            value: The cell value to impute if missing.

        Returns:
            Any: The original value or the imputed replacement if it was missing.
        """
        isna = pd.isna(value)
        if isinstance(isna, bool) and isna:
            return self.value
        return value


@dataclass
class ReplaceStringsTransform(MultiColumnTransform):
    """
    Replaces strings in selected columns according to a specified mapping.

    Useful for ad hoc fixes such as replacing '\\n' with '\n'.

    Attributes:
        columns (List[str] | str): The column(s) to which replacements are applied.
        mapping (Dict[str, str]): A dictionary of old-to-new string replacements.
        case (bool): Whether the replacements should be case-sensitive.
    """

    mapping: Dict[str, str]
    case: bool

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replaces occurrences of keys in the specified columns with their mapped values.

        Args:
            df (pd.DataFrame): The DataFrame to transform.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        self.validate(df)
        for column in self.columns:
            for source, target in self.mapping.items():
                df[column] = df[column].str.replace(source, target, case=self.case, regex=False)
        return df


@dataclass
class MapStringsTransform(MultiColumnTransform):
    """
    Maps values in selected columns to new values according to a specified dictionary.

    Attributes:
        columns (List[str] | str): The column(s) to apply the map.
        mapping (Dict[str, str]): A dictionary mapping old strings to new strings.
    """

    mapping: Dict[str, str]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Maps each value in the specified columns using the provided dictionary.

        Args:
            df (pd.DataFrame): The DataFrame to transform.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        self.validate(df)
        for column in self.columns:
            df[column] = df[column].map(self.mapping)
        return df


@dataclass
class PrependStringTransform(MultiColumnTransform):
    """
    Prepends a specified string to the values in selected columns.

    Attributes:
        columns (List[str] | str): The columns to transform.
        string (str): The string to prepend.
    """

    string: str

    def _transform(self, value):
        """
        Prepends the specified string to a given value.

        Args:
            value: The cell value to transform.

        Returns:
            Any: The value with the string prepended.
        """
        if isinstance(value, list):
            value = [self.string + val for val in value]
        else:
            value = self.string + value
        return value


@dataclass
class RegexTransform(MultiColumnTransform):
    """
    Extracts a regex match from string values in selected columns.

    Attributes:
        columns (List[str] | str): The columns to transform.
        prompt_pattern (str): The regex pattern to use for matching.
        ignore_case (bool): Whether to ignore case in the regex search.
        occurrence (str): Which occurrence to return, either "last" or "first".
    """

    prompt_pattern: str
    ignore_case: bool = False
    occurrence: str = "last"

    def _transform(self, sentence):
        """
        Finds and returns the specified occurrence of a regex match in a string.

        Args:
            sentence (str): The string to search.

        Returns:
            str or None: The matched pattern or None if not found.
        """
        if self.ignore_case:
            results = re.findall(self.prompt_pattern, sentence, flags=re.IGNORECASE)
        else:
            results = re.findall(self.prompt_pattern, sentence)
        if results:
            if self.occurrence == "first":
                return results[0]
            elif self.occurrence == "last":
                return results[len(results) - 1]
        else:
            return None


@dataclass
class ASTEvalTransform(MultiColumnTransform):
    """
    Parses strings in selected columns using ast.literal_eval.

    Attributes:
        columns (List[str] | str): The columns to transform.
    """

    def _transform(self, string):
        """
        Parses a string using ast.literal_eval.

        Args:
            string (str): The string to parse.

        Returns:
            Any: The parsed Python object (e.g., list, dict, etc.).
        """
        list_strings = ast.literal_eval(string)
        return list_strings


@dataclass
class TokenCounterTransform(MultiColumnTransform):
    """
    Counts the number of tokens in the selected columns using tiktoken.

    Attributes:
        columns (List[str] | str): The columns whose token counts will be computed.
    """

    def transform(self, df: pd.DataFrame, encoding="cl100k_base") -> pd.DataFrame:
        """
        Counts tokens in each specified column using tiktoken.

        Args:
            df (pd.DataFrame): The input DataFrame with textual data.
            encoding (str, optional): The tiktoken encoding to use. Defaults to "cl100k_base".

        Returns:
            pd.DataFrame: DataFrame with additional columns named "<column>_token_count".
        """
        self.validate(df)
        encoding = tiktoken.get_encoding(encoding)
        for column in self.columns:
            token_count = df[column].apply(lambda x: len(encoding.encode(x)))
            token_count_column = f"{column}_token_count"
            df[token_count_column] = token_count
        return df


@dataclass
class MajorityVoteTransform:
    """
    Applies a majority vote transformation on the specified model output column grouped by ID.

    Attributes:
        model_output_col (str): The column name for model outputs.
        model_label_column (str): The column name for corresponding labels or scores.
        id_col (str): The column name for IDs.
        majority_vote_col (str): The column name for storing the majority vote result.
        majority_label_col (str): The column name for storing the label of the majority vote output.
    """

    model_output_col: str = "model_output"
    model_label_column: str = None
    id_col: str = "data_point_id"
    majority_vote_col: str = "majority_vote"
    majority_label_col: str = "majority_label"

    def transform(self, df: pd.DataFrame, random_state: int = 0) -> pd.DataFrame:
        """
        Calculates the majority vote of model outputs for each group identified by 'id_col'.

        If the model output is NaN, it will be dropped before calculating the majority vote.

        Args:
            df (pd.DataFrame): The DataFrame containing model outputs and IDs.
            random_state (int, optional): The random seed for tie-breaking among the mode values.
                Defaults to 0.

        Returns:
            pd.DataFrame: The transformed DataFrame, including majority vote columns.
        """
        result_df = df.groupby(self.id_col).apply(
            self.majority_vote,
            self.model_output_col,
            self.model_label_column,
            self.majority_vote_col,
            self.majority_label_col,
            random_state=random_state,
        )
        return result_df

    @staticmethod
    def majority_vote(
        group, model_output_col, model_label_col, majority_vote_col, majority_label_col, random_state: int = 0
    ):
        """
        Calculates the majority vote for each group and optionally its corresponding label.

        Args:
            group (pd.DataFrame): The DataFrame group containing model outputs.
            model_output_col (str): The model output column name.
            model_label_col (str): The model label column name corresponding to outputs.
            majority_vote_col (str): The column name for storing the majority vote result.
            majority_label_col (str): The column name for storing the label of the majority vote output.
            random_state (int, optional): The random seed for tie-breaking among the mode values.
                Defaults to 0.

        Returns:
            pd.DataFrame: The group DataFrame with columns for the majority vote and optional label.
        """
        x = group[model_output_col]
        majority_value = (
            x.dropna().mode().sample(n=1, random_state=random_state).iloc[0] if not x.dropna().mode().empty else pd.NA
        )
        group[majority_vote_col] = majority_value
        if model_label_col:
            group[majority_label_col] = group.loc[group[model_output_col] == majority_value, model_label_col].iloc[0]
        return group


@dataclass
class ExtractUsageTransform:
    """
    Extracts token usage (completion tokens) for models and stores it in a new column.

    Attributes:
        model_config (ModelConfig): The model configuration used for the experiment.
        usage_completion_output_col (str): The column name where completion token usage is stored.
        usage_column (str): The column name where usage information is stored.
        n_tokens_column (str): The column name where the number of tokens is stored.
    """

    model_config: ModelConfig
    usage_completion_output_col: str = "usage_completion"
    usage_column: str = "usage"
    n_tokens_column: str = "n_output_tokens"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts token usage from the DataFrame based on the model configuration.

        Args:
            df (pd.DataFrame): The DataFrame containing inference results.

        Returns:
            pd.DataFrame: The transformed DataFrame with completion token usage in
            'usage_completion_output_col'.
        """
        usage_completion_read_col = None
        if self.model_config.class_name is GeminiModel:
            usage_completion_read_col = "candidates_token_count"
        elif self.model_config.class_name is ClaudeModel or self.model_config.class_name is ClaudeReasoningModel:
            usage_completion_read_col = "output_tokens"
        elif (
            self.model_config.class_name is AzureOpenAIOModel
            or self.model_config.class_name is AzureOpenAIModel
            or self.model_config.class_name is LlamaServerlessAzureRestEndpointModel
            or self.model_config.class_name is MistralServerlessAzureRestEndpointModel
            or self.model_config.class_name is DeepseekR1ServerlessAzureRestEndpointModel
            or self.model_config.class_name is DirectOpenAIModel
            or self.model_config.class_name is DirectOpenAIOModel
            or self.model_config.class_name is TogetherModel
        ):
            usage_completion_read_col = "completion_tokens"
        else:
            logging.warn(
                f"Model {self.model_config.class_name} is not recognized for extracting completion token usage."
            )
        self.validate(df, usage_completion_read_col)
        if usage_completion_read_col:
            df[self.usage_completion_output_col] = df.apply(
                lambda x: self._extract_usage(x, usage_completion_read_col), axis=1
            )
        elif self.n_tokens_column in df.columns:
            df[self.usage_completion_output_col] = df[self.n_tokens_column]
        else:
            df[self.usage_completion_output_col] = np.nan
        return df

    def validate(self, df: pd.DataFrame, usage_completion_read_col: str):
        """
        Validates that necessary usage columns are present in the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame to validate.
            usage_completion_read_col (str): The column name used for completion token usage.

        Raises:
            ValueError: If necessary usage columns are missing.
        """
        if usage_completion_read_col and self.usage_column not in df.columns:
            raise ValueError(f"The {self.usage_column} column is not present in the data frame.")
        elif self.n_tokens_column not in df.columns:
            raise ValueError(f"The {self.n_tokens_column} column is not present in the data frame.")

    def _extract_usage(self, row, usage_completion_read_col):
        """
        Extracts the completion token usage from a single row.

        Args:
            row (pd.Series): A row from the DataFrame.
            usage_completion_read_col (str): The column name used to extract the token usage.

        Returns:
            int or float: The extracted token count if present, otherwise NaN.
        """
        if not pd.isna(row[self.usage_column]) and usage_completion_read_col in row[self.usage_column]:
            return row[self.usage_column][usage_completion_read_col]
        return np.nan