import ast
import re
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd


@dataclass
class DFTransformBase:
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...


@dataclass
class SequenceTransform(DFTransformBase):
    transforms: List[DFTransformBase]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for transform in self.transforms:
            df = transform.transform(df)
        return df


@dataclass
class ColumnRename(DFTransformBase):
    name_mapping: Dict[str, str]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns=self.name_mapping)


@dataclass
class RunPythonTransform(DFTransformBase):
    """Runs arbitrary python code on the data frame.
    args:
        python_code: str: The python code to run on the data frame.
        global_imports: list: A list of modules to import in global scope, if needed. Default is an empty list.
                       Local (to the python_code scope) imports can be included in the python_code. Such global
                       scope imports are needed when the python_code uses a lambda function, for example, since
                       imports in the python_code scope are not available to the lambda function.
    returns:
        df: pd.DataFrame: The transformed data frame.
    """

    python_code: str
    global_imports: list = field(default_factory=list)

    def __post_init__(self):
        # To avoid disastrous consequences, we only allow operations on the data frame.
        # Therefore, every statement in the python_code should be in the form of df['column_name'] = ... or df = ...
        self.allowed_statement_prefixes = ["df = ", "df[", "import "]
        # Similarly, we only allow a limited set of imports. To add to this safe list, create a PR.
        self.allowed_imports = ["ast", "math", "numpy"]
        self.validate()

    def validate(self):
        # First, splits the python code into python statements and strips whitespace.
        statements = [s.strip() for s in self.python_code.split(";")]
        # Checks that each statement starts with an allowed prefix.
        for statement in statements:
            if not any(statement.startswith(prefix) for prefix in self.allowed_statement_prefixes):
                raise ValueError("For security reasons, only imports and operations on the data frame are allowed.")
        if self.global_imports:
            for module_name in self.global_imports:
                if module_name not in self.allowed_imports:
                    raise ValueError(f"Importing {module_name} in RunPythonTransform is not allowed.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Adds 'df' to the global scope of exec so that it can be overwritten during exec if needed.
        exec_globals = {"df": df}
        exec_globals.update(globals())

        # Adds safe imports to exec_globals.
        for module_name in self.global_imports:
            exec_globals[module_name] = __import__(module_name)

        exec(self.python_code, exec_globals)

        return exec_globals["df"]


@dataclass
class SamplerTransform(DFTransformBase):
    random_seed: int
    sample_count: int
    stratify_by: List[str] = None

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.stratify_by:
            return df.groupby(self.stratify_by, group_keys=False).apply(
                lambda x: x.sample(n=self.sample_count, random_state=self.random_seed)
            )
        else:
            return df.sample(n=self.sample_count, random_state=self.random_seed)


@dataclass
class MultiplyTransform(DFTransformBase):
    """
    Repeats each row n times, and adds a column to the data frame indicating the repeat number.
    Also adds a column to the data frame indicating the data point id that will be the same for
    all repeats of the same data point.
    """

    n_repeats: int

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        dfs = []
        for i in range(self.n_repeats):
            df_copy = df.copy()
            df_copy["data_repeat_id"] = f"repeat_{i}"
            df_copy["data_point_id"] = df_copy.index
            dfs.append(df_copy)
        return pd.concat(dfs, ignore_index=True)


@dataclass
class AddColumn(DFTransformBase):
    column_name: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.column_name] = ""
        return df


@dataclass
class AddColumnAndData(DFTransformBase):
    column_name: str
    data: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.column_name] = str(self.data)

        return df


@dataclass
class CopyColumn(DFTransformBase):
    """Copy a src column's data to a new dst names column."""

    column_name_src: str
    column_name_dst: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.column_name_dst] = df[self.column_name_src]

        return df


@dataclass
class MultiColumnTransform(DFTransformBase):
    """
    Transform class to apply a function to multiple columns.
    This class' validate method checks that the columns to be transformed are present in the data frame,
    and its transform method applies the _transform method to each column.

    This class is meant to be subclassed, and the subclass should implement the _transform method.
    """

    columns: List[str] | str

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check that all columns to be transformed are present actually in the data frame."""
        # if columns is not a list, make it a list
        if not isinstance(self.columns, list):
            self.columns = [self.columns]
        extra_columns = set(self.columns) - set(df.columns)
        if extra_columns:
            msg = ", ".join(sorted(extra_columns))
            raise ValueError(f"The following columns are not present in the data frame: {msg}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the transform to the columns."""
        self.validate(df)
        for column in self.columns:
            df[column] = df[column].apply(self._transform)
        return df


@dataclass
class ImputeNA(MultiColumnTransform):
    """Impute missing values in selected columns with a specified value."""

    columns: List[str] | str
    value: str

    def _transform(self, value):
        if pd.isna(value):
            return self.value
        return value


@dataclass
class ReplaceStringsTransform(MultiColumnTransform):
    """
    Replaces strings in selected columns.  Useful for adhoc fixes, i.e., \\n to \n.
    """

    columns: List[str] | str
    mapping: Dict[str, str]
    case: bool

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate(df)
        for column in self.columns:
            for source, target in self.mapping.items():
                df[column] = df[column].str.replace(source, target, case=self.case, regex=False)

        return df
    

@dataclass
class MapStringsTransform(MultiColumnTransform):
    """
    Map values in certain columns of a pandas dataframe according to a mapping dictionary.
    """

    columns: List[str] | str
    mapping: Dict[str, str]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate(df)
        for column in self.columns:            
            df[column] = df[column].map(self.mapping)

        return df


@dataclass
class PrependStringTransform(MultiColumnTransform):
    """
    Prepends a string for selected columns.
    args:
        columns: List of str or str, Column(s) to apply transform to.
        string: str, string to prepend
    """

    columns: List[str] | str
    string: str

    def _transform(self, value):
        if isinstance(value, list):
            value = [self.string + val for val in value]
        else:
            value = self.string + value
        return value


@dataclass
class RegexTransform(MultiColumnTransform):
    """
    Find occurrence of the pattern in selected columns.
    """

    columns: List[str] | str
    prompt_pattern: str
    case: bool

    def _transform(self, sentence):
        results = re.findall(self.prompt_pattern, sentence)
        return results[0] if results else None


@dataclass
class ASTEvalTransform(MultiColumnTransform):
    """
    Applies ast.literal_eval to parse strings in selected columns
    """

    columns: List[str]

    def _transform(self, string):
        list_strings = ast.literal_eval(string)
        return list_strings