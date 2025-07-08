"""Module containing data loader classes, reader/writer classes, and supporting utilities
for local or remote data ingestion and consumption.
"""

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from io import BytesIO
from typing import List, Optional

import jsonlines
import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobClient, ContainerClient
from datasets import load_dataset, load_from_disk
from PIL import Image
from tqdm import tqdm

from eureka_ml_insights.secret_management import get_secret

from .encoders import NumpyEncoder
from .transform import DFTransformBase

log = logging.getLogger("data_reader")


class AzureStorageLogger:
    """Provides a logger for Azure Blob Storage operations."""

    def get_logger(self, level=logging.WARNING):
        """Returns a logger for Azure Blob Storage.

        Args:
            level: The logging level to set for the logger.

        Returns:
            logging.Logger: The configured logger instance.
        """
        logger = logging.getLogger("azure.storage")
        logger.setLevel(level)
        return logger


class DataReaderBase(ABC):
    """Base class for data readers."""

    @abstractmethod
    def read(self):
        """Abstract method to read data from a source.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        raise NotImplementedError


class DataLoader:
    """Loads data from local sources for model inference.

    DataLoader is used to feed data to models at inference time from local data sources.
    """

    def __init__(self, path, total_lines=None):
        """Initializes DataLoader.

        Args:
            path (str): The path to the data file.
            total_lines (int, optional): The total number of lines in the data file. Defaults to None.
        """
        self.path = path
        self.total_lines = total_lines

    def __enter__(self):
        self.reader = jsonlines.open(self.path, "r", loads=json.loads)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.reader.close()

    def __len__(self):
        if self.total_lines is None:
            log.info("Total data lines not provided, iterating through the data to get the total lines.")
            with jsonlines.open(self.path, "r", loads=json.loads) as reader:
                self.total_lines = sum(1 for _ in reader)
        return self.total_lines

    def __iter__(self):
        for data in self.reader.iter(skip_empty=True, skip_invalid=True):
            yield self.prepare_model_input(data)

    def prepare_model_input(self, row):
        """Prepares a single row of data for model input.

        Args:
            row (dict): A single data row from the JSON lines file.

        Returns:
            tuple: A tuple containing the data row, model arguments, and model keyword arguments.
        """
        query_text = row["prompt"]
        model_args = (query_text,)
        model_kwargs = {}
        return row, model_args, model_kwargs

    def get_sample_model_input(self):
        """Gets a sample data row and model arguments from the JSON lines reader.

        Returns:
            tuple: A tuple containing the sample data row, model arguments, and model keyword arguments.
        """
        row = next(self.reader.iter(skip_empty=True, skip_invalid=True))
        return self.prepare_model_input(row)


class MMDataLoader(DataLoader):
    """Base class for loading images from a local dataset for multimodal tasks.

    MMDataLoader supports loading images that are referenced in the local dataset in addition to textual data.
    """

    def __init__(
        self,
        path: str,
        mm_data_path_prefix: str = "",
        total_lines: int = None,
        load_images: bool = True,
        image_column_names: List[str] = None,
        image_column_search_regex: str = "image",
        misc_columns: List[str] = None,
    ):
        """Initializes an MMDataLoader.

        Args:
            path (str): Local path to the dataset JSONL file.
            mm_data_path_prefix (str, optional): Local path prefix to prepend to the image file names in the JSONL file.
            total_lines (int, optional): The total number of lines in the dataset file. Defaults to None.
            load_images (bool, optional): Whether images should be loaded. Defaults to True.
            image_column_names (List[str], optional): Names of columns containing images. Defaults to None.
            image_column_search_regex (str, optional): Regex pattern used to identify image columns. Defaults to "image".
            misc_columns (List[str], optional): Names of other columns to include in model inputs. Defaults to None.
        """
        super().__init__(path, total_lines)
        self.mm_data_path_prefix = mm_data_path_prefix
        self.load_images = load_images
        self.image_column_names = image_column_names
        self.image_column_search_regex = image_column_search_regex
        self.misc_columns = misc_columns

    def prepare_model_input(self, row):
        """Prepares a single row of data (including images if required) for model input.

        Args:
            row (dict): A data record from the JSON lines file.

        Returns:
            tuple: A tuple containing the data row, model arguments, and model keyword arguments.
        """
        query_text = row["prompt"]
        model_args = (query_text,)

        if self.load_images:
            if self.image_column_names:
                image_column_names = self.image_column_names
            else:
                image_column_names = self._search_for_image_columns(row)

            if image_column_names:
                images = self._gather_image_file_names(row, image_column_names)
                query_images = self._load_images(images)
                model_args = (query_text, query_images)

        model_kwargs = {}
        if self.misc_columns:
            for column in self.misc_columns:
                model_kwargs[column] = row[column]
        return row, model_args, model_kwargs

    def _search_for_image_columns(self, data_row) -> list:
        """Searches for columns that contain images in a data row.

        Args:
            data_row (dict): A data record from the JSONL file.

        Returns:
            list: A list of column names that contain images.
        """
        if "image_column_names" in data_row:
            image_column_names = data_row["image_column_names"]
        else:
            if self.image_column_search_regex:
                image_column_names = []
                for col_name in data_row.keys():
                    match_obj = re.match(self.image_column_search_regex, col_name)
                    if match_obj:
                        image_column_names.append(col_name)
            else:
                log.warning(
                    "No image search string was provided, and no image columns were found. Thus no images will be loaded."
                )
                image_column_names = None

        return image_column_names

    def _gather_image_file_names(self, data, image_column_names: List[str] | str) -> list:
        """Gets all image file names from a data record.

        Args:
            data (dict): A data record from the JSONL file.
            image_column_names (List[str] | str): Names of columns that contain images.

        Returns:
            list: A list of image file paths (or a single path if not a list).
        """
        if not isinstance(image_column_names, list):
            images = data[image_column_names]
        else:
            if len(image_column_names) == 1:
                images = data[image_column_names[0]]
            else:
                images = [
                    data[image_column_name] for image_column_name in image_column_names if (data[image_column_name])
                ]

        if not images:
            log.warning("No image files names were found in the data row. Thus no images will be passed to the model.")

        return images

    def _load_images(self, images: List[str] | str) -> list:
        """Loads image files from local paths.

        Args:
            images (List[str] | str): One or more image file paths.

        Returns:
            list: A list of loaded PIL Images.
        """
        if not isinstance(images, list):
            images = [images]

        query_images = []
        for image in images:
            query_image = self.load_image(image)
            query_images.append(query_image)

        return query_images

    def load_image(self, image_file_name):
        """Loads an image file from a local path.

        Args:
            image_file_name (str): The image file path.

        Returns:
            PIL.Image.Image: The loaded image as a PIL Image.
        """
        full_image_file_path = os.path.join(self.mm_data_path_prefix, image_file_name)
        query_image = Image.open(full_image_file_path).convert("RGB")
        return query_image


class AzureDataAuthenticator:
    """Handles Azure authentication parameters for blob storage."""

    def get_query_string(self, query_string=None, secret_key_params=None):
        """Obtains an Azure query string for authentication.

        One of the two arguments (query_string or secret_key_params) must be provided.

        Args:
            query_string (str, optional): The query string to authenticate with Azure Blob Storage. Defaults to None.
            secret_key_params (dict, optional): Parameters for retrieving the query string from secrets. Defaults to None.

        Raises:
            ValueError: If neither query_string nor secret_key_params are provided.
        """
        self.query_string = query_string
        self.secret_key_params = secret_key_params
        if self.query_string is None and self.secret_key_params is None:
            raise ValueError("Either provide query_string or secret_key_params to load data from Azure.")
        if self.query_string is None:
            self.query_string = get_secret(**secret_key_params)


class AzureMMDataLoader(MMDataLoader):
    """Allows for image loading from Azure Blob Storage based on references in a local dataset.

    AzureMMDataLoader extends MMDataLoader for loading images from an Azure Blob Storage container.
    """

    def __init__(
        self,
        path,
        account_url,
        blob_container,
        total_lines=None,
        image_column_names=None,
        image_column_search_regex="image",
        misc_columns=None,
    ):
        """Initializes AzureMMDataLoader.

        Args:
            path (str): The local path or reference to the dataset JSONL file.
            account_url (str): The Azure storage account URL.
            blob_container (str): The Azure storage container name.
            total_lines (int, optional): The number of lines in the dataset file. Defaults to None.
            image_column_names (list, optional): The names of columns containing image references. Defaults to None.
            image_column_search_regex (str, optional): The regex string used to find image columns. Defaults to "image".
            misc_columns (list, optional): The names of other columns to include in model inputs. Defaults to None.
        """
        super().__init__(
            path,
            total_lines,
            image_column_names=image_column_names,
            image_column_search_regex=image_column_search_regex,
            misc_columns=misc_columns,
        )
        self.logger = AzureStorageLogger().get_logger()
        self.account_url = account_url
        self.blob_container = blob_container
        self.container_client = ContainerClient(
            account_url=self.account_url,
            container_name=self.blob_container,
            credential=DefaultAzureCredential(),
            logger=self.logger,
        )

    def load_image(self, image_file_name):
        """Loads an image from Azure Blob Storage.

        Args:
            image_file_name (str): The name of the image file in the blob storage.

        Returns:
            PIL.Image.Image: The loaded image as a PIL Image.
        """
        image_bytes = self.container_client.download_blob(image_file_name).readall()
        query_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return query_image


class JsonLinesWriter:
    """Writes data to a JSON lines file."""

    def __init__(self, out_path, mode="w"):
        """Initializes a JsonLinesWriter.

        Args:
            out_path (str): The output file path.
            mode (str, optional): The file open mode. Defaults to "w".
        """
        self.out_path = out_path
        directory = os.path.dirname(out_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.writer = None
        self.mode = mode

    def __enter__(self):
        self.writer = jsonlines.open(self.out_path, mode=self.mode, dumps=NumpyEncoder().encode)
        return self.writer

    def __exit__(self, exc_type, exc_value, traceback):
        self.writer.close()


class JsonReader(DataReaderBase):
    """Reads JSON or JSONL data from a local file."""

    def __init__(self, path):
        """Initializes a JsonReader.

        Args:
            path (str): The local path to the JSON or JSONL file.
        """
        self.path = path
        _, ext = os.path.splitext(self.path)
        self.format = ext.lower()

    def read(self):
        """Reads the data from the file.

        Returns:
            list or dict: The data read from the file.

        Raises:
            ValueError: If the file format is not .json or .jsonl.
        """
        if self.format == ".json":
            with open(self.path, mode="r") as reader:
                data = json.load(reader)
        elif self.format == ".jsonl":
            with jsonlines.open(self.path, mode="r", loads=json.loads) as reader:
                data = list(reader)
        else:
            raise ValueError("JsonReader currently only supports json and jsonl format.")
        return data


class AzureBlobReader:
    """Provides functionality to read data from an Azure Storage blob via a full blob URL."""

    def read_azure_blob(self, blob_url) -> str:
        """Reads content from an Azure Storage blob.

        Args:
            blob_url (str): The full URL of the Azure blob.

        Returns:
            str: The content of the blob as a string.
        """
        blob_client = BlobClient.from_blob_url(blob_url, credential=DefaultAzureCredential(), logger=self.logger)
        file = blob_client.download_blob().readall()
        file = file.decode("utf-8")
        return file


class AzureJsonReader(JsonReader, AzureBlobReader):
    """Reads JSON or JSONL data from an Azure Storage blob and returns the content as a dict."""

    def __init__(
        self,
        account_url: str,
        blob_container: str,
        blob_name: str,
    ):
        """Initializes an AzureJsonReader.

        Args:
            account_url (str): The Azure storage account URL.
            blob_container (str): The Azure storage container name.
            blob_name (str): The Azure storage blob name.
        """
        self.blob_url = f"{account_url}/{blob_container}/{blob_name}"
        super().__init__(self.blob_url)
        self.logger = AzureStorageLogger().get_logger()

    def read(self) -> dict:
        """Reads the data from the Azure Storage blob.

        Returns:
            dict or jsonlines.Reader: The data loaded from the blob.

        Raises:
            ValueError: If the file format is not .json or .jsonl.
        """
        file = super().read_azure_blob(self.blob_url)
        if self.format == ".json":
            data = json.loads(file)
        elif self.format == ".jsonl":
            data = jsonlines.Reader(file.splitlines(), loads=json.loads)
        else:
            raise ValueError("AzureJsonReader currently only supports json and jsonl format.")
        return data


class HFJsonReader(JsonReader):
    """Reads JSON or JSONL data from Hugging Face repositories."""

    def __init__(self, repo_id, repo_type, filename):
        """Initializes an HFJsonReader.

        Args:
            repo_id (str): The Hugging Face repository ID.
            repo_type (str): The type of the repository (e.g., dataset).
            filename (str): The name of the file in the repository.
        """
        from huggingface_hub import hf_hub_download

        cached_file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
        super().__init__(cached_file_path)


class Writer:
    """Base class for writing data to files."""

    def __init__(self, out_path):
        """Initializes a Writer.

        Args:
            out_path (str): The output file path.
        """
        self.out_path = out_path
        directory = os.path.dirname(out_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def write(self, data):
        """Writes data to a file.

        This method should be implemented by subclasses.

        Args:
            data: The data to be written.
        """


class TXTWriter(Writer):
    """Writes data to a text file."""

    def write(self, data):
        """Writes data to a text file.

        Args:
            data (list, dict, or str): The data to be written.
        """
        with open(self.out_path, "w") as f:
            if isinstance(data, list):
                for item in data:
                    f.write("%s\n" % item)
            elif isinstance(data, dict):
                for key in data:
                    f.write("%s\t%s\n" % (key, data[key]))
            else:
                f.write("%s\n" % data)


class DataReader:
    """Base class that loads data from a local file in various formats (Parquet, CSV, JSONL).

    It can optionally apply a transform to the loaded data.
    """

    def __init__(
        self,
        path: str,
        format: str = None,
        transform: Optional[DFTransformBase] = None,
        **kwargs,
    ):
        """Initializes a DataReader.

        Args:
            path (str): The local path to the dataset file.
            format (str, optional): The file format (e.g., .parquet, .csv, .jsonl). Defaults to None.
            transform (DFTransformBase, optional): A transform to apply after loading data. Defaults to None.
            **kwargs: Additional keyword arguments for the data reading process.
        """
        self.path = path
        _, ext = os.path.splitext(self.path)
        if format:
            self.format = format
        else:
            self.format = ext.lower()
        self.transform = transform
        self.kwargs = kwargs

    def load_dataset(self) -> pd.DataFrame:
        """Loads the dataset into a pandas DataFrame.

        Returns:
            pd.DataFrame: The loaded and transformed dataset.
        """
        df = self._load_dataset()
        df = df.reset_index(drop=True)
        log.info(f"Loaded dataset with shape: {df.shape}")
        if not df.empty and self.transform is not None:
            df = self.transform.transform(df)
        df = df.reset_index(drop=True)
        log.info(f"Transformed dataset has shape: {df.shape}")
        return df

    def _load_dataset(self) -> pd.DataFrame:
        """Loads the dataset from the specified path based on the file format.

        Returns:
            pd.DataFrame: The dataset as a DataFrame.
        """
        if self.format == ".parquet":
            log.info(f"Loading Parquet Data From {self.path}.")
            df = pd.read_parquet(self.path, **self.kwargs)
        elif self.format == ".csv":
            log.info(f"Loading CSV Data From {self.path}.")
            df = pd.read_csv(self.path, **self.kwargs)
        elif self.format == ".jsonl":
            log.info(f"Loading JSONL Data From {self.path}.")
            df = pd.read_json(self.path, lines=True, convert_dates=False, convert_axes=False, **self.kwargs)
        else:
            log.info(f"Data format is: {self.format}, default to read as csv.")
            df = pd.read_csv(self.path, **self.kwargs)
        return df


class AzureDataReader(DataReader, AzureBlobReader):
    """Loads JSONL data from an Azure Blob Storage URL into a pandas DataFrame."""

    def __init__(
        self,
        account_url: str,
        blob_container: str,
        blob_name: str,
        format: str = None,
        transform: Optional[DFTransformBase] = None,
        **kwargs,
    ):
        """Initializes an AzureDataReader.

        Args:
            account_url (str): The Azure storage account URL.
            blob_container (str): The Azure storage container name.
            blob_name (str): The Azure storage blob name.
            format (str, optional): The file format (only .jsonl is currently supported). Defaults to None.
            transform (DFTransformBase, optional): A transform to apply after loading data. Defaults to None.
            **kwargs: Additional keyword arguments for the data reading process.
        """
        self.blob_url = f"{account_url}/{blob_container}/{blob_name}"
        super().__init__(self.blob_url, format, transform, **kwargs)
        self.logger = AzureStorageLogger().get_logger()

    def _load_dataset(self) -> pd.DataFrame:
        """Loads JSONL data from the Azure Blob Storage specified by the blob URL.

        Returns:
            pd.DataFrame: The loaded dataset as a DataFrame.

        Raises:
            ValueError: If the file format is not .jsonl.
        """
        file = super().read_azure_blob(self.blob_url)
        if self.format == ".jsonl":
            jlr = jsonlines.Reader(file.splitlines(), loads=json.loads)
            df = pd.DataFrame(jlr.iter(skip_empty=True, skip_invalid=True))
        else:
            raise ValueError("AzureDataReader currently only supports jsonl format.")
        return df


class HFDataReader(DataReader):
    """DataReader that leverages Hugging Face datasets for loading data.

    It can download data from Hugging Face or load from a local HF dataset directory,
    then convert the data to a pandas DataFrame.
    """

    def __init__(
        self,
        path: str,
        split: List[str] | str = "test",
        tasks: List[str] | str = None,
        transform: Optional[DFTransformBase] = None,
        cache_dir: str = None,
        load_data_from_disk: bool = False,
        **kwargs,
    ):
        """Initializes an HFDataReader.

        Args:
            path (str): The Hugging Face dataset path or repository ID.
            split (str or list, optional): The split(s) of the dataset to load. Defaults to "test".
            tasks (str or list, optional): The tasks or dataset configurations to load. Defaults to None.
            transform (DFTransformBase, optional): A transform to apply after loading. Defaults to None.
            cache_dir (str, optional): The local cache directory for the dataset. Defaults to None.
            load_data_from_disk (bool, optional): Whether to load the dataset from a local directory. Defaults to False.
            **kwargs: Additional keyword arguments for dataset loading.
        """
        super().__init__(path=path, transform=transform, **kwargs)
        self.split = split
        self.tasks = tasks
        self.cache_dir = cache_dir
        self.load_data_from_disk = load_data_from_disk

    def _save_base64_to_image_file(self, image_base64: dict, cache_path: str) -> str:
        """Saves a base64-encoded image to a local file.

        Args:
            image_base64 (dict): A dictionary containing the byte string and file name of the image.
            cache_path (str): The local directory path where the image should be saved.

        Returns:
            str: The full path to the saved image file.
        """
        file_path = ""

        if image_base64:
            file_path = os.path.join(cache_path, image_base64["path"])
            if not os.path.exists(file_path):
                buffered = BytesIO(image_base64["bytes"])
                query_image = Image.open(buffered).convert("RGB")
                dir_path = os.path.dirname(file_path)
                os.makedirs(dir_path, exist_ok=True)
                query_image.save(file_path)

        return file_path

    def _save_images(self, df: pd.DataFrame, cache_path: str, image_columns) -> pd.DataFrame:
        """Saves base64-encoded images from the dataframe columns to the local cache directory.

        Args:
            df (pd.DataFrame): The DataFrame containing base64-encoded images.
            cache_path (str): The directory path where images are saved.
            image_columns (list): The names of columns containing base64-encoded images.

        Returns:
            pd.DataFrame: The updated DataFrame with replaced image paths.
        """
        tqdm.pandas()
        for column in tqdm(image_columns, desc="Image Saving Progress:"):
            df[column] = df[column].progress_apply(self._save_base64_to_image_file, args=(cache_path,))
        return df

    def _hf_to_dataframe(self, hf_dataset):
        """Converts a Hugging Face dataset object to a pandas DataFrame.

        If the dataset contains base64-encoded images, they are saved to disk and replaced with file paths.

        Args:
            hf_dataset: The Hugging Face dataset object.

        Returns:
            pd.DataFrame: The resulting pandas DataFrame.
        """
        df = hf_dataset.to_pandas()

        if hasattr(hf_dataset, "features"):
            image_columns = [
                col
                for col in hf_dataset.features
                if hasattr(hf_dataset.features[col], "dtype") and hf_dataset.features[col].dtype == "PIL.Image.Image"
            ]
            if image_columns:
                cache_path = os.path.dirname(hf_dataset.cache_files[0]["filename"])
                df = self._save_images(df, cache_path, image_columns)
                df["image_column_names"] = df.apply(lambda x: image_columns, axis=1)

        return df

    def _load_dataset(self) -> pd.DataFrame:
        """Loads one or multiple Hugging Face datasets, converts them to pandas DataFrames, and merges them.

        Returns:
            pd.DataFrame: The combined dataset as a single DataFrame.
        """
        if not isinstance(self.split, list):
            self.split = [self.split]
        if self.tasks is not None and not isinstance(self.tasks, list):
            self.tasks = [self.tasks]
        df_frames = []
        if self.tasks is None:
            if self.load_data_from_disk:
                dataset_dict = load_from_disk(self.path)
                hf_dataset = [dataset_dict[split] for split in self.split]
            else:
                hf_dataset = load_dataset(self.path, cache_dir=self.cache_dir, split=self.split)
            for i, data_split in enumerate(hf_dataset):
                task_df = self._hf_to_dataframe(data_split)
                task_df["__hf_split"] = self.split[i]
                df_frames.append(task_df)
        else:
            for task in self.tasks:
                if self.load_data_from_disk:
                    dataset_dict = load_from_disk(self.path)
                    hf_dataset = [dataset_dict[task][split] for split in self.split]
                else:
                    hf_dataset = load_dataset(self.path, task, cache_dir=self.cache_dir, split=self.split)
                for i, data_split in enumerate(hf_dataset):
                    task_df = self._hf_to_dataframe(data_split)
                    task_df["__hf_task"] = task
                    task_df["__hf_split"] = self.split[i]
                    df_frames.append(task_df)
        return pd.concat(df_frames)
