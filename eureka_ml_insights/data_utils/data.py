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
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from eureka_ml_insights.secret_management import get_secret

from .encoders import NumpyEncoder
from .transform import DFTransformBase

log = logging.getLogger("data_reader")


class AzureStorageLogger:
    def get_logger(self, level=logging.WARNING):
        logger = logging.getLogger("azure.storage")
        logger.setLevel(level)
        return logger


class DataReaderBase(ABC):
    @abstractmethod
    def read(self):
        raise NotImplementedError


class DataLoader:
    """Dataloaders are used to feed data to models in inference time from LOCAL data sources."""

    def __init__(self, path, total_lines=None):
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
        query_text = row["prompt"]
        model_args = (query_text,)
        model_kwargs = {}
        return row, model_args, model_kwargs

    def get_sample_model_input(self):
        """Get a sample data row and model_args from the jsonlines reader."""
        row = next(self.reader.iter(skip_empty=True, skip_invalid=True))
        return self.prepare_model_input(row)


class MMDataLoader(DataLoader):
    """This data loader class is a base class for those that allow for loading images that are referenced in the local dataset."""

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
        super().__init__(path, total_lines)
        self.mm_data_path_prefix = mm_data_path_prefix
        self.load_images = load_images
        self.image_column_names = image_column_names
        self.image_column_search_regex = image_column_search_regex
        self.misc_columns = misc_columns
        """
        Initializes an MMDataLoader.
        args:
            path: str, local path to the dataset jsonl file.
            mm_data_path_prefix: str, local path prefix that will be prepended to the mm data location (e.g. image file name) stored in the jsonl file.
            total_lines: option int, the number of lines of the dataset file.
            load_images: optional bool, specify if images should be loaded.
            image_column_names: optional List of str, names of columns that have images in them.
            image_column_search_regex: optional Regex str, to search for which columns have images in them.
            misc_columns: optional List of str, names of other columns from the data to include in the model inputs.
        """

    def prepare_model_input(self, row):
        # Given a row from the jsonl file, prepare the data for the model.
        query_text = row["prompt"]
        model_args = (query_text,)

        # if images are present load them
        if self.load_images:
            # if the user passed in a list of image column names when creating the class, use it
            if self.image_column_names:
                image_column_names = self.image_column_names
            # otherwise search for the image columns
            else:
                image_column_names = self._search_for_image_columns(row)

            # if found load them from disk and add to model inputs
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
        """
        Search for columns that contain images for a datarow.
        args:
            data_row: dict, a record (row) from the jsonl
        returns:
            image_columns: List of str, column names.
        """
        # look in the data to see if it was stored
        if "image_column_names" in data_row:
            # fetch which columns are images
            image_column_names = data_row["image_column_names"]
        # if not stored try to search for the columns
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
        """
        Get all image file names from the data dict and return as a list.
        args:
            image_column_names: List of str or str, names of column(s) that have images in them.
        returns:
            images: List of str or str, images path(s).
        """
        # if is not a list, get the single image file name with the column name
        if not isinstance(image_column_names, list):
            images = data[image_column_names]
        else:
            if len(image_column_names) == 1:
                # some datasets store multiple images in one column as a list
                images = data[image_column_names[0]]
            else:
                # some datasets store multiple images in multiple columns
                images = [
                    data[image_column_name] for image_column_name in image_column_names if (data[image_column_name])
                ]

        if not images:
            log.warning("No image files names were found in the data row. Thus no images will be passed to the model.")

        return images

    def _load_images(self, images: List[str] | str) -> list:
        """
        Load images files with load_image.
        args:
            images: List of str or str, images path(s).
        returns:
            query_images: List of PIL Images.
        """
        # if is not a list, make it a list
        if not isinstance(images, list):
            images = [images]

        query_images = []
        for image in images:
            query_image = self.load_image(image)
            query_images.append(query_image)

        return query_images

    def load_image(self, image_file_name):
        """
        Load image file from local path.
        args:
            image_file_name: str, images path.
        returns:
            query_image: PIL Image.
        """
        # prepend the local path prefix
        full_image_file_path = os.path.join(self.mm_data_path_prefix, image_file_name)
        query_image = Image.open(full_image_file_path).convert("RGB")
        return query_image


class AzureDataAuthenticator:
    def get_query_string(self, query_string=None, secret_key_params=None):
        """
        One of the two arguments must be provided.
        args:
            query_string: str, query string to authenticate with Azure Blob Storage.
            secret_key_params: dict, dictionary containing the paramters to call get_secret with.
        """
        self.query_string = query_string
        self.secret_key_params = secret_key_params
        if self.query_string is None and self.secret_key_params is None:
            raise ValueError("Either provide query_string or secret_key_params to load data from Azure.")
        if self.query_string is None:
            self.query_string = get_secret(**secret_key_params)


class AzureMMDataLoader(MMDataLoader):
    """This data loader allows for loading images that are referenced in the local dataset from Azure Blob Storage."""

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
        """
        Initializes an AzureMMDataLoader.
        args:
            path: str, The Azure storage account URL.
            account_url: str, The Azure storage account URL.
            blob_container: str, Azure storage container name.
            total_lines: option int, the number of lines of the dataset file.
            image_column_names: optional List of str, names of columns that have images in them.
            image_column_search_regex: optional Regex str, to search for which columns have images in them.
            misc_columns: optional List of str, names of other columns from the data to include in the model inputs.
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
        image_bytes = self.container_client.download_blob(image_file_name).readall()
        query_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return query_image


class JsonLinesWriter:
    def __init__(self, out_path, mode="w"):
        self.out_path = out_path
        # if the directory does not exist, create it
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
    """
    This is a DataReader that loads a json or jsonl data file from a local path.
    """

    def __init__(self, path):
        self.path = path
        _, ext = os.path.splitext(self.path)
        self.format = ext.lower()

    def read(self):
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
    """Reads an Azure storage blob from a full URL to a str"""

    def read_azure_blob(self, blob_url) -> str:
        """
        Reads an Azure storage blob..
        args:
            blob_url: str, The Azure storage blob full URL.
        """
        blob_client = BlobClient.from_blob_url(blob_url, credential=DefaultAzureCredential(), logger=self.logger)
        # real all the bytes from the blob
        file = blob_client.download_blob().readall()
        file = file.decode("utf-8")
        return file


class AzureJsonReader(JsonReader, AzureBlobReader):
    """
    This is a Azure storage blob DataReader that loads a json data
    file hosted on a blob and returns the contents as a dict.
    """

    def __init__(
        self,
        account_url: str,
        blob_container: str,
        blob_name: str,
    ):
        """
        Initializes an AzureJsonReader.
        args:
            account_url: str, The Azure storage account URL.
            blob_container: str, Azure storage container name.
            blob_name: str, Azure storage blob name.
        """
        self.blob_url = f"{account_url}/{blob_container}/{blob_name}"
        super().__init__(self.blob_url)
        self.logger = AzureStorageLogger().get_logger()

    def read(self) -> dict:
        file = super().read_azure_blob(self.blob_url)
        if self.format == ".json":
            data = json.loads(file)
        elif self.format == ".jsonl":
            data = jsonlines.Reader(file.splitlines(), loads=json.loads)
        else:
            raise ValueError("AzureJsonReader currently only supports json and jsonl format.")
        return data


class HFJsonReader(JsonReader):
    """
    This is a DataReader that loads a json or jsonl data file from HuggingFace.
    """

    def __init__(self, repo_id, repo_type, filename):
        """
        Initializes an HFJsonReader.
        args:
            repo_id: str, The HF repo id.
            repo_type: str, The HF repo_type.
            filename: str, The HF filename.
        """
        from huggingface_hub import hf_hub_download

        cached_file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
        super().__init__(cached_file_path)


class Writer:
    def __init__(self, out_path):
        self.out_path = out_path
        # if the directory does not exist, create it
        directory = os.path.dirname(out_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def write(self, data):
        pass


class TXTWriter(Writer):
    def write(self, data):
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
    """This is the base DataReader that loads a jsonl data file from a local path"""

    def __init__(
        self,
        path: str,
        format: str = None,
        transform: Optional[DFTransformBase] = None,
        **kwargs,
    ):
        """
        Initializes an DataReader.
        args:
            path: str, local path to the dataset file.
            format: optional str, to specify file format (parquet, csv, and jsonl).
            transform: optional Transform, to apply after loading.
            kwargs: addtional arguments.
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
        df = self._load_dataset()
        df = df.reset_index(drop=True)
        log.info(f"Loaded dataset with shape: {df.shape}")
        if not df.empty and self.transform is not None:
            df = self.transform.transform(df)
        df = df.reset_index(drop=True)
        log.info(f"Transformed dataset has shape: {df.shape}")
        return df

    def _load_dataset(self) -> pd.DataFrame:
        if self.format == ".parquet":
            log.info(f"Loading Parquet Data From {self.path}.")
            df = pd.read_parquet(self.path, **self.kwargs)
        elif self.format == ".csv":  # TODO: remove
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
    """This is a Azure storage blob DataReader that loads a jsonl data file hosted on a blob."""

    def __init__(
        self,
        account_url: str,
        blob_container: str,
        blob_name: str,
        format: str = None,
        transform: Optional[DFTransformBase] = None,
        **kwargs,
    ):
        """
        Initializes an AzureDataReader.
        args:
            account_url: str, The Azure storage account URL.
            blob_container: str ,Azure storage container name.
            blob_name: str, Azure storage blob name.
            format: optional str, specifies file format (only jsonl currently supported).
            transform: optional Transform, to apply after loading.
            kwargs: addtional arguments.
        """
        self.blob_url = f"{account_url}/{blob_container}/{blob_name}"
        super().__init__(self.blob_url, format, transform, **kwargs)
        self.logger = AzureStorageLogger().get_logger()

    def _load_dataset(self) -> pd.DataFrame:
        file = super().read_azure_blob(self.blob_url)
        if self.format == ".jsonl":
            jlr = jsonlines.Reader(file.splitlines(), loads=json.loads)
            df = pd.DataFrame(jlr.iter(skip_empty=True, skip_invalid=True))
        else:
            raise ValueError("AzureDataReader currently only supports jsonl format.")
        return df


class HFDataReader(DataReader):
    """This is a HuggingFace DataReader that downloads data hosted on HuggingFace to infer on
    using HF load_dataset method."""

    def __init__(
        self,
        path: str,
        split: List[str] | str = "test",
        tasks: List[str] | str = None,
        transform: Optional[DFTransformBase] = None,
        cache_dir: str = None,
        **kwargs,
    ):
        """
        Initializes a HFDataReader.
        args:
            path: str, Huggingface specific path.
            split: optional str or List of str, names of splits (e.g., val, test,...).
            tasks: optional str or List of str, names of tasks (e.g., Math, Art,...).
            transform: optional list of Transforms, to apply after loading.
            cache_dir: optional str, local cache path.
        """
        super().__init__(path=path, transform=transform, **kwargs)
        self.split = split
        self.tasks = tasks
        self.cache_dir = cache_dir

    def _save_base64_to_image_file(self, image_base64: dict, cache_path: str) -> str:
        """
        Saves a base64 encoded image to a local cache path and returns the path to be stored.
        args:
            image_base64: dict, that contains the byte string and file name.
            cache_path: str, that is the directory to save the image.
        returns:
            file_path: str, full path to saved image.
        """
        file_path = ""

        if image_base64:
            # create path to save image
            file_path = os.path.join(cache_path, image_base64["path"])

            # only do this if the image doesn't already exist
            if not os.path.exists(file_path):
                # base64 string to binary image data
                buffered = BytesIO(image_base64["bytes"])
                query_image = Image.open(buffered).convert("RGB")

                # save image and make the dir path if needed (need for paths with nested new dirs)
                dir_path = os.path.dirname(file_path)
                os.makedirs(dir_path, exist_ok=True)
                query_image.save(file_path)

        return file_path

    def _save_images(self, df: pd.DataFrame, cache_path: str, image_columns) -> pd.DataFrame:
        """
        Saves all base64 encoded image columns to a local cache path and updates the data frame.
        args:
            df: Panda dataframe, to save images for.
            cache_path: str, that is the directory to save the image.
            image_columns: List of str, with names of columns that have images in them.
        returns:
            df: Pandas dataframe, with cached image path updated in the column
        """

        tqdm.pandas()

        for column in tqdm(image_columns, desc="Image Saving Progress:"):
            df[column] = df[column].progress_apply(self._save_base64_to_image_file, args=(cache_path,))

        return df

    def _hf_to_dataframe(self, hf_dataset):
        """
        Converts a huggingface dataset object to a Pandas dataframe.
        If images are present in the dataset (as base64 encoded images),
        those images will be cached to the local path where the dataset files reside.
        args:
            hf_dataset: huggingface dataset, object to convert.
        returns:
            df: Pandas dataframe, converted from a huggingface dataset.
        """

        df = hf_dataset.to_pandas()

        if hasattr(hf_dataset, "features"):
            # find which columns contain images
            image_columns = [
                col
                for col in hf_dataset.features
                if hasattr(hf_dataset.features[col], "dtype") and hf_dataset.features[col].dtype == "PIL.Image.Image"
            ]

            if image_columns:

                # get the dir where the dataset is cached
                cache_path = os.path.dirname(hf_dataset.cache_files[0]["filename"])
                df = self._save_images(df, cache_path, image_columns)
                # store the names of the images columns in the data frame for later retrieval
                df["image_column_names"] = df.apply(lambda x: image_columns, axis=1)

        return df

    def _load_dataset(self) -> pd.DataFrame:
        """
        Loads a set of huggingface datasets specified as a list of splits or tasks (as provided to the class init).
        Each dataset is loaded, processed to a Pandas dataframe and then merged to a single dataframe
        Each dataframe has the task and split name added to a column before the merge
        returns:
            pd.concat(df_frames): Pandas dataframe, concatenated Pandas dataframe.
        """
        if not isinstance(self.split, list):
            self.split = [self.split]
        if self.tasks is not None and not isinstance(self.tasks, list):
            self.tasks = [self.tasks]
        df_frames = []
        if self.tasks is None:
            hf_dataset = load_dataset(self.path, cache_dir=self.cache_dir, split=self.split)
            for i, data_split in enumerate(hf_dataset):
                task_df = self._hf_to_dataframe(data_split)
                task_df["__hf_split"] = self.split[i]
                df_frames.append(task_df)
        else:
            for task in self.tasks:
                hf_dataset = load_dataset(self.path, task, cache_dir=self.cache_dir, split=self.split)
                for i, data_split in enumerate(hf_dataset):
                    task_df = self._hf_to_dataframe(data_split)
                    task_df["__hf_task"] = task
                    task_df["__hf_split"] = self.split[i]
                    df_frames.append(task_df)
        return pd.concat(df_frames)
