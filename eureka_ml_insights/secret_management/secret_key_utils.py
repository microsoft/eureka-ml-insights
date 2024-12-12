import json
import logging
import os
from typing import Dict, Optional

from azure.keyvault.secrets import SecretClient

logging.basicConfig(level=logging.INFO, format="%(filename)s - %(funcName)s - %(message)s")


def get_secret(key_name: str, local_keys_path:Optional[str]=None, key_vault_url:Optional[str]=None, credential_func=lambda _: None) -> Optional[str]:
    """This function retrieves a key from key vault or if it is locally cached in a file.
    args:
        key_name: str, the name of the key to retrieve.
        local_keys_path: str, the path to the keys file where the key is cached or should be cached by this method.
        key_vault_url: str, the url of the key vault to retrieve the key from if not found in the local keys file.
    Returns:
        key_value: str, the value of the key if found, otherwise None.
    """

    keys_dict = {}

    # make sure one of local_keys_path or key_vault_url is provided
    if local_keys_path is None and key_vault_url is None:
        logging.error("One of local_keys_path or key_vault_url must be provided.")
        return None

    # if local_keys_path is not provided, create a file path to cache the keys
    if local_keys_path is None:
        logging.info("Local keys file path not provided, caching keys in keys/keys.json")
        local_keys_path = os.path.join("keys", "keys.json")

    # Try to get the key from local cache file.
    key_value, keys_dict = get_key_from_local_file(key_name, local_keys_path)

    # if the key wasn't found in cache file, try to get it from Azure
    if key_value is None:
        if key_vault_url is None:
            raise ValueError(
                f"Key [{key_name}] not found in local keys file [{local_keys_path}] and key_vault_url is not provided."
            )
        else:
            key_value = get_key_from_azure(key_name, key_vault_url, credential_func=credential_func)

    # if the key still wasn't found, raise an error
    if key_value is None:
        error_str = (
            f"Failed to get key [{key_name}] from both local keys file [{local_keys_path}] or "
            f"Azure key vault [{key_vault_url}]."
        )
        raise ValueError(error_str)

    # update the local keys file with the new key
    os.makedirs(os.path.dirname(local_keys_path), exist_ok=True)
    keys_dict[key_name] = key_value
    with open(local_keys_path, "w") as file:
        json.dump(keys_dict, file, indent=4)
    return key_value


def get_key_from_azure(key_name: str, key_vault_url: str, credential_func=lambda _: None) -> Optional[str]:
    """This function retrieves a key from azure key vault.
    args:
        key_name: str, the name of the key to retrieve.
        key_vault_url: str, the url of the key vault to retrieve the key from.
    Returns:
        key_value: str, the value of the key if found, otherwise None.
    """
    logging.getLogger("azure").setLevel(logging.ERROR)
    try:
        logging.info(f"Trying to get the key from Azure Key Vault {key_vault_url} using provided func")
        credential = credential_func(additionally_allowed_tenants=["*"])
        client = SecretClient(vault_url=key_vault_url, credential=credential)
        retrieved_key = client.get_secret(key_name)
        return retrieved_key.value
    except Exception as e:
        logging.info(f"Failed to get the key from Azure Key Vault {key_vault_url} using provided func")
        logging.info("The error is caused by: {}".format(e))
        return None


def get_key_from_local_file(key_name: str, local_keys_path: str) -> tuple[Optional[str], Dict[str, str]]:
    """This function retrieves a key from a local file.
    args:
        key_name: str, the name of the key to retrieve.
        local_keys_path: str, the path to the local keys file.
    Returns:
        key_value: str, the value of the key if found in the local file, otherwise None.
        keys_dict: dict, a dictionary containing the keys cached in the local file.
    """
    keys_dict = {}
    key_value = None
    if os.path.exists(local_keys_path):
        keys_dict = get_cached_keys_dict(local_keys_path)
        if key_name in keys_dict:
            key_value = keys_dict[key_name]
        else:
            logging.info(f"Key [{key_name}] not found in local keys file {local_keys_path}.")
    return key_value, keys_dict


def get_cached_keys_dict(local_keys_path: str) -> Dict[str, str]:
    """This function retrieves the keys cached in local json file.
    args:
        local_keys_path: str, the path to the local keys file.
    Returns:
        keys_dict: dict, a dictionary containing the keys cached in the local file
                   or an empty dictionary if the file does not exist or can't be decoded.
    """
    try:
        with open(local_keys_path, "r") as file:
            keys_dict = json.load(file)
    except Exception as e:
        logging.error(e)
        keys_dict = {}
    return keys_dict


if __name__ == "__main__":
    key_name = "aifeval-datasets"
    key_vault_url = "https://aifeval.vault.azure.net/"
    get_secret(key_name, key_vault_url=key_vault_url)
