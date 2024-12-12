# This file was authored by authors of the Kitab dataset (https://huggingface.co/datasets/microsoft/kitab)
# All code in this file is copied from the original source repository and then adapted to fit this repository.
# The original license for this code is Community Data License Agreement - Permissive - Version 2.0

import ast
import logging
import re
import string
import time

import numpy as np
import requests
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import (
    HttpResponseError,
    ServiceRequestError,
    ServiceResponseError,
)
from fuzzywuzzy import fuzz

from eureka_ml_insights.metrics import CompositeMetric
from eureka_ml_insights.secret_management import get_secret


class KitabMetric(CompositeMetric):
    stopwords = set(["a", "an", "the", "in", "is", "of", "on", "for", "with", "to", "and"])

    def __init__(self, temp_path_names, azure_lang_service_config):
        """
        args:
            temp_path_names: path where to store the pre extracted human names via gpt-4. The file is then used to complement human names extracted via gpt-4 with those extracted via the Azure Language service.
            azure_lang_service_config: config dict for the Azure Language Service that will be used for evaluating human and city name constraints.
        """
        super().__init__()
        self.gpt4_names = kitab_utils.get_gpt4_preprocessed_names(
            "https://huggingface.co/datasets/microsoft/kitab/raw/main/code/utils/gpt_4_name_data_processed.csv",
            temp_path_names,
        )
        self.credential_func = azure_lang_service_config["secret_key_params"].get("credential_func", lambda _: None)
        # requires an Azure Cognitive Services Endpoint
        # (ref: https://learn.microsoft.com/en-us/azure/ai-services/language-service/)
        self.key = get_secret(
            key_name=azure_lang_service_config["secret_key_params"].get("key_name", None),
            local_keys_path=azure_lang_service_config["secret_key_params"].get("local_keys_path", None),
            key_vault_url=azure_lang_service_config["secret_key_params"].get("key_vault_url", None),
            credential_func=self.credential_func
        )
        self.endpoint = azure_lang_service_config["url"]
        self.text_analytics_credential = self.get_verified_credential()

    def get_verified_credential(self):
        model_version = "latest"
        try:
            text_analytics_client = TextAnalyticsClient(endpoint=self.endpoint, credential=AzureKeyCredential(self.key))
            text_analytics_client.recognize_entities(["New York City"], model_version=model_version)
            return AzureKeyCredential(self.key)
        except Exception as e:
            logging.info(f"Failed to create the TextAnalyticsClient using AzureKeyCredential")
            logging.info("The error is caused by: {}".format(e))
        try:
            text_analytics_client = TextAnalyticsClient(endpoint=self.endpoint, credential=self.credential_func())
            text_analytics_client.recognize_entities(["New York City"], model_version=model_version)
            return self.credential_func()
        except Exception as e:
            logging.info(f"Failed to create the TextAnalyticsClient using provided credential func")
            logging.info("The error is caused by: {}".format(e))
        return None

    def __evaluate__(self, row):
        if not row["is_valid"]:
            return "none"
        return self.process_row(row, self.gpt4_names)

    def process_row(self, row, gpt4_names):
        """
        Process a row of data to identify correct, incorrect, and hallucinated book titles based on given constraints.
        Args:
            row (dict): A dictionary containing the input row data with columns 'mapped_books', 'model_books',
                        'all_books', 'raw_books', 'constraint_type', and 'constraints'.
            gpt4_names (list): A list of human names used by the GPT-4 model for comparison.
        Returns:
            tuple: A tuple containing three elements:
                - A dictionary containing the processed results including correct, incorrect, and hallucinated book
                     titles, counts, and mappings.
                - An integer representing the number of unmapped raw books.
                - An integer representing the original count of model books before processing.
        Raises:
            ValueError: If the input row or constraints are not in the expected format.
        Note:
            This function assumes the following format for input row:
            - 'mapped_books', 'all_books', 'raw_books' are lists of book titles in string format.
            - 'model_books' is either a list of book titles in string format or a dictionary containing 'titles' key
                 with a list of book titles.
            Constraints can be of the following types:
            - 'starts-with': Check if the model books start with a specified prefix.
            - 'ends-with': Check if the model books end with a specified suffix.
            - 'word-count': Check if the model books have a specified word count.
            - 'publishing-year': Check if the model books' publishing year falls within a specified range.
            - 'human-name': Check if the model books contain a specified human name.
            - 'city-name': Check if the model books contain a specified city name.
        """
        satisfied = []
        unsatisfied = []
        not_from_author = []
        mapped_books = []
        model_books = []
        all_books = []
        raw_unmapped = []

        mapped_books = [self.process_title(book) for book in ast.literal_eval(row["mapped_books"])]
        model_books = (
            [self.process_title(book) for book in row["model_books"]]
            if isinstance(row["model_books"], list)
            else [self.process_title(book) for book in row["model_books"]["titles"]]
        )
        all_books = [self.process_title(self.process_all_books(book)) for book in ast.literal_eval(row["all_books"])]
        raw_books = [self.process_title(book) for book in ast.literal_eval(row["raw_books"])]

        len(model_books)

        # check for not_from_author, map model books to data books
        existing_titles_model_titles = {}
        for book in model_books.copy():
            if book == "":
                continue

            existing_title = ""
            if not any(book in item for item in all_books) and not any(item in book for item in all_books):
                close_enough, existing_title = self.fuzzy_compare(book, all_books, threshold=80)
                if not close_enough:
                    # book not in raw_books:
                    if not any(book in item for item in raw_books) and not any(item in book for item in raw_books):
                        close_enough_raw, _ = self.fuzzy_compare(book, raw_books, threshold=80)
                        if not close_enough_raw:
                            not_from_author.append(book)
                            continue
                    raw_unmapped.append(book)
                    model_books.remove(book)
                    continue
            # book in all_books. So Check if in mapped_books and then:
            if existing_title == "":
                existing_title = next((item for item in all_books if book in item or item in book), None)

            if existing_title not in existing_titles_model_titles.keys():
                existing_titles_model_titles[existing_title] = []

            existing_titles_model_titles[existing_title].append(book)

        # check for satisfaction for non-hallucinated books
        constraint_list = row["constraints"].split(",")
        constraint_count = len(constraint_list)
        if constraint_count == 1:
            constraint_type_list = [row["constraint_type"]]
        else:
            constraint_type_list = ast.literal_eval(row["constraint_type"])
        for c in range(constraint_count):
            satisfaction_search = self.compute_satisfaction_unsatisfaction(
                existing_titles_model_titles, constraint_list[c], constraint_type_list[c], all_books, row["all_books"]
            )
            satisfied_c = satisfaction_search["satisfied"]
            unsatisfied_c = satisfaction_search["unsatisfied"]
            if c == 0:
                satisfied = satisfied_c
                unsatisfied = unsatisfied_c
            else:
                # the list of books that satisfy both constraints is the intersection of the respective lists
                # that satisfy each constraint individually
                satisfied = list(set(satisfied_c).intersection(satisfied))
                # the list of books that do not satisfy at least one of constraints is the (distinct) union of the respective lists
                # that do not satisfy each constraint individually
                unsatisfied = list(set(unsatisfied_c).union(unsatisfied))
        not_from_author = list(set(not_from_author))
        satisfied = list(set(satisfied))
        unsatisfied = list(set(unsatisfied))
        count_mapped_books = len(mapped_books)
        count_all_books = len(all_books)
        count_model_books = len(model_books)
        count_satisfied = len(satisfied)
        count_unsatisfied = len(unsatisfied)
        count_not_from_author = len(not_from_author)
        count_raw_unmapped = len(raw_unmapped)
        number_of_clusters = count_not_from_author + len(existing_titles_model_titles.keys())
        # query constrainedness is defined as \kappa = 1 - \frac{S}{N}
        # where S is the number of ground truth books that satisfy the constraint
        # N is the total number of books for the author
        # the higher the constrainedness of a query, the more difficult the query is
        constrainedness = 1 - count_mapped_books / count_all_books

        # compute satisfaction, unsatisfaction, and not from author rates based on counts
        if number_of_clusters > 0:
            satisfied_rate = count_satisfied / number_of_clusters
            unsatisfied_rate = count_unsatisfied / number_of_clusters
            not_from_author_rate = count_not_from_author / number_of_clusters
        else:
            satisfied_rate = np.nan
            unsatisfied_rate = np.nan
            not_from_author_rate = np.nan

        # compute completeness and correctness
        if ast.literal_eval(row["mapped_books"]):
            set_mapped_books = set(self.process_title(book) for book in ast.literal_eval(row["mapped_books"]))
            set_satisfied = set(self.process_title(book) for book in ast.literal_eval(str(satisfied)))
            for book in satisfied:
                for alternative_title in existing_titles_model_titles[book]:
                    set_satisfied = set_satisfied.union(set([alternative_title]))
            completeness = 1 - len(set_mapped_books - set_satisfied) / len(set_mapped_books)
        else:
            completeness = np.nan

        all_correct = int((completeness == 1) & (satisfied_rate == 1) & (not_from_author_rate == 0))

        # handle corner cases
        if row["mapped_books"] == "[]" and row["model_books"] == "[]":
            completeness = 1
            satisfied_rate = 1
            unsatisfied_rate = 0
            not_from_author_rate = 0
            all_correct = 1
        elif row["mapped_books"] == "[]" and row["model_books"] != "[]":
            completeness = np.nan
            # the rest is unchanged
        elif row["mapped_books"] != "[]" and row["model_books"] == "[]":
            completeness = 0
            satisfied_rate = np.nan
            unsatisfied_rate = np.nan
            not_from_author_rate = np.nan
            all_correct = 0

        return {
            "model_books": f"{model_books}",
            "model_to_data": f"{existing_titles_model_titles}",
            "raw_unmapped": f"{raw_unmapped}",
            "satisfied": str(satisfied),
            "unsatisfied": str(unsatisfied),
            "not_from_author": str(not_from_author),
            "count_mapped_books": count_mapped_books,
            "count_all_books": count_all_books,
            "count_model_books": count_model_books,
            "count_satisfied": count_satisfied,
            "count_unsatisfied": count_unsatisfied,
            "count_not_from_author": count_not_from_author,
            "count_raw_unmapped": count_raw_unmapped,
            "number_of_clusters": number_of_clusters,
            "constrainedness": constrainedness,
            "satisfied_rate": satisfied_rate,
            "unsatisfied_rate": unsatisfied_rate,
            "not_from_author_rate": not_from_author_rate,
            "completeness": completeness,
            "all_correct": all_correct,
        }

    def compute_satisfaction_unsatisfaction(
        self, existing_titles_model_titles, single_constraint, constraint_type, all_books, str_all_books
    ):
        satisfied = []
        unsatisfied = []
        if single_constraint[-1] != ".":
            single_constraint = single_constraint + "."
        for existing_title, model_book_list in existing_titles_model_titles.items():
            if constraint_type == "starts-with":
                l = single_constraint[-2]
                if self.check_starts_with(model_book_list, l):
                    satisfied.append(existing_title)
                else:
                    unsatisfied.append(existing_title)
            elif constraint_type == "ends-with":
                l = single_constraint[-2]
                if self.check_ends_with(model_book_list, l):
                    satisfied.append(existing_title)
                else:
                    unsatisfied.append(existing_title)
            elif constraint_type == "word-count":
                c = re.search(r"(\d+)\s+word", single_constraint).group(1)
                if self.check_word_count(model_book_list, int(c)):
                    satisfied.append(existing_title)
                else:
                    unsatisfied.append(existing_title)
            elif constraint_type == "publishing-year":
                pub_year_search = re.search(
                    r"\((\d{3,4})\)", ast.literal_eval(str_all_books)[all_books.index(existing_title)]
                )
                if pub_year_search is None:
                    unsatisfied.append(existing_title)
                else:
                    pub_year = pub_year_search.group(1)
                    year_range = [int(year) for year in re.findall(r"\b(\d{1,4})\b", single_constraint)][1:]
                    if self.check_publishing_year(int(pub_year), year_range):
                        satisfied.append(existing_title)
                    else:
                        unsatisfied.append(existing_title)
            elif constraint_type == "human-name":
                if "doesn't" not in single_constraint:
                    if self.check_human_name(model_book_list + [existing_title], self.gpt4_names):
                        satisfied.append(existing_title)
                    else:
                        unsatisfied.append(existing_title)
                elif "doesn't" in single_constraint:
                    if self.check_human_name(model_book_list + [existing_title], self.gpt4_names):
                        unsatisfied.append(existing_title)
                    else:
                        satisfied.append(existing_title)

            elif constraint_type == "city-name":
                if "doesn't" not in single_constraint:
                    if self.check_city_name(model_book_list):
                        satisfied.append(existing_title)
                    else:
                        unsatisfied.append(existing_title)
                elif "doesn't" in single_constraint:
                    if self.check_city_name(model_book_list):
                        unsatisfied.append(existing_title)
                    else:
                        satisfied.append(existing_title)

        return {"satisfied": satisfied, "unsatisfied": unsatisfied}

    def process_title(self, title):
        """
        Process a book title by converting it to lowercase, replacing '&' with 'and',
        removing punctuation, and excluding common starting words ('the', 'a', 'an').
        Parameters:
        title (str): Input book title.
        Returns:
        str: Processed book title.
        """
        # Convert string to lowercase
        title = title.lower()
        # Replace '&' with 'and'
        title = title.replace("&", "and")

        # Remove punctuation
        translator = str.maketrans("", "", string.punctuation)
        title = title.translate(translator)

        # Remove first word if it's in ['the', 'a', 'an']
        first_word = title.split()[0] if title.split() else ""
        if first_word in ["the", "a", "an"]:
            title = " ".join(title.split()[1:])

        return title

    def process_all_books(self, title):
        """
        Process a book title by removing the (xxxx) format at the end of the title.
        Parameters:
        title (str): Input book title.
        Returns:
        str: Processed book title.
        """
        # Use a regex pattern to remove the (xxxx) format
        pattern = r"\(\d{3,4}\)$"
        processed_title = re.sub(pattern, "", title).strip()
        return processed_title

    def fuzzy_compare(self, title, list_of_title, threshold=90):
        """
        Perform fuzzy string comparison between the input title and a list of titles.
        Parameters:
        title (str): Input book title.
        list_of_titles (list): List of book titles for comparison.
        threshold (int): Minimum similarity score required for a match (default is 90).
        Returns:
        tuple: A tuple containing a boolean indicating if a match was found and the matched title (if found).
        Example: (True, 'Matching Title') or (False, '')
        """
        for compare_title in list_of_title:
            if fuzz.ratio(compare_title, title) >= threshold:
                return True, compare_title
        return False, ""

    def extract_cities(self, text):
        """
        Extract cities mentioned in the input text using Azure Text Analytics and external data source.
        Parameters:
        text (str): Input text containing city names.
        Returns:
        list: A list of extracted city names.
        """
        error_flag = True
        max_tries = 10
        tries = 0
        location_entities = []
        while error_flag and tries < max_tries:
            try:
                tries += 1
                text_analytics_client = TextAnalyticsClient(
                    endpoint=self.endpoint, credential=self.text_analytics_credential
                )

                # Use the given text as the input
                input_texts = [text]

                result = text_analytics_client.recognize_entities(input_texts, model_version="latest")

                error_flag = any([review.is_error for review in result])
                result = [review for review in result if not review.is_error]

                # Extract location entities
                location_entities = []
                for review in result:
                    for entity in review.entities:
                        if entity.category == "Location":
                            location_entities.append(entity.text)
                if error_flag and tries < max_tries:
                    time.sleep(1)
            except (
                HttpResponseError,
                ServiceRequestError,
                ServiceResponseError,
                requests.exceptions.ConnectionError,
            ) as e:
                self.handle_azure_language_service_exception(e)
                continue

        cities = []
        for loc in location_entities:
            url = f"https://public.opendatasoft.com/api/records/1.0/search/?dataset=geonames-all-cities-with-a-population-1000&q=name:{loc.replace(' ', '+')}&sort=-name&facet=feature_code&facet=cou_name_en&facet=timezone"  # noqa
            response = requests.get(url)
            data = response.json()
            if "records" in data.keys():
                if len(data["records"]) > 1:
                    cities.append(loc)
        return cities

    def extract_persons(self, text):
        """
        Extract persons mentioned in the input text using Azure Text Analytics service.
        Parameters:
        text (str): Input text containing person names.
        Returns:
        list: A list of extracted person names.
        """
        error_flag = True
        max_tries = 10
        tries = 0
        while error_flag and tries < max_tries:
            try:
                tries += 1
                text_analytics_client = TextAnalyticsClient(
                    endpoint=self.endpoint, credential=self.text_analytics_credential, api_version="2023-04-01"
                )
                # Use the given text as the input
                input_texts = [text]

                result = text_analytics_client.recognize_entities(input_texts, model_version="2023-04-15-preview")

                error_flag = any([review.is_error for review in result])
                result = [review for review in result if not review.is_error]

                persons = []
                for review in result:
                    for entity in review.entities:
                        if entity.category == "Person":
                            persons.append(entity.text)

                if len(persons) == 0:
                    time.sleep(1)
                    input_texts = [text.lower()]
                    text_analytics_client = TextAnalyticsClient(
                        endpoint=self.endpoint, credential=self.text_analytics_credential, api_version="2023-04-01"
                    )
                    result = text_analytics_client.recognize_entities(input_texts, model_version="2023-04-15-preview")

                    error_flag = any([review.is_error for review in result])
                    result = [review for review in result if not review.is_error]

                    persons = []
                    for review in result:
                        for entity in review.entities:
                            if entity.category == "Person":
                                persons.append(entity.text)
                if error_flag and tries < max_tries:
                    time.sleep(1)
            except (
                HttpResponseError,
                ServiceRequestError,
                ServiceResponseError,
                requests.exceptions.ConnectionError,
            ) as e:
                self.handle_azure_language_service_exception(e)
                continue
        return persons

    def handle_azure_language_service_exception(self, e):
        logging.warning(f"Azure Language Service call failed: {e}")
        time.sleep(1)

    def check_starts_with(self, books, l):
        """
        Check if any book title in the given list starts with the specified letter or word.
        Parameters:
        books (list): List of book titles.
        l (str): Letter or word to check for at the beginning of the titles.
        stopwords (list): List of stopwords to ignore (default is an empty list).
        Returns:
        bool: True if any title starts with the specified letter or word, False otherwise.
        """
        for s in books:
            words = s.split()
            if words[0].lower().startswith(l.lower()):
                return True
            if words[0].lower() in self.stopwords:
                words.pop(0)
            if words and words[0].lower().startswith(l.lower()):
                return True
        return False

    def check_ends_with(self, books, l):
        """
        Check if any book title in the given list ends with the specified letter or word.
        Parameters:
        books (list): List of book titles.
        l (str): Letter or word to check for at the end of the titles.
        Returns:
        bool: True if any title ends with the specified letter or word, False otherwise.
        """
        for s in books:
            words = s.split()
            if words and words[-1].lower().endswith(l.lower()):
                return True
        return False

    def check_word_count(self, books, c, delta=1):
        """
        Check if any book title in the given list has a word count within a specified range.
        Parameters:
        books (list): List of book titles.
        c (int): Target word count to check against.
        delta (int): Allowable difference from the target word count (default is 1).
        Returns:
        bool: True if any title has a word count within the specified range, False otherwise.
        """
        for s in books:
            word_count = len(s.split())
            if c - delta <= word_count <= c + delta:
                return True
        return False

    def check_publishing_year(self, pub_year, year_range):
        """
        Check if the given publishing year falls within the specified year range.
        Parameters:
        pub_year (int): The publishing year to be checked.
        year_range (tuple): A tuple containing two integers representing the start and end of the allowed year range.
        Returns:
        bool: True if the publishing year is within the specified range, False otherwise.
        """
        if pub_year >= year_range[0] and pub_year <= year_range[1]:
            return True
        else:
            return False

    def check_human_name(self, books, gpt4_names):
        """
        Check if any book title contains a human name, either by direct extraction or fuzzy comparison.
        Parameters:
        books (list): List of book titles to check.
        gpt4_names (set): Set of human names generated by GPT-4 for fuzzy comparison.
        Returns:
        bool: True if any title contains a human name, False otherwise.
        """
        for book in books:
            if len(self.extract_persons(book)) > 0 or self.fuzzy_compare(book, gpt4_names, 80)[0]:
                return True
        return False

    def check_city_name(self, books):
        """
        Check if any book title contains a city name.
        Parameters:
        books (list): List of book titles to check.
        Returns:
        bool: True if any title contains a city name, False otherwise.
        """
        for book in books:
            if len(self.extract_cities(book)) > 0:
                return True
        return False
