@dataclass
class OpenAICommonRequestResponseMixIn:
    """
    Defines the request and response handling for most OpenAI models.

    This mixin provides methods to create a chat request body and parse the response
    from the OpenAI API.
    """

    def create_request(self, text_prompt, query_images=None, system_message=None, previous_messages=None):
        """
        Creates a request dictionary for use with the OpenAI chat API.

        Args:
            text_prompt (str): The user-provided text prompt.
            query_images (Optional[List[str]]): A list of images to encode and include, if any.
            system_message (Optional[str]): The system message to include in the conversation, if any.
            previous_messages (Optional[List[Dict]]): A list of previous messages in the conversation.

        Returns:
            Dict: A request body dictionary that can be passed to the OpenAI API.
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        if previous_messages:
            messages.extend(previous_messages)
        user_content = text_prompt
        if query_images:
            encoded_images = self.base64encode(query_images)
            user_content = [
                {"type": "text", "text": text_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_images[0]}",
                    },
                },
            ]
        messages.append({"role": "user", "content": user_content})
        request_body = {"messages": messages}
        for kwarg in {"extra_body"}:
            if hasattr(self, kwarg):
                request_body[kwarg] = getattr(self, kwarg)
        return request_body

    def get_response(self, request):
        """
        Sends a chat completion request to the OpenAI API and returns the parsed response.

        Args:
            request (Dict): The request body to send to the OpenAI API.

        Returns:
            Dict: A dictionary containing the model output, response time, and optional usage information.
        """
        start_time = time.time()
        completion = self.client.chat.completions.create(
            model=self.model_name,
            top_p=self.top_p,
            seed=self.seed,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **request,
        )
        end_time = time.time()
        openai_response = completion.model_dump()
        model_output = openai_response["choices"][0]["message"]["content"]
        response_time = end_time - start_time
        response_dict = {
            "model_output": model_output,
            "response_time": response_time,
        }
        if "usage" in openai_response:
            usage = openai_response["usage"]
            response_dict.update({"usage": usage})
            if isinstance(usage, dict) and "completion_tokens" in usage:
                response_dict.update({"n_output_tokens": usage["completion_tokens"]})
        return response_dict


class AzureOpenAIClientMixIn:
    """
    Provides Azure OpenAI-specific client methods and error handling.
    """

    def get_client(self):
        """
        Retrieves the Azure OpenAI client.

        Returns:
            AzureOpenAI: The Azure OpenAI client instance.
        """
        from openai import AzureOpenAI

        token_provider = get_bearer_token_provider(AzureCliCredential(), self.auth_scope)
        return AzureOpenAI(
            azure_endpoint=self.url,
            api_version=self.api_version,
            azure_ad_token_provider=token_provider,
        )

    def handle_request_error(self, e):
        """
        Handles an error that occurs while making a request to the Azure OpenAI service.

        If the error is due to content filtering, logs a warning and returns (None, False, True).
        Otherwise, logs the exception and returns False.

        Args:
            e (Exception): The exception raised during the request.

        Returns:
            Union[Tuple[None, bool, bool], bool]: Either a tuple indicating a content filter block
            or False indicating the request can be retried.
        """
        # if the error is due to a content filter, there is no need to retry
        if hasattr(e, "code") and e.code == "content_filter":
            logging.warning("Content filtered.")
            response = None
            return response, False, True
        else:
            logging.warning(str(e))
        return False


class DirectOpenAIClientMixIn(KeyBasedAuthMixIn):
    """
    Provides client retrieval and error handling for direct OpenAI usage.
    """

    def get_client(self):
        """
        Retrieves the direct OpenAI client.

        Returns:
            OpenAI: The direct OpenAI client instance.
        """
        from openai import OpenAI

        return OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def handle_request_error(self, e):
        """
        Handles an error that occurs while making a request to the direct OpenAI service.

        Args:
            e (Exception): The exception that was raised.

        Returns:
            bool: Always returns False, indicating the request can be retried.
        """
        logging.warning(e)
        return False


@dataclass
class AzureOpenAIModel(OpenAICommonRequestResponseMixIn, AzureOpenAIClientMixIn, EndpointModel):
    """
    Interacts with Azure OpenAI models using the provided configuration.

    Attributes:
        url (str): The Azure OpenAI endpoint URL.
        model_name (str): The name of the model.
        temperature (float): The temperature setting for the model.
        max_tokens (int): The maximum number of tokens to generate.
        top_p (float): The top-p sampling parameter.
        frequency_penalty (float): The frequency penalty.
        presence_penalty (float): The presence penalty.
        seed (int): An optional seed for reproducibility.
        api_version (str): The API version used for Azure OpenAI.
        auth_scope (str): The authentication scope for Azure OpenAI.
    """

    url: str = None
    model_name: str = None
    temperature: float = 0
    max_tokens: int = 2000
    top_p: float = 0.95
    frequency_penalty: float = 0
    presence_penalty: float = 0
    seed: int = 0
    api_version: str = "2023-06-01-preview"
    auth_scope: str = "https://cognitiveservices.azure.com/.default"

    def __post_init__(self):
        """
        Initializes the AzureOpenAIModel instance by obtaining the Azure OpenAI client.
        """
        self.client = self.get_client()


@dataclass
class DirectOpenAIModel(OpenAICommonRequestResponseMixIn, DirectOpenAIClientMixIn, EndpointModel):
    """
    Interacts directly with OpenAI models using the provided configuration.

    Attributes:
        model_name (str): The name of the model.
        temperature (float): The temperature setting for the model.
        max_tokens (int): The maximum number of tokens to generate.
        top_p (float): The top-p sampling parameter.
        frequency_penalty (float): The frequency penalty.
        presence_penalty (float): The presence penalty.
        seed (int): An optional seed for reproducibility.
        api_version (str): The API version used by OpenAI.
        base_url (str): The base URL for the OpenAI API.
        extra_body (dict): Additional data to include in the request body.
    """

    model_name: str = None
    temperature: float = 0
    max_tokens: int = 2000
    top_p: float = 0.95
    frequency_penalty: float = 0
    presence_penalty: float = 0
    seed: int = 0
    api_version: str = "2023-06-01-preview"
    base_url: str = "https://api.openai.com/v1"
    extra_body: dict = None

    def __post_init__(self):
        """
        Initializes the DirectOpenAIModel instance by obtaining the API key and direct OpenAI client.
        """
        self.api_key = self.get_api_key()
        self.client = self.get_client()


class OpenAIOModelsRequestResponseMixIn:
    """
    Defines request creation and response handling for OpenAI O1 models.
    """

    def create_request(self, text_prompt, query_images=None, system_message=None, previous_messages=None):
        """
        Creates a request dictionary for use with OpenAI O1 chat models.

        Args:
            text_prompt (str): The text prompt to send to the model.
            query_images (Optional[List[str]]): A list of images to encode and include, if supported by the model.
            system_message (Optional[str]): A system or developer message to pass to the model.
            previous_messages (Optional[List[Dict]]): A list of previous conversation messages.

        Returns:
            Dict: The request body dictionary to pass to the OpenAI API.
        """
        messages = []
        if system_message and "o1-preview" in self.model_name:
            logging.warning("System and developer messages are not supported by OpenAI O1 preview model.")
        elif system_message:
            # Developer messages are the new system messages:
            # Starting with o1-2024-12-17, o1 models support developer messages rather than system messages,
            # to align with the chain of command behavior described in the model spec.
            messages.append({"role": "developer", "content": system_message})
        if previous_messages:
            messages.extend(previous_messages)

        user_content = text_prompt
        if query_images and "o1-preview" in self.model_name:
            logging.warning("Images are not supported by OpenAI O1 preview model.")
        elif query_images:
            encoded_images = self.base64encode(query_images)
            user_content = [
                {"type": "text", "text": text_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_images[0]}",
                    },
                },
            ]

        messages.append({"role": "user", "content": user_content})
        request_body = {"messages": messages}
        for kwarg in {"extra_body"}:
            if hasattr(self, kwarg):
                request_body[kwarg] = getattr(self, kwarg)
        return request_body

    def get_response(self, request):
        """
        Sends the request to the OpenAI O1 model and returns the parsed response.

        Depending on whether the model is an O1 preview or not, it may or may not support
        certain parameters such as developer/system messages or reasoning effort.

        Args:
            request (Dict): The request body for the chat completion.

        Returns:
            Dict: A dictionary containing the model output, response time, and optional usage details.
        """
        start_time = time.time()
        if "o1-preview" in self.model_name:
            if self.reasoning_effort == "high":
                logging.error("Reasoning effort is not supported by OpenAI O1 preview model.")
            completion = self.client.chat.completions.create(
                model=self.model_name,
                seed=self.seed,
                temperature=self.temperature,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                **request,
            )
        else:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                seed=self.seed,
                temperature=self.temperature,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                reasoning_effort=self.reasoning_effort,
                **request,
            )
        end_time = time.time()
        openai_response = completion.model_dump()
        model_output = openai_response["choices"][0]["message"]["content"]
        response_time = end_time - start_time
        response_dict = {
            "model_output": model_output,
            "response_time": response_time,
        }
        if "usage" in openai_response:
            response_dict.update({"usage": openai_response["usage"]})
        return response_dict


@dataclass
class DirectOpenAIOModel(OpenAIOModelsRequestResponseMixIn, DirectOpenAIClientMixIn, EndpointModel):
    """
    Interacts directly with OpenAI O1 models using the provided configuration.

    Attributes:
        model_name (str): The name of the model.
        temperature (float): The temperature setting for generation.
        max_completion_tokens (int): Not currently used due to API constraints.
        top_p (float): The top-p sampling parameter.
        seed (int): An optional seed for reproducibility.
        frequency_penalty (float): The frequency penalty.
        presence_penalty (float): The presence penalty.
        reasoning_effort (str): The level of reasoning effort requested.
        base_url (str): The base URL for the OpenAI API.
        extra_body (dict): Additional data to include in the request.
    """

    model_name: str = None
    temperature: float = 1
    max_completion_tokens: int = 2000
    top_p: float = 1
    seed: int = 0
    frequency_penalty: float = 0
    presence_penalty: float = 0
    reasoning_effort: str = "medium"
    base_url: str = "https://api.openai.com/v1"
    extra_body: dict = None

    def __post_init__(self):
        """
        Initializes the DirectOpenAIOModel instance by obtaining the API key and direct OpenAI client.
        """
        self.api_key = self.get_api_key()
        self.client = self.get_client()


@dataclass
class AzureOpenAIOModel(OpenAIOModelsRequestResponseMixIn, AzureOpenAIClientMixIn, EndpointModel):
    """
    Interacts with Azure OpenAI O1 models using the provided configuration.

    Attributes:
        url (str): The Azure endpoint for O1 models.
        model_name (str): The name of the O1 model.
        temperature (float): The temperature setting for generation.
        max_completion_tokens (int): Not currently used due to API constraints.
        top_p (float): The top-p sampling parameter.
        seed (int): An optional seed for reproducibility.
        frequency_penalty (float): The frequency penalty.
        presence_penalty (float): The presence penalty.
        reasoning_effort (str): The level of reasoning effort requested.
        api_version (str): The API version for Azure capabilities.
        auth_scope (str): The scope for Azure authentication.
    """

    url: str = None
    model_name: str = None
    temperature: float = 1
    max_completion_tokens: int = 64000
    top_p: float = 1
    seed: int = 0
    frequency_penalty: float = 0
    presence_penalty: float = 0
    reasoning_effort: str = "medium"
    api_version: str = "2023-06-01-preview"
    auth_scope: str = "https://cognitiveservices.azure.com/.default"

    def __post_init__(self):
        """
        Initializes the AzureOpenAIOModel instance by obtaining the Azure OpenAI client.
        """
        self.client = self.get_client()


@dataclass
class GeminiModel(EndpointModel, KeyBasedAuthMixIn):
    """
    Interacts with Gemini models through the Python API.

    Attributes:
        timeout (int): The API request timeout in seconds.
        model_name (str): The name of the Gemini model.
        temperature (float): The temperature setting for generation.
        max_tokens (int): The maximum number of tokens to generate.
        top_p (float): The top-p sampling parameter.
    """

    timeout: int = 600
    model_name: str = None
    temperature: float = 0
    max_tokens: int = 2000
    top_p: float = 0.95

    def __post_init__(self):
        """
        Initializes the GeminiModel by configuring the generative AI client with the provided API key
        and safety settings. Also sets up the generation config.
        """
        super().__post_init__()
        import google.generativeai as genai
        from google.generativeai.types import HarmBlockThreshold, HarmCategory

        genai.configure(api_key=self.api_key)
        # Safety config, turning off all filters for direct experimentation with the model only
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }
        self.gen_config = genai.GenerationConfig(
            max_output_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p
        )

    def create_request(self, text_prompt, query_images=None, system_message=None, previous_messages=None):
        """
        Creates a request for generating content with Gemini models.

        Args:
            text_prompt (str): The text prompt to send to the model.
            query_images (Optional[List[str]]): Image data to pass to the model.
            system_message (Optional[str]): An optional system instruction to pass to the model.
            previous_messages (Optional[List[Dict]]): A list of previous conversation messages (unused).

        Returns:
            Union[str, List[str]]: The prompt alone, or a list combining the prompt and images.
        """
        import google.generativeai as genai

        if self.model_name == "gemini-1.0-pro":
            if system_message:
                logging.warning("System messages are not supported for Gemini 1.0 Pro.")
            self.model = genai.GenerativeModel(self.model_name)
        else:
            self.model = genai.GenerativeModel(self.model_name, system_instruction=system_message)

        if query_images:
            return [text_prompt] + query_images
        else:
            return text_prompt

    def get_response(self, request):
        """
        Sends the request to the Gemini model and returns the parsed response.

        Args:
            request (Union[str, List[str]]): The text prompt or a combined prompt and images.

        Returns:
            Dict: A dictionary containing the model output, response time, and usage metadata if available.
        """
        start_time = time.time()
        gemini_response = None
        try:
            gemini_response = self.model.generate_content(
                request,
                generation_config=self.gen_config,
                request_options={"timeout": self.timeout},
                safety_settings=self.safety_settings,
            )
            end_time = time.time()
            model_output = gemini_response.parts[0].text
            response_time = end_time - start_time
        except Exception as e:
            self.handle_gemini_error(e, gemini_response)

        response_dict = {
            "model_output": model_output,
            "response_time": response_time,
        }
        if hasattr(gemini_response, "usage_metadata"):
            try:
                response_dict.update(
                    {
                        "usage": {
                            "prompt_token_count": gemini_response.usage_metadata.prompt_token_count,
                            "candidates_token_count": gemini_response.usage_metadata.candidates_token_count,
                            "total_token_count": gemini_response.usage_metadata.total_token_count,
                        }
                    }
                )
            except AttributeError:
                logging.warning("Usage metadata not found in the response.")
        return response_dict

    def handle_gemini_error(self, e, gemini_response):
        """
        Handles exceptions originating from making requests to the Gemini API.

        If the model explicitly or implicitly blocks the prompt, logs a warning
        indicating the block reason and raises the exception again.

        Args:
            e (Exception): The exception that occurred.
            gemini_response (Any): The response object from the Gemini model.

        Raises:
            Exception: Re-raises the provided exception after handling.
        """
        # Handling cases where the model explicitly blocks prompts and provides a reason for it.
        if e.__class__.__name__ == "ValueError" and gemini_response.prompt_feedback.block_reason > 0:
            logging.warning(
                f"Attempt failed due to explicitly blocked input prompt: {e} Block Reason {gemini_response.prompt_feedback.block_reason}"
            )

        # Handling cases where the model implicitly blocks prompts and does not provide an explicit block reason.
        elif e.__class__.__name__ == "IndexError" and len(gemini_response.parts) == 0:
            logging.warning(f"Attempt failed due to implicitly blocked input prompt and empty model output: {e}")
            if len(gemini_response.candidates) > 0:
                logging.warning(f"The response is not empty and has : {len(gemini_response.candidates)} candidates")
                logging.warning(
                    f"Finish Reason for the first answer candidate is: {gemini_response.candidates[0].finish_reason}"
                )
                logging.warning(
                    f"Safety Ratings for the first answer candidate are: {gemini_response.candidates[0].safety_ratings}"
                )

        raise e

    def handle_request_error(self, e):
        """
        Handles an error that occurs while making a request to the Gemini model.

        Any error case not handled in handle_gemini_error will be attempted again.

        Args:
            e (Exception): The exception that was raised.

        Returns:
            bool: Always returns False, indicating the request can be retried.
        """
        return False


@dataclass
class TogetherModel(OpenAICommonRequestResponseMixIn, KeyBasedAuthMixIn, EndpointModel):
    """
    Interacts with Together models through the together Python API.

    Attributes:
        timeout (int): The API request timeout in seconds.
        model_name (str): The name of the Together model.
        temperature (float): The temperature setting for generation.
        max_tokens (int): The maximum number of tokens to generate.
        top_p (float): The top-p sampling parameter.
        presence_penalty (float): The presence penalty.
        stop (List[str]): A list of stop tokens for generation.
    """

    timeout: int = 600
    model_name: str = None
    temperature: float = 0
    max_tokens: int = 65536
    top_p: float = 0.95
    presence_penalty: float = 0
    stop = ["<｜end▁of▁sentence｜>"]

    def __post_init__(self):
        """
        Initializes the TogetherModel by setting up the Together client with the provided API key.
        """
        from together import Together

        self.api_key = self.get_api_key()
        self.client = Together(api_key=self.api_key)

    def get_response(self, request):
        """
        Sends the request to the Together model and returns the parsed response.

        Args:
            request (Dict): The request body for the Together chat completion.

        Returns:
            Dict: A dictionary containing the model output, response time, and optional usage details.
        """
        start_time = time.time()
        completion = self.client.chat.completions.create(
            model=self.model_name,
            top_p=self.top_p,
            presence_penalty=self.presence_penalty,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=self.stop,
            **request,
        )

        end_time = time.time()
        openai_response = completion.model_dump()
        model_output = openai_response["choices"][0]["message"]["content"]
        response_time = end_time - start_time
        response_dict = {
            "model_output": model_output,
            "response_time": response_time,
        }
        if "usage" in openai_response:
            response_dict.update({"usage": openai_response["usage"]})
        return response_dict

    def handle_request_error(self, e):
        """
        Handles an error that occurs while making a request to the Together service.

        Args:
            e (Exception): The exception that was raised.

        Returns:
            bool: Always returns False, indicating the request can be retried.
        """
        return False