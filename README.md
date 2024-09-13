# Eureka ML Insights Framework
<p align="left">
  <a href='https://aka.ms/eureka-ml-insights-report'>
    <img src=docs/figures/eureka_logo.png width="16">
    Technical Report  
  </a>
  <a href='https://aka.ms/eureka-ml-insights-blog'>
    <img src=docs/figures/msr_blog.png width="16">
    Blog Post
  </a>
  <a href='https://microsoft.github.io/eureka-ml-insights'>
    <img src=docs/figures/github.png width="16">
    Project Website
  </a>
</p>
This repository contains the code for the Eureka ML Insights framework. The framework is designed to help researchers and practitioners run reproducible evaluations of generative models using a variety of benchmarks and metrics efficiently. The framework allows the user to define custom pipelines for data processing, inference, and evaluation, and provides a set of pre-defined evaluation pipelines for key benchmarks.

Certainly! Here's a simple markdown table for you:

| Benchmark <br> #prompts       | Modality  | Capability           |Logs| Pipeline Config |
|-------------------------------|---------------|----------------------|------|-----|
| GeoMeter 1086            | Image -> Text | Geometric Reasoning  | Link | [geometer.py](eureka_ml_insights/configs/geometer.py) |
| MMMU 900                 | Image -> Text | Multimodal QA        | Link |[mmmu.py](eureka_ml_insights/configs/mmmu.py)|
| Image Understanding <br> 10249| Image -> Text | Object Recognition <br> Object Detection <br> Visual Prompting <br> Spatial Reasoning | Link <br> Link <br> Link <br> Link | [object_recognition.py](eureka_ml_insights/configs/spatial_understanding/object_recognition.py) <br> [object_detection.py](eureka_ml_insights/configs/spatial_understanding/object_detection.py) <br> [visual_prompting.py](eureka_ml_insights/configs/spatial_understanding/visual_prompting.py) <br> [spatial_reasoning.py](eureka_ml_insights/configs/spatial_understanding/spatial_reasoning.py) |
| Vision Language <br> 13500    | Image -> Text | Spatial Understanding <br> Navigation <br> Counting| Link <br> Link <br> Link |[spatial_map.py](eureka_ml_insights/configs/vision_language/spatial_map.py) <br> [maze.py](eureka_ml_insights/configs/vision_language/maze.py) <br> [spatial_grid.py](eureka_ml_insights/configs/vision_language/spatial_grid.py)|
| IFEval 541                 | Text -> Text | Instruction Following        | Link |[ifeval.py](eureka_ml_insights/configs/ifeval.py)|
| FlenQA 12000               | Text -> Text | Long Context Multi-hop QA | Link |[flenQA.py](eureka_ml_insights/configs/flenqa.py)|
| Kitab 34217                | Text -> Text | Information Retrieval        | Link |[kitab.py](eureka_ml_insights/configs/kitab.py)|
| Toxigen 10500              | Text -> Text | Toxicity Detection <br> Safe Language Generation         | Link |[toxigen.py](eureka_ml_insights/configs/toxigen.py)|

## Installation
To get started, clone this repository to your local machine and navigate to the project directory.

### 🐍Installing with Conda
1. Make sure you have [Conda](https://docs.anaconda.com/free/miniconda/) installed on your system.
2. Open a terminal and create a new Conda environment using the `environment.yml` file:\
    ```conda env create --name myenv --file environment.yml```
3. Activate your newly created environment:\
    `conda activate myenv`
4. [Optional] Install GPU packages if you have a GPU machine and want to self-host models:\
    ```conda env update --file environment_gpu.yml```

### 📦 Installing with pip + editable for development
1. ```python3 -m venv .venv```
2. Activate venv.
3. ```pip install -e .```

### 📦 Generate wheel package to share with others
1. *activate venv*
2. Update version inside setup.py if needed.
3. ```python setup.py bdist_wheel```
4. Fetch from dir dist/ the .whl
5. This file can be installed via `pip install eureka_ml_insights.whl`

## 💥 Quick start
To reproduce the results of a pre-defined experiment pipeline, you can run the following command:

```python main.py --exp_config exp_config_name --model_config model_config_name --exp_logdir your_log_dir```

For example, to run the `KITAB_ONE_BOOK_CONSTRAINT_PIPELINE` experiment pipeline defined in `eureka_ml_insights/configs/kitab.py` using the OpenAI GPT4 1106 Preview model, you can run the following command:

```python main.py --exp_config KITAB_ONE_BOOK_CONSTRAINT_PIPELINE --model_config OAI_GPT4_1106_PREVIEW_CONFIG --exp_logdir gpt4_1106_preveiw```

The results of the experiment will be saved in the `logs/KITAB_ONE_BOOK_CONSTRAINT_PIPELINE/gpt4_1106_preveiw` directory.
For other available experiment pipelines and model configurations, see the `eureka_ml_insights/configs` directory.

## 🔧 Configuring a Custom Experiment Pipeline
![Components](./docs/figures/transparent_uml.png)
You can find examples of experiment pipeline configurations in `configs`. To create a new experiment configuration, you need to define a class that inherits from `ExperimentConfig` and implements the `configure_pipeline` method. In the `configure_pipeline` method you define the Pipeline config (arrangement of Components) for your Experiment. Once your class is ready, add it to `configs/__init__.py` import list.


Your Pipeline can use any of the available Components which can be found under the `core` directory:
- `PromptProcessing`: you can use this component to prepare your data for inference, apply transformation, or apply a Jinja prompt template.
- `DataProcessing`: you can use this component to to post-process the model outputs.
- `Inference`: you can use this component to run your model on any processed data, for example running inference on the model subject to evaluation, or another model that is involved in the evaluation pipeline as an evaluator or judge.
- `EvalReporting`: you can use this component to evaluate the model outputs using various metrics, aggregators and visualizers, and generate a report.
- `DataJoin`: you can use this component to join two sources of data, for example to join the model outputs with the ground truth data for evaluation.

Note that:
- You can inherit from one of the existing experiment config classes and override the necessary attributes to reduce the amount of code you need to write. You can find examples of this too in `configs/spatial_reasoning.py`.
- Your pipeline does not need to use all of the components. You can use only the components you need. And you can use the components multiple times in the pipeline.
- Make sure the input of each component matches the output of the previous component in the pipeline. The components are run sequentially in the order they are defined in the pipeline configuration.
- For standard scenarios you do not need to implement new components for your pipeline, but you do need to configure the existing components to use the correct utility classes for your scenario.

### 🔧 Utility Classes Used in Components
The components in your pipeline need to use the corrent utility classes for your scenario. In standard scenarios do not need to implement new components for your pipeline, but you do need to configure the existing components to work with the correct utility classes. If you need a functionality that is not provided by the existing utility classes, you can implement a new utility class and use it in your pipeline.

In general, to find out what utility classes and other attributes need to be configured for a component, you can look at the component's corresponding Config dataclass in `configs/config.py`. For example, if you are configuring the `DataProcessing` component, you can look at the `DataProcessingConfig` dataclass in `configs/config.py`.

Utility classes are also configurable by providing the name of the class and the initialization arguments. For example see ModelConfig in `configs/config.py` that can be initialized with the model class name and the model initialization arguments.

Our current components use the following utility classes: `DataReader`, `DataLoader`, `Model`, `Metric`, `Aggregator`. You can use the existing utility classes or implement new ones as needed to configure your components.

### 🔧 Configuring the Data Processing Component
This component is used for general data processing tasks.

- `data_reader_config`: Configuration for the DataReader that is used to load the data into a pandas dataframe, apply any necessary processing on it (optional), and return the processed data. We currently support local and Azure Blob Storage data sources.
    - Transformations: you can find the available transformations in `data_utils/transforms.py`. If you need to implement new transform classes, add them to this file.
- `output_dir`: This is the folder name where the processed data will be saved. This folder will automatically be created under the experiment log directory and the processed data will be saved in a file called `processed_data.jsonl`.
- `transformed_data_columns` (OPTIONAL): This is the list of columns to save in transformed_data.jsonl. By default, all columns are saved.

### 🔧 Configuring the Prompt Processing Component
This component inherits from the DataProcessing component and is used specifically for prompt processing tasks, such as applying a Jinja prompt template. If a prompt template is provided, the processed data will have a 'prompt' column that is expected by the inference component. Otherwise the input data is expected to already have a 'prompt' column. This component also reserves the "model_output" column for the model outputs so if it already exists in the input data, it will be removed. 

In addition to the attributes of the DataProcessing component, the PromptProcessing component has the following attributes:
- `prompt_template_path` (OPTIONAL): This template is used to format your data for model inference in case you need prompt templating or system prompts. Provide your jinja prompt template path to this component. See for example `prompt_templates/basic.jinja`. The prompt template processing step adds a 'prompt' column to the processed data, which is expected by the inference component. If you do not need prompt templating, make sure your data already does have a 'prompt' column.
- `ignore_failure` (OPTIONAL): Whether to ignore the failure of prompt processing on a row and move on to the next, or to raise an exception. Default is False.

### 🔧 Configuring the Inference Component
- `model_config`: Configuration of the model class to use for inference. You can find the available models in `models/`.
- `data_config`: Configuration of the data_loader class to use for inference. You can find the available data classes in `data_utils/data.py`.
- `output_dir`: This is the folder name where the model outputs will be saved. This folder will automatically be created under the experiment log directory and the model outputs will be saved in a file called `inference_result.jsonl`.

### 🔧 Configuring the Evaluation Reporting  Component
- `data_reader_config`: Configuration object for the DataReader that is used to load the data into a pandas dataframe. This is the same type of utility class used in the DataProcessing component.
- `metric_config`: a MetricConfig object to specify the metric class to use for evaluation. You can find the available metrics in `metrics/`. If you need to implement new metric classes, add them to this directory.
- `aggregator_configs`/`visualizer_configs`: List of configs for aggregators/visualizers to apply to the metric results. These classes that take metric results and aggragate/analyze/vizualize them and save them. You can find the available aggregators and visualizers in `metrics/reports.py`.
- `output_dir`: This is the folder name where the evaluation results will be saved.

# ✋ How to contribute:
- To contribute to the framework, please create a new branch.
- Implement your pipeline configuration class under `configs` and any utility classes that your pipeline requires.
- Please add end-to-end tests for your contributions in the `tests` directory.
- Please add unit tests for any new utility classes you implement in the `tests` directory.
- Please add documentation to your classes and methods in form of docstrings.
- Use `git add filename` to add the files you want to commit, and ONLY the files you want to commit.  
- Then use `make format-inplace` to format the files you have changed. This will only work on files that git is tracking, so make sure to `git add` any newly created files before running this command.
- Use `make linters` to check any remaining style or format issues and fix them manually.
- Use `make test` to run the tests and make sure they all pass.
- Finally, submit a pull request.

# ✒️ Citation
If you use this framework in your research, please cite the following paper:
```
@article{eureka2024,
  title={Eureka: Evaluating and Understanding Large Foundation Models},
  author={Vidhisha Balachandran, Jingya Chen, Neel Joshi, Besmira Nushi, Hamid Palangi, Eduardo Salinas, Vibhav Vineet, James Woffinden-Luey, Safoora Yousefi},
  year={2024}
  eprint={TODO},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={TODO}, 
}

```