import os
import json
import re


def coallate_kitlab_results(directory_path, file_pattern):
    data = []

    experiments = os.listdir(directory_path)
    for experiment in experiments:
        model_families = os.listdir(os.path.join(directory_path, experiment))
        for model_family in model_families:
            models = os.listdir(os.path.join(directory_path, experiment, model_family))
            for model in models:
                runs = os.listdir(os.path.join(directory_path, experiment, model_family, model))
                merged_runs = {}
                for run in runs:
                    report_files = [f for f in os.listdir(os.path.join(directory_path, experiment, model_family, model, run, 'eval_report')) if file_pattern.match(f)]
                    for report in report_files:
                        file_path = os.path.join(directory_path, experiment, model_family, model, run, 'eval_report', report)
                        with open(file_path, 'r') as f:
                            file_contents = f.read()
                            scores = json.loads(file_contents)
                            for metric in scores:
                                merged_runs[metric] = merged_runs[metric] + [scores[metric]] if metric in merged_runs else [scores[metric]]
                for metric in merged_runs:
                    scores = merged_runs[metric]
                    data.append({
                        'benchmark': 'Kitab',
                        'experiment': experiment,
                        'model_family': model_family,
                        'model': model,
                        'metric': metric.replace("KitabMetric_", ''),
                        'value': sum(scores) / len(scores)
                    })
    
    # Write the final JSON file
    with open('website\\static\\compiled_results.json', 'w') as f:
        json.dump(data, f, indent=2)

    transformed = []  
    for item in data:  
        # Check if we already have an entry for this benchmark, experiment, and metric  
        existing_item = next((x for x in transformed if x["benchmark"] == item["benchmark"] and x["experiment"] == item["experiment"] and x["metric"] == item["metric"]), None)  
        if existing_item:  
            # If we do, add to the scores list  
            existing_item[item["model"]] = item["value"]  
        else:  
            # If we don't, create a new entry  
            transformed.append({  
                "benchmark": item["benchmark"],  
                "experiment": item["experiment"],  
                "metric": item["metric"],  
                item["model"]: item["value"]
            })  
    with open('website\\static\\compiled_results_transformed.json', 'w') as f:  
        json.dump(transformed, f, indent=2)



def create_config(directory_path, file_pattern):
    data = { "benchmarks": ["Kitab"], "experiments": [], "model_list":[] }
    modelsSet = {}

    experiments = os.listdir(directory_path)
    for experiment in experiments:
        model_families = os.listdir(os.path.join(directory_path, experiment))
        metrics = set()
        for model_family in model_families:
            if model_family not in modelsSet:
                modelsSet[model_family] = set()
            models = os.listdir(os.path.join(directory_path, experiment, model_family))
            modelsSet[model_family].update(models)
            for model in models:
                runs = os.listdir(os.path.join(directory_path, experiment, model_family, model))
                for run in runs:
                    report_files = [f for f in os.listdir(os.path.join(directory_path, experiment, model_family, model, run, 'eval_report')) if file_pattern.match(f)]
                    for report in report_files:
                        file_path = os.path.join(directory_path, experiment, model_family, model, run, 'eval_report', report)
                        with open(file_path, 'r') as f:
                            file_contents = f.read()
                            scores = json.loads(file_contents)
                            for metric in scores:
                                metrics.add(metric.replace("KitabMetric_", ''))
        data['experiments'].append({"experiment": experiment, "metrics": list(set(metrics))})

    for model_family in modelsSet:
        data['model_list'].append({"model_family": model_family, "models": list(modelsSet[model_family])})

    # Write the final JSON file
    with open('website\\static\\config.json', 'w') as f:
        json.dump(data, f, indent=2)


# Example usage
directory_path = 'C:\\Users\\jluey\\Downloads\\reports\\release\\Kitab'
file_pattern = re.compile(r'^(?!.*by).*\.json$', re.IGNORECASE)
coallate_kitlab_results(directory_path, file_pattern)
# create_config(directory_path, file_pattern)