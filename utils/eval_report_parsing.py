import os
import json
import math
import re

# Usage instructions (internal use only):
# 1. Download the "release" folder from blob storage and extract it to a local directory
# 2. Point release_directory_path to the release directory
# 3. Run 'python utils/eval_report_parsing.py'
# 4. The compiled results will be written to 'website/static/compiled_results.json'

def coallate_results(release_directory_path, config):
    file_pattern = re.compile(r'^(?!.*by).*\.json$', re.IGNORECASE)
    mapping = config["capability_mapping"]
    model_family_list = config["model_families"]
    data = {
        "language": {
            "capabilities": [ ]
        },
        "multimodal": {
            "capabilities": [ ]
        }
    }

    model_list = [model["model"] for model in config["model_list"]]

    # these models have different names in some of the report folders
    model_list.append("GPT-4o_2024_05_13_450K")
    model_list.append("GPT-4o_2024_05_13")
    model_list.append("LLaVA-34B")
    model_list.append("GPT-4")

    for capability in mapping:
        name = capability["capability"]
        modality = capability["modality"]
        description = capability["description"]
        
        model_scores = []
        model_families = os.listdir(os.path.join(release_directory_path, *capability["path"]))
        for model_family in model_families:
            if model_family.lower() not in model_family_list:
                continue
            models = os.listdir(os.path.join(release_directory_path, *capability["path"], model_family))
            for model in models:
                if model not in model_list:
                    continue
                if capability["run"] == "average":
                    runs = os.listdir(os.path.join(release_directory_path, *capability["path"], model_family, model))
                else:
                    runs = [capability["run"]]
                
                sum = 0.0
                num = 0 # there's a chance that one of the runs doesn't have the correct output file so need to keep track separately
                for run in runs:
                    try:
                        file_pattern = re.compile(r'^(?!.*by).*\.json$', re.IGNORECASE)
                        if name == "Long Context QA Longest Context (3K)":
                            file_pattern = re.compile(r'^.*by_ctx_size_normalized.*\.json$', re.IGNORECASE)
                        report = [f for f in os.listdir(os.path.join(release_directory_path, *capability["path"], model_family, model, run, 'eval_report')) if file_pattern.match(f)][0]
                        file_path = os.path.join(release_directory_path, *capability["path"], model_family, model, run, 'eval_report', report)
                        with open(file_path, 'r') as f:
                            file_contents = f.read()
                            scores = json.loads(file_contents)
                            if(name == 'Object Detection (AP50)'):
                                scores = scores[0]
                            for metric in capability["metric"]:
                                scores = scores[metric]
                            if type(scores) == float:
                                sum += scores
                            else:
                                sum += scores["correct"]
                            # sum += scores
                        num += 1
                        break
                    except FileNotFoundError:
                        continue
                    except Exception as e:
                        print("Error processing file: " + file_path)
                        print(e)
                        continue

                if model == 'GPT-4o_2024_05_13_450K':
                    model = 'GPT-4o-2024-05-13'
                if model == 'GPT-4o_2024_05_13':
                    model = 'GPT-4o-2024-05-13'
                if model == "LLaVA-34B":
                    model = "Llava-1_6-34B"
                if model == "GPT-4":
                    model = "GPT-4-1106-Preview"
                model_scores.append({   
                    "name": model,
                    "score": round(sum  * 100.0 / num, 1)
                })
        data[modality]["capabilities"].append({
            "name": name,
            "description": description,
            "models": model_scores
        })
    
    # Write the final JSON file
    with open('website\\static\\compiled_results.json', 'w') as f:
        json.dump(data, f, indent=2)


def create_benchmark_breakdown(release_directory_path, config):
    mapping = config["benchmarks"]
    model_family_list = config["model_families"]
    model_list = [model["model"] for model in config["model_list"]]

    # these models have different names in some of the report folders
    model_list.append("GPT-4o_2024_05_13_450K")
    model_list.append("GPT-4o_2024_05_13")
    model_list.append("LLaVA-34B")
    model_list.append("GPT-4")

    data = { }
    for benchmark in mapping:
        name = benchmark["name"]
        data[name] = {"experiments": []}
        for experiment in benchmark["experiments"]:
            file_pattern = re.compile(experiment["filePattern"], re.IGNORECASE)
            experiment_json = {
                "title": experiment["title"],
                "categories": [],
                "series": []
            }
            
            for graph in experiment["series"]:
                graph_json = {
                    "title": graph["title"],
                    "values": []
                }
                path = graph["path"]
                model_families = os.listdir(os.path.join(release_directory_path, *path))
                
                for model_family in model_families:
                    if model_family.lower() not in model_family_list:
                        continue
                    models = os.listdir(os.path.join(release_directory_path, *path, model_family))
                    for model in models:
                        if model not in model_list:
                            continue
                        runs = os.listdir(os.path.join(release_directory_path, *path, model_family, model))
                        model_scores = {}
                        for run in runs:
                            try:
                                report_folder = 'eval_report'
                                if graph["title"] == "Instruction Following":
                                    report_folder = 'instruction_level_eval_report'

                                report = [f for f in os.listdir(os.path.join(release_directory_path, *path, model_family, model, run, report_folder)) if file_pattern.match(f)]
                                if len(report) == 0:
                                    continue
                                else:
                                    report = report[0]
                                file_path = os.path.join(release_directory_path, *path, model_family, model, run, report_folder, report)
                                with open(file_path, 'r') as f:
                                    file_contents = f.read()
                                    scores = json.loads(file_contents)
                                    for metric in graph["metric"]:
                                        try:
                                            scores = scores[metric]
                                        except TypeError: # some of the reports are a list instead of a json object (Obj Det (single)/claude opus)
                                            scores = scores[0][metric]
                                    for category in scores:
                                        if category == "none" or category == "incorrect":
                                            continue
                                        if category not in experiment_json["categories"]:
                                            experiment_json["categories"].append(category)
                                        if category not in model_scores:
                                            model_scores[category] = []
                                        if type(scores[category]) == float:
                                            score = scores[category]
                                        else:
                                            if name == "Geometric Reasoning (GeoMeter)": # geometric reasoning reports count instead of percentage
                                                none = 0
                                                if "none" in scores and (scores["none"] == scores["none"]): # check for NaN
                                                    none = scores["none"]
                                                score = scores[category]["correct"] * 1.0 / (scores[category]["correct"] + scores[category]["incorrect"] + none)
                                            else:
                                                score = scores[category]["correct"]
                                        model_scores[category].append(score)
                                break
                            except FileNotFoundError:
                                print("Error finding " + file_path)
                                continue
                            
                        if model == 'GPT-4o_2024_05_13_450K':
                            model = 'GPT-4o-2024-05-13'
                        if model == 'GPT-4o_2024_05_13':
                            model = 'GPT-4o-2024-05-13'
                        if model == "LLaVA-34B":
                            model = "Llava-1_6-34B"
                        if model == "GPT-4":
                            model = "GPT-4-1106-Preview"

                        if len(model_scores) == 0:
                            print("No scores found for model: " + model + " in experiment: " + experiment_json["title"])
                            continue
                        scores = []
                        for category in experiment_json["categories"]:
                            scores.append(round(math.fsum(model_scores[category])  * 100.0 / len(model_scores[category]), 1))
                        graph_json["values"].append({
                            "name": model,
                            "scores": scores
                        })
                experiment_json["series"].append(graph_json)
            data[name]["experiments"].append(experiment_json)
    with open('website/static/benchmark_results.json', 'w') as f:
        json.dump(data, f, indent=2)


# Example usage
release_directory_path = '/release'
config_path = 'website/static/config.json'

coallate_results(release_directory_path, json.load(open(config_path)))
create_benchmark_breakdown(release_directory_path, json.load(open(config_path)))