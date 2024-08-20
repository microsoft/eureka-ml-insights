import os
import json
import re


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
                runs = os.listdir(os.path.join(release_directory_path, *capability["path"], model_family, model))
                for run in runs:
                    try:
                        report = [f for f in os.listdir(os.path.join(release_directory_path, *capability["path"], model_family, model, run, 'eval_report')) if file_pattern.match(f)][0]
                        break
                    except FileNotFoundError:
                        continue
                file_path = os.path.join(release_directory_path, *capability["path"], model_family, model, run, 'eval_report', report)
                with open(file_path, 'r') as f:
                    file_contents = f.read()
                    scores = json.loads(file_contents)
                    for metric in capability["metric"]:
                        scores = scores[metric]
                    model_scores.append({
                        "name": model,
                        "score": scores
                    })
        data[modality]["capabilities"].append({
            "name": name,
            "description": description,
            "models": model_scores
        })
    
    # Write the final JSON file
    with open('website\\static\\compiled_results.json', 'w') as f:
        json.dump(data, f, indent=2)

# Example usage
release_directory_path = 'C:\\Users\\jluey\\Downloads\\reports\\release'
config_path = 'website\\static\\config.json'

coallate_results(release_directory_path, json.load(open(config_path)))