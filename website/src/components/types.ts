export interface VisualizationConfig {
    benchmarks: string[];
    experiments: Experiment[];
    models: Model[];
    model_families: ModelFamily[];
}

export interface VisualizationFilter {
    benchmark: string[];
    experiment: string[];
    model_family: string[];
    model: string[];
    metric: string[];
}

export interface Experiment {
    experiment: string;
    metrics: string[];
}

export interface ModelFamily {
    model_family: string;
    models: string[];
}

export interface Model {
    model: string;
    model_family: string;
    color: string;
}

export interface TransformedResult {
    benchmark: string;  
    experiment: string;  
    metric: string;  
    [key: string]: string | number; 
}

export interface ModelResult {  
    name: string;
    data: number[];
    pointPlacement: string;
    type: string;
    color: string;
}