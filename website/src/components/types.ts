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
}

export interface CompiledResult {
    benchmark: string;
    experiment: string;
    model_family: string;
    model: string;
    metric: string;
    value: number;
}

export interface TransformedResult {
    benchmark: string;  
    experiment: string;  
    metric: string;  
    [key: string]: string | number; 
}