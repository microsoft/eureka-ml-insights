import React from "react";

export interface VisualizationConfig {
    benchmarks: string[];
    experiments: Experiment[];
    models: ModelConfig[];
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
    name: string;
    score: number;
}

export interface ModelConfig {
    model_family: string;
    model: string;
    color: string;
    modalities: string[];
}

export interface Capability{
    name: string;
    description: string;
    models: Model[];
}

export interface TableEntry{
    key: React.Key;
    model: string;
    [capability: string]: string | number;
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