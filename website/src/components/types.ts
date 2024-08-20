import React from "react";

export interface Config {
    benchmarks: string[];
    models: ModelConfig[];
    model_families: string[];
}

export interface ModelScore {
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
    models: ModelScore[];
}