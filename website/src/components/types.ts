import React from "react";

export interface Config {
    benchmarks: string[];
    models: ModelConfig[];
    model_families: string[];
    capability_mapping: Capability[];
}

export interface Capability {
    "capability": string;
    "modality": string;
    "path": string[]
    "metric": string[],
    "description": string[];
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

export interface CapabilityScores {
    name: string;
    description: string;
    models: ModelScore[];
}