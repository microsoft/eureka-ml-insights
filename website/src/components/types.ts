import React from "react";

export interface EurekaConfig {
    benchmarks: Benchmark[];
    models: ModelConfig[];
    model_families: string[];
    capability_mapping: Capability[];
}
export interface Benchmark {
    name: string;
    modality: string;
    filePattern: string;
    description: string;
    graphs: GraphDetails[];
}

export interface GraphDetails {
    title: string;
    description: string;
}

export interface Capability {
    capability: string;
    modality: string;
    path: string[];
    metric: string[];
    description: string[];
}

export interface BenchmarkGraph {
    title: string;
    models: ModelScore[];
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