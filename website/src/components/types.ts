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
    benchmarkDescription: string;
    capabilityImportance: string;
    experiments: Experiment[];
}

export interface Experiment {
    title: string;
    experimentDescription: string;
}

export interface Capability {
    capability: string;
    benchmark: string;
    modality: string;
    path: string[];
    metric: string[];
    description: string[];
}

export interface BenchmarkExperiment {
    title: string;
    categories: string[];
    series: BenchmarkGraph[];
}

export interface ModelScore {
    name: string;
    score: number;
}

export interface BenchmarkGraph {
    title: string;
    values: BenchmarkResult[];
}

export interface BenchmarkResult {
    name: string;
    scores: Number[];
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