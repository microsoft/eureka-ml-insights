import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, Legend, Label } from 'recharts';  
import * as d3 from 'd3';
import Layout from '@theme/Layout';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { ControlRowView} from "../components/atoms";
import { Button, Checkbox, CheckboxProps, Divider, Drawer, Input, Modal, Select, Tabs, message, theme } from "antd";
import { Experiment, Model, ModelFamily, ModelResult, TransformedResult, VisualizationConfig, VisualizationFilter } from '../components/types';
import Highcharts, { SeriesOptionsType, color } from 'highcharts';
import HighchartsReact from 'highcharts-react-official';
import HC_more from 'highcharts/highcharts-more';

HC_more(Highcharts); 

const Visualization = ({config}: {config: VisualizationConfig}) => {
    if (!config) {  
        // config is still null, probably still fetching data
        return <div>Loading...</div>;
    }
    const defaultFilter: VisualizationFilter = {  
        benchmark: config.benchmarks,  
        experiment: config.experiments.map((d: Experiment) => d.experiment),  
        model_family: config.model_families.map((d: ModelFamily) => d.model_family),  
        model: config.models.map((d: Model) => d.model),  
        metric: [],  
    };  

    const [filteredData, setFilteredData] = useState<TransformedResult[]>([]);
    const [visualizationFilter, setVisualizationFilter] = useState<VisualizationFilter>(defaultFilter);
    const [filteredModels, setFilteredModels] = useState<Model[]>([]);
    const [dummyData, setDummyData] = useState<SeriesOptionsType[]>([]);

    // TODO: Remove
    const generateDummyData = (length: number) => {
        const data = [];
        for (let i = 0; i < length; i++) {
            data.push(Math.random());
        }
        return data;
    }


    React.useEffect(() => {  
        fetch('/compiled_results_transformed.json')
            .then(response => response.json())
            .then(fetchedData => 
                {
                    console.log(config.experiments)
                    for (const key in visualizationFilter) {
                        if (visualizationFilter[key]) {
                            if (key.startsWith('model'))
                            {
                                const filtered_models = key === "model" ? visualizationFilter["model"] : config.models.filter((d: Model) => visualizationFilter.model_family.some(x => x === d.model_family)).map((d: Model) => d.model);
                                fetchedData = fetchedData.map(d => {  
                                    return Object.keys(d)  
                                        .filter(key => key === "benchmark" || key === "experiment" || key === "metric" || filtered_models.includes(key))  
                                        .reduce((obj, key) => {  
                                            obj[key] = d[key];  
                                            return obj;  
                                        }, {});  
                                }); 
                            }
                            else
                            {
                                fetchedData = fetchedData.filter((d: TransformedResult) => visualizationFilter[key].some(x => x === d[key]));
                            }
                        }
                    }
                    setFilteredData(fetchedData);
                    const len = visualizationFilter.experiment.length;
                    const results = config.models.filter((d: Model) => visualizationFilter.model.some(x => x === d.model))
                                                 .map((d: Model): SeriesOptionsType => { 
                                                    return {
                                                        name: d.model, 
                                                        data: generateDummyData(len), 
                                                        pointPlacement: 'on', 
                                                        type: 'line',
                                                        color: d.color
                                                        };
                                                    }
                                                );
                    setDummyData(results);
                })
            .catch(error => console.error(error));
    }, [visualizationFilter]);

    React.useEffect(() => {  
        if (visualizationFilter?.model_family) {  
            const filtered = config.models.filter((d: Model) => visualizationFilter.model_family.some(x => x === d.model_family));  
            setFilteredModels(filtered);  
        } else {  
            setFilteredModels(config.models);  
        }  
    }, [visualizationFilter?.model_family, config.models]);

    const visualizationFilterOnChange = (list: string[], filterKey: string) => {
        const updatedFilter = { ...visualizationFilter, [filterKey]: list };
        setVisualizationFilter(updatedFilter);  
    };  

    const onCheckAllChange = (e, filterKey, defaultList) => {
        const list = e.target.checked ? defaultList : [];
        const updatedFilter = { ...visualizationFilter, [filterKey]: list };
        setVisualizationFilter(updatedFilter);  
    };

    const onModelFamilyChange = (list: string[]) => {
        const filtered_models = config.models.filter((d: Model) => list.some(x => x === d.model_family)).map((d: Model) => d.model);
        visualizationFilterOnChange(filtered_models, 'model');
        visualizationFilterOnChange(list, 'model_family');
    };

    const options: Highcharts.Options = {
        title: {
            text: 'My chart'
        },
        chart: {
            polar: true,
            type: 'area'
        },
        series: dummyData,
        xAxis: {
            categories: visualizationFilter.experiment,
            tickmarkPlacement: 'on',
        },
        yAxis: {
            gridLineInterpolation: 'polygon',
            lineWidth: 0,  
            min: 0,
            max: 1,
        },
        legend: {
            align: 'center',
            verticalAlign: 'bottom',
            layout: 'horizontal'
        },
    };
    

    return (
        <div>
            <div>
                <HighchartsReact highcharts={Highcharts} options={options} />
            </div>
            <div>
                <ControlRowView
                    title="Benchmark"
                    className="mb-2"
                    description="Filter by Benchmark"
                    value = ""
                    control={
                      <div>
                        <Checkbox.Group 
                            options={config.benchmarks} 
                            value={visualizationFilter?.benchmark} 
                            onChange={(list) => visualizationFilterOnChange(list, 'benchmark')}  />
                        <Divider />
                      </div>
                    }
                />
                <ControlRowView
                    title="Experiment"
                    className="mb-2"
                    description="Filter by Experiments"
                    value = ""
                    control={
                      <div>
                        <Select
                            className='mt-2 w-full'
                            onChange={(e) => visualizationFilterOnChange([e.target.value], 'experiment')}
                            options={
                                config.experiments.map((d: Experiment) => ({
                                    value: d.experiment,
                                    label: d.experiment
                                }))
                            }
                        />
                      </div>
                    }
                />
                <ControlRowView
                    title="Model Family"
                    className="mb-2"
                    description="Filter by Model Families"
                    value = ""
                    control={
                      <div>
                        <Checkbox 
                            indeterminate={visualizationFilter?.model_family?.length > 0 && visualizationFilter?.model_family?.length < config.model_families.length} 
                            onChange={(e) => onCheckAllChange(e, 'model_family', config.model_families.map((d: ModelFamily) => d.model_family))} 
                            checked={visualizationFilter?.model_family?.length === config.model_families.length}>
                          Check all
                        </Checkbox>
                        <Checkbox.Group 
                            options={config.model_families.map((d: ModelFamily) => d.model_family)} 
                            value={visualizationFilter?.model_family} 
                            onChange={onModelFamilyChange}  />
                        <Divider />
                      </div>
                    }
                />
                <ControlRowView
                    title="Model"
                    className="mb-2"
                    description="Filter by Model"
                    value = ""
                    control={
                      <div>
                        <Checkbox 
                            indeterminate={visualizationFilter?.model?.length > 0 && visualizationFilter?.model?.length < config.models.length} 
                            onChange={(e) => onCheckAllChange(e, 'model', config.models.map((d: Model) => d.model))} 
                            checked={visualizationFilter?.model?.length === config.models.length}>
                          Check all
                        </Checkbox>
                        <Checkbox.Group 
                            options={filteredModels.map((d: Model) => d.model)}
                            value={visualizationFilter?.model} 
                            onChange={(list) => visualizationFilterOnChange(list, 'model')}  />
                        <Divider />
                      </div>
                    }
                />
            </div>
        </div>
    );
}

export default Visualization;