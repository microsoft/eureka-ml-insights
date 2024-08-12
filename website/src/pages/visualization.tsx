import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';  
import * as d3 from 'd3';
import Layout from '@theme/Layout';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { ControlRowView} from "../components/atoms";
import { Button, Checkbox, CheckboxProps, Divider, Drawer, Input, Modal, Select, Tabs, message, theme } from "antd";
import { CompiledResult, Experiment, Model, ModelFamily, VisualizationFilter } from '../components/types';



const App: React.FC = () => {
    const {siteConfig} = useDocusaurusContext();
    return (
        <Layout
            title={`Hello from ${siteConfig.title}`}
            description="Description will go into a meta tag in <head />">
            <main>
                <div>
                    <h1>Your Webpage Title</h1>
                    <div>
                        <Visualization/>
                    </div>
                </div>
            </main>
        </Layout>
    );
};

const Visualization = () => {
    const [filteredData, setFilteredData] = useState<CompiledResult[]>([]);
    const [visualizationFilter, setVisualizationFilter] = useState<VisualizationFilter>();
    const [benchmarks, setBenchmarks] = useState<string[]>([]);
    const [experiments, setExperiments] = useState<Experiment[]>([]);
    const [modelFamilies, setModelFamilies] = useState<string[]>([]);
    const [models, setModels] = useState<Model[]>([]);

    React.useEffect(() => {  
        fetch('/config.json')
            .then(response => response.json())
            .then(fetchedData => 
                {
                    setBenchmarks(fetchedData.benchmarks);
                    setExperiments(fetchedData.experiments);
                    setModelFamilies(fetchedData.model_list.map((d: ModelFamily) => d.model_family));
                    setModels(fetchedData.model_list.map((d: ModelFamily) => d.models.map((model: string) => ({ model, model_family: d.model_family }))).flat());
                })
            .catch(error => console.error(error));
    }, []);

    React.useEffect(() => {  
        fetch('/compiled_results.json')
            .then(response => response.json())
            .then(fetchedData => 
                {
                    for (const key in visualizationFilter) {
                        if (visualizationFilter[key]) {
                            fetchedData = fetchedData.filter((d: CompiledResult) => visualizationFilter[key].some(x => x === d[key]));
                        }
                    }
                    setFilteredData(fetchedData);
                })
            .catch(error => console.error(error));
    }, [visualizationFilter]);

    const [modelFamilyChecked, setModelFamilyChecked] = useState<string[]>(modelFamilies);

    const checkAll = modelFamilies.length === modelFamilyChecked.length;
    const indeterminate = modelFamilyChecked.length > 0 && modelFamilyChecked.length < modelFamilies.length;


    // const experimentFilterOnChange = (list: string[]) => {  
    //     setModelFamilyChecked(list);  
    //     console.log(list);  
    //     const updatedFilter = { ...visualizationFilter, model_family: list };  
    //     setVisualizationFilter(updatedFilter);  
    // }; 
    const visualizationFilterOnChange = (list: string[], filterKey: string) => {
        const updatedFilter = { ...visualizationFilter, [filterKey]: list };
        setVisualizationFilter(updatedFilter);  
    };  

    const onCheckAllChange = (e, filterKey, defaultList) => {
        const list = e.target.checked ? defaultList : [];
        const updatedFilter = { ...visualizationFilter, [filterKey]: list };
        setVisualizationFilter(updatedFilter);  
    };

    return (
        <div>
            <div>
                <ControlRowView
                    title="Benchmark"
                    className="mt-4 mb-2"
                    description="Filter by Benchmark"
                    value = ""
                    control={
                      <div>
                        <Checkbox.Group 
                            options={benchmarks} 
                            value={visualizationFilter?.benchmark} 
                            onChange={(list) => visualizationFilterOnChange(list, 'benchmark')}  />
                        <Divider />
                      </div>
                    }
                />
                <ControlRowView
                    title="Experiment"
                    className="mt-4 mb-2"
                    description="Filter by Experiments"
                    value = ""
                    control={
                      <div>
                        <Checkbox 
                            indeterminate={indeterminate} 
                            onChange={(e) => onCheckAllChange(e, 'experiment', experiments.map((d: Experiment) => d.experiment))} 
                            checked={visualizationFilter?.experiment?.length === experiments.length}>
                          Check all
                        </Checkbox>
                        <Checkbox.Group 
                            options={experiments.map((d: Experiment) => d.experiment)} 
                            value={visualizationFilter?.experiment} 
                            onChange={(list) => visualizationFilterOnChange(list, 'experiment')}  />
                        <Divider />
                      </div>
                    }
                />
                <ControlRowView
                    title="Model Family"
                    className="mt-4 mb-2"
                    description="Filter by Model Families"
                    value = ""
                    control={
                      <div>
                        <Checkbox 
                            indeterminate={indeterminate} 
                            onChange={(e) => onCheckAllChange(e, 'model_family', modelFamilies)} 
                            checked={visualizationFilter?.model_family?.length === modelFamilies.length}>
                          Check all
                        </Checkbox>
                        <Checkbox.Group 
                            options={modelFamilies} 
                            value={visualizationFilter?.model_family} 
                            onChange={(list) => visualizationFilterOnChange(list, 'model_family')}  />
                        <Divider />
                      </div>
                    }
                />
                <ControlRowView
                    title="Model"
                    className="mt-4 mb-2"
                    description="Filter by Model"
                    value = ""
                    control={
                      <div>
                        <Checkbox 
                            indeterminate={indeterminate} 
                            onChange={(e) => onCheckAllChange(e, 'model', models.map((d: Model) => d.model))} 
                            checked={visualizationFilter?.model?.length === models.length}>
                          Check all
                        </Checkbox>
                        <Checkbox.Group 
                            options={models.map((d: Model) => d.model)} 
                            value={visualizationFilter?.model} 
                            onChange={(list) => visualizationFilterOnChange(list, 'model')}  />
                        <Divider />
                      </div>
                    }
                />
            </div>
            <BarChart  
                width={500}  
                height={300}  
                data={filteredData}  
                margin={{  
                    top: 5, right: 30, left: 20, bottom: 5,  
                }}  
            >  
                <CartesianGrid strokeDasharray="3 3" />  
                <XAxis dataKey="model" />  
                <YAxis />  
                <Tooltip />  
                <Legend />  
                <Bar dataKey="value" fill="#8884d8" />  
            </BarChart>
        </div>
    );
}

interface BarChartProps {  
    data: { name: string; value: number }[];
    width: number;  
    height: number;  
}  
  

// const BarChart: React.FC<BarChartProps> = ({ data, width, height }) => {  
//     const ref = React.useRef<SVGSVGElement>(null);  
  
//     React.useEffect(() => {  
//         const svg = d3.select(ref.current);  
//         svg.selectAll("*").remove(); // Clear svg before re-rendering
  
//         let margin = {top: 20, right: 20, bottom: 60, left: 40}, // increase bottom margin to create space for labels  
//         width = 960 - margin.left - margin.right,  
//         height = 500 - margin.top - margin.bottom;  

//         // Create scale  
//         const xScale = d3.scaleBand().range([0, width]).padding(0.4).domain(data.map(d => d.name));  
//         const yScale = d3.scaleLinear().range([height, 0]).domain([0, d3.max(data, d => d.value) || 0]);  
    
//         d3.select("body").append("svg")  
//             .attr("width", width + margin.left + margin.right)  
//             .attr("height", height + margin.top + margin.bottom) // increased height  
//             .append("g")  
//             .attr("transform", "translate(" + margin.left + "," + margin.top + ")");  
        
//         // Your code to draw bars goes here  
//         // Draw bars  
//         svg.append('g')  
//             .attr('transform', `translate(0, ${height})`)  
//             .call(d3.axisBottom(xScale));  
//         svg.append('g')  
//             .call(d3.axisLeft(yScale).tickFormat(d3.format('.2s')).ticks(5));  
//         const bars = svg.selectAll('.bar')  
//             .data(data)  
//             .enter().append('rect')  
//             .attr('class', 'bar')  
//             .attr('x', d => xScale(d.name) || 0)  
//             .attr('width', xScale.bandwidth())  
//             .attr('y', d => yScale(d.value))  
//             .attr('height', d => height - yScale(d.value));  
//         // Add labels  
//         svg.selectAll('.label')  
//             .data(data)  
//             .enter().append('text')  
//             .attr('class', 'label')  
//             .attr('x', (d, i) => xScale(i) + xScale.bandwidth() / 2)  
//             .attr('y', d => yScale(d) + 3) // adjust this to change the position of the labels  
//             .attr('dy', '-.7em') // adjust this to change the position of the labels  
//             .attr('text-anchor', 'middle')  
//             .text(d => d.name);  
//     }, [data, width, height]);  
  
//     return <svg ref={ref} width={width} height={height} />;  
// };  

export default App;