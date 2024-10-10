import config from "@generated/docusaurus.config";
import React, { useState } from "react";
import { BenchmarkGraph, ModelScore, ModelConfig, EurekaConfig, Benchmark } from "../types";
import Highcharts from "highcharts";
import HighchartsReact from "highcharts-react-official";
import { Col, Row } from "antd";
import Heading from '@theme/Heading';

const BenchmarkChart = ({benchmark, config}: {benchmark: string, config: EurekaConfig}) => {
    const [isLoading, setIsLoading] = useState(true);  
    const [highchartsLoading, setHighchartsLoading] = useState(true);  
    const [benchmarkGraphSeries, setBenchmarkGraphSeries] = useState<[]>([]);
    const [graphTitles, setGraphTitles] = useState<[]>([]);
    const [benchmarkDescription, setBenchmarkDescription] = useState("");

    React.useEffect(() => {  
        const loadHighchartsMore = async () => {  
          const HC_more = await import('highcharts/highcharts-more');  
          HC_more.default(Highcharts);  
          setHighchartsLoading(false);
        };  
        loadHighchartsMore();
      }, []); 

    React.useEffect(() => {
        if (!benchmark) return;  // Ensure benchmark is not null  
  
        setIsLoading(true);  // Reset loading state  
        setBenchmarkGraphSeries([]);  // Clear previous data  
        setGraphTitles([]);

        fetch('benchmark_results.json')
        .then(response => response.json())
        .then(benchmarkResults => {
            const graphs = benchmarkResults[benchmark]["graphs"];
            const allGraphSeries = [];
            const titles = [];
            graphs.forEach((d: BenchmarkGraph) => {
                const modelScores = {};
                titles.push(d.title);
                d.models.forEach((model: ModelScore) => {
                    if (!modelScores[model.name]) {
                        modelScores[model.name] = [];
                    }
                    modelScores[model.name].push(model.score); 
                });
                const series = [];
                for (const key in modelScores) {
                    series.push({
                        name: key,
                        data: modelScores[key],
                        color: config?.models?.find((d: ModelConfig) => d.model === key)?.color || 'black',
                    });
                }
                console.log(series);
                allGraphSeries.push(series);
            });
            
            setBenchmarkGraphSeries(allGraphSeries);
            setGraphTitles(titles)
            setIsLoading(false);
        })
        .catch(error => console.error('Error fetching compiled results:', error));
    }, [benchmark, config]); 

    React.useEffect(() => {
        if (!config || !benchmark) return; 
        const matchingBenchmark = config.benchmarks.find((b: Benchmark) => b.name === benchmark); 
        setBenchmarkDescription(matchingBenchmark.description);
    }, [benchmark, config]); 

    if (isLoading || highchartsLoading) {  
        return <div>Loading...</div>;  
    }

    return (
        <div style={{width: '100%', paddingBottom: '4em'}}>
            <Heading as="h4" className="hero__title" style={{textAlign: "center", fontSize: '2.5em'}}>
                Benchmark Breakdown
            </Heading>
            <Heading as="h5">
                {benchmarkDescription}
            </Heading>
            <br/>
            <div style={{width: '100%'}}>
                <Row justify="space-between" style={{display: 'flex', justifyContent: 'center'}}>
                    {benchmarkGraphSeries.map((series, index) => {  
                        const chartOptions = {  
                            chart: {  
                                type: 'column'  
                            },  
                            title: {  
                                text: graphTitles[index],
                            },  
                            xAxis: {  
                                title: {  
                                    text: "Model"  
                                }  
                            },  
                            yAxis: {  
                                min: 0,  
                                title: {  
                                    text: 'Score',  
                                },  
                                labels: {  
                                    overflow: 'justify'  
                                }  
                            },
                            series: series,
                            credits: {
                                enabled: false
                            }  
                        };  
    
                        return (  
                            <Col xs={24} md={12} style={{ minWidth: '40em' }} key={index}>  
                                <div style={{ width: '90%', margin: '0 auto' }}>  
                                    <HighchartsReact highcharts={Highcharts} options={chartOptions} />  
                                </div>  
                            </Col>  
                        );  
                    })} 
                </Row>
            </div>
        </div>
    )
}

export default BenchmarkChart;