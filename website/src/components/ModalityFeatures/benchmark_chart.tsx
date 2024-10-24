import React, { useState } from "react";
import { BenchmarkGraph, ModelScore, ModelConfig, EurekaConfig, Benchmark } from "../types";
import Highcharts from "highcharts";
import HighchartsReact from "highcharts-react-official";
import { Col, Row } from "antd";

const BenchmarkChart = ({benchmark, subcategory, config}: {benchmark: string, subcategory: string, config: EurekaConfig}) => {
    const [isLoading, setIsLoading] = useState(true);  
    const [highchartsLoading, setHighchartsLoading] = useState(true);  
    const [chartOptions, setChartOptions] = useState<any>();

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

        fetch('benchmark_results.json')
        .then(response => response.json())
        .then(benchmarkResults => {
            const graph = benchmarkResults[benchmark]["graphs"].find((d: BenchmarkGraph) => d.title === subcategory);
            const modelScores = {};
            graph.models.forEach((model: ModelScore) => {
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
            const tempChartOptions = {  
                chart: {  
                    type: 'column'
                },  
                title: {  
                    text: graph.title,
                },  
                xAxis: {  
                    title: {  
                        text: "Model"  
                    }  
                },  
                yAxis: {  
                    min: 0,
                    max: 100,
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

            setChartOptions(tempChartOptions);
            setIsLoading(false);
        })
        .catch(error => console.error('Error fetching compiled results:', error));
    }, [benchmark, subcategory, config]); 

    if (isLoading || highchartsLoading) {  
        return <div>Loading...</div>;  
    }

    return (
        <div style={{width: '100%', paddingBottom: '4em'}}>
            <HighchartsReact highcharts={Highcharts} options={chartOptions} />  
        </div>
    )
}

export default BenchmarkChart;