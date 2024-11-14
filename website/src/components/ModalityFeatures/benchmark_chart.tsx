import React, { useState } from "react";
import { BenchmarkGraph, ModelScore, ModelConfig, EurekaConfig, Benchmark, BenchmarkResult, BenchmarkExperiment } from "../types";
import Highcharts from "highcharts";
import HighchartsReact from "highcharts-react-official";
import { Col, Row } from "antd";

const BenchmarkChart = ({benchmark, experiment, config}: {benchmark: string, experiment: string, config: EurekaConfig}) => {
    const [isLoading, setIsLoading] = useState(true);  
    const [highchartsLoading, setHighchartsLoading] = useState(true);  
    const [chartOptions, setChartOptions] = useState<any[]>([]);

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
        if (!experiment) return;  // Ensure experiment is not null
        if (!config) return;  // Ensure config is not null
  
        setIsLoading(true);  // Reset loading state  

        fetch('benchmark_results.json')
        .then(response => response.json())
        .then(benchmarkResults => {
            const selectedExperiment = benchmarkResults[benchmark]["experiments"].find((d: BenchmarkExperiment) => d.title === experiment);
            const chartOptionsArray = [];
            console.log(selectedExperiment);
            for (const i in selectedExperiment.series) {
                const series = [];
                const selectedSeries = selectedExperiment.series[i];
                console.log(selectedSeries);
                for (const j in selectedSeries.values) {
                    series.push({
                        name: selectedSeries.values[j].name,
                        type: 'column',
                        data: selectedSeries.values[j].scores,
                        color: config?.models?.find((d: ModelConfig) => d.model === selectedSeries.values[j].name)?.color || 'black',
                    });
                }

                const tempChartOptions = {  
                    chart: {  
                        type: 'column'
                    },  
                    title: {  
                        text: selectedSeries.title,
                    },  
                    xAxis: {  
                        title: {  
                            text: "Model"  
                        },
                        categories: selectedExperiment.categories
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
                chartOptionsArray.push(tempChartOptions);
            }

            setChartOptions(chartOptionsArray);
            setIsLoading(false);
        })
        .catch(error => console.error('Error fetching compiled results:', error));
    }, [benchmark, experiment, config]); 

    if (isLoading || highchartsLoading) {  
        return <div>Loading...</div>;  
    }

    return (
        <div style={{width: '100%', paddingBottom: '4em'}}>
            {chartOptions.map((options, index) => (  
                <HighchartsReact key={index} highcharts={Highcharts} options={options} />  
            ))} 
        </div>
    )
}

export default BenchmarkChart;