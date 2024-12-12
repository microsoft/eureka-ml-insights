import React, { useState } from "react";
import { ModelScore, EurekaConfig, CapabilityScores, ModelConfig } from "../types";
import Highcharts from "highcharts";
import HighchartsReact from "highcharts-react-official";
import Exporting from "highcharts/modules/exporting";  
import ExportData from "highcharts/modules/export-data";  

const OverallBenchmarkChart = ({benchmark, config}: {benchmark: string, config: EurekaConfig}) => {
    const [isLoading, setIsLoading] = useState(true);  
    const [highchartsLoading, setHighchartsLoading] = useState(true);  
    const [chartOptions, setChartOptions] = useState<any[]>([]);

    React.useEffect(() => {  
        const loadHighchartsMore = async () => {  
          const HC_more = await import('highcharts/highcharts-more');  
          HC_more.default(Highcharts);  
          setHighchartsLoading(false);
          Exporting(Highcharts);  
          ExportData(Highcharts);  
        };  
        loadHighchartsMore();
      }, []); 

    React.useEffect(() => {
        if (!benchmark) return;  // Ensure benchmark is not null  
        if (!config) return;  // Ensure config is not null
  
        setIsLoading(true);  // Reset loading state  

        fetch('compiled_results.json')
        .then(response => response.json())
        .then(compiledResults => {
            const modality = config.benchmarks.find((d) => d.name === benchmark).modality;
            const capabilities = config.capability_mapping.filter((d) => d.benchmark === benchmark).map((d) => d.capability);

            let capabilityScores = [];
            if (modality === 'language') {
                capabilityScores = compiledResults.language.capabilities.filter((d) => capabilities.includes(d.name));
            }
            else {
                capabilityScores = compiledResults.multimodal.capabilities.filter((d) => capabilities.includes(d.name));
            }

            const chartOptionsArray = [];
            capabilityScores.forEach((d: CapabilityScores) => {
                const series = [];
                const categories = [];
                d.models.forEach((model: ModelScore) => {
                    series.push({
                        name: model.name,
                        type: 'column',
                        data: [model.score],
                        color: config?.models?.find((d: ModelConfig) => d.model === model.name)?.color || 'black',
                    });
                    categories.push(model.name);
                });
                const tempChartOptions = {  
                    chart: {  
                        type: 'column'
                    },  
                    title: {  
                        text: d.name,
                    },  
                    xAxis: {  
                        categories: [d.name + " Score"]
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
            });
            setChartOptions(chartOptionsArray);
            setIsLoading(false);
        })
        .catch(error => console.error('Error fetching compiled results:', error));
    }, [benchmark, config]); 

    if (isLoading || highchartsLoading) {  
        return <div>Loading...</div>;  
    }

    return (
        <div style={{width: '100%', paddingBottom: '4em', display:'flex', flexWrap: 'wrap', justifyContent:'center'}}>
            {chartOptions.map((options, index) => (  
                <div key={index} style={{ flex: '1 1 0', minWidth: '300px', padding: '1em' }}> 
                    <HighchartsReact key={index} highcharts={Highcharts} options={options} />  
                </div>
            ))} 
        </div>
    )
}

export default OverallBenchmarkChart;