import React, { useState } from 'react';
import { CapabilityScores, ModelScore, ModelConfig, EurekaConfig } from '../components/types';
import Highcharts, { SeriesOptionsType } from 'highcharts';
import HighchartsReact from 'highcharts-react-official';
import { Col, Row } from 'antd';
import Heading from '@theme/Heading';

const OverallVisualization = ({config}: {config: EurekaConfig}) => {
    if (!config) {  
        // config is still null, probably still fetching data
        return <div>Loading...</div>;
    }

    React.useEffect(() => {  
        const loadHighchartsMore = async () => {  
          const HC_more = await import('highcharts/highcharts-more');  
          HC_more.default(Highcharts);  
        };  
        loadHighchartsMore();  
      }, []); 

    const [languageCapabilties, setLanguageCapabilties] = useState<CapabilityScores[]>([]);
    const [langOverallSeries, setLangOverallSeries] = useState<SeriesOptionsType[]>([]);
    const [multimodalCapabilties, setMultimodalCapabilties] = useState<CapabilityScores[]>([]);
    const [multimodalOverallSeries, setMultimodalOverallSeries] = useState<SeriesOptionsType[]>([]);
    const [isLoading, setIsLoading] = useState(true);  

    const parseResultCategory = (capabilities: CapabilityScores[], setCapabilityFunction, setSeriesFunction) => {
        setCapabilityFunction(capabilities);
        const modelScores = {};
        capabilities.forEach((d: CapabilityScores) => {
            d.models.forEach((model: ModelScore) => {
                if (!modelScores[model.name]) {
                    modelScores[model.name] = [];
                }
                modelScores[model.name].push(model.score); 
            });
        });
        const series = [];
        for (const key in modelScores) {
            series.push({
                name: key,
                data: modelScores[key],
                pointPlacement: 'on',
                type: 'line',
                color: config?.models?.find((d: ModelConfig) => d.model === key)?.color || 'black',
            });
        }
        setSeriesFunction(series);
    };

    React.useEffect(() => {
            fetch('compiled_results.json')
                .then(response => response.json())
                .then(compiledResults => {
                    parseResultCategory(compiledResults.language.capabilities, setLanguageCapabilties, setLangOverallSeries);
                    parseResultCategory(compiledResults.multimodal.capabilities, setMultimodalCapabilties, setMultimodalOverallSeries);
                    setIsLoading(false);
                })
                .catch(error => console.error('Error fetching compiled results:', error));
            }, []);

    if (isLoading) {  
        return <div>Loading...</div>;  
    }  
    const languageChartOptions: Highcharts.Options = {
        title: {
            text: 'Language Performance',
            style: {
                fontSize: '1.7em',
                paddingbottom: '2em',
            },
        },
        chart: {
            polar: true,
            type: 'area',
            height: '100%'
        },
        series: langOverallSeries,
        xAxis: {
            categories: languageCapabilties?.map(d => d.name),
            tickmarkPlacement: 'on',
            labels: {
                padding: 20,
                style: {
                    fontSize: '1em'
                }
            }
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
            layout: 'horizontal',
            itemMarginTop: 10,
        },
        credits: {
            enabled: false
        }
    };
    
    const multimodalChartOptions: Highcharts.Options = {
        title: {
            text: 'Multimodal Performance',
            style: {
                fontSize: '1.7em',
                paddingbottom: '2em'
            }
        },
        chart: {
            polar: true,
            type: 'area',
            height: '100%'
        },
        series: multimodalOverallSeries,
        xAxis: {
            categories: multimodalCapabilties?.map(d => d.name),
            tickmarkPlacement: 'on',
            labels: {
                padding: 20,
                style: {
                    fontSize: '1em',
                }
            }
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
            layout: 'horizontal',
            itemMarginTop: 10,
        },
        credits: {
            enabled: false
        }
    };

    return (
        <div style={{width: '100%', paddingBottom: '4em'}}>
            <Heading as="h4" className="hero__title" style={{textAlign: "center", fontSize: '2.5em'}}>
                Overall Performance 
            </Heading>
            <br/>
            <div style={{width: '100%'}}>
                <Row justify="space-between">
                    <Col xs={24} md={12}>
                        <HighchartsReact highcharts={Highcharts} options={languageChartOptions} />
                    </Col>
                    <Col xs={24} md={12}>
                        <HighchartsReact highcharts={Highcharts} options={multimodalChartOptions} />
                    </Col>
                </Row>
            </div>
        </div>
    );
}

export default OverallVisualization; 