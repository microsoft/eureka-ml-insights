import React, { useState } from 'react';
import { CapabilityScores, ModelScore, ModelConfig, EurekaConfig } from '../components/types';
import Highcharts, { SeriesOptionsType } from 'highcharts';
import HighchartsReact from 'highcharts-react-official';
import { Button, Col, Row } from 'antd';
import styles from './homepage_header.module.css';
import Heading from '@theme/Heading';
import { useHistory } from '@docusaurus/router';
import Link from '@docusaurus/Link';

const OverallVisualization = ({config}: {config: EurekaConfig}) => {
    if (!config) {  
        // config is still null, probably still fetching data
        return <div>Loading...</div>;
    }

    const [isLoading, setIsLoading] = useState(true);  
    const [highchartsLoading, setHighchartsLoading] = useState(true);  

    React.useEffect(() => {  
        const loadHighchartsMore = async () => {  
          const HC_more = await import('highcharts/highcharts-more');  
          HC_more.default(Highcharts);  
          setHighchartsLoading(false);
        };  
        loadHighchartsMore();
      }, []); 

    const [languageCapabilties, setLanguageCapabilties] = useState<CapabilityScores[]>([]);
    const [langOverallSeries, setLangOverallSeries] = useState<SeriesOptionsType[]>([]);
    const [multimodalCapabilties, setMultimodalCapabilties] = useState<CapabilityScores[]>([]);
    const [multimodalOverallSeries, setMultimodalOverallSeries] = useState<SeriesOptionsType[]>([]);

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

    if (isLoading || highchartsLoading) {  
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
                style: {
                    fontSize: '1em'
                }
            }
        },
        yAxis: {
            gridLineInterpolation: 'polygon',
            lineWidth: 0,  
            min: 0,
            max: 100,
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
                style: {
                    fontSize: '1em',
                }
            }
        },
        yAxis: {
            gridLineInterpolation: 'polygon',
            lineWidth: 0,  
            min: 0,
            max: 100,
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

    const history = useHistory();  
    const navigateToBenchmarks = (modality) => {  
        history.push(`/eureka-ml-insights/${modality}`);  
    }  

    return (
        <div style={{width: '100%', paddingBottom: '4em'}}>
            <Heading as="h4" className="hero__title" style={{textAlign: "center", fontSize: '2.5em'}}>
                Overall Performance 
            </Heading>
            <br/>
            <div style={{width: '100%'}}>
                <Row justify="space-between" style={{display: 'flex', justifyContent: 'center'}}>
                    <Col xs={24} md={12} style={{ minWidth: '40em'}}>
                        <div style={{ width: '90%', margin: '0 auto'}}>
                            <HighchartsReact highcharts={Highcharts} options={languageChartOptions}/>
                        </div>
                        <div style={{display: 'flex', justifyContent: 'center'}}>
                            <Button shape='round' className={`${styles.buttons}`} style={{outline: "black"}}>
                                <Link to="/eureka-ml-insights/Language"><strong>Explore Language Results</strong></Link>
                            </Button>
                        </div>
                    </Col>
                    <Col xs={24} md={12} style={{ minWidth: '40em'}}>
                        <div style={{ width: '90%', margin: '0 auto'}}>
                            <HighchartsReact highcharts={Highcharts} options={multimodalChartOptions}/>
                        </div>
                        <div style={{display: 'flex', justifyContent: 'center'}}>
                            <Button shape='round' className={`${styles.buttons}`} style={{outline: "black"}}>
                                <Link to="/eureka-ml-insights/Multimodal"><strong>Explore Multimodal Results</strong></Link>
                            </Button>
                        </div>
                    </Col>
                </Row>
            </div>
        </div>
    );
}

export default OverallVisualization; 