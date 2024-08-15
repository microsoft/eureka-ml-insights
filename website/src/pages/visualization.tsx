import React, { useState } from 'react';
import { Capability, Model, ModelConfig, VisualizationConfig } from '../components/types';
import Highcharts, { SeriesOptionsType } from 'highcharts';
import HighchartsReact from 'highcharts-react-official';
import HC_more from 'highcharts/highcharts-more';
import { Button, Card, Col, Row } from 'antd';
import { ChartBarIcon, NewspaperIcon, PhotoIcon, ShieldCheckIcon } from "@heroicons/react/24/outline";

HC_more(Highcharts); 

const Visualization = ({config}: {config: VisualizationConfig}) => {
    if (!config) {  
        // config is still null, probably still fetching data
        return <div>Loading...</div>;
    }

    const [languageCapabilties, setLanguageCapabilties] = useState<Capability[]>([]);
    const [langOverallSeries, setLangOverallSeries] = useState<SeriesOptionsType[]>([]);
    const [multimodalCapabilties, setMultimodalCapabilties] = useState<Capability[]>([]);
    const [multimodalOverallSeries, setMultimodalOverallSeries] = useState<SeriesOptionsType[]>([]);

    const parseResultCategory = (capabilities: Capability[], setCapabilityFunction, setSeriesFunction) => {
        setCapabilityFunction(capabilities);
        const modelScores = {};
        capabilities.forEach((d: Capability) => {
            d.models.forEach((model: Model) => {
                if (!modelScores[model.name]) {
                    modelScores[model.name] = [];
                }
                modelScores[model.name].push(model.score); 
            });
        });
        const langSeries = [];
        for (const key in modelScores) {
            langSeries.push({
                name: key,
                data: modelScores[key],
                pointPlacement: 'on',
                type: 'line',
                color: config.models.find((d: ModelConfig) => d.model === key).color,
            });
        }
        setSeriesFunction(langSeries);
    };

    React.useEffect(() => {
            fetch('/compiled_results.json')
                .then(response => response.json())
                .then(compiledResults => {
                    parseResultCategory(compiledResults.language.capabilities, setLanguageCapabilties, setLangOverallSeries);
                    parseResultCategory(compiledResults.multimodal.capabilities, setMultimodalCapabilties, setMultimodalOverallSeries);
                });
            }, []);

    const languageChartOptions: Highcharts.Options = {
        title: {
            text: 'Language Task Performance',
        },
        chart: {
            polar: true,
            type: 'area'
        },
        series: langOverallSeries,
        xAxis: {
            categories: languageCapabilties.map(d => d.name),
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
    
    const multimodalChartOptions: Highcharts.Options = {
        title: {
            text: 'Multi-Modal Task Performance',
        },
        chart: {
            polar: true,
            type: 'area'
        },
        series: multimodalOverallSeries,
        xAxis: {
            categories: multimodalCapabilties.map(d => d.name),
            tickmarkPlacement: 'on',
//             labels: {  
//                 formatter: function () {  
//                   return `<div class="tooltip">${this.value}  
//   <div class="tooltiptext">Tooltip text</div>
// </div>  `;  
//                 },  
//                 useHTML: true  
//               }  
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
                <Card bodyStyle={{ padding: 0 }}>  
                    <Row align='middle'> 
                        <Col span={3}>  
                            <p style={{ fontWeight: 'bold', fontSize: '20px' }}>{config.model_families.length}</p>  
                            <p style={{ color: 'grey', fontSize: '14px' }}>Model Families</p>
                        </Col>  
                        <Col span={3}>
                            <p style={{ fontWeight: 'bold', fontSize: '20px' }}>{config.models.length}</p>  
                            <p style={{ color: 'grey', fontSize: '14px' }}>Models</p>
                        </Col>  
                        <Col span={3}>  
                            <p style={{ fontWeight: 'bold', fontSize: '20px' }}>{config.benchmarks.length}</p>  
                            <p style={{ color: 'grey', fontSize: '14px' }}>Benchmarks</p>
                        </Col>  
                        <Col span={3}>  
                            <div style={{fontSize: '20px'}}><NewspaperIcon className='inline-block mr-1' width={40} height={40}/></div>
                            <p style={{ color: 'grey', fontSize: '14px' }}>Language tasks</p>
                        </Col>  
                        <Col span={3}>  
                            <div style={{fontSize: '20px'}}><PhotoIcon className='inline-block mr-1' width={40} height={40}/></div>
                            <p style={{ color: 'grey', fontSize: '14px' }}>Multimodal tasks</p>
                        </Col>  
                        <Col span={3}>  
                            <Button type='primary'><ChartBarIcon width={20} height={20}/>AI Quality</Button>
                        </Col>  
                        <Col>  
                            <Button type='default'><ShieldCheckIcon width={20} height={20}/>AI Safety</Button>
                        </Col>  
                    </Row>  
                </Card>
                <HighchartsReact highcharts={Highcharts} options={languageChartOptions} />
                <HighchartsReact highcharts={Highcharts} options={multimodalChartOptions} />
            </div>
        </div>
    );
}

export default Visualization;