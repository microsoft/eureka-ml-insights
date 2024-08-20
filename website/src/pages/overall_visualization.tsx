import React, { useState } from 'react';
import { Capability, ModelScore, ModelConfig, Config } from '../components/types';
import Highcharts, { SeriesOptionsType } from 'highcharts';
import HighchartsReact from 'highcharts-react-official';
import HC_more from 'highcharts/highcharts-more';
import { Button, Card, Col, Row } from 'antd';
import Heading from '@theme/Heading';

HC_more(Highcharts); 

const OverallVisualization = ({config}: {config: Config}) => {
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
            d.models.forEach((model: ModelScore) => {
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
            <Heading as="h1" className="hero__title" style={{textAlign: "center"}}>
                Overall Performance 
            </Heading>
            <div>
                <Row>
                    <Col>
                        <HighchartsReact highcharts={Highcharts} options={languageChartOptions} />
                    </Col>
                    <Col>
                        <HighchartsReact highcharts={Highcharts} options={multimodalChartOptions} />
                    </Col>
                </Row>
            </div>
        </div>
    );
}

export default OverallVisualization;