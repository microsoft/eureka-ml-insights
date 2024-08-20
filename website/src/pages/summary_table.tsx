import React, { Children, useState } from 'react';
import { Capability, Model, ModelConfig, VisualizationConfig } from '../components/types';
import Highcharts, { SeriesOptionsType } from 'highcharts';
import { Layout, Table, CollapseProps, Collapse } from 'antd';
import Title from 'antd/es/skeleton/Title';
import { ColumnsType } from 'antd/es/table';

const SummaryTable = ({config}: {config: VisualizationConfig}) => {
    if (!config) {  
        // config is still null, probably still fetching data
        return <div>Loading...</div>;
    }

    const [languageCapabilties, setLanguageCapabilties] = useState<ColumnsType<never>>([]);
    const [langOverallSeries, setLangOverallSeries] = useState<[]>([]);
    const [multimodalCapabilties, setMultimodalCapabilties] = useState<ColumnsType<never>>([]);
    const [multimodalOverallSeries, setMultimodalOverallSeries] = useState<[]>([]);

    const parseResultCategory = (capabilities: Capability[], setCapabilityFunction, setSeriesFunction) => {
        const width = 100 / (capabilities.length + 3);
        const temp = capabilities.map((d: Capability) => {  
            return {
                title: d.name, 
                dataIndex: d.name, 
                key: d.name, 
                width: `${width}%`,
                sorter: (a, b) => a[d.name] - b[d.name],
                defaultSortOrder: 'descend',
            };  
        });
        
        const modelScores = {};
        capabilities.forEach((d: Capability) => {
            d.models.forEach((model: Model) => {
                if (!modelScores[model.name]) {
                    modelScores[model.name] = [];
                }
                modelScores[model.name].push(model.score); 
            });
        });
        let filters = Object.keys(modelScores).map(model => ({text: model, value: model}));
        let modelColumn = {
            title: 'model', 
            dataIndex: 'model', 
            key: 'model', 
            width: `${width}%`, 
            sorter: (a, b) => a.model.localeCompare(b.model),
            onFilter: (value, record) => record.model.indexOf(value as string) === 0,
            filters: filters,
        };
        setCapabilityFunction([modelColumn, ...temp]);
        const tableRows = [];
        for (const key in modelScores) {
            if (key === 'Llava-1_6-34B') {
                console.log(modelScores[key]);
            }
            const tableRow = {};
            tableRow['model'] = key;
            for (let i = 0; i < modelScores[key].length; i++) {
                tableRow[capabilities[i].name] = modelScores[key][i];
            }
            tableRows.push(tableRow);
        }
        setSeriesFunction(tableRows);
    };

    React.useEffect(() => {
        fetch('/compiled_results.json')
            .then(response => response.json())
            .then(compiledResults => {
                parseResultCategory(compiledResults.language.capabilities, setLanguageCapabilties, setLangOverallSeries);
                parseResultCategory(compiledResults.multimodal.capabilities, setMultimodalCapabilties, setMultimodalOverallSeries);
            });
        }, []);
    
    const items: CollapseProps['items'] = [
        {
            key: '1',
            label: 'Language Task Performance',
            children: <Table columns={languageCapabilties} dataSource={langOverallSeries} style={{ width: '100%' }}/>
        },
        {
            key: '2',
            label: 'Multimodal Task Performance',
            children: <Table columns={multimodalCapabilties} dataSource={multimodalOverallSeries} style={{ width: '100%' }}/>
        }
        
    ]

    return (
        <div style={{ width: '100%' }}>
            <Collapse items={items} defaultActiveKey={['1', '2']} style={{ width: '100%' }}/>;
        </div>
    )
};

export default SummaryTable;