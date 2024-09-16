import React, { useState } from 'react';
import { CapabilityScores, ModelScore, EurekaConfig } from '../components/types';
import { Table } from 'antd';
import { ColumnsType } from 'antd/es/table';

const SummaryTable = ({config}: {config: EurekaConfig}) => {
    if (!config) {  
        // config is still null, probably still fetching data
        return <div>Loading...</div>;
    }

    const [languageCapabilties, setLanguageCapabilties] = useState<ColumnsType<never>>([]);
    const [langOverallSeries, setLangOverallSeries] = useState<[]>([]);
    const [multimodalCapabilties, setMultimodalCapabilties] = useState<ColumnsType<never>>([]);
    const [multimodalOverallSeries, setMultimodalOverallSeries] = useState<[]>([]);

    const parseResultCategory = (capabilities: CapabilityScores[], setCapabilityFunction, setSeriesFunction) => {
        const width = 100 / (capabilities.length + 3);
        const temp = capabilities.map((d: CapabilityScores) => {  
            return {
                title: d.name, 
                dataIndex: d.name, 
                key: d.name, 
                width: `${width}%`,
                sorter: (a, b) => a[d.name] - b[d.name],
                defaultSortOrder: 'descend',
                render: (text, record) => text.toFixed(1),  
            };  
        });
        
        const modelScores = {};
        capabilities.forEach((d: CapabilityScores) => {
            d.models.forEach((model: ModelScore) => {
                if (!modelScores[model.name]) {
                    modelScores[model.name] = [];
                }
                modelScores[model.name].push(Number(model.score.toFixed(1))); 
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
        fetch('compiled_results.json')
            .then(response => response.json())
            .then(compiledResults => {
                parseResultCategory(compiledResults.language.capabilities, setLanguageCapabilties, setLangOverallSeries);
                parseResultCategory(compiledResults.multimodal.capabilities, setMultimodalCapabilties, setMultimodalOverallSeries);
            });
        }, []);

    return (
        <div style={{ width: '100%' }}>
            <div>
                <br/>
                <h2>Language Performance</h2>
                <Table columns={languageCapabilties} dataSource={langOverallSeries} pagination={false}/>
            </div>
            <div>
                <br/>
                <h2>Multimodal Performance</h2>
                <Table columns={multimodalCapabilties} dataSource={multimodalOverallSeries} pagination={false}/>
            </div>
        </div>
    )
};

export default SummaryTable;