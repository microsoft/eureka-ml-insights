import React from "react";
import { DownOutlined } from '@ant-design/icons';
import { BenchmarkGraph, EurekaConfig, GraphDetails } from "../types";
import { Button, Dropdown, Menu, MenuProps, message, Row, Space } from "antd";
import BenchmarkChart from "./benchmark_chart";

const BenchmarkDetails = ({benchmark, config}: {benchmark: string, config: EurekaConfig}) => {
    const [benchmarkDescription, setBenchmarkDescription] = React.useState<string>('');
    const [subcategories, setSubcategories] = React.useState<GraphDetails[]>([]);
    const [selectedSubcategory, setSelectedSubcategory] = React.useState(null);
    const [selectedSubcategoryDescription, setSelectedSubcategoryDescription] = React.useState<string>('');

    React.useEffect(() => {
        if (!benchmark) return;  // Ensure benchmark is not null  

        setBenchmarkDescription(config.benchmarks.find((d) => d.name === benchmark).description);
        const graphs = config.benchmarks.find((d) => d.name === benchmark).graphs;
        setSubcategories(graphs);
        setSelectedSubcategory(graphs[0].title);
        setSelectedSubcategoryDescription(graphs[0].description);

    }, [benchmark, config]); 

    const onClick: MenuProps['onClick'] = ({ key }) => {
        setSelectedSubcategory(key);
        setSelectedSubcategoryDescription(subcategories.find((d) => d.title === key).description);
    }; 

    const menuItems: MenuProps['items'] = subcategories.map(subcategory => (  
        {  
            key: subcategory.title,  
            label: subcategory.title,  
        }  
    ));  
    

    return (
        <div style={{width: '100%', paddingBottom: '4em'}}>
            <h2>{benchmark}</h2>
            <h4>{benchmarkDescription}</h4>
            
            <div style={{display: 'flex'}}>
                <span style={{marginRight: '0.5em'}}>Select Subcategory:</span>
                <Dropdown menu={{items: menuItems, onClick: onClick}} >  
                    <span>
                        {selectedSubcategory}
                        <DownOutlined />
                    </span>
                </Dropdown>
            </div>
            <br/>
            <div>
                <span>{selectedSubcategoryDescription}</span>
            </div>
            <br/>
            <div style={{width: '100%'}}>
                <Row justify="space-between" style={{display: 'flex', justifyContent: 'center'}}>
                    <BenchmarkChart benchmark={benchmark} subcategory={selectedSubcategory} config={config}></BenchmarkChart>
                </Row>
            </div>
        </div>
    )
}

export default BenchmarkDetails;