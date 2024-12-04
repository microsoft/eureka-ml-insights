import React from "react";
import { DownOutlined } from '@ant-design/icons';
import { BenchmarkGraph, EurekaConfig, Experiment } from "../types";
import { Button, Dropdown, Menu, MenuProps, message, Row, Space } from "antd";
import BenchmarkChart from "./benchmark_chart";
import OverallBenchmarkChart from "./overall_benchmark_chart";

const BenchmarkDetails = ({benchmark, config}: {benchmark: string, config: EurekaConfig}) => {
    const [benchmarkDescription, setBenchmarkDescription] = React.useState<string>('');
    const [capabilityImportance, setCapabilityImportance] = React.useState<string>('');
    const [subcategories, setSubcategories] = React.useState<Experiment[]>([]);
    const [selectedSubcategory, setSelectedSubcategory] = React.useState(null);
    const [selectedSubcategoryDescription, setSelectedSubcategoryDescription] = React.useState<string>('');

    React.useEffect(() => {
        if (!benchmark) return;  // Ensure benchmark is not null  
        if (!config) return;  // Ensure config is not null

        const benchmarkObject = config.benchmarks.find((d) => d.name === benchmark);  
        setBenchmarkDescription(benchmarkObject ? benchmarkObject.benchmarkDescription : '');  
        setCapabilityImportance(benchmarkObject ? benchmarkObject.capabilityImportance : '');
        const experiments = config.benchmarks.find((d) => d.name === benchmark).experiments;
        if (experiments.length === 0) {
            return;
        }

        setSubcategories(experiments);
        setSelectedSubcategory(experiments[0].title);
        setSelectedSubcategoryDescription(experiments[0].experimentDescription);

    }, [benchmark, config]); 

    const onClick: MenuProps['onClick'] = ({ key }) => {
        setSelectedSubcategory(key);
        setSelectedSubcategoryDescription(subcategories.find((d) => d.title === key).experimentDescription);
    }; 

    const menuItems: MenuProps['items'] = subcategories.map(subcategory => (  
        {  
            key: subcategory.title,  
            label: subcategory.title,  
        }  
    ));  
    

    return (
        <div style={{width: '100%', paddingBottom: '4em'}}>
            <h2 style={{paddingTop: '0.75em'}}>{benchmark}</h2>
            <h3>Task Description</h3>
            <span>{benchmarkDescription}</span>
            <h3 style={{paddingTop: '1em'}}>Capability Importance</h3>
            <span>{capabilityImportance}</span>
            <br/>
            <div>
                <h3 style={{paddingTop: '1em'}}>Overall Performance</h3>
                <div style={{width: '100%'}}>
                <Row justify="space-between" style={{display: 'flex', justifyContent: 'center'}}>
                    <OverallBenchmarkChart benchmark={benchmark} config={config}></OverallBenchmarkChart>
                </Row>
            </div>
            </div>
            <div>
                <h3 style={{paddingTop: '1em'}}>Metrics Description</h3>
                {menuItems.length > 1 ? ( 
                <div style={{display: 'flex', marginBottom: '0.5em', alignItems:'center'}}>
                    <span style={{marginRight: '0.5em'}}>Select Experiment:</span>
                    <Dropdown 
                        menu={{items: menuItems, onClick: onClick}}
                        overlayStyle={{}}
                    >  
                        <span style={{border: '1px solid #000000', borderRadius:'0.75em', padding:'0.25em'}}>
                            {selectedSubcategory}
                            <DownOutlined />
                        </span>
                    </Dropdown>
                </div> ) : (<div></div>)}
            <span>{selectedSubcategoryDescription}</span>
            </div>
            <br/>
            <div style={{width: '100%'}}>
                <Row justify="space-between" style={{display: 'flex', justifyContent: 'center'}}>
                    <BenchmarkChart benchmark={benchmark} experiment={selectedSubcategory} config={config}></BenchmarkChart>
                </Row>
            </div>
        </div>
    )
}

export default BenchmarkDetails;