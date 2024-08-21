import ReactMarkdown from 'react-markdown';
import Layout from '@theme/Layout';
import { Capability, CapabilityScores, Config } from "../components/types";
import Heading from '@theme/Heading';
import styles from './index.module.css';
import StatsBar from './stats_bar';
import { ControlRowView } from '../components/atoms';
import { Select, Table } from 'antd';
import { Label } from 'recharts';
import React from 'react';


const DetailedLanguageView = () => {
    const [config, setConfig] = React.useState<Config | null>(null);
    const [selectedCapability, setSelectedCapability] = React.useState<Capability>();

    React.useEffect(() => {  
            fetch('/config.json')
        .then(response => response.json())
        .then(fetchedData => 
            {
                const benchmarks = fetchedData.benchmarks;
                const models = fetchedData.model_list;
                const model_families = fetchedData.model_families;
                const capabilities = fetchedData.capability_mapping;
                setSelectedCapability(capabilities[0]);   
                setConfig({benchmarks: benchmarks, models: models, model_families: model_families, capability_mapping: capabilities});
            })
        .catch(error => console.error(error));
    }, []);

    if (!config) {  
        // config is still null, probably still fetching data
        return <div>Loading...</div>;
    }

    

    return (
        <Layout>
            <header className={styles.hero}>  
                <div className={styles.heroBackground}></div>  
                <div className="container">  
                    <Heading as="h1" className="hero__title">  
                    LFM Model Benchmarking
                    </Heading>  
                    <p className="hero__subtitle">AI Frontiers Evaluation and Understanding</p>  
                </div>
                
            </header>
            <StatsBar config={config}/>
            <main>
                <div>
                    <p className='hero__subtitle'>Capability to explore:
                        <Select
                            className="mt-2 w-full"
                            value={selectedCapability.capability}
                            onChange={(selectedValue: any) => {
                                let newSelection = config.capability_mapping.filter(x => x.capability === selectedValue)[0]
                                setSelectedCapability(newSelection);
                            }}
                            options={config.capability_mapping.map((c: Capability) => (
                                {
                                    value: c.capability, 
                                    label: c.capability
                                }))
                            }
                        />
                    </p>
                    
                </div>
                <div>
                    <p className='hero__subtitle'>{selectedCapability.capability}</p>
                    <ReactMarkdown>{selectedCapability.description}</ReactMarkdown>
                </div>
                <p>Insert cool chart here</p>
                <div style={{ width: '100%' }}>
                    <Table columns={[{title: 'model', 
                                    dataIndex: 'model', 
                                    key: 'model'}]} 
                            dataSource={[{model:"hello"}]}
                            style={{ width: '100%' }}/>
                </div>
            </main>
        </Layout>
    );
}

export default DetailedLanguageView;