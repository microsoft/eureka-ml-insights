import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { Layout as AntdLayout, Menu, Select} from "antd";
import Layout from '@theme/Layout';
import React from "react";
import { EurekaConfig } from "../types";
import BenchmarkChart from "./benchmark_chart";
import Sider from "antd/es/layout/Sider";
import { Header } from "antd/es/layout/layout";
import Link from "@docusaurus/Link";
import { ArrowLeftCircleIcon, ArrowLeftIcon } from "@heroicons/react/24/outline";

const ModalityBreakdown = ({modality}: {modality: string}) => {
    const {siteConfig} = useDocusaurusContext();
    const [config, setConfig] = React.useState<EurekaConfig | null>(null);
    const [items, setItems] = React.useState<MenuItem[]>([]);
    const [selectedBenchmark, setSelectedBenchmark] = React.useState(null);  

    React.useEffect(() => {  
          fetch('config.json')
         .then(response => response.json())
         .then(fetchedData => 
            {
                const benchmarks = fetchedData.benchmarks.filter((benchmark) => benchmark.modality === modality);
                const models = fetchedData.model_list;
                const model_families = fetchedData.model_families;
                const capabilities = fetchedData.capability_mapping;  
                setConfig({benchmarks: benchmarks, models: models, model_families: model_families, capability_mapping: capabilities});
                setItems(benchmarks.map((benchmark, index) => ({label: benchmark.name, key: benchmark.name})));
                
                if (benchmarks.length > 0) {  
                  
                  setSelectedBenchmark(benchmarks[0].name);  
                } 
            })
         .catch(error => console.error(error));
    }, []);

    return (
       <Layout>
          <Header style={{ alignItems: 'left', color: 'white' }}>
            <Link to="/eureka-ml-insights"><strong><ArrowLeftIcon style={{width: '2em'}}/>Multimodal Task Performance</strong></Link>
          </Header>
          <AntdLayout>
            <Sider>
              <Menu
                selectedKeys={[selectedBenchmark]} 
                mode="inline"
                onClick={({ key }) => setSelectedBenchmark(key)}  
                >
                  {items.map(item => (  
                    <Menu.Item key={item.key}>  
                      {item.label}  
                    </Menu.Item>  
                  ))}  
                </Menu>
            </Sider>
            <main style={{ flex: 1 }}>
              <section>
                <div className="container">
                {selectedBenchmark ? (<BenchmarkChart benchmark={selectedBenchmark} config={config}></BenchmarkChart>) : (<div>Loading data...</div>)}  
                </div>
              </section>
            </main>
        </AntdLayout>
       </Layout>
    );
  }

export default ModalityBreakdown;