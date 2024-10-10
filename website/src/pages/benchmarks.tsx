import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { Layout, Select, Space } from "antd";
import React from "react";
import HomepageHeader from "../components/HomepageFeatures/homepage_header";
import StatsBar from "../components/HomepageFeatures/stats_bar";
import { EurekaConfig } from "../components/types";
import styles from './index.module.css';
import BenchmarkChart from "../components/HomepageFeatures/benchmark_chart";

export default function Home(): JSX.Element {
    const {siteConfig} = useDocusaurusContext();
    const [config, setConfig] = React.useState<EurekaConfig | null>(null);

    React.useEffect(() => {  
          fetch('config.json')
         .then(response => response.json())
         .then(fetchedData => 
            {
                const benchmarks = fetchedData.benchmarks;
                const models = fetchedData.model_list;
                const model_families = fetchedData.model_families;
                const capabilities = fetchedData.capability_mapping;  
                setConfig({benchmarks: benchmarks, models: models, model_families: model_families, capability_mapping: capabilities});
            })
         .catch(error => console.error(error));
    }, []);

    const [selectedBenchmark, setSelectedBenchmark] = React.useState(null);  
  
    React.useEffect(() => {  
        if (config && config.benchmarks.length > 0) {  
            setSelectedBenchmark(config.benchmarks[0].name);  
        }  
    }, [config]);  

    return (
       <Layout
        title={`${siteConfig.title}`}
        description="Welcome to the page for Eureka Model Benchmarks">
          <div className={styles.fullWidthContainer}>
            <div className={styles.heroBackground} style={{zIndex: '0'}}/>
            <div className={styles.splashSvg}/>
            <div className="container" style={{position: 'relative'}}>
              <div className={styles.heroContent}>
                <HomepageHeader mainPage={false}/>
                <StatsBar config={config}/>
              </div>
            </div>
          </div>
          <main>
            <section className={styles.features}>
              <div className="container">
                <div>
                    Please select benchmark:
                    <Select 
                        value={selectedBenchmark}
                        popupMatchSelectWidth={false}
                        onChange={(value) => setSelectedBenchmark(value)}
                        style={{marginLeft: '.5em'}}
                    >
                        {config && config.benchmarks.map((benchmark, index) => (  
                            <Select.Option key={index} value={benchmark.name}>{benchmark.name}</Select.Option>  
                        ))}  
                    </Select>  
                </div>
              </div>
            </section>
            <section className={styles.features}>
              <div className="container">
                {config ? (<BenchmarkChart benchmark={selectedBenchmark} config={config}></BenchmarkChart>) : (<div>Loading data...</div>)}
              </div>
            </section>
          </main>
       </Layout>
    );
  }