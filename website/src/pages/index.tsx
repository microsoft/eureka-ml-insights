import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import styles from './index.module.css';
import { Button } from 'antd';
import StatsBar from './stats_bar';
import ExecutiveSummary from './executive_summary';
import OverallVisualization from './overall_visualization';
import SummaryTable from './summary_table';
import React from 'react';
import { Config } from '../components/types';

function HomepageHeader() {
  return (
    <header className={styles.hero}>  
      {/* <div className={styles.heroBackground}></div>   */}
      <div className="container">  
        <Heading as="h1" className="hero__title">  
          Eureka ML Insights
        </Heading>  
        <p className="hero__subtitle">Evaluating and Understanding Large Foundation Models</p> 
        <p>Eureka is an open-source framework for standardizing evaluations of large foundation models, beyond single-score 
          reporting and rankings. We report in-depth evaluation and analysis of 12 state-of-the-art models across a collection 
          of language and multimodal benchmarks. These benchmarks test fundamental but overlooked capabilities that are still 
          challenging for even the most capable models.</p>
        <br/>
        <Button shape='round' className={`${styles.buttons} ${styles.fullReportButton}`}><strong>Read full report</strong></Button>
        <Button shape='round' className={`${styles.buttons}`} href={'https://github.com/microsoft/eureka-ml-insights'} style={{outline: "black"}}>
          <strong>Github</strong>
          <img src='img/link_icon.svg' alt="External Link to Github"/>
        </Button>
      </div>
    </header>
  );
}

export default function Home(): JSX.Element {
  const {siteConfig} = useDocusaurusContext();
  const [config, setConfig] = React.useState<Config | null>(null);
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
  
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="Welcome to the page for Eureka Model Benchmarks">
      <div className={styles.fullWidthContainer}>
        <div className={styles.heroBackground}></div>
        <div className={styles.splashSvg}></div>
        <div className="container" style={{position: 'relative'}}>
          <div className={styles.heroContent}>
            <br/>
            <br/>
            <HomepageHeader />
            <br/>
            <br/>
            <br/>
            <StatsBar config={config}/>
            <br/>
            <br/>
            <br/>
            <br/>
          </div>
        </div>
      </div>
      <main>
        <section className={styles.features}>
          <div className="container">
            <OverallVisualization config={config}/>
            <br/>
            <br/>
            <ExecutiveSummary/>
            <br/>
            <SummaryTable config={config}/>
          </div>
        </section>
      </main>
    </Layout>
  );
}
