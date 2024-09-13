import React from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import styles from './index.module.css';
import { Button, Col, Row } from 'antd';
import StatsBar from '../components/HomepageFeatures/stats_bar';
import ExecutiveSummary from '../components/HomepageFeatures/executive_summary';
import OverallVisualization from '../components/HomepageFeatures/overall_visualization';
import SummaryTable from '../components/HomepageFeatures/summary_table';
import { EurekaConfig } from '../components/types';

function HomepageHeader() {
  return (
    <header className={styles.hero} style={{paddingTop: '2em', paddingBottom: '4em'}}>  
      <div className="container">  
        <div>
          <Row style={{ display: 'flex', alignItems: 'center' }}>
            <Col style={{paddingRight: '1.8em', paddingBottom: '.9em'}}><img src="img/eureka_logo.png" alt="Eureka Logo" style={{objectFit:'cover', height: '5em'}}/></Col>
            <Col><Heading as="h1" className="hero__title">Eureka ML Insights</Heading></Col>
          </Row>
        </div>
        <p className="hero__subtitle">Evaluating and Understanding Large Foundation Models</p> 
        <p style={{fontSize: '1.3em'}}>Eureka is an open-source framework for standardizing evaluations of large foundation models, beyond single-score 
          reporting and rankings. We report in-depth evaluation and analysis of 12 state-of-the-art models across a collection 
          of language and multimodal benchmarks. These benchmarks test fundamental but overlooked capabilities that are still 
          challenging for even the most capable models.</p>
        <br/>
        <Button shape='round' className={`${styles.buttons} ${styles.fullReportButton}`} href={'https://aka.ms/eureka-ml-insights-blog'}>
          <strong>Read full report</strong>
        </Button>
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
  
  return (
     <Layout
      title={`${siteConfig.title}`}
      description="Welcome to the page for Eureka Model Benchmarks">
        <div className={styles.fullWidthContainer}>
          <div className={styles.heroBackground}/>
          <div className={styles.splashSvg}/>
          <div className="container" style={{position: 'relative'}}>
            <div className={styles.heroContent}>
              <HomepageHeader />
              <StatsBar config={config}/>
            </div>
          </div>
        </div>
        <main>
          <section className={styles.features}>
            <div className="container">
              <OverallVisualization config={config}/>
              <ExecutiveSummary/>
              <br/>
              <SummaryTable config={config}/>
            </div>
          </section>
        </main>
     </Layout>
  );
}
