import React from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import styles from './index.module.css';
import StatsBar from '../components/HomepageFeatures/stats_bar';
import ExecutiveSummary from '../components/HomepageFeatures/executive_summary';
import HomepageHeader from '../components/HomepageFeatures/homepage_header';
import OverallVisualization from '../components/HomepageFeatures/overall_visualization';
import SummaryTable from '../components/HomepageFeatures/summary_table';
import { EurekaConfig } from '../components/types';

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
              <HomepageHeader/>
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
