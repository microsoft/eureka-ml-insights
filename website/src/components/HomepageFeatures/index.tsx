import clsx from 'clsx';
import styles from './styles.module.css';
import OverallVisualization from '@site/src/pages/overall_visualization';
import ExecutiveSummary from '@site/src/pages/executive_summary';
import StatsBar from '@site/src/pages/stats_bar';
import SummaryTable from '@site/src/pages/summary_table';
import React from 'react';
import { Config } from '../types';


export default function HomepageFeatures(): JSX.Element {
  const [config, setConfig] = React.useState<Config | null>(null);
  React.useEffect(() => {  
        fetch('/config.json')
       .then(response => response.json())
       .then(fetchedData => 
          {
              const benchmarks = fetchedData.benchmarks;
              const models = fetchedData.model_list;
              const model_families = fetchedData.model_families;
              const capabilities = fetchedData.capability_mapping;  
              console.log(capabilities);
              setConfig({benchmarks: benchmarks, models: models, model_families: model_families, capability_mapping: capabilities});
          })
       .catch(error => console.error(error));
  }, []);

  return (
    <section className={styles.features}>
      <div className="container">
        <div>
          <div className={styles.heroBackground}></div>
          <StatsBar config={config}/>
          <OverallVisualization config={config}/>
        </div>
        <br/>
        <ExecutiveSummary/>
        <br/>
        <SummaryTable config={config}/>
      </div>
    </section>
  );
}
