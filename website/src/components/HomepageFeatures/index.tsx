import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';
import Visualization from '@site/src/pages/visualization';
import React from 'react';
import { ModelFamily, VisualizationConfig } from '../types';

export default function HomepageFeatures(): JSX.Element {
  const [config, setConfig] = React.useState<VisualizationConfig | null>(null);
  React.useEffect(() => {  
        fetch('/config.json')
       .then(response => response.json())
       .then(fetchedData => 
          {
            console.log(fetchedData);
              const benchmarks = fetchedData.benchmarks;
              const experiments = fetchedData.experiments;
              const models = fetchedData.model_list;
              console.log(models);
              const modelMap = fetchedData.model_list.reduce((acc, curr) => {  
                  if(!acc[curr.model_family]) {  
                    acc[curr.model_family] = [];  
                  }  
                  acc[curr.model_family].push(curr.model);  
                  return acc;  
                }, {});  
              const model_families: ModelFamily[] = Object.keys(modelMap).map(model_family => ({  
                  model_family,  
                  models: modelMap[model_family]  
                })); 
              
              setConfig({benchmarks: benchmarks, experiments: experiments, models: models, model_families: model_families});
          })
       .catch(error => console.error(error));
  }, []);

  return (
    <section className={styles.features}>
      <div className="container">
        <Visualization config={config}/>
      </div>
    </section>
  );
}
