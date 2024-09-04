import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';
import styles from './index.module.css';
import { Button, Card, Col, Row } from 'antd';

function HomepageHeader() {
  return (
    <header className={styles.hero}>  
      <div className={styles.heroBackground}></div>  
      <div className="container">  
        <Heading as="h1" className="hero__title">  
          Eureka ML Insights
        </Heading>  
        <p className="hero__subtitle">Evaluating and Understanding Large Foundation Models</p> 
        <p>TODO: Insert Blurb</p>
        <Button>Read full report</Button>
        <Button>Github</Button>
      </div>
      <div>
         
      </div>
    </header>  
  );
}

export default function Home(): JSX.Element {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="Description will go into a meta tag in <head />">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
