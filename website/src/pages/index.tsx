import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';
import styles from './index.module.css';
import { Card, Col, Row } from 'antd';

function HomepageHeader() {
  return (
    <header className={styles.hero}>  
      <div className={styles.heroBackground}></div>  
      <div className="container">  
        <Heading as="h1" className="hero__title">  
          LFM Model Benchmarking
        </Heading>  
        <p className="hero__subtitle">AI Frontiers Evaluation and Understanding</p>  
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
