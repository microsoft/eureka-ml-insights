import { Button, Col, Row } from "antd";
import Heading from '@theme/Heading';
import styles from './homepage_header.module.css';

function HomepageHeader(mainPage: boolean) {
    return (
      <header className={styles.hero} style={{paddingTop: '2em', paddingBottom: '4em'}}>  
        <div className="container">  
          <div>
            <Row style={{ display: 'flex', alignItems: 'center' }}>
              <Col style={{paddingRight: '1.8em', paddingBottom: '.9em'}}><img src="img/eureka_logo.png" alt="" style={{objectFit:'cover', height: '5em'}}/></Col>
              <Col><Heading as="h1" className="hero__title">Eureka ML Insights</Heading></Col>
            </Row>
          </div>
          <h2>Evaluating and Understanding Large Foundation Models</h2> 
          <p style={{fontSize: '1.3em'}}>Eureka is an open-source framework for standardizing evaluations of large foundation models, beyond single-score 
            reporting and rankings. We report in-depth evaluation and analysis of 12 state-of-the-art models across a collection 
            of language and multimodal benchmarks. These benchmarks test fundamental but overlooked capabilities that are still 
            challenging for even the most capable models.</p>
          <br/>
          <Button shape='round' className={`${styles.buttons} ${styles.fullReportButton}`} href={'https://aka.ms/eureka-ml-insights-report'}>
            <strong>Read full report</strong>
          </Button>
          <Button shape='round' className={`${styles.buttons}`} href={'https://github.com/microsoft/eureka-ml-insights'} style={{outline: "black"}}>
            <strong>Github</strong>
            <img src='img/link_icon.svg' alt=""/>
          </Button>
        </div>
      </header>
    );
  }

export default HomepageHeader;