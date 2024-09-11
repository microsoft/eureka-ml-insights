import { Card, Col, Row } from "antd";
import React from "react";
import { EurekaConfig } from "../components/types";

const statsHeader: React.CSSProperties = {  
    fontWeight: 'bold', 
    fontSize: '2.5em', 
    marginBottom: '.08em'
}; 

const statsIcon: React.CSSProperties = {  
    fontWeight: 'bold', 
    fontSize: '2em', 
    marginBottom: '-.07em'
}; 

const statsLabel: React.CSSProperties = {
    color: 'grey', 
    fontSize: '1.5em', 
    marginBottom: '0'
}

const StatsBar = ({config}: {config: EurekaConfig}) => {
  if (!config) {  
      // config is still null, probably still fetching data
      return <div>Loading...</div>;
  }

  return (
    <div style={{paddingBottom: '4em'}}>
        <Card>  
            <Row align='middle' > 
                <Col span={4} >  
                    <p style={statsHeader}>{config.model_families?.length}</p>  
                    <p style={statsLabel}>Model Families</p>
                </Col>  
                <Col span={4} >
                    <p style={statsHeader}>{config.models?.length}</p>  
                    <p style={statsLabel}>Models</p>
                </Col>  
                <Col span={4} >  
                    <p style={statsHeader}>{config.benchmarks?.length}</p>  
                    <p style={statsLabel}>Benchmarks</p>
                </Col>
                <Col span={4} >
                    <p style={statsHeader}>{config.capability_mapping?.length}</p>  
                    <p style={statsLabel}>Capabilities</p>
                </Col>  
                <Col span={4} >  
                    <div style={statsIcon}><img src='img/language.svg' alt="Language Tasks"/></div>
                    <p style={statsLabel}>Language tasks</p>
                </Col>  
                <Col span={4} >  
                    <div style={statsIcon}><img src='img/multimodal.svg' alt="Language Tasks"/></div>
                    <p style={statsLabel}>Multimodal tasks</p>
                </Col>
            </Row>  
        </Card>
    </div>
  );
}

export default StatsBar;