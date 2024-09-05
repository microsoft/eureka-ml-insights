import { Button, Card, Col, Row } from "antd";
import { ChartBarIcon, NewspaperIcon, PhotoIcon, ShieldCheckIcon } from "@heroicons/react/24/outline";
import React from "react";
import { Config } from "../components/types";

const statsHeader: React.CSSProperties = {  
    fontWeight: 'bold', 
    fontSize: '2.5em', 
    marginBottom: '.08em'
}; 

const statsLabel: React.CSSProperties = {
    color: 'grey', 
    fontSize: '1.5em' 
}

const StatsBar = ({config}: {config: Config}) => {
  if (!config) {  
      // config is still null, probably still fetching data
      return <div>Loading...</div>;
  }
  return (
    <Card style={{marginTop: 0, marginBottom: 0}}>  
        <Row align='middle' > 
            <Col span={4} >  
                <p style={statsHeader}>{config.model_families.length}</p>  
                <p style={statsLabel}>Model Families</p>
            </Col>  
            <Col span={4} >
                <p style={statsHeader}>{config.models.length}</p>  
                <p style={statsLabel}>Models</p>
            </Col>  
            <Col span={4} >  
                <p style={statsHeader}>{config.benchmarks.length}</p>  
                <p style={statsLabel}>Benchmarks</p>
            </Col>
            <Col span={4} >
                <p style={statsHeader}>{config.capability_mapping.length}</p>  
                <p style={statsLabel}>Capabilities</p>
            </Col>  
            <Col span={4} >  
                <div style={statsHeader}><NewspaperIcon className='inline-block mr-1' width={40} height={40}/></div>
                <p style={statsLabel}>Language tasks</p>
            </Col>  
            <Col span={4} >  
                <div style={statsHeader}><PhotoIcon className='inline-block mr-1' width={40} height={40}/></div>
                <p style={statsLabel}>Multimodal tasks</p>
            </Col>
        </Row>  
    </Card>
  );
}

export default StatsBar;