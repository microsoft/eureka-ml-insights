import { Button, Card, Col, Row } from "antd";
import { ChartBarIcon, NewspaperIcon, PhotoIcon, ShieldCheckIcon } from "@heroicons/react/24/outline";
import React from "react";
import { Config } from "../components/types";

const StatsBar = ({config}: {config: Config}) => {
  if (!config) {  
      // config is still null, probably still fetching data
      return <div>Loading...</div>;
  }
  return (
    <Card>  
        <Row align='middle'> 
            <Col span={4} >  
                <p style={{ fontWeight: 'bold', fontSize: '30px' }}>{config.model_families.length}</p>  
                <p style={{ color: 'grey', fontSize: '18px' }}>Model Families</p>
            </Col>  
            <Col span={3} >
                <p style={{ fontWeight: 'bold', fontSize: '30px' }}>{config.models.length}</p>  
                <p style={{ color: 'grey', fontSize: '18px' }}>Models</p>
            </Col>  
            <Col span={3} >  
                <p style={{ fontWeight: 'bold', fontSize: '30px' }}>{config.benchmarks.length}</p>  
                <p style={{ color: 'grey', fontSize: '18px' }}>Benchmarks</p>
            </Col>  
            <Col span={3} >  
                <div style={{fontSize: '30px'}}><NewspaperIcon className='inline-block mr-1' width={40} height={40}/></div>
                <p style={{ color: 'grey', fontSize: '18px' }}>Language tasks</p>
            </Col>  
            <Col span={4} >  
                <div style={{fontSize: '30px'}}><PhotoIcon className='inline-block mr-1' width={40} height={40}/></div>
                <p style={{ color: 'grey', fontSize: '18px' }}>Multimodal tasks</p>
            </Col>  
            {/* <Col span={4} style={{ flexDirection: 'column', justifyContent: 'center' }}>  
                <Button style={{fontSize: '18px'}} size='large' type='primary'><ChartBarIcon width={30} height={30}/>AI Quality</Button>
            </Col>  
            <Col span={3} style={{ flexDirection: 'column', justifyContent: 'center' }}>  
                <Button style={{fontSize: '18px'}} size='large' type='default'><ShieldCheckIcon width={30} height={30}/>AI Safety</Button>
            </Col>   */}
        </Row>  
    </Card>
  );
}

export default StatsBar;