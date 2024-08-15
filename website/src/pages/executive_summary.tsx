import React from 'react';
import { Layout, Button, CollapseProps, Collapse } from 'antd';


const siderStyle: React.CSSProperties = {  
    display: 'flex',  
    flexDirection: 'column',  
    alignItems: 'center',  
    justifyContent: 'center',  
    color: '#000000',  
    backgroundColor: '#F9FAFF',  
    fontSize: '20px',  
  };  
  
  const headerStyle: React.CSSProperties = {  
    marginBottom: '20px', // adjust this value as needed  
  }; 

const ExecutiveSummary = () => {
    const executiveSummaryItems: CollapseProps['items'] = [
        {
          key: '1',
          label: '01\t\tPerformance for multi modal capabilities',
          children: 'Performance lorem ipsom',
        },
        {
          key: '2',
          label: '02\t\t Performance for language capabilities',
          children: 'This is the model families',
        },
        {
          key: '3',
          label: 'Models',
          children: 'This is the models'
        }
      ];

    return (
        <div>
          <Layout>
            <Layout.Sider width="20%" style={siderStyle}>
              <div style={headerStyle}>
                Executive Summary
              </div>
              <Button>
                Read More
              </Button>
            </Layout.Sider>
            <Layout.Content>
              <Collapse items={executiveSummaryItems} />
            </Layout.Content>
          </Layout>
        </div>
    )
}

export default ExecutiveSummary;