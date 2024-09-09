import ReactMarkdown from 'react-markdown';
import React from 'react';
import { Layout, Button, CollapseProps, Collapse } from 'antd';
import { MinusIcon, NewspaperIcon, PlusIcon } from '@heroicons/react/24/outline';


const siderStyle: React.CSSProperties = {  
    display: 'flex',  
    flexDirection: 'column',  
    alignItems: 'center',  
    justifyContent: 'center',  
    color: '#000000',  
    backgroundColor: '#F9FAFF',
    minWidth: '22%',
    // fontSize: '2em',  
  };  
  
const headerStyle: React.CSSProperties = {  
  marginTop: '.4em', 
  marginLeft: '.2em'
}; 

const ExecutiveSummary = () => {
    const languageItems: CollapseProps['items']  = [
      {
        key: '1',
        label: '01\t\tFaster improvements for instruction following across all model families',
        children: <ReactMarkdown>Amongst the studied language capabilities, instruction following 
          is where most models are improving faster, potentially due to strong investments in 
          instruction tuning processes, with most models now having an instruction following rate 
          of higher than 75%.</ReactMarkdown>,
      },
      {
        key: '2',
        label: '02\t\tAll models\' performance in question answering drops with longer context',
        children: <ReactMarkdown>Contrary to “needle-in-a-haystack” experiments, testing state-of-the-art
           models on tasks that involve reasoning over long-context shows significant performance 
           drops as context size grows. Amongst all models, GPT-4o 2024-05-13 and Llama 3.1 405B 
           have the lowest drop in performance for longer context.</ReactMarkdown>,
      },
      {
        key: '3',
        label: '03\t\tMajor gaps in factuality and grounding for information retrieval from parametric knowledge or input context',
        children: <ReactMarkdown>Models exhibit query constraint satisfaction rates (i.e. fact 
          precision) of lower than 55\%, completeness rates of lower than 25% (i.e. fact recall), 
          and information irrelevance rates of higher than 20% (potentially fabricated information). 
          Llama 3.1 405B, GPT-4o 2024-05-13, and Claude 3.5 Sonnet are the best performing models in 
          this area across different conditions.</ReactMarkdown>,
      },
      {
        key: '4',
        label: '04\t\tHigh refusal rates. Lower accuracy in detecting toxic content vs. neutral content for most models.',
        children: <ReactMarkdown>While several models have high accuracy rates for toxicity detection, 
          others (Gemini 1.5 Pro, Claude 3.5 Sonnet, Claude 3 Opus, and Llama 3.1 405B) exhibit low 
          accuracy in classifying toxic content and a high amount of refusal. During the safe language 
          generation evaluation, models like GPT-4 1106 Preview and Mistral Large 2407 have the highest 
          toxicity rates. GPT-4o 2024-05-13 is the only model that has both a high toxicity detection 
          accuracy and a low toxicity score for safe language generation, as shown in the discriminative 
          and generative evaluations respectively. </ReactMarkdown>,
      }
    ]

    const multimodalItems: CollapseProps['items'] = [
      {
        key: '1',
        label: '01\t\tState-of-the-art multimodal models struggle with geometric reasoning',
        children: <ReactMarkdown>Models perform worse in reasoning about height than about depth. Claude 3.5 
          Sonnet and Gemini 1.5 Pro are the best performing models for this task with Claude 3.5 Sonnet being 
          the most accurate model for depth ordering and Gemini 1.5 Pro the most accurate for height ordering. </ReactMarkdown>,
      },
      {
        key: '2',
        label: '02\t\tMultimodal capabilities lag language capabilities',
        children: <ReactMarkdown>On tasks which can be described either as a multimodal task or as 
          language-only, the performance of most tested models is higher for the language-only condition. 
          GPT-4o is the only model that consistently achieves better results when presented with both 
          vision and language information, showing therefore that it can better fuse the two data modalities.</ReactMarkdown>,
      },
      {
        key: '3',
        label: '03\t\tComplementary performance across models for fundamental multimodal skills',
        children: <ReactMarkdown>Claude 3.5 Sonnet, GPT-4o 2024-05-13, and GPT-4 Turbo 2024-04-09 have comparable 
          performance in multimodal question answering (MMMU). In tasks like object recognition and visual prompting, 
          the performance of Claude 3.5 Sonnet is better or comparable to GPT-4o 2024-05-13, but Gemini 1.5 Pro 
          outperforms them both. Finally, in tasks like object detection and spatial reasoning, GPT-4o 2024-05-13 
          is the most accurate model. </ReactMarkdown>,
      }
    ]

    const customExpandIcon = (props) => {  
      if (props.isActive) {  
        return <MinusIcon className='inline-block' width={20} height={20}/>;  
      }
      return <PlusIcon className='inline-block' width={20} height={20} />;  
    }  

    return (
        <div>
          <Layout>
            <Layout.Sider style={siderStyle}>
              <div>
                <h1 style={{marginLeft: '1em', marginTop: '1em', wordBreak: 'break-all'}}>Executive Summary</h1>
              </div>
            </Layout.Sider>
            <Layout.Content>
              <div>
                <div>
                  <h3 style={headerStyle}>Language Evaluation</h3>
                  <p style={{marginLeft: '.3em'}}>The evaluation through Eureka shows that there have been important 
                    advances from state-of-the-art models in the language capabilities of instruction following, long 
                    context question answering, information retrieval, and safety. The analysis also discovers major 
                    differences and gaps between models related to robustness to context length, factuality and 
                    grounding for information retrieval, and refusal behavior.</p>
                  <Collapse expandIconPosition='end' expandIcon={customExpandIcon}>
                    {languageItems.map(item => (  
                      <Collapse.Panel   
                        key={item.key}   
                        header={<div style={{ fontWeight: 'bold', fontSize: '16px', whiteSpace: 'pre', wordWrap: 'break-word', overflowWrap: 'break-word', hyphens: 'auto' }}>{item.label}</div>}   
                      >
                        {item.children}  
                      </Collapse.Panel>  
                    ))}  
                  </Collapse>
                </div>
                <div>
                  <h3 style={headerStyle}>Multimodal Evaluation</h3>
                  <p style={{marginLeft: '.3em'}}>State-of-the-art models are still fairly limited in their 
                    multimodal abilities, specifically when it comes to detailed image understanding. For 
                    example, these models struggle with localizing objects, geometric and spatial reasoning, 
                    and navigation, which are all examples of capabilities that are most needed in truly 
                    multimodal scenarios that require physical awareness, visual grounding and localization.</p>
                  <Collapse expandIconPosition='end' expandIcon={customExpandIcon}>
                    {multimodalItems.map(item => (  
                      <Collapse.Panel   
                        key={item.key}   
                        header={<div style={{ fontWeight: 'bold', fontSize: '16px', whiteSpace: 'pre' }}>{item.label}</div>}   
                      >
                        {item.children}  
                      </Collapse.Panel>  
                    ))}  
                  </Collapse>
                </div>
              </div>
            </Layout.Content>
          </Layout>
        </div>
    )
}

export default ExecutiveSummary;