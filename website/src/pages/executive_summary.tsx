import ReactMarkdown from 'react-markdown';
import React from 'react';
import { Layout, Button, CollapseProps, Collapse } from 'antd';
import { MinusIcon, PlusIcon } from '@heroicons/react/24/outline';


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
  marginTop: '.4em', 
  marginLeft: '.2em'
}; 

const ExecutiveSummary = () => {
    const languageItems: CollapseProps['items']  = [
      {
        key: '1',
        label: '01 Faster improvements for instruction following across all model families',
        children: <ReactMarkdown>Amongst the studied language capabilities, instruction following 
          is where most models are improving faster, potentially due to strong investments in 
          instruction tuning processes, with most models now having an instruction following rate 
          of higher than 75%.</ReactMarkdown>,
      },
      {
        key: '2',
        label: '02 All models\' performance in question answering drops with longer context',
        children: <ReactMarkdown>When state-of-the-art models are compared in "needle-in-a-haystack" 
          tasks, they seem to all perform equally well. However, testing the models on tasks that 
          involve reasoning over long-context, we see that all models\' performance drops as context 
          size grows. Amongst all models, GPT-4o and Llama-3_1-405B have the lowest drop in 
          performance for longer context.</ReactMarkdown>,
      },
      {
        key: '3',
        label: '03 Major gaps in factuality and grounding for information retrieval from parametric knowledge or input context',
        children: <ReactMarkdown>For example, we observe query constraint satisfaction rates (i.e. 
          fact precision) of lower than 55%, completeness rates of lower than 25% (i.e. fact recall), 
          and information irrelevance rates of higher than 20% (potentially information fabrication). 
          Llama-3_1-405B, GPT-4o, and Claude-3_5-Sonnet are the best performing models in this task 
          across different conditions. GPT-4o and Claude-3_5-Sonnet in particular have significantly 
          lower information irrelevance rates (associated with better factuality). Llama-3_1-405B
          has better constraint satisfaction rates (associated with better constrained text generation and grounding).</ReactMarkdown>,
      },
      {
        key: '4',
        label: '04 High refusal rates and low accuracy in detecting neutral content for some models',
        children: <ReactMarkdown>While several models have high accuracy rates for toxicity detection, 
          others (Gemini-1_5-Pro, Claude-3_5-Sonnet, Claude-3-Opus, and Llama-3_1-405B) exhibit a high amount of
          refusal and low accuracy in classifying neutral content, leading therefore to erasure risks. 
          During the safe language generation evaluation, models like GPTFourPrev and MistralLargeTwo 
          have the highest toxicity rates. GPT-4o is the only model that has both a high toxicity 
          detection accuracy and a low toxicity score for safe language generation, as shown in the 
          discriminative and generative evaluations respectively.</ReactMarkdown>,
      }
    ]

    const multimodalItems: CollapseProps['items'] = [
      {
        key: '1',
        label: '01 State-of-the-art multimodal models struggle with geometric reasoning',
        children: <ReactMarkdown>Reasoning about height is more difficult than about depth. Claude-3_5-Sonnet 
          and Gemini-1_5-Pro are the best performing models for this task with Claude-3_5-Sonnet being the most 
          accurate model for depth ordering and Gemini-1_5-Pro the most accurate for height ordering.</ReactMarkdown>,
      },
      {
        key: '2',
        label: '02 Multimodal capabilities lag language capabilities',
        children: <ReactMarkdown>On tasks which can be described either as a multimodal task or as 
          language-only, the performance of most tested models is higher for the language-only condition. 
          GPT-4o is the only model that consistently achieves better results when presented with both 
          vision and language information, showing therefore that it can better fuse the two data modalities.</ReactMarkdown>,
      },
      {
        key: '3',
        label: '03 Complementary performance across models for fundamental multimodal skills',
        children: <ReactMarkdown>For example, Claude-3_5-Sonnet, GPT-4o, and GPT-4o-2025-13 have comparable 
          performance in multimodal question answering (MMMU) but they outperform all other models by at
          least 15%. There are tasks like object recognition and visual prompting where the performance of 
          Claude-3_5-Sonnet is better or comparable to GPT-4o, Gemini-1_5-Pro but outperforms them both. Finally, 
          in tasks like object detection and spatial reasoning, GPT-4o is the most accurate model.</ReactMarkdown>,
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
            <Layout.Sider width="22%" style={siderStyle}>
              <div>
                <h2>Executive Summary</h2>
              </div>
            </Layout.Sider>
            <Layout.Content>
              <div>
                <div>
                  <h3 style={headerStyle}>Language Evaluation</h3>
                  <p style={{marginLeft: '.3em'}}>The evaluation shows that there have been important advances from state-of-the-art LFMs 
                    in the language capabilities of instruction following, long context question answering, 
                    information retrieval, and safety.</p>
                  <Collapse items={languageItems} expandIconPosition='end' expandIcon={customExpandIcon}/>
                </div>
                <div>
                  <h3 style={headerStyle}>Multimodal Evaluation</h3>
                  <p style={{marginLeft: '.3em'}}>Evaluations on important vision-language capabilities such as geometric and spatial 
                    reasoning, object recognition and detection, multimodal question answering, and navigation 
                    demonstrate increased capabilities of most recent models when compared to their previous 
                    versions. For example, improvements over range between 3%-20%. Yet, state-of-the-art models 
                    are still fairly limited in their multimodal abilities, specifically when it comes to detailed 
                    image understanding (e.g. localization of objects, geometric and spatial reasoning, and navigation),
                    which is most needed in truly multimodal scenarios that require physical awareness and localization.</p>
                  <Collapse items={multimodalItems} expandIconPosition='end' expandIcon={customExpandIcon}/>
                </div>
              </div>
            </Layout.Content>
          </Layout>
        </div>
    )
}

export default ExecutiveSummary;