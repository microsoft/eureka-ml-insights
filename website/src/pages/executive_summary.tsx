import ReactMarkdown from 'react-markdown';
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
    const languageItems: CollapseProps['items']  = [
      {
        key: '1',
        label: 'Faster improvements for instruction following across all model families.',
        children: <ReactMarkdown>Amongst the studied language capabilities, instruction following is where most models are improving faster, potentially due to strong investments in instruction tuning processes, with most models now having an instruction following rate of higher than 75%.\n'
          </ReactMarkdown>,
      }
    ]

    const executiveSummaryItems: CollapseProps['items'] = [
        {
          key: '1',
          label: 'Language Evaluation',
          children: <Collapse items={languageItems}/>
          // <ReactMarkdown>{'The evaluation shows that there have been important advances from state-of-the-art LFMs in the language capabilities of instruction following, long context question answering, information retrieval, and safety.\n' +  
          //   '\n' +  
          //   '1. **Faster improvements for instruction following across all model families.** Amongst the studied language capabilities, instruction following is where most models are improving faster, potentially due to strong investments in instruction tuning processes, with most models now having an instruction following rate of higher than 75%.\n' +  
          //   '\n' +  
          //   '2. **All models\' performance in question answering drops with longer context.** When state-of-the-art models are compared in "needle-in-a-haystack" tasks, they seem to all perform equally well. However, testing the models on tasks that involve reasoning over long-context, we see that all models\' performance drops as context size grows. Amongst all models, and have the lowest drop in performance for longer context.\n' +  
          //   '\n' +  
          //   '3. **Major gaps in factuality and grounding for information retrieval from parametric knowledge or input context.** For example, we observe query constraint satisfaction rates (i.e. fact precision) of lower than 55%, completeness rates of lower than 25% (i.e. fact recall), and information irrelevance rates of higher than 20% (potentially information fabrication). , , and are the best performing models in this task across different conditions. and in particular have significantly lower information irrelevance rates (associated with better factuality). has better constraint satisfaction rates (associated with better constrained text generation and grounding).\n' +  
          //   '\n' +  
          //   '4. **High refusal rates and low accuracy in detecting neutral content for some models.** While several models have high accuracy rates for toxicity detection, others (, , , and ) exhibit a high amount of refusal and low accuracy in classifying neutral content, leading therefore to erasure risks. During the safe language generation evaluation, models like and have the highest toxicity rates. is the only model that has both a high toxicity detection accuracy and a low toxicity score for safe language generation, as shown in the discriminative and generative evaluations respectively.\n'}  
          //   </ReactMarkdown>,
        },
        {
          key: '2',
          label: 'Multimodal Evaluation',
          children: <ReactMarkdown>
              Evaluations on important vision-language
capabilities such as geometric and spatial reasoning, object recognition
and detection, multimodal question answering, and navigation demonstrate
increased capabilities of most recent models when compared to their
previous versions. For example, improvements over range between 3%-20%.
Yet, state-of-the-art models are still fairly limited in their
multimodal abilities, specifically when it comes to detailed image
understanding (e.g. localization of objects, geometric and spatial
reasoning, and navigation), which is most needed in truly multimodal
scenarios that require physical awareness and localization.


1.  **State-of-the-art multimodal models struggle with geometric
    reasoning.** Reasoning about height is more difficult than about
    depth. and are the best performing models for this task with being
    the most accurate model for depth ordering and the most accurate for
    height ordering.


2.  **Multimodal capabilities lag language capabilities.** On tasks
    which can be described either as a multimodal task or as
    language-only, the performance of most tested models is higher for
    the language-only condition. is the only model that consistently
    achieves better results when presented with both vision and language
    information, showing therefore that it can better fuse the two data
    modalities.


3.  **Complementary performance across models for fundamental multimodal
    skills.** For example, , , and have comparable performance in
    multimodal question answering (MMMU) but they outperform all other
    models by at least 15%. There are tasks like object recognition and
    visual prompting where the performance of is better or comparable to
    , but outperforms them both. Finally, in tasks like object detection
    and spatial reasoning, is the most accurate model.
            </ReactMarkdown>,
        }
      ];

    return (
        <div>
          <Layout>
            <Layout.Sider width="20%" style={siderStyle}>
              <div style={headerStyle}>
                Executive Summary
              </div>
              <p style={{fontSize: '16px', marginRight: '2px'}}>
                These results show a complementary picture of capabilities of different models and that there is no single model that outperforms all others in most tasks. 
                However, ClaudeSonnet, GPT-4o, and Llama-31Large repeatedly outperform others in several capabilities.
              </p>
            </Layout.Sider>
            <Layout.Content>
              <Collapse items={executiveSummaryItems} />
            </Layout.Content>
          </Layout>
        </div>
    )
}

export default ExecutiveSummary;