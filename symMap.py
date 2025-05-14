from typing import List, Dict
import pandas as pd
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import Tool
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
import re
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class SyndromeAgent:
    def __init__(self):
        self.symmap_data = pd.read_csv('./symMap_file.csv')
        
    def extract_syndromes(self, user_input: str) -> List[Dict]:
        """Extract syndrome keywords using GPT-4"""
        messages = [
            {"role": "system", "content": "You are a TCM expert. Extract syndrome names from the input in both Chinese and English."},
            {"role": "user", "content": user_input}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0
        )
        
        extracted_text = response.choices[0].message.content
        
        # Convert response to structured format
        syndromes = []
        for line in extracted_text.split('\n'):
            if line.strip():
                chinese = re.findall(r'[\u4e00-\u9fff]+', line)
                english = re.findall(r'[a-zA-Z\s-]+', line)
                if chinese and english:
                    syndromes.append({
                        'chinese': chinese[0].strip(),
                        'english': english[0].strip()
                    })
        return syndromes

    def find_syndrome_ids(self, syndromes: List[Dict]) -> List[str]:
        """Find syndrome IDs from symmap_data"""
        syndrome_ids = []
        
        try:
            for syndrome in syndromes:
                matches = self.symmap_data[
                    self.symmap_data['Syndrome_name'].str.contains(syndrome['chinese'], na=False) |
                    self.symmap_data['Syndrome_name'].str.contains(syndrome['english'], na=False, case=False)
                ]
                
                if not matches.empty:
                    syndrome_ids.extend(matches['Syndrome_id'].tolist())
        except Exception as e:
            print(f"Error processing syndromes: {e}")
            
        return list(set(syndrome_ids))

    

# 定义状态模型
class WorkflowState(BaseModel):
    input: str
    syndromes: List[dict] = []
    ids: List[str] = []
    files: List[str] = []

def create_agent_workflow():
    """Create LangGraph workflow"""
    agent = SyndromeAgent()
    
    # 使用 Pydantic 模型定义状态架构
    workflow = StateGraph(state_schema=WorkflowState)
    
    # Add nodes
    workflow.add_node("extract", agent.extract_syndromes)
    workflow.add_node("find_ids", agent.find_syndrome_ids)
    workflow.add_node("download", agent.download_targets)
    
    # 定义工作流的起点
    workflow.set_entry_point("extract")

    # Define edges and conditions
    workflow.add_edge("extract", "find_ids")
    workflow.add_edge("find_ids", "download")
    workflow.add_edge("download", END)
    
    return workflow.compile()   


workflow = create_agent_workflow()

# 使用 Pydantic 模型创建初始状态
initial_state = WorkflowState(
    input="请帮我分析阴虚火旺和肝肾不足的基因靶点",
    syndromes=[],
    ids=[],
    files=[]
)

# 运行工作流
result = workflow.invoke(initial_state)
print(result)

