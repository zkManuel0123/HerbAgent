from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentType, initialize_agent, create_react_agent
from langchain.prompts import StringPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from openai import OpenAI
from typing import List, Dict, Union, Optional, Any
from dataclasses import dataclass
from enum import Enum
import re
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
import asyncio
import pandas as pd
import logging
import traceback
import time
load_dotenv()
from langchain.tools import Tool
import json

# 在文件开头设置日志级别
logging.basicConfig(level=logging.DEBUG)

# 配置 QWen-Max
llm = ChatOpenAI(
    model_name="qwen-max",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0,
    max_tokens=2048,
    top_p=0.8,
    request_timeout=120,
)

# 定义中药靶点查找工具
class HerbTargetTool:
    def __init__(self, tool_path: str = "D:/HerbAgent/tools/"):
        import sys
        import os
        
        # 添加工具目录到系统路径
        tool_dir = os.path.dirname(tool_path)
        if tool_dir not in sys.path:
            sys.path.append(tool_dir)
            
        from tools.Herb_target_search import HerbTargetSearch
        self.searcher = HerbTargetSearch()
    
    def search_targets(self, herbs: Dict[str, List[str]]) -> Dict:
        """
        搜索中药靶点信息
        
        Args:
            herbs: 格式如 {"补血汤": ["当归", "白芍"], "方剂2": ["当归", "白芍", "熟地黄"]}
                  或 {"当归": ["当归"], "白术": ["白术"]}
        
        Returns:
            Dict: 包含靶点信息的字典
        """
        try:
            # 直接将传入的字典传给搜索器
            results = self.searcher.search(herbs)
            return results
            
        except Exception as e:
            logging.error(f"搜索中药靶点时发生错误: {str(e)}")
            logging.error(traceback.format_exc())
            return {"error": f"搜索中药靶点时发生错误: {str(e)}"}
            
            
            
            
       

# 将中药靶点工具封装为 LangChain Tool
def create_herb_target_tool() -> Tool:
    herb_tool = HerbTargetTool()
    return Tool(
        name="herb_target_search",
        func=herb_tool.search_targets,  # 直接使用 search_targets 方法
        description="Search the target information of Chinese medicine. Input format is a dictionary."
    )

# Agent 1: 用户输入处理 Agent
class InputProcessingAgent:
    def __init__(self, llm):
        self.llm = llm
        self.chat_mode = True
        self.analysis_mode = False
        self.timing_stats = {
            'chat_processing': 0,
            'analysis_processing': 0,
            'total_processing': 0
        }

        self.chat_prompt = """You are a friendly and professional Chinese medicine assistant. Please analyze user input, detect user intent, and respond accordingly.

User input: {user_input}

Please judge and respond according to the following rules:

Case 1: If the user input is ordinary chat content: Please respond to the user in a friendly way and prompt that they can input Chinese medicine, formula, syndrome, or disease information for professional analysis at any time.

Case 2: If the user input contains any of the following information or intent:
    - Formula name and corresponding Chinese medicine
    - Single Chinese medicine name
    - Syndrome information
    - Disease information
    - Explicitly express the intention to perform analysis
     Please only return "ANALYSIS_MODE", as a switch to help the system switch to analysis mode, no need to reply any other text and content.

Case 3: If the user expresses the intention to exit the system or end the conversation: Please return "SYSTEM_EXIT", as a switch to help the system exit, no need to reply any other text and content.

Remember: For case 1, return normal conversation reply. When detecting case 2 and case 3, only return "ANALYSIS_MODE" or "SYSTEM_EXIT" code, no need to reply any other text and content.
"""


        self.prompt_template = """You are a professional Chinese medicine information processing assistant. Your task is to accurately identify keywords from the user's natural language input in Chinese or English or Chinese Pinyin, extract them as the closest standard formula name, standard Chinese medicine name, standard syndrome name, standard disease name, and organize them into the specified JSON format.

Please carefully analyze the user input: {user_input}

Processing rules:
1. If the user mentioned the formula name and the corresponding Chinese medicine name, then use the formula name as the key and the Chinese medicine name list as the value
2. If the user only mentioned the Chinese medicine name without mentioning the formula, then use each Chinese medicine name as the key and the corresponding Chinese medicine name as a single element list value
3. If the user mentioned to form a formula from multiple Chinese medicines but did not specify the formula name, then use "formula" as the key name
4. If the user mentioned the syndrome information, all syndromes need to be extracted and placed in the syndromes list
5. If the user mentioned the disease information, all diseases need to be extracted and placed in the diseases list. Note: If the user input is the English disease name, the English name needs to be retained without translation.
6. Must strictly return the JSON format below, no need to include any other explanatory text:
{{
    "formula_herbs": {{key-value pairs}},
    "syndromes": [syndrome list],
    "diseases": [disease list],
    
}}

Example input and corresponding output:
Input: "Analyze the target points of当归 and白术"
Output: {{
    "formula_herbs": {{"当归": ["当归"], "白术": ["白术"]}},
    "syndromes": [],
    "diseases": [],
    
}}

Input: "Analyze the target points of补血汤(当归,白术) and阳和汤(人参,甘草), as well as 阴虚火旺 syndrome"
Output: {{
    "formula_herbs": {{"补血汤": ["当归", "白术"], "阳和汤": ["人参", "甘草"]}},
    "syndromes": ["阴虚火旺"],
    "diseases": [],
    
}}

输入："分析当归和白术这两个药组成的方子的靶点，以及证候亡阳和外感，以及疾病甲状腺癌Thyroid Cancer"
输出：{{
    "formula_herbs": {{"方剂": ["当归", "白术"]}},
    "syndromes": ["亡阳", "外感"],
    "diseases": ["Thyroid Cancer"],
    
}}

Remember:
- The return must be a valid JSON format
- Only return the extracted JSON data, no need to include any explanatory text
- Both Chinese medicine name and formula name need to be enclosed in double quotes
- Syndrome information needs to be placed in the syndrome list
- Disease information needs to be placed in the disease list
- Drug information needs to be placed in the drugs list
- If the user did not mention Chinese medicine information, return empty dictionary{{}}
- If the user did not mention syndrome information, return empty list[]
- If no disease information was extracted, return empty list[]
- Even if the user's expression method is different, the specified format needs to be accurately extracted and returned
- If the user did not mention any valid information or analysis requirements, nothing will be returned, and the system will prompt the user to re-enter information, and enter the idle chat mode with the user until the user's valid information is captured.
"""
        
    def process(self, user_input: str) -> Dict[str, List[str]]:
        try:
            start_time = time.time()
            
            # 系统开始自动进入闲聊模式
            chat_start = time.time()
            chat_response = self.llm.invoke(self.chat_prompt.format(user_input=user_input))
            response = chat_response.content.strip()
            self.timing_stats['chat_processing'] = time.time() - chat_start

            # 检查用户是否想要退出系统
            if response == "SYSTEM_EXIT":
                self.timing_stats['total_processing'] = time.time() - start_time
                return {
                    "status": "exit",
                    "message": "Thank you for using, goodbye!",
                    "formula_herbs": {},
                    "syndromes": [],
                    "diseases": [],
                    "timing_stats": self.timing_stats
                }
            
            # 如果用户想要进行分析
            elif response == "ANALYSIS_MODE":
                analysis_start = time.time()
                self.analysis_mode = True
                self.chat_mode = False

                analysis_response = self.llm.invoke(self.prompt_template.format(user_input=user_input))
                response_text = analysis_response.content.strip()
                self.timing_stats['analysis_processing'] = time.time() - analysis_start
                
                try:
                    # 修改这里：改进JSON清理和解析逻辑
                    # 1. 移除多余的换行和空格
                    cleaned_response = re.sub(r'\s+', ' ', response_text)
                    # 2. 确保JSON格式正确
                    cleaned_response = re.sub(r',\s*}', '}', cleaned_response)
                    # 3. 移除可能存在的额外逗号
                    cleaned_response = re.sub(r',(\s*[}\]])', r'\1', cleaned_response)
                    
                    # 解析JSON
                    result = json.loads(cleaned_response)
                    
                    # 确保所有必需的键都存在
                    result = {
                        "formula_herbs": result.get("formula_herbs", {}),
                        "syndromes": result.get("syndromes", []),
                        "diseases": result.get("diseases", [])
                    }
                    
                    # 在返回结果前添加计时统计
                    self.timing_stats['total_processing'] = time.time() - start_time
                    result['timing_stats'] = self.timing_stats
                    return result
                        
                except json.JSONDecodeError as e:
                    logging.error(f"JSON parsing error: {str(e)}")
                    logging.error(f"Original response text: {response_text}")
                    logging.error(f"Cleaned response text: {cleaned_response}")
                    # 尝试手动构建结果
                    try:
                        # 使用正则表达式提取关键信息
                        formula_herbs_match = re.search(r'"formula_herbs":\s*({[^}]+})', cleaned_response)
                        syndromes_match = re.search(r'"syndromes":\s*(\[[^\]]+\])', cleaned_response)
                        diseases_match = re.search(r'"diseases":\s*(\[[^\]]+\])', cleaned_response)
                        
                        result = {
                            "formula_herbs": json.loads(formula_herbs_match.group(1)) if formula_herbs_match else {},
                            "syndromes": json.loads(syndromes_match.group(1)) if syndromes_match else [],
                            "diseases": json.loads(diseases_match.group(1)) if diseases_match else []
                        }
                        return result
                    except Exception:
                        return {
                            "status": "error",
                            "error": "JSON parsing failed",
                            "formula_herbs": {},
                            "syndromes": [],
                            "diseases": []
                        }
            else:
                # 修改这里：正确处理闲聊模式的返回
                self.chat_mode = True
                self.analysis_mode = False
                self.timing_stats['total_processing'] = time.time() - start_time
                return {
                    "status": "chat",
                    "chat_mode": True,
                    "message": response,
                    "formula_herbs": {},
                    "syndromes": [],
                    "diseases": [],
                    "timing_stats": self.timing_stats
                }
                
        except Exception as e:
            self.timing_stats['total_processing'] = time.time() - start_time
            logging.error(f"Input processing error: {str(e)}")
            logging.error(traceback.format_exc())
            return {
                "status": "error",
                "error": f"Input processing failed: {str(e)}",
                "formula_herbs": {},
                "syndromes": [],
                "diseases": [],
                "timing_stats": self.timing_stats
            }

# 创建中药靶点分析Agent
class HerbTargetAnalysisAgent:
    def __init__(self, tool: Tool):
        self.tool = tool
    
    def analyze(self, herbs_dict: Dict[str, List[str]]) -> Dict:
        try:
            # 直接调用工具并返回结果
            logging.debug(f"Analyzing Chinese medicine: {herbs_dict}")
            results = self.tool.func(herbs_dict)
            logging.debug(f"Results: {results}")
            return results
            
        except Exception as e:
            logging.error(f"Error: {str(e)}")
            logging.error(traceback.format_exc())
            return {"error": f"Target analysis failed: {str(e)}"}

# 定义证候靶点查找工具
class SyndromeTool:
    def __init__(self, tool_path: str = "D:/HerbAgent/tools/"):
        import sys
        import os
        
        # 添加工具目录到系统路径
        tool_dir = os.path.dirname(tool_path)
        if tool_dir not in sys.path:
            sys.path.append(tool_dir)
            
        from tools.Syndrome_target_search import search_targets
        self.searcher = search_targets
    
    def search_targets(self, syndromes: List[str]) -> Dict:
        """
        搜索证候靶点信息
        
        Args:
            syndromes: 证候列表，如 ["亡阳", "外感"]
        
        Returns:
            Dict: 包含靶点信息的字典
        """
        try:
            results = self.searcher(syndromes)
            return results
            
        except Exception as e:
            logging.error(f"Search syndrome target error: {str(e)}")
            logging.error(traceback.format_exc())
            return {"error": f"Search syndrome target error: {str(e)}"}

# 将证候靶点工具封装为 LangChain Tool
def create_syndrome_target_tool() -> Tool:
    syndrome_tool = SyndromeTool()
    return Tool(
        name="syndrome_target_search",
        func=syndrome_tool.search_targets,
        description="Search the target information of syndrome. Input format is a syndrome list."
    )

# 创建证候靶点分析Agent
class SyndromeTargetAnalysisAgent:
    def __init__(self, tool: Tool):
        self.tool = tool
    
    def analyze(self, syndromes: List[str]) -> Dict:
        try:
            # 直接调用工具并返回结果
            logging.debug(f"Analyzing syndrome: {syndromes}")
            results = self.tool.func(syndromes)
            logging.debug(f"Results: {results}")
            return results
            
        except Exception as e:
            logging.error(f"Error: {str(e)}")
            logging.error(traceback.format_exc())
            return {"error": f"Syndrome target analysis failed: {str(e)}"}

# 定义疾病靶点查找工具
class DiseaseTool:
    def __init__(self, tool_path: str = "D:/HerbAgent/tools/"):
        import sys
        import os
        
        # 添加工具目录到系统路径
        tool_dir = os.path.dirname(tool_path)
        if tool_dir not in sys.path:
            sys.path.append(tool_dir)
            
        from tools.Disease_target_search import DiseaseTargetSearch
        self.searcher = DiseaseTargetSearch()
    
    def search_targets(self, diseases: List[str]) -> Dict:
        """
        搜索疾病靶点信息
        
        Args:
            diseases: 疾病列表，如 ["甲状腺癌"]
        
        Returns:
            Dict: 包含靶点信息的字典
        """
        try:
            results = self.searcher.process(
                diseases,
                include_genecards=True
            )
            return results
            
        except Exception as e:
            logging.error(f"Search disease target error: {str(e)}")
            logging.error(traceback.format_exc())
            return {"error": f"Search disease target error: {str(e)}"}

# 将疾病靶点工具封装为 LangChain Tool
def create_disease_target_tool() -> Tool:
    disease_tool = DiseaseTool()
    return Tool(
        name="disease_target_search",
        func=disease_tool.search_targets,
        description="Search the target information of disease. Input format is a disease list."
    )

# 创建疾病靶点分析Agent
class DiseaseTargetAnalysisAgent:
    def __init__(self, tool: Tool):
        self.tool = tool
    
    def analyze(self, diseases: List[str]) -> Dict:
        try:
            # 直接调用工具并返回结果
            logging.debug(f"Analyzing disease: {diseases}")
            results = self.tool.func(diseases)
            logging.debug(f"Results: {results}")
            return results
            
        except Exception as e:
            logging.error(f"Error: {str(e)}")
            logging.error(traceback.format_exc())
            return {"error": f"Disease target analysis failed: {str(e)}"}

# 定义药物靶点查找工具
class DrugTool:
    def __init__(self, tool_path: str = "D:/HerbAgent/tools/"):
        import sys
        import os
        
        # 添加工具目录到系统路径
        tool_dir = os.path.dirname(tool_path)
        if tool_dir not in sys.path:
            sys.path.append(tool_dir)
            
        from tools.drug_target import search_disease_drug_targets
        self.searcher = search_disease_drug_targets
    
    def search_targets(self, disease_keywords: List[str]) -> Dict:
        """
        搜索疾病相关药物靶点信息
        
        Args:
            disease_keywords: 疾病关键词列表，如 ["Thyroid Cancer", "Thyroid carcinoma"]
        
        Returns:
            Dict: 包含靶点信息的字典
        """
        try:
            results = self.searcher(disease_keywords)
            return results
            
        except Exception as e:
            logging.error(f"Search drug target error: {str(e)}")
            logging.error(traceback.format_exc())
            return {"error": f"Search drug target error: {str(e)}"}

# 将药物靶点工具封装为 LangChain Tool
def create_drug_target_tool() -> Tool:
    drug_tool = DrugTool()
    return Tool(
        name="drug_target_search",
        func=drug_tool.search_targets,
        description="Search the target information of drug related to disease. Input format is a disease keyword list."
    )

# 创建药物靶点分析Agent
class DrugTargetAnalysisAgent:
    def __init__(self, tool: Tool):
        self.tool = tool
    
    def analyze(self, diseases: List[str]) -> Dict:
        try:
            # 添加更详细的日志
            logging.info(f"Start drug target analysis, disease list: {diseases}")
            
            # 直接调用工具并返回结果
            logging.debug(f"Analyzing drug related to disease: {diseases}")
            results = self.tool.func(diseases)
            logging.info(f"Drug target analysis completed, results: {results}")
            return results
            
        except Exception as e:
            error_msg = f"Drug target analysis failed: {str(e)}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            return {"error": error_msg}

# 定义PPI分析工具
class PPITool:
    def __init__(self):
        import sys
        import os
        
        # 添加项目根目录到系统路径
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(PROJECT_ROOT)
        
        # 导入PPI pipeline模块
        from tools.PPI_pipeline import run_ppi_pipeline
        self.pipeline = run_ppi_pipeline
    
    def run_pipeline(self) -> Dict:
        try:
            # 运行pipeline并获取结果
            success, results = self.pipeline()
            
            if not success:
                logging.error("PPI pipeline returned failure status")
                return {
                    "status": "error",
                    "error": "PPI pipeline execution failed"
                }
                
            # 解包结果，忽略输出路径
            result_data, _ = results
            
            if result_data is None:
                logging.error("PPI analysis result is empty")
                return {
                    "status": "error",
                    "error": "PPI analysis result is empty"
                }
            
            print(f"\nSuccessfully obtained PPI analysis results, data item count: {len(result_data)}")
            return {
                "status": "success",
                "results": result_data
            }
            
        except Exception as e:
            error_msg = f"PPI analysis process error: {str(e)}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            return {
                "status": "error",
                "error": error_msg
            }

# 创建PPI分析Agent
class PPIPipelineAgent:
    def __init__(self, tool: Tool, report_agent=None):
        self.tool = tool
        self.report_agent = report_agent
    
    def analyze(self) -> Dict:
        try:
            # 移除 debug 日志，改用 print 输出状态
            print("\nStarting PPI network analysis...")
            response = self.tool.func()
            
            if response.get("status") == "error":
                return {
                    "status": "error",
                    "error": response.get("error", "PPI analysis failed")
                }
            
            results = response.get("results")
            
            # 修改这里：正确处理DataFrame类型的结果
            if isinstance(results, pd.DataFrame):
                if results.empty:
                    return {
                        "status": "error",
                        "error": "No valid PPI analysis results obtained"
                    }
                print(f"\nPPI analysis completed, analyzed {len(results)} target relationships")
                
                # 生成分析报告
                analysis_report = None
                if self.report_agent:
                    print("\nGenerating PPI analysis report...")
                    analysis_report = self.report_agent.analyze_results(results)
                
                # 修改返回格式，确保报告被包含在返回结果中
                return {
                    "status": "success",
                    "results": results.to_dict('records'),  # 转换DataFrame为可序列化格式
                    "analysis_report": analysis_report.get("report") if analysis_report else None,
                    "message": "PPI analysis completed successfully"
                }
            else:
                return {
                    "status": "error",
                    "error": "PPI analysis result format incorrect"
                }
            
        except Exception as e:
            print(f"\nError during PPI analysis: {str(e)}")
            return {
                "status": "error",
                "error": f"PPI analysis failed: {str(e)}"
            }

# 将PPI工具封装为 LangChain Tool
def create_ppi_tool() -> Tool:
    ppi_tool = PPITool()
    return Tool(
        name="ppi_pipeline",
        func=ppi_tool.run_pipeline,
        description="Run PPI network analysis process"
    )

# 定义结果分析报告 Agent
class PPIReportAgent:
    def __init__(self, llm):
        self.llm = llm
        self.analysis_prompt = """You are a professional medical statistical analysis expert. Please analyze the PPI network analysis results and combine drug and disease information to focus on statistical information (such as Z-Value and P-Value), and provide professional analysis insights.

Analysis data:
{ppi_results}

Please analyze according to the following structure:
1. Statistical overview
   - Summarize statistical indicators in the data
   - Focus on the distribution characteristics of Z-Value and P-Value

2. Significance analysis
   - Identify and summarize significant associations
   - Focus on the most significant few associations

3. Biological significance
   - Key node identification
   - Potential biological pathways
   - Possible mechanisms of action

4. Conclusion and suggestions
   - Main findings
   - Research limitations
   - Suggestions for subsequent research

Please write a professional but easy-to-understand analysis report.
"""

    def analyze_results(self, ppi_results: pd.DataFrame) -> Dict:
        """分析PPI结果并生成报告"""
        try:
            print("\nStart analyzing PPI network results...")
            
            # 将DataFrame转换为字典列表
            results_dict = ppi_results.to_dict('records')
            
            # 生成一个更易读的字符串表示
            readable_results = "\n".join([
                f"file_name: {row['file_name']}\n"
                f"drug: {row['drug']}\n"
                f"disease: {row['disease']}\n"
                f"z-value: {row['z-value']:.4f}\n"
                f"p-value: {row['p-value']:.4f}\n"
                for row in results_dict
            ])
            
            report_response = self.llm.invoke(
                self.analysis_prompt.format(
                    ppi_results=readable_results
                )
            )
            

            print("\nAnalysis report generation completed:\n", report_response.content)
            
            return {
                "status": "success",
                "report": report_response.content,
                "message": "Analysis report generation completed"
            }
            
            

        except Exception as e:
            logging.error(f"Result analysis process error: {str(e)}")
            logging.error(traceback.format_exc())
            return {
                "status": "error",
                "error": f"Result analysis failed: {str(e)}"
            }

# 定义重启随机游走分析工具
class RandomWalkTool:
    def __init__(self, tool_path: str = "D:/HerbAgent/Random_Walk/"):
        import sys
        import os
        
        # 添加工具目录到系统路径
        tool_dir = os.path.dirname(tool_path)
        if tool_dir not in sys.path:
            sys.path.append(tool_dir)
            
         # 修改导入语句
        from Random_Walk.random_walk_results import run_analysis
        self.rw_main = run_analysis  # 使用 run_analysis 函数
    
    def run_analysis(self) -> Dict:
        """
        运行重启随机游走分析
        
        Returns:
            Dict: 包含分析结果的字典
        """
        try:
            # 运行分析并获取结果
            results = self.rw_main()  # 返回值包含 'success', 'summary_df', 'transformed_df', 'message'
            
            # 添加详细的日志记录
            logging.debug(f"Random walk original result type: {type(results)}")
            
            # 检查结果格式
            if not isinstance(results, dict):
                raise ValueError("Return result is not a dictionary format")
            
            if not results['success']:
                return {
                    "status": "error",
                    "error": results['message']
                }
            
            # 获取转换后的数据框
            transformed_df = results.get('transformed_df')
            if transformed_df is None:
                raise ValueError("No transformed data found")
                
            logging.debug(f"transformed_df类型: {type(transformed_df)}")
            
            # 确保数据是DataFrame格式并转换
            if isinstance(transformed_df, pd.DataFrame):
                data_dict = transformed_df.to_dict('records')
                logging.debug(f"Transformed data example: {data_dict[:2] if data_dict else 'empty'}")
                
                return {
                    "status": "success",
                    "results": data_dict  # 返回可序列化的字典列表
                }
            else:
                raise ValueError(f"Expected DataFrame format, actually obtained: {type(transformed_df)}")
            
        except Exception as e:
            logging.error(f"Error occurred during random walk analysis: {str(e)}")
            logging.error(traceback.format_exc())
            return {
                "status": "error",
                "error": f"Random walk analysis failed: {str(e)}"
            }

# 将重启随机游走工具封装为 LangChain Tool
def create_random_walk_tool() -> Tool:
    rw_tool = RandomWalkTool()
    return Tool(
        name="random_walk_analysis",
        func=rw_tool.run_analysis,
        description="Run random walk analysis"
    )

# 创建重启随机游走分析Agent
class RandomWalkAnalysisAgent:
    def __init__(self, tool: Tool, report_agent=None):
        self.tool = tool
        self.report_agent = report_agent
    
    def analyze(self) -> Dict:
        try:
            # logging.debug("Start random walk analysis")
            response = self.tool.func()
            # logging.debug(f"RandomWalkTool returned result: {response}")
            
            if response.get("status") == "error":
                return response
            
            results = response.get("results")
            if not results:
                return {
                    "status": "error",
                    "error": "No valid random walk analysis results obtained"
                }
            
            # 生成分析报告
            analysis_report = None
            if self.report_agent:
                print("\nGenerating random walk analysis report...")
                analysis_report = self.report_agent.analyze_results(results)
                # logging.debug(f"Random walk analysis report generation result: {analysis_report}")
            
            return {
                "status": "success",
                "results": results,
                "analysis_report": analysis_report,
                "message": "Random walk analysis completed, results saved"
            }
            
        except Exception as e:
            logging.error(f"Random walk analysis failed: {str(e)}")
            logging.error(traceback.format_exc())
            return {
                "status": "error",
                "error": f"Random walk analysis failed: {str(e)}"
            }

# 定义重启随机游走结果分析报告 Agent
class RandomWalkReportAgent:
    def __init__(self, llm):
        self.llm = llm
        self.analysis_prompt = """You are a professional network pharmacology analysis expert. This is an analysis result containing Top10 chemical components (Ingredients) and their core flavors in formulas, please analyze and interpret the results of restarting random walk analysis, focusing on the importance ranking (Total_Score) and Top_Ingredients_Count.

Analysis data:
{rw_results}

Please analyze and interpret according to the following structure:
1. Core chemical component analysis and its core flavor analysis in formulas
   - Identify and explain the most important core chemical components and core flavors
   - Analyze the relationship between them

2. Feature analysis
   - Analyze the overall characteristics of core chemical components and their formulas
   - Identify key information

3. Biological significance
   - Explain the biological functions of core chemical components and core flavors
   - Analyze possible mechanisms of action

4. Conclusion and suggestions
   - Main findings
   - Potential research directions
   - Application suggestions

Please write a professional but easy-to-understand analysis report.
"""

    def analyze_results(self, rw_results: List[Dict]) -> Dict:
        """分析重启随机游走结果并生成报告"""
        try:
            print("\nStart analyzing random walk results...")
            
            # 将结果转换为更易读的格式
            readable_results = json.dumps(rw_results, ensure_ascii=False, indent=2)
            
            report_response = self.llm.invoke(
                self.analysis_prompt.format(
                    rw_results=readable_results
                )
            )
            
        
            print("\nRestart random walk analysis report generated\n", report_response.content)
            
            return {
                "status": "success",
                "report": report_response.content,
                "message": "Analysis report generated"
            }
            
        except Exception as e:
            logging.error(f"Error during result analysis: {str(e)}")
            logging.error(traceback.format_exc())
            return {
                "status": "error",
                "error": f"Failed to analyze results: {str(e)}"
            }

# 主流程协调器
class HerbAnalysisCoordinator:
    def __init__(self):
        self.llm = llm
        # 保存项目根目录
        self.project_root = "D:/HerbAgent"
        self.original_working_dir = os.getcwd()
        
        # 初始化各个Agent时确保在正确的工作目录
        os.chdir(self.project_root)
        
        self.input_agent = InputProcessingAgent(self.llm)
        
        # 创建PPI相关组件（确保在正确目录下初始化）
        self.herb_tool = create_herb_target_tool()
        self.herb_analysis_agent = HerbTargetAnalysisAgent(self.herb_tool)
        self.syndrome_tool = create_syndrome_target_tool()
        self.syndrome_analysis_agent = SyndromeTargetAnalysisAgent(self.syndrome_tool)
        self.disease_tool = create_disease_target_tool()
        self.disease_analysis_agent = DiseaseTargetAnalysisAgent(self.disease_tool)
        self.drug_tool = create_drug_target_tool()
        self.drug_analysis_agent = DrugTargetAnalysisAgent(self.drug_tool)
        
        # 创建报告agents
        self.ppi_report_agent = PPIReportAgent(self.llm)
        self.rw_report_agent = RandomWalkReportAgent(self.llm)
        
        # 创建PPI工具时确保在主目录
        os.chdir(self.project_root)
        self.ppi_tool = create_ppi_tool()
        self.ppi_agent = PPIPipelineAgent(self.ppi_tool, self.ppi_report_agent)
        
        # 创建随机游走工具时切换到Random_Walk目录
        os.chdir(os.path.join(self.project_root, "Random_Walk"))
        self.rw_tool = create_random_walk_tool()
        self.rw_agent = RandomWalkAnalysisAgent(self.rw_tool, self.rw_report_agent)
        
        # 创建交互agent
        self.interaction_agent = AnalysisInteractionAgent(
            self.llm, 
            ppi_agent=self.ppi_agent,
            rwr_agent=self.rw_agent
        )
        
        # 最后恢复到原始工作目录
        os.chdir(self.original_working_dir)
    
    def process(self, user_input: str) -> Dict:
        try:
            parsed_data = self.input_agent.process(user_input)
            
            # 首先检查是否是退出状态
            if parsed_data.get("status") == "exit":
                return {
                "status": "exit",
                "message": "Thank you for using, goodbye!"
                }
            
            # 处理错误状态
            if "error" in parsed_data:
                return {
                    "status": "error",
                    "error": parsed_data["error"]
                }       
            
            # 2. 处理闲聊模式
            if parsed_data.get("chat_mode"):
                return {
                    "status": "chat",
                    "message": parsed_data.get("message", ""),
                    "chat_mode": True
                }
            
           # 3. 处理分析模式
            # 验证是否有足够的分析数据
            has_analysis_data = any([
                parsed_data.get("formula_herbs"),
                parsed_data.get("syndromes"),
                parsed_data.get("diseases")
            ])
        
            if not has_analysis_data:
                return {
                    "status": "no_data",
                    "message": "No valid analysis data detected, please re-enter",
                    "chat_mode": False
                }
            
            # 4. 执行分析流程
            logging.debug(f"Processing analysis data: {parsed_data}")
            
            # 修改这部分：先进行疾病分析，如果成功则一定执行药物分析
            disease_analysis = None
            drug_analysis = None
            if parsed_data["diseases"]:
                logging.debug(f"Start disease analysis, disease list: {parsed_data['diseases']}")
                disease_analysis = self.disease_analysis_agent.analyze(parsed_data["diseases"])
                logging.debug(f"Disease analysis result: {disease_analysis}")
                
                # 修改这里：只要疾病分析成功就执行药物分析，不需要额外的条件判断
                if disease_analysis and not isinstance(disease_analysis.get("error", None), str):
                    logging.debug("Start drug analysis")
                    drug_analysis = self.drug_analysis_agent.analyze(parsed_data["diseases"])
                    logging.debug(f"Drug analysis result: {drug_analysis}")
            
            # 然后执行其他分析
            herb_analysis = self.herb_analysis_agent.analyze(parsed_data["formula_herbs"])
            
            syndrome_analysis = None
            if parsed_data["syndromes"]:
                syndrome_analysis = self.syndrome_analysis_agent.analyze(parsed_data["syndromes"])
            
            # 3. 整理所有分析结果
            analysis_results = {
                "herb_analysis": herb_analysis,
                "syndrome_analysis": syndrome_analysis,
                "disease_analysis": disease_analysis,
                "drug_analysis": drug_analysis
            }
            
            # 4. 进入交互式分析循环
            while True:
                interaction_result = self.interaction_agent.interact(analysis_results)
                
                # 处理退出状态
                if interaction_result.get("status") == "exit":
                    return {
                        "status": "exit",
                        "message": "Thank you for using, goodbye!"
                    }
                
                if interaction_result.get("status") == "error":
                    return interaction_result
                
                # 处理新增的状态
                if interaction_result.get("status") == "restart":
                    return {
                        "status": "restart",
                        "message": "Please re-enter analysis information"
                    }
                
                # 如果没有可用的分析或用户选择跳过，退出循环
                if not self.interaction_agent.get_available_analyses() or \
                   interaction_result.get("status") == "skipped":
                    break
                
                # 更新分析结果
                if interaction_result.get("ppi_results"):
                    analysis_results["ppi_analysis"] = {
                        "status": "success",
                        "analysis_report": interaction_result["ppi_results"].get("analysis_report", {}).get("report", ""),
                        "results": interaction_result["ppi_results"].get("results", [])
                    }
                if interaction_result.get("rwr_results"):
                    analysis_results["rwr_analysis"] = {
                        "status": "success",
                        "analysis_report": interaction_result["rwr_results"].get("analysis_report", {}).get("report", ""),
                        "results": interaction_result["rwr_results"].get("results", [])
                    }
            
            # 5. 返回最终分析结果
            result = {
                "status": "completed",
                "analysis_reports": {
                    "ppi_analysis": analysis_results.get("ppi_analysis", {}),
                    "rwr_analysis": analysis_results.get("rwr_analysis", {})
                },
                "message": "All analyses completed"
            }
            
            # 修改结果处理部分
            # if 'analysis_reports' in result:
            #     # 确保两种分析报告都被正确处理
            #     if result['analysis_reports'].get('ppi_analysis'):
            #         print("\nPPI Network Analysis Report:")
            #         print(json.dumps(result['analysis_reports']['ppi_analysis'], 
            #                        ensure_ascii=False, indent=2))
                    
            #     if result['analysis_reports'].get('rwr_analysis'):
            #         print("\nRandom Walk Analysis Report:")
            #         print(json.dumps(result['analysis_reports']['rwr_analysis'], 
            #                        ensure_ascii=False, indent=2))
            
            return result
            
        except Exception as e:
            logging.error(f"Analysis process error: {str(e)}")
            logging.error(traceback.format_exc())
            return {
                "status": "error",
                "error": f"Analysis failed: {str(e)}"
            }
        finally:
            # 确保最后恢复到原始工作目录
            os.chdir(self.original_working_dir)

    

# 创建分析交互Agent
class AnalysisInteractionAgent:
    def __init__(self, llm, ppi_agent=None, rwr_agent=None):
        self.llm = llm
        self.ppi_agent = ppi_agent
        self.rwr_agent = rwr_agent
        self.analysis_state = {
            "ppi_executed": False,
            "rwr_executed": False
        }
        self.timing_stats = {
            'prompt_generation': 0,
            'llm_processing': 0,
            'analysis_execution': 0,
            'total_interaction': 0
        }
        
        # 添加退出意图检测提示
        self.exit_check_prompt = """As an AI assistant, please judge whether the user's input expresses the intention to exit the system.

User input: {user_input}

If the user expresses the intention to exit, end the conversation, or not continue, return "EXIT"
If the user is in normal conversation or asking, return "CONTINUE"

Only return "EXIT" or "CONTINUE", no other text.
"""

        self.initial_prompt_template = """Based on the completed target analysis results, ask the user whether they need further analysis in English.

Completed analysis results:
{analysis_results}

Please ask the user in English whether they need the following further analysis:
1. PPI network proximity analysis
2. Restart random walk analysis
3. Exit system

Please describe the current results and ask the user's choice in concise language by using English.
"""
        self.insufficient_data_prompt = """You are a professional Traditional Chinese Medicine analysis assistant. The current situation is as follows:

Current completed analysis:
{current_results}

To perform further network analysis, we need to meet one of the following conditions:
1. Herb result + disease result
2. Herb result + syndrome result
3. Herb result + syndrome result + disease result

Remind the user in English that they can:
1. Supplement more information to continue analysis
2. Exit system

Please explain the current situation in English to the user in a friendly way and ask the user whether they want to supplement more information to continue analysis, or end this conversation.
"""

# 添加意图判断提示模板
        self.intent_analysis_prompt = """As an AI assistant, please analyze the user's reply to express what intention.

User reply: {user_input}

Possible intentions:
1. Continue analysis: User indicates wanting to continue and supplement more information
2. End conversation: User indicates wanting to end the current conversation

Please analyze the user's intention and return one of the following options:
- "continue": If the user wants to continue and supplement information
- "exit": If the user wants to end the conversation

Only return one of the above options, no other text.
"""
        self.follow_up_prompt_template = """Based on the current analysis status, provide the next step choice for the user.

Completed analysis:
{completed_analyses}

Optional next analysis:
{available_analyses}

Exit system

Please use concise language in English to ask the user whether they want to perform the remaining analysis. Remind the user that they can enter "exit" at any time to end analysis.
"""

        self.decision_prompt = """As an analysis assistant, please judge the user's reply to express what analysis intention.

User reply: {user_input}

Current available analysis options:
{available_options} 
Or exit system

Please analyze the user's intention and give a judgment:
1. If the user expresses the intention to perform PPI network analysis, return "execute_ppi"
2. If the user expresses the intention to perform restart random walk analysis, return "execute_rwr"
3. If the user expresses the intention to exit the system or not want to continue analysis, return "exit"


Only return one of the above options, no other text.
"""
        

    def get_available_analyses(self) -> List[str]:
        """获取当前可用的分析选项"""
        available = []
        if not self.analysis_state["ppi_executed"]:
            available.append("PPI network proximity analysis")
        if not self.analysis_state["rwr_executed"]:
            available.append("Restart random walk analysis")
        return available

    def get_completed_analyses(self) -> List[str]:
        """获取已完成的分析列表"""
        completed = []
        if self.analysis_state["ppi_executed"]:
            completed.append("PPI network proximity analysis")
        if self.analysis_state["rwr_executed"]:
            completed.append("Restart random walk analysis")
        return completed

    def _validate_analysis_requirements(self, analysis_results: Dict) -> Dict:
        """验证分析结果是否满足继续分析的要求"""
        # 检查各个分析结果是否存在且有效
        has_herb = (analysis_results.get("herb_analysis") and 
                   "error" not in analysis_results["herb_analysis"])
        has_syndrome = (analysis_results.get("syndrome_analysis") and 
                       "error" not in analysis_results["syndrome_analysis"])
        has_disease = (analysis_results.get("disease_analysis") and 
                      "error" not in analysis_results["disease_analysis"])

        # 检查是否满足继续分析的条件
        valid_combinations = [
            has_herb and has_disease,  # 中药 + 疾病
            has_herb and has_syndrome,  # 中药 + 证候
            has_herb and has_syndrome and has_disease  # 中药 + 证候 + 疾病
        ]

        current_results = []
        if has_herb:
            current_results.append("Herb analysis")
        if has_syndrome:
            current_results.append("Syndrome analysis")
        if has_disease:
            current_results.append("Disease analysis")

        return {
            "is_valid": any(valid_combinations),
            "current_results": ", ".join(current_results) if current_results else "No valid results"
        }

    def interact(self, analysis_results: Dict) -> Dict:
        try:
            interaction_start = time.time()
            
            # 添加调试日志，查看传入的数据
            logging.debug(f"interact method received analysis_results: {analysis_results}")
            
            # 验证分析要求
            validation_result = self._validate_analysis_requirements(analysis_results)
            
            if not validation_result["is_valid"]:
                # 记录提示生成时间
                prompt_start = time.time()
                response = self.llm.invoke(
                    self.insufficient_data_prompt.format(
                        current_results=validation_result["current_results"]
                    )
                )
                self.timing_stats['prompt_generation'] = time.time() - prompt_start
                print("\n" + response.content)
                
                # 获取用户自然语言回复
                user_response = input("> ")
                
                # 记录LLM处理时间
                llm_start = time.time()
                intent_response = self.llm.invoke(
                    self.intent_analysis_prompt.format(
                        user_input=user_response
                    )
                )
                self.timing_stats['llm_processing'] = time.time() - llm_start
                user_intent = intent_response.content.strip()
                
                self.timing_stats['total_interaction'] = time.time() - interaction_start
                
                if user_intent == "continue":
                    return {
                        "status": "restart",
                        "message": "OK, let's continue to supplement more information.",
                        "timing_stats": self.timing_stats
                    }
                else:  # user_intent == "exit"
                    return {
                        "status": "exit",
                        "message": "Thank you for using, goodbye!",
                        "timing_stats": self.timing_stats
                    }

            # 验证通过后，继续原有的交互逻辑
            if not self._verify_analysis_completion(analysis_results):
                self.timing_stats['total_interaction'] = time.time() - interaction_start
                return {
                    "message": "Some analysis is not completed, please wait for all analyses to complete before further operation.",
                    "status": "incomplete",
                    "timing_stats": self.timing_stats
                }
            
            # 修改这里：在传入LLM之前截断或简化分析结果
            def truncate_analysis_results(results: Dict) -> Dict:
                logging.debug(f"truncate_analysis_results received data: {results}")
                truncated = {}
                
                # 添加证候ID到名称的转换函数
                def convert_syndrome_id_to_name(syndrome_ids: List[str]) -> List[str]:
                    try:
                        syndrome_map_df = pd.read_excel(
                            "D:/HerbAgent/data/SymMap_syndrome_name.xlsx",
                            usecols=[0, 1]
                        )
                        id_to_name = dict(zip(
                            syndrome_map_df.iloc[:, 0].astype(str),
                            syndrome_map_df.iloc[:, 1]
                        ))
                        syndrome_names = [
                            id_to_name.get(str(sid), f"Unknown Syndrome ID:{sid}") 
                            for sid in syndrome_ids
                        ]
                        return syndrome_names
                    except Exception as e:
                        logging.error(f"Syndrome ID conversion failed: {str(e)}")
                        return syndrome_ids
                
                # 检查herb_analysis
                if results.get("herb_analysis"):
                    herb_data = results["herb_analysis"]
                    # 只显示方剂名，不显示靶点数据
                    truncated["herb_analysis"] = f"Completed herb analysis: {list(herb_data.keys())}"
                    logging.debug("Herb target data omitted")
                else:
                    logging.debug("No herb data found")
                    truncated["herb_analysis"] = "Completed herb analysis"
                
                # 检查syndrome_analysis
                if results.get("syndrome_analysis"):
                    syndrome_data = results["syndrome_analysis"]
                    syndrome_ids = list(syndrome_data.keys())
                    syndrome_names = convert_syndrome_id_to_name(syndrome_ids)
                    # 只显示证候名，不显示靶点数据
                    truncated["syndrome_analysis"] = f"Completed syndrome analysis: {syndrome_names}"
                    logging.debug("Syndrome target data omitted")
                else:
                    logging.debug("No syndrome data found")
                    truncated["syndrome_analysis"] = "Completed syndrome analysis"
                
                # 检查disease_analysis
                if results.get("disease_analysis") is not None:
                    # 只显示疾病分析状态，不显示靶点数据
                    truncated["disease_analysis"] = "Completed disease analysis"
                    logging.debug("Disease target data omitted")
                else:
                    logging.debug("No disease data found")
                    truncated["disease_analysis"] = "Completed disease analysis"
                
                # 检查drug_analysis
                if results.get("drug_analysis") is not None:
                    # 只显示药物分析状态，不显示靶点数据
                    truncated["drug_analysis"] = "Completed drug analysis"
                    logging.debug("Drug target data omitted")
                else:
                    logging.debug("No drug data found")
                    truncated["drug_analysis"] = "Completed drug analysis"
                
                logging.debug(f"Truncated result (target data omitted): {truncated}")
                return truncated
            
            prompt_start = time.time()  # 记录提示生成时间
            if not any(self.analysis_state.values()): # 如果所有分析都未完成
                logging.debug("Preparing to truncate analysis results")
                # 直接使用传入的analysis_results
                simplified_results = truncate_analysis_results(analysis_results)
                logging.debug(f"Truncated result: {simplified_results}")
                prompt = self.initial_prompt_template.format(
                    analysis_results=json.dumps(simplified_results, ensure_ascii=False)
                )
            else:
                prompt = self.follow_up_prompt_template.format(
                    completed_analyses=", ".join(self.get_completed_analyses()),
                    available_analyses=", ".join(self.get_available_analyses())
                )
            
            # 生成交互提示
            response = self.llm.invoke(prompt)
            self.timing_stats['prompt_generation'] = time.time() - prompt_start
            print("\n" + response.content)
            
            # 获取用户输入
            choice = input("Please enter your choice > ")
            
            # 记录LLM处理时间
            llm_start = time.time()
            exit_check = self.llm.invoke(self.exit_check_prompt.format(user_input=choice))
            self.timing_stats['llm_processing'] += time.time() - llm_start
            
            if exit_check.content.strip() == "EXIT":
                self.timing_stats['total_interaction'] = time.time() - interaction_start
                return {
                    "status": "exit",
                    "message": "Thank you for using, goodbye!",
                    "timing_stats": self.timing_stats
                }
            
            # 记录分析执行时间
            analysis_start = time.time()
            result = self.process_user_choice(choice, analysis_results)
            self.timing_stats['analysis_execution'] = time.time() - analysis_start
            
            # 如果 process_user_choice 返回退出状态，直接返回
            if result.get("status") == "exit":
                return result
            self.timing_stats['total_interaction'] = time.time() - interaction_start
            
            # 更新分析状态
            if result.get("executed_analysis") == "ppi":
                self.analysis_state["ppi_executed"] = True
            elif result.get("executed_analysis") == "rwr":
                self.analysis_state["rwr_executed"] = True
            # 在返回结果中添加时间统计
            if isinstance(result, dict):
                result['timing_stats'] = self.timing_stats
            
            return result
            
        except Exception as e:
            self.timing_stats['total_interaction'] = time.time() - interaction_start
            logging.error(f"Interaction process error: {str(e)}")
            logging.error(traceback.format_exc())
            return {
                "status": "error",
                "message": f"Interaction process error: {str(e)}",
                "timing_stats": self.timing_stats
            }

    def process_user_choice(self, choice: str, analysis_results: Dict) -> Dict:
        """处理用户对进一步分析的选择"""
        try:
            # 首先检查是否是直接退出命令
            if choice.lower() in ['exit', 'quit']:
                return {
                    "status": "exit",
                    "message": "Thank you for using, goodbye!"
                }

            # 获取当前可用的分析选项
            available_options = self.get_available_analyses()
            
            # 使用LLM判断用户意图
            response = self.llm.invoke(
                self.decision_prompt.format(
                    user_input=choice,
                    available_options=", ".join(available_options)
                )
            )
            decision = response.content.strip()
            
            logging.debug(f"LLM决策结果: {decision}")
            
            # 处理退出意图
            if decision == "exit":
                return {
                    "status": "exit",
                    "message": "Thank you for using, goodbye!"
                }
            
            if decision == "execute_ppi" and not self.analysis_state["ppi_executed"]:
                if self.ppi_agent:
                    print("\nExecuting PPI network analysis...")
                    ppi_results = self.ppi_agent.analyze()
                    
                    # 确保结果包含分析报告
                    if ppi_results.get("status") == "success":
                        return {
                            "status": "success",
                            "executed_analysis": "ppi",
                            "ppi_results": {
                                "analysis_report": ppi_results.get("analysis_report"),
                                "results": ppi_results.get("results"),
                                "message": ppi_results.get("message")
                            },
                            "message": "PPI analysis completed"
                        }
                    else:
                        return ppi_results  # 返回错误状态
                    
            elif decision == "execute_rwr" and not self.analysis_state["rwr_executed"]:
                if self.rwr_agent:
                    logging.debug("Start restart random walk analysis")
                    rwr_results = self.rwr_agent.analyze()
                    logging.debug(f"Restart random walk analysis completed: {rwr_results}")
                    return {
                        "status": "success",
                        "executed_analysis": "rwr",
                        "rwr_results": rwr_results,
                        "message": "Restart random walk analysis completed"
                    }
                else:
                    return {
                        "status": "error",
                        "message": "Restart random walk analysis agent not initialized"
                    }
            else:
                return {
                    "status": "skipped",
                    "message": "User chose to skip analysis"
                }
                
        except Exception as e:
            logging.error(f"Error processing user choice: {str(e)}")
            logging.error(traceback.format_exc())
            return {
                "status": "error",
                "message": f"Failed to process: {str(e)}"
            }

    def _verify_analysis_completion(self, analysis_results: Dict) -> bool:
        """验证所有必要的分析是否都已完成"""
        # 检查每个分析结果是否存在且不包含错误
        required_analyses = [
            "herb_analysis",
            "syndrome_analysis",
            "disease_analysis",
            "drug_analysis"
        ]
        
        for analysis in required_analyses:
            if analysis not in analysis_results:
                return False
            if analysis_results[analysis] is not None and "error" in analysis_results[analysis]:
                return False
                
        return True



# 使用示例
def main():
    coordinator = HerbAnalysisCoordinator()
    
    print("Welcome to HerbAgent Traditional Chinese Medicine Target Analysis System! I'm HerbAgent, a professional network pharmacology analysis assistant, how can I help you?")
    print("Please enter the formula, herb, syndrome, and disease you want to analyze:")
    
    while True:
        user_input = input("> ")
        
        # 记录开始时间
        start_time = time.time()
        
        # 获取分析结果
        result = coordinator.process(user_input)
        # print("~~!!!特别标记!!!~~Result结果是:", result)
        
        # 处理特殊状态
        if result.get("status") == "restart":
            print("\n" + result["message"])
            continue
        elif result.get("status") == "exit":
            print("\n" + result["message"])
            break
            
        # 修改这里：正确处理闲聊模式
        if result.get("status") == "chat" and result.get("chat_mode"):
            print("\n" + result["message"])  # 直接输出闲聊回复
            end_time = time.time()
            print(f"\nResponse time: {end_time - start_time:.2f} seconds")
            continue
            
        # 递归地将所有DataFrame转换为可序列化的格式
        # def convert_dataframes(obj):
        #     if isinstance(obj, pd.DataFrame):
        #         return obj.to_dict('records')
        #     elif isinstance(obj, dict):
        #         return {k: convert_dataframes(v) for k, v in obj.items()}
        #     elif isinstance(obj, list):
        #         return [convert_dataframes(item) for item in obj]
        #     return obj
        
        # # 转换结果中的所有DataFrame
        # serializable_result = convert_dataframes(result)
        
        # 修改这里：分别处理不同类型的分析结果
        # print("\nComplete analysis results:")
        
        # 特别处理 PPI 网络分析结果
        # if 'ppi_results' in serializable_result:
        #     print("\nPPI Network Analysis Results:")
        #     ppi_report = serializable_result['ppi_results'].get('analysis_report')
        #     if ppi_report:
        #         print("PPI Analysis Report:")
        #         print(json.dumps(ppi_report, ensure_ascii=False, indent=2))
        
        # # 特别处理 Random Walk 分析结果
        # if 'rwr_results' in serializable_result:
        #     print("\nRandom Walk Analysis Results:")
        #     rwr_report = serializable_result['rwr_results'].get('analysis_report')
        #     if rwr_report:
        #         print("Random Walk Analysis Report:")
        #         print(json.dumps(rwr_report, ensure_ascii=False, indent=2))
        
        # 输出完整的序列化结果
        # print("\nAll Analysis Results:")

        # 将分析结果以格式化的 JSON 形式输出到前端
        # print(json.dumps(serializable_result, ensure_ascii=False, indent=2))
        
        print("\nPlease continue to enter the content to be analyzed (or exit):")

if __name__ == "__main__":
    main()