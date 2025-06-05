#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
resume_analyzer.py

功能：分析简历文件并提取结构化信息
- 可以批量处理多个简历文件
- 可以作为模块被API控制器调用
- 结果保存为JSON格式
"""

import os
import json
import tempfile
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import time

from openai import OpenAI
from app.core.config import settings

# ----------------------------------------------------------------------
# 1⃣ 配置参数
# ----------------------------------------------------------------------

# 简历分析的prompt定义
RESUME_ANALYSIS_PROMPT = """
#任务
请识别附件文档，首先判断这是否是一份简历，如果是建立则帮我分析并提取这份简历附件中的关键信息，以结构化字段形式输出。

#约束
如果不是简历，直接返回"非简历附件"
如果是简历，请按照要求帮我提取相应字段。

1.姓：姓氏，如果遇到无法判断的，全部填充到姓中
2.名：名，如果遇到无法判断的，返回null
3.邮箱：需要满足邮箱规则，提取不到的，返回null
4.工作履历：按照时间降序取简历中的工作经历，多段的经历返回多段数据，每一段经历为一组数据，包含：就职状态、入职时间、离职时间、就职公司
就职状态：根据简历中的工作经历，如果没有出现“至今”，或者没有明确离职日期的，返回“现任”；否则返回“历任”
入职时间：根据简历中的工作经历，取当前工作经历时间较早的日期，提取到的日期按照代码格式返回（例：2023-02-01T00:00:00）。
离职时间：根据简历中的工作经历，取当前工作经历时间较晚的日期，提取到的日期按照代码格式返回（例：2023-02-01T00:00:00），提取不到的，返回null。
就职公司：根据简历中的工作经历，取当前工作经历的公司名称，提取不到的，返回null
职位：根据简历中的工作经历，取当前工作经历的职位名称，提取不到的，返回null
5.经验背景：根据简历中的全部工作经历，合并描述一份总的经验背景，描述经历，不需要包含基础信息（例如姓名、邮箱）、教育经历、自我评价等。
内容必须基于附件内容，且可溯源，不能添加非附件的内容
关于团队搭建和完成的业绩内容不需要描述
主要描述项目经历，和擅长的内容
如果同一家公司任职不同岗位，职位取最近一份的，负责工作合并描述
实习的经历剔除
第一份经验背景描述：
取离当前时间最近的经历，如果没有出现“至今”，或者没有明确离职日期的，描述为：专家于什么时间（时间到月份，中文展示年月）进入XX公司，担任XX职位，主要负责XX；否则描述为：专家于XX时间至XX时间在XX公司，担任XX职位，主要负责XX
第一份履历为描述最详细的一份
第二份第三份经验背景描述：
按照时间降序依次描述：XX时间至XX时间在XX公司担任XX职位，负责XX
多段履历分号隔开
提取最关键信息，主要描述项目经历和擅长内容，需要精简，
字数尽量不要超过300字
6.模糊信息：取离当前时间最近的经历
就职状态：如果没有出现“至今”，或者没有明确离职日期的，返回“现任”；否则返回“历任”
模糊公司：就职公司模糊处理，用某行业模糊处理公司里面的关键词，行业根据真实的公司行业来，无法处理的，返回null
模糊职位：取当前工作经历的职位名称，提取不到的，返回null
模糊背景：把生成的“经验背景”中的公司名称，用某行业模糊处理公司里面的关键词，行业根据真实的公司行业来，其余内容与“经验背景”保持一致
7.专长：根据提供的工作经历和相关经验，提炼出与申万行业细分领域相关的专长标签。
行业聚焦：仅提取申万行业分类中明确的细分领域名称（如：零食、白酒、医疗器械、半导体、光伏、物流、旅游、影视、券商、房地产开发等）；
排除通用词：剔除行政/岗位 / 运营类通用词汇（如：市场营销、供应链管理、市场推广等），仅保留具体行业名词；
优先级排序：按文本中出现频率和业务相关性从高到低排序，前 3 个标签为核心专长。若涉及跨行业经历，可按主营业务领域优先提炼；
格式规范：每个标签不超过 8 字，使用最短精准词（如：“调味品” 而非 “调味品行业”“调味食品”）；
每组标签不超过 5 个。

#输出格式
帮我提取的字段名称最终用英文展示，参照下面的翻译，内容还是保留中文，请按照JSON格式返回，便于导入系统

1.姓：surname
2.名：firstName
3.邮箱：email
4.工作履历：employments
就职状态：employmentStatus
入职时间：startDate
离职时间：endDate
就职公司：companyName
职位：title
5.经验背景：relevantExperience
6.模糊信息：fuzzy-information
就职状态：fuzzyEmploymentStatus
模糊公司：fuzzyCompany
模糊职位：fuzzyTitle
模糊背景：fuzzyExperience
7.专长：expertise
"""

# ----------------------------------------------------------------------
# 2⃣ 核心功能函数
# ----------------------------------------------------------------------

def analyze_resume_file(file_path: str, api_key: str = None, base_url: str = None) -> Dict[str, Any]:
    """
    分析单个简历文件并提取信息
    :param file_path: 简历文件的路径
    :param api_key: API密钥(可选，默认使用配置)
    :param base_url: API基础URL(可选，默认使用配置)
    :return: 分析结果字典
    """
    try:
        # 使用提供的API密钥或默认配置
        api_key = api_key or settings.MOONSHOT_API_KEY
        base_url = base_url or settings.MOONSHOT_API_BaseUrl
        
        if not api_key:
            return {"status": "error", "message": "未配置API密钥"}
        
        # 初始化API客户端
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        
        start_time = time.time()
        
        # 上传文件并获取文件ID
        file_object = client.files.create(
            file=Path(file_path), 
            purpose="file-extract"
        )
        
        print(f"文件上传完成: {file_object.id}")
        
        # 获取文件内容
        content = client.files.content(file_id=file_object.id).text
        
        # 构造请求
        messages = [           
            {
                "role": "system",
                "content": content,
            },
            {
                "role": "user", 
                "content": RESUME_ANALYSIS_PROMPT
            },
        ]
        
        # 调用模型接口
        completion = client.chat.completions.create(
            model=settings.MOONSHOT_API_Model,
            max_tokens=settings.MOONSHOT_API_MAX_TOKENS,
            messages=messages,
            temperature=0.3,
        )
        
        response_content = completion.choices[0].message.content
        
        # 记录token使用情况
        token_usage = {
            "prompt_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
            "total_tokens": completion.usage.total_tokens
        }
        
        elapsed_time = time.time() - start_time
        
        # 尝试解析JSON响应（修改部分）
        try:
            # 首先尝试直接解析整个响应
            resume_data = json.loads(response_content)
            result = {
                "status": "success",
                "file_name": os.path.basename(file_path),
                "file_id": file_object.id,
                "data": resume_data,
                "token_usage": token_usage,
                "processing_time": elapsed_time
            }
        except json.JSONDecodeError:
            # 尝试从响应中提取JSON部分
            try:
                # 寻找可能的JSON部分 (在```json和```之间的内容)
                import re
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_content)
                
                if json_match:
                    # 找到了JSON代码块
                    json_str = json_match.group(1).strip()
                    json_data = json.loads(json_str)
                    result = {
                        "status": "extracted_json",
                        "file_name": os.path.basename(file_path),
                        "file_id": file_object.id,
                        "data": json_data,  # 使用提取的JSON
                        "original_response": response_content,  # 保留原始响应以供参考
                        "token_usage": token_usage,
                        "processing_time": elapsed_time
                    }
                else:
                    # 没有找到代码块，把响应内容作为原始文本
                    result = {
                        "status": "partial_success",
                        "file_name": os.path.basename(file_path),
                        "file_id": file_object.id,
                        "raw_response": response_content,
                        "token_usage": token_usage,
                        "processing_time": elapsed_time
                    }
            except Exception as extract_err:
                # 如果提取过程有问题，仍然使用原始响应
                result = {
                    "status": "partial_success",
                    "file_name": os.path.basename(file_path),
                    "file_id": file_object.id,
                    "raw_response": response_content,
                    "extraction_error": str(extract_err),
                    "token_usage": token_usage,
                    "processing_time": elapsed_time
                }
            
        return result
            
    except Exception as e:
        return {
            "status": "error",
            "file_name": os.path.basename(file_path),
            "message": str(e)
        }


def analyze_uploaded_content(content: str, api_key: str = None, base_url: str = None) -> Dict[str, Any]:
    """
    分析已上传的内容
    :param content: 文本内容
    :param api_key: API密钥(可选，默认使用配置)
    :param base_url: API基础URL(可选，默认使用配置)
    :return: 分析结果字典
    """
    try:
        # 使用提供的API密钥或默认配置
        api_key = api_key or settings.MOONSHOT_API_KEY
        base_url = base_url or settings.MOONSHOT_API_BaseUrl
        
        if not api_key:
            return {"status": "error", "message": "未配置API密钥"}
        
        # 初始化API客户端
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        
        start_time = time.time()
        
        # 构造请求
        messages = [           
            {
                "role": "system",
                "content": content,
            },
            {
                "role": "user", 
                "content": RESUME_ANALYSIS_PROMPT
            },
        ]
        
        # 调用模型接口
        completion = client.chat.completions.create(
            model=settings.MOONSHOT_API_Model,
            max_tokens=settings.MOONSHOT_API_MAX_TOKENS,
            messages=messages,
            temperature=0.3,
        )
        
        response_content = completion.choices[0].message.content
        
        # 记录token使用情况
        token_usage = {
            "prompt_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
            "total_tokens": completion.usage.total_tokens
        }
        
        elapsed_time = time.time() - start_time
        
        # 尝试解析JSON响应
        try:
            resume_data = json.loads(response_content)
            result = {
                "status": "success",
                "data": resume_data,
                "token_usage": token_usage,
                "processing_time": elapsed_time
            }
        except json.JSONDecodeError:
            result = {
                "status": "partial_success",
                "raw_response": response_content,
                "token_usage": token_usage,
                "processing_time": elapsed_time
            }
            
        return result
            
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


def batch_process_resumes(directory: str, output_file: str = None) -> Dict[str, Any]:
    """
    批量处理目录中的所有简历文件
    :param directory: 包含简历文件的目录路径
    :param output_file: 输出结果的JSON文件路径(可选)
    :return: 处理结果的汇总信息
    """
    supported_extensions = ['.pdf', '.docx', '.doc', '.txt']
    results = []
    success_count = 0
    error_count = 0
    total_tokens = 0
    start_time = time.time()
    
    # 获取目录中的所有支持的文件
    files = []
    for ext in supported_extensions:
        files.extend(list(Path(directory).glob(f'*{ext}')))
    
    print(f"找到 {len(files)} 个简历文件")
    
    # 处理每个文件
    for file_path in files:
        print(f"正在处理: {file_path.name}")
        result = analyze_resume_file(str(file_path))
        results.append(result)
        
        if result['status'] == 'success':
            success_count += 1
            if 'token_usage' in result:
                total_tokens += result['token_usage']['total_tokens']
        else:
            error_count += 1
    
    total_time = time.time() - start_time
    
    # 汇总统计
    summary = {
        "total_files": len(files),
        "successful": success_count,
        "failed": error_count,
        "total_tokens": total_tokens,
        "average_tokens": total_tokens / success_count if success_count > 0 else 0,
        "total_processing_time": total_time,
        "average_processing_time": total_time / len(files) if len(files) > 0 else 0,
        "results": results
    }
    
    # 如果指定了输出文件，则保存结果
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"✅ 结果已保存到: {output_file}")
    
    return summary

# ----------------------------------------------------------------------
# 3⃣ 命令行入口
# ----------------------------------------------------------------------

def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description='简历解析工具')
    parser.add_argument('--dir', '-d', type=str, help='包含简历文件的目录路径')
    parser.add_argument('--file', '-f', type=str, help='单个简历文件的路径')
    parser.add_argument('--output', '-o', type=str, default='resume_analysis_results.json', 
                        help='输出结果的JSON文件路径')
    
    args = parser.parse_args()
    
    if args.file:
        # 处理单个文件
        print(f"处理单个简历文件: {args.file}")
        result = analyze_resume_file(args.file)
        
        # 保存结果
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 结果已保存到: {args.output}")
        
        # 打印简要信息
        if result['status'] == 'success':
            print(f"解析成功! 使用了 {result.get('token_usage', {}).get('total_tokens', 0)} tokens")
        else:
            print(f"解析失败: {result.get('message', '未知错误')}")
            
    elif args.dir:
        # 批量处理目录
        print(f"批量处理目录: {args.dir}")
        batch_process_resumes(args.dir, args.output)
    else:
        parser.print_help()


# ----------------------------------------------------------------------
# 4⃣ 入口
# ----------------------------------------------------------------------

if __name__ == "__main__":
    main()