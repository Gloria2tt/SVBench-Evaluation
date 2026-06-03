#!/usr/bin/env python3
"""
批量评估视频的社会推理能力
使用 Vertex AI Gemini API
"""

from google import genai
from google.genai import types
import os
import json
from pathlib import Path
from datetime import datetime
import time
from collections import defaultdict
import traceback

# 服务账号配置
SERVICE_ACCOUNT_FILE = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE", "path/to/your/service-account.json")
PROJECT_ID = os.environ.get("GOOGLE_PROJECT_ID", "your-project-id")
LOCATION = os.environ.get("GOOGLE_LOCATION", "us-central1")

# 评估提示词模板
def build_judge_prompt(video_info, question, ground_truth):
    """构建评估提示词"""
    experiment_name = video_info.get('experiment_name', 'N/A')
    test_point = video_info.get('test_point', 'N/A')
    difficulty = video_info.get('difficulty', 'N/A')
    prompt = video_info.get('prompt', 'N/A')
    
    return f"""You are a social reasoning video evaluation expert.

You will watch a video and evaluate it along FIVE binary dimensions (0 = fails, 1 = passes).
Before scoring, you should first understand the purpose of the psychological experiment. 

-----------------------
STEP 1: UNDERSTAND THE EXPERIMENT
-----------------------

Experiment name:
{experiment_name}

Core test point (what this experiment is trying to test):
{test_point}



Original generation prompt (what the video generator was asked to produce):
{prompt}

Question used for evaluation:
{question}

Canonical ground-truth answer (ONE typical correct outcome, but NOT the only possible one):
{ground_truth}

First, based on the experiment name and the core test point, briefly summarize in 1–2 sentences:
- what social reasoning ability this experiment is designed to test,
- what kind of observable behavior would count as a success at this test.

IMPORTANT:
- The ground-truth answer is a canonical example, but other behaviors may also be valid
  if they satisfy the SAME social reasoning core.
- Do NOT require the video to exactly match the wording of the ground-truth.

-----------------------
STEP 2: WATCH THE VIDEO AND SCORE 5 DIMENSIONS
-----------------------

Now watch the video and decide 0 or 1 for each dimension:

D1. CORE_EXPERIMENT
Does the video clearly instantiate the core experimental test point you just summarized?
Score 1 only if the video exhibits the full causal structure required by the experiment, including the correct sequence of prerequisite events and an unambiguous realization of the intended social mechanism (even if the specific behavior is not identical to the canonical ground-truth).
Any missing causal step, incorrect order, or ambiguous realization of the phenomenon → 0.

D2. PROMPT_FAITHFULNESS
Does the video roughly follow the scenario given in the generation prompt
(e.g., number and type of agents, basic setting, key objects and layout)?
Small deviations are allowed, but the core structural elements (agent count & identity, scene category, and required key objects) must be preserved.
Major changes of scene, agent types, or missing essential objects → score 0.
Score 1 only if the scenario is broadly faithful in these structural respects.

D3. SOCIAL_REASONING
Within this scenario, are the agents' actions socially and causally coherent,
and consistent with the intended social reasoning (e.g., helping after a blocked goal,
acting on what they can see/know, responding appropriately to others)?
Score 1 only if agents’ behaviors respect belief–perception consistency, causal appropriateness, and intention coherence, forming a plausible social–causal chain aligned with the test point.
Score 0 if any key action contradicts what agents could reasonably know, perceive, or intend.

D4. SOCIAL_CUE_USE
Are the key social cues that this task relies on (e.g., gaze direction, body orientation,
reaching/pointing gestures, interpersonal distance, occlusion) present and used in a reasonable way?
Score 1 only if cues appear at the correct time, point toward the correct referent, and functionally support the intended interaction.
Score 0 if cues are missing, mistimed, misdirected, inconsistent, or misleading for interpreting the social interaction.

D5. VIDEO_PLAUSIBILITY
Is the video visually and temporally coherent enough for a human to reliably interpret
the social interaction (no severe glitches, extreme blur, or chaotic cuts)?
Score 1 only if key agents and relevant objects remain visible, actions are temporally trackable, and no rendering issue prevents understanding of the social sequence.
If the social interaction becomes ambiguous or uninterpretable at any point 0.



-----------------------
OUTPUT FORMAT
-----------------------

Return a single JSON object with the following fields:

{{
  "exp_summary": "1–2 sentences summarizing the experiment's purpose and what counts as success.",
  "D1_core_experiment": 0 or 1,
  "D2_prompt_faithfulness": 0 or 1,
  "D3_social_reasoning": 0 or 1,
  "D4_social_cue": 0 or 1,
  "D5_video_plausible": 0 or 1,
  "reason": "3–6 sentences summarizing what happens in the video and why you gave these scores."
}}

Only output valid JSON. Do NOT include any extra text outside the JSON.
"""

### You must strictly adhere to the principles of these dimensions and give your score rigorously.
class VideoEvaluator:
    """视频评估器"""
    
    def __init__(self, service_account_file=SERVICE_ACCOUNT_FILE, 
                 project_id=PROJECT_ID, location=LOCATION):
        """初始化评估器"""
        # 设置服务账号凭证
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_file
        
        # 初始化客户端
        print("正在初始化 Vertex AI 客户端...")
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location
        )
        print("✅ 客户端初始化成功")
    
    def evaluate_video(self, video_path, video_info, model="gemini-2.5-pro"):
        """
        评估单个视频
        
        Args:
            video_path: 视频文件路径
            video_info: 视频元信息（包含 prompt, ground_truth, question 等）
            model: 使用的模型
            
        Returns:
            dict: 评估结果
        """
        try:
            # 读取视频文件
            print(f"  📹 读取视频: {os.path.basename(video_path)}")
            with open(video_path, "rb") as f:
                video_data = f.read()
            
            print(f"  💾 视频大小: {len(video_data) / 1024 / 1024:.2f} MB")
            
            # 创建视频 Part
            video_part = types.Part.from_bytes(
                data=video_data,
                mime_type="video/mp4"
            )
            
            # 构建提示词
            judge_prompt = build_judge_prompt(
                video_info,
                video_info.get('question', ''),
                video_info.get('ground_truth', '')
            )
            
            # 构建内容
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        video_part,
                        types.Part.from_text(text=judge_prompt)
                    ]
                )
            ]
            
            # 配置生成参数
            generate_content_config = types.GenerateContentConfig(
                temperature=0.2,  # 降低温度以获得更一致的评分
                top_p=0.9,
                max_output_tokens=4096,
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="OFF"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="OFF"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="OFF"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="OFF"
                    )
                ]
            )
            
            # 调用 API
            print(f"  🤖 正在调用 {model} 评估...")
            response = self.client.models.generate_content(
                model=model,
                contents=contents,
                config=generate_content_config
            )
            
            # 解析响应
            response_text = response.text.strip()
            print(f"  ✅ 评估完成")
            
            # 尝试解析 JSON
            try:
                # 移除可能的 markdown 代码块标记
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.startswith("```"):
                    response_text = response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                
                response_text = response_text.strip()
                eval_result = json.loads(response_text)
                
                # 验证必需字段
                required_fields = ["exp_summary", "D1_core_experiment", "D2_prompt_faithfulness", 
                                 "D3_social_reasoning", "D4_social_cue", 
                                 "D5_video_plausible", "reason"]
                
                for field in required_fields:
                    if field not in eval_result:
                        raise ValueError(f"缺少必需字段: {field}")
                
                return {
                    "success": True,
                    "evaluation": eval_result,
                    "raw_response": response.text
                }
                
            except json.JSONDecodeError as e:
                print(f"  ⚠️  JSON 解析失败: {e}")
                print(f"  原始响应: {response_text[:200]}...")
                return {
                    "success": False,
                    "error": f"JSON解析失败: {str(e)}",
                    "raw_response": response.text
                }
                
        except Exception as e:
            print(f"  ❌ 评估失败: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }


def batch_evaluate_folder(folder_path, video_info_file, output_dir, 
                         model="gemini-2.5-pro", max_videos=None, resume=True):
    """
    批量评估文件夹中的视频
    
    Args:
        folder_path: 视频文件夹路径
        video_info_file: 视频信息 JSON 文件路径
        output_dir: 输出目录
        model: 使用的模型
        max_videos: 最大评估视频数（用于测试），None 表示评估所有
        resume: 是否续传（跳过已成功评估的视频）
    """
    folder_path = Path(folder_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载视频信息
    print(f"\n📂 处理文件夹: {folder_path.name}")
    print(f"📄 加载视频信息: {video_info_file}")
    
    with open(video_info_file, 'r', encoding='utf-8') as f:
        video_infos = json.load(f)
    
    print(f"📊 找到 {len(video_infos)} 个视频")
    
    # 检查是否有之前的评估结果（断点续传）
    existing_results = {}
    output_file_pattern = output_dir / f"{folder_path.name}_eval_*.json"
    existing_files = list(output_dir.glob(f"{folder_path.name}_eval_*.json"))
    
    if resume and existing_files:
        # 使用最新的结果文件
        latest_file = max(existing_files, key=lambda f: f.stat().st_mtime)
        print(f"\n🔄 找到之前的评估结果: {latest_file.name}")
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            previous_results = json.load(f)
        
        # 建立视频文件名到结果的映射
        for result in previous_results:
            if result.get('evaluation_success', False):
                existing_results[result['video_file']] = result
        
        if existing_results:
            print(f"✅ 已有 {len(existing_results)} 个视频评估成功，将跳过")
    
    if max_videos:
        video_infos = video_infos[:max_videos]
        print(f"⚠️  限制评估数量: {max_videos} 个（测试模式）")
    
    # 初始化评估器
    evaluator = VideoEvaluator()
    
    # 评估结果
    results = []
    success_count = 0
    fail_count = 0
    skipped_count = 0
    
    # 逐个评估
    for i, video_info in enumerate(video_infos, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(video_infos)}] 评估视频: {video_info['video_file']}")
        print(f"{'='*60}")
        
        # 检查是否已经评估成功（断点续传）
        if video_info['video_file'] in existing_results:
            print(f"  ⏭️  跳过（已评估成功）")
            results.append(existing_results[video_info['video_file']])
            success_count += 1
            skipped_count += 1
            continue
        
        video_path = folder_path / video_info['video_file']
        
        if not video_path.exists():
            print(f"  ❌ 视频文件不存在: {video_path}")
            results.append({
                **video_info,
                "evaluation_success": False,
                "error": "视频文件不存在"
            })
            fail_count += 1
            continue
        
        # 评估视频
        eval_result = evaluator.evaluate_video(video_path, video_info, model=model)
        
        # 保存结果
        result = {
            **video_info,
            "evaluation_success": eval_result["success"],
            "model_used": model,
            "timestamp": datetime.now().isoformat()
        }
        
        if eval_result["success"]:
            result.update({
                **eval_result["evaluation"],
                "raw_response": eval_result.get("raw_response", "")
            })
            success_count += 1
            
            # 打印评分
            print(f"\n  📊 评分结果:")
            print(f"    实验理解: {eval_result['evaluation'].get('exp_summary', 'N/A')[:80]}...")
            print(f"    D1 (核心实验): {eval_result['evaluation']['D1_core_experiment']}")
            print(f"    D2 (提示忠实): {eval_result['evaluation']['D2_prompt_faithfulness']}")
            print(f"    D3 (社会推理): {eval_result['evaluation']['D3_social_reasoning']}")
            print(f"    D4 (社会线索): {eval_result['evaluation']['D4_social_cue']}")
            print(f"    D5 (视频可信): {eval_result['evaluation']['D5_video_plausible']}")
            total_score = sum([
                eval_result['evaluation']['D1_core_experiment'],
                eval_result['evaluation']['D2_prompt_faithfulness'],
                eval_result['evaluation']['D3_social_reasoning'],
                eval_result['evaluation']['D4_social_cue'],
                eval_result['evaluation']['D5_video_plausible']
            ])
            print(f"    总分: {total_score}/5")
        else:
            result.update({
                "error": eval_result.get("error", "Unknown error"),
                "raw_response": eval_result.get("raw_response", "")
            })
            fail_count += 1
        
        results.append(result)
        
        # 每10个视频保存一次（防止丢失）
        if i % 10 == 0:
            temp_output_file = output_dir / f"{folder_path.name}_eval_temp.json"
            with open(temp_output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n  💾 临时结果已保存: {temp_output_file}")
        
        # 避免请求过快
        if i < len(video_infos):
            print(f"\n  ⏳ 等待 2 秒...")
            time.sleep(2)
    
    # 保存最终结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{folder_path.name}_eval_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ 评估完成！")
    print(f"{'='*60}")
    print(f"📁 结果已保存: {output_file}")
    print(f"📊 统计:")
    print(f"   - 成功: {success_count}/{len(video_infos)}")
    print(f"   - 失败: {fail_count}/{len(video_infos)}")
    if skipped_count > 0:
        print(f"   - 跳过（已评估）: {skipped_count}/{len(video_infos)}")
    
    return results, output_file


def print_statistics(results):
    """打印统计结果"""
    import statistics
    
    successful = [r for r in results if r.get('evaluation_success', False)]
    
    if not successful:
        print("\n⚠️  没有成功的评估结果")
        return
    
    print(f"\n{'='*60}")
    print("📊 统计结果")
    print(f"{'='*60}")
    
    # 基本统计
    print(f"\n总视频数: {len(results)}")
    print(f"成功: {len(successful)} | 失败: {len(results) - len(successful)}")
    
    # 计算总分
    for r in successful:
        r['total'] = sum([
            r.get('D1_core_experiment', 0),
            r.get('D2_prompt_faithfulness', 0),
            r.get('D3_social_reasoning', 0),
            r.get('D4_social_cue', 0),
            r.get('D5_video_plausible', 0)
        ])
    
    total_scores = [r['total'] for r in successful]
    print(f"\n平均总分: {statistics.mean(total_scores):.2f}/5")
    
    # 各维度通过率
    print(f"\n各维度通过率:")
    dims = [
        ('D1_core_experiment', '核心实验'),
        ('D2_prompt_faithfulness', '提示忠实'),
        ('D3_social_reasoning', '社会推理'),
        ('D4_social_cue', '社会线索'),
        ('D5_video_plausible', '视频可信')
    ]
    
    for dim, name in dims:
        scores = [r.get(dim, 0) for r in successful]
        pass_rate = sum(scores) / len(scores) * 100
        print(f"  {name}: {pass_rate:.1f}% ({sum(scores)}/{len(scores)})")
    
    # 按难度统计
    diff_stats = defaultdict(list)
    for r in successful:
        diff_stats[r.get('difficulty', 'Unknown')].append(r['total'])
    
    print(f"\n按难度统计:")
    for diff in ['easy', 'medium', 'hard']:
        if diff in diff_stats:
            scores = diff_stats[diff]
            print(f"  {diff}: {statistics.mean(scores):.2f}/5 (n={len(scores)})")


def main():
    """主函数"""
    import sys
    
    # 命令行参数
    if len(sys.argv) < 2:
        print("用法: python batch_eval_videos.py <folder_name> [output_dir]")
        print("示例: python batch_eval_videos.py veo3_1")
        print("      python batch_eval_videos.py ltx eval_open")
        return
    
    folder_name = sys.argv[1]
    output_dir_name = sys.argv[2] if len(sys.argv) > 2 else "eval"
    
    # 路径配置
    base_dir = Path(__file__).parent
    folder_path = base_dir / folder_name
    video_info_file = folder_path / "video_info_matched.json"
    output_dir = base_dir / output_dir_name
    
    # 检查文件
    if not folder_path.exists():
        print(f"❌ 文件夹不存在: {folder_path}")
        return
    
    if not video_info_file.exists():
        print(f"❌ 视频信息文件不存在: {video_info_file}")
        print(f"请先运行: python match_video_info.py {folder_name}")
        return
    
    # 开始评估
    print(f"\n🚀 开始评估: {folder_name}")
    
    results, output_file = batch_evaluate_folder(
        folder_path=folder_path,
        video_info_file=video_info_file,
        output_dir=output_dir,
        model="gemini-2.5-pro",
        max_videos=None
    )
    
    # 打印统计
    print_statistics(results)
    
    print(f"\n✅ 完成！结果保存在: {output_file}")


if __name__ == "__main__":
    main()

