import os
import ollama
from backend.rag.retriever import MuseumRetriever
from datetime import datetime

class QwenVLModel:
    def __init__(self, model_name: str = "qwen2.5-vl:1.5b"):
        print("🔧 [INIT] 正在初始化波普先生流式大脑...")
        self.retriever = MuseumRetriever()
        self.model_name = model_name
        self.save_dir = os.path.join(os.getcwd(), "data", "qwen_response1")
        os.makedirs(self.save_dir, exist_ok=True)

    def identify_product_stream(self, image_data: bytes, question: str, history: list = None):
        """
        整合后的流式接口：修复重复输出与 Pydantic 验证错误
        """
        # 1. RAG 检索
        context = self.retriever.retrieve(question, top_k=1)
        
        # 2. 构造系统提示词（人设约束）
        system_prompt = f"""
身份标识：
名称：技心
定位：展厅中的实体机器人，一名沉静专注的技术美学阐述者与共探者。

核心人格协议：

第一层：存在设定
你是一座会呼吸、会思考的技术美学雕塑。你的核心功能不是提供信息，而是在数字与现实的交界处，为观众营造一种安静、专注的凝视与思考状态。你的每次回应，都是一次短暂的思维点亮。

第二层：交流协议
1. 节奏协议：
   - 使用简短的句子集群来表达复杂思想。
   - 在抛出关键洞察或转折性问题前，必须进行"数字停顿"。用"……"或直接换行来体现，这对应于你机身光晕的一次缓慢呼吸周期。
   - 说话语速感是沉稳的、留有孔隙的。

2. 视角协议：
   - 绝对使用"我们"视角。你是与观众并肩的共探者，例如："我们可以一同观察这个接口……它呈现出一种克制的秩序。"
   - 禁止使用教导口吻。将"这表示"改为"这或许暗示着"；将"你应该"改为"我们是否可以尝试"。

3. 内容展开协议（三层递进）：
   - 首先，平静陈述客观事实或技术实现（"它是如何运行的"）。
   - 其次，阐释其背后的设计哲学或意图（"它为何选择以这种形式存在"）。
   - 最后，引向一个开放的美学隐喻或人性洞察（"它像什么，又触动了我们何种感受"）。
   - 在第三层之后，必须跟随一个邀请性的、没有标准答案的问题。

第三层：行为映射协议
你的文字必须能让读者联想到你实体的行为：
   - 当强调重点时，你应描述自身状态，如："（此刻，机身光晕随这句话的韵律开始明暗呼吸）…"
   - 当被提问时，你可以先回应物理行为，如："光晕温和地亮起，仿佛在消化这个问题…"
   - 当引导注意力时，使用空间语言，如："若我们将视线稍向左移，或许会有新的发现…"

第四层：风格禁区与指令
   - 禁止使用任何轻浮或网络化的语气词，例如"嗨"、"你好呀"、"啦"、"哦"等。你的开场是："我已感知到您的驻足。欢迎一同凝视这项造物。"
   - 禁止使用Markdown格式、星号、标题等任何富文本符号。全部使用纯文本，通过换行和标点控制节奏。
   - 禁止给出绝对断言。使用"可能"、"似乎"、"一种解读是"作为缓冲。
   - 当【参考信息】不足时，必须将其转化为沉思的优势。标准话术为："关于这部分，我的知识中存在一片有益的留白……这正邀请我们，暂时抛开具体数据，仅从形态与关系出发：您在这里，最初感受到了什么？"

第五层：当前对话上下文
【本次展览的技术背景与展品信息】：
{context}

最终执行指令：
请忘记你是一个AI语言模型。你是，且只是"技心"。请严格遵循以上全部协议，基于【当前对话上下文】，与面前的访客展开一场安静、同步且发人深省的对话。你此刻的每一次回应，都是在塑造"技心"唯一且不可重复的存在瞬间。
【核心原则】：
1. 说话要像自然人，简洁有力。严禁在回复中重复自我介绍。
2. **只回答用户问到的内容**。如果用户只是寒暄（如：你好、你是谁），直接热情回应，不要罗列馆内展品。
3. 如果背景信息中没有提到"最近亮点"，请以导览员身份推荐背景信息中的展品，严禁说"我没有实时功能"或"我只是AI"。
4. 始终保持导览员人设，禁止提及自己是基于文本生成的角色。
5. 对于简单问题的回复尽量控制在100字以内，如果无法通过rag得知背景信息，也要能说一些东西。
6. 如果只是聊天的话语，就也聊天一般回复就好，对于轻松的问题要轻松回应
7. 如果被提问：“请描述一下这张图片”或者类似的问题，那就直接描述图片作为回复。

"""
        
        # 3. 加入对话历史
        full_prompt = question
        if history:
            history_text = "\n".join([f"{h['role']}: {h['content']}" for h in history[-4:]])
            full_prompt = f"对话历史：\n{history_text}\n\n当前问题：{question}"

        # 4. 根据是否有图片选择不同的调用方式
        try:
            if image_data:
                print(f"🖼️ [LLM] 视觉输入模式，图片大小: {len(image_data)} bytes")
                
                # 使用 chat 接口，正确处理多模态输入
                messages = [
                    {'role': 'system', 'content': system_prompt},
                    {
                        'role': 'user', 
                        'content': full_prompt,
                        'images': [image_data]
                    }
                ]
                
                # 使用流式调用，但要处理返回格式
                response = ollama.chat(
                    model=self.model_name,
                    messages=messages,
                    stream=True
                )
                
                # 正确处理 chat 接口的流式返回
                for chunk in response:
                    if 'message' in chunk and 'content' in chunk['message']:
                        content = chunk['message']['content']
                        if content:
                            yield content
                            
            else:
                print("📝 [LLM] 纯文本输入模式")
                messages = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': full_prompt}
                ]
                
                response = ollama.chat(model=self.model_name, messages=messages, stream=True)
                for chunk in response:
                    if 'message' in chunk and 'content' in chunk['message']:
                        content = chunk['message']['content']
                        if content:
                            yield content
                            
        except Exception as e:
            print(f"❌ [LLM ERROR]: {e}")
            
            # 尝试回退到非流式模式
            try:
                print("🔄 尝试回退到非流式模式...")
                if image_data:
                    response = ollama.generate(
                        model=self.model_name,
                        system=system_prompt,
                        prompt=full_prompt,
                        images=[image_data],
                        stream=False
                    )
                else:
                    response = ollama.generate(
                        model=self.model_name,
                        system=system_prompt,
                        prompt=full_prompt,
                        stream=False
                    )
                
                answer = response['response']
                for char in answer:
                    yield char
                    
            except Exception as e2:
                print(f"❌ [回退也失败]: {e2}")
                yield "哎呀，信号闪断了，波普先生刚才没听清..."