from typing import Dict, Any, Optional, List
import logging
from config import Config

logger = logging.getLogger("AgentManager")

class AgentManager:
    """
    负责场景分发与 Agent 状态调度
    """
    def __init__(self):
        logger.info("🧭 [INIT] 正在初始化 Agent 管理器...")
        self.agents = {}
        self.agent_states = {}
        self._register_default_agents()
    
    def _register_default_agents(self):
        """注册默认 Agent"""
        try:
            from agents.scene_analyzer_agent import SceneAnalyzerAgent
            from agents.dialogue_agent import DialogueAgent
            from agents.action_agent import ActionAgent
            from agents.intro_agent import IntroAgent
            from agents.chat_agent import ChatAgent
            from agents.smalltalk_agent import SmallTalkAgent
            
            self.register_agent("scene_analyzer", SceneAnalyzerAgent())
            self.register_agent("dialogue", DialogueAgent())
            self.register_agent("action", ActionAgent())
            self.register_agent("intro", IntroAgent())
            self.register_agent("chat", ChatAgent())
            self.register_agent("smalltalk", SmallTalkAgent())
            
            logger.info("✅ [AGENT] 默认 Agent 注册完成")
        except Exception as e:
            logger.warning("⚠️ [AGENT] 默认 Agent 注册失败: %s", e)
    
    def register_agent(self, agent_name: str, agent_instance: Any):
        """
        注册新的 Agent 实例
        """
        self.agents[agent_name] = agent_instance
        self.agent_states[agent_name] = "ready"
        logger.info("📝 [AGENT] 注册 Agent: %s", agent_name)
    
    def dispatch_task(self, scene_type: str, task_data: Dict[str, Any]) -> Optional[Any]:
        """
        根据场景类型分发任务给合适的 Agent
        """
        agent_mapping = {
            "scene_analysis": "scene_analyzer",
            "dialogue": "dialogue",
            "action": "action",
            "introduction": "intro",
            "chat": "chat",
            "smalltalk": "smalltalk"
        }
        
        agent_name = agent_mapping.get(scene_type, "dialogue")
        if agent_name in self.agents:
            logger.info("🚚 [AGENT] 分发任务到 Agent: %s, 场景: %s", agent_name, scene_type)
            self.update_agent_status(agent_name, "working")
            try:
                result = self.agents[agent_name].process(task_data)
                self.update_agent_status(agent_name, "ready")
                return result
            except Exception as e:
                logger.error("❌ [AGENT] Agent 执行失败: %s, 错误: %s", agent_name, e)
                self.update_agent_status(agent_name, "error")
                return None
        else:
            logger.warning("⚠️ [AGENT] 未找到对应 Agent: %s", agent_name)
            return None
    
    def get_agent_state(self, agent_name: str) -> Optional[str]:
        """
        获取指定 Agent 的当前状态
        """
        return self.agent_states.get(agent_name)
    
    def update_agent_status(self, agent_name: str, status: str):
        """
        更新 Agent 状态
        """
        if agent_name in self.agent_states:
            self.agent_states[agent_name] = status
            logger.info("🔄 [AGENT] 更新 Agent 状态: %s -> %s", agent_name, status)
    
    def coordinate_agents(self, complex_task: Dict[str, Any]) -> Dict[str, Any]:
        """
        协调多个 Agent 共同完成复杂任务
        """
        results = {}
        
        # 示例：先进行场景分析，再进行对话，最后执行动作
        if "scene_data" in complex_task:
            scene_result = self.dispatch_task("scene_analysis", complex_task["scene_data"])
            results["scene_analysis"] = scene_result
        
        if "dialogue_data" in complex_task:
            dialogue_result = self.dispatch_task("dialogue", complex_task["dialogue_data"])
            results["dialogue"] = dialogue_result
        
        if "action_data" in complex_task:
            action_result = self.dispatch_task("action", complex_task["action_data"])
            results["action"] = action_result
        
        logger.info("🤝 [AGENT] 多 Agent 协调完成，结果: %s", results)
        return results
    
    def select_agent(self, query: str, context: Optional[Dict[str, Any]] = None):
        """
        根据 query 和 context 选择合适的 Agent
        """
        # 基于关键词的简单路由逻辑
        if "介绍" in query or "看" in query:
            return self.agents.get("intro")
        elif len(query) > 10:  # 假设长句子是深度对话
            return self.agents.get("chat")
        else:
            return self.agents.get("smalltalk")
    
    def shutdown(self):
        """
        优雅关闭所有 Agent
        """
        for agent_name in self.agents:
            try:
                if hasattr(self.agents[agent_name], "shutdown"):
                    self.agents[agent_name].shutdown()
                self.update_agent_status(agent_name, "shutdown")
                logger.info("🛑 [AGENT] 关闭 Agent: %s", agent_name)
            except Exception as e:
                logger.error("❌ [AGENT] 关闭 Agent 失败: %s, 错误: %s", agent_name, e)

