import os
from abc import ABC, abstractmethod

from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from logger import LOG
from chat_history import get_session_history


chat_4o_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    streaming=True,
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)


deepseek_model = ChatDeepSeek(
    # deepseek-chat 对应 DeepSeek-V3；deepseek-reasoner 对应 DeepSeek-R1。
    model="deepseek-chat",
    temperature=0.2,
    streaming=True,
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)


class ChatBase(ABC):
    """
    聊天机器人基类，提供聊天功能。
    """
    def __init__(self, model, prompt_file, session_id=None, with_history=False):
        self.model = model
        self.prompt_file = prompt_file
        self.session_id = session_id if session_id else f"{model}_{prompt_file}_default_session_id"
        self.prompt = self.load_prompt()
        self.with_history = with_history

    def load_prompt(self):
        """
        从文件加载系统提示语。
        """
        try:
            with open(self.prompt_file, "r", encoding="utf-8") as file:
                return file.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到提示文件 {self.prompt_file}!")

    def create_chatbot(self):
        system_prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])

        chatbot = system_prompt | self.model

        if self.with_history:
            # 将聊天机器人与消息历史记录关联
            self.chatbot = RunnableWithMessageHistory(chatbot, get_session_history)
        else:
            self.chatbot = chatbot
