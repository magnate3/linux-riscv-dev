import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# 初始化语言模型
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)
basic_prompt = "请用一句话解释一下什么是 Multi-Turn Prompts ? "

print(llm.invoke(basic_prompt).content)
structured_prompt = PromptTemplate(
    input_variables=["topic"],
    template="请提供 {topic} 的定义，然后解释其重要性。"
)

chain = structured_prompt | llm
input_variables = {"topic": "Multi-Turn Prompts"}
print(chain.invoke(input_variables).content)