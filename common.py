from langchain import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationSummaryBufferMemory

MAX_TRACKED_TOKENS = 4096
API_KEY = "./api_key"
with open(API_KEY, "r") as fp:
    key_content = fp.read().strip()

embeddings = OpenAIEmbeddings(openai_api_key=key_content)
llm = OpenAI(openai_api_key=key_content, temperature=0)
memory = ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history",
                                         return_messages=True, max_token_limit=MAX_TRACKED_TOKENS)
