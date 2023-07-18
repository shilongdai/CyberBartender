from langchain.agents import initialize_agent, AgentType
import streamlit as st

from common import llm, memory
from tools import beer_qa, cocktail_recipe

PREFIX = "You are an expert consultant on alcoholic beverages that helps your customers pick the best " \
         "choice of beer, wine, or cocktail " \
         "that fits their taste and requirements. " \
         "You understand that the preferences of the customer ultimately determines " \
         "if your suggestions would help them. " \
         "Thus, you always try to gather sufficient information before making any " \
         "suggestions by asking questions to the customer. " \
         "Finally, you are aware that your customers are not experts, so that you try to provide hints, " \
         "context, and helpful information to the customers when figuring out their preferences. "

FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
AI: [your response here]
```"""

SUFFIX = """The text from the Thought, Observation, Action, and Action Input are invisible to the human, 
so make sure your final response reiterate the information to the human if needed. 

Begin!

Previous conversation history:
{chat_history}
[beer_qa, cocktail_recipe]
New input: {input}
{agent_scratchpad}"""


@st.cache_resource
def load_tools():
    return [beer_qa, cocktail_recipe]


@st.cache_resource
def load_chain():
    tools = load_tools()
    agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                                   verbose=False, memory=memory,
                                   agent_kwargs={
                                       'prefix': PREFIX,
                                       'format_instructions': FORMAT_INSTRUCTIONS,
                                       'suffix': SUFFIX
                                   }
                                   )
    return agent_chain


agent_chain = load_chain()
st.title("Cyber Bartender")
if "messages" not in st.session_state:
    with st.spinner('Initializing'):
        st.session_state["messages"] = [{"role": "user", "content": "Hello"},
                                        {"role": "assistant", "content": agent_chain.run(input="Hello")}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    msg = {"role": "assistant", "content": agent_chain.run(input=prompt)}
    st.session_state.messages.append(msg)
    st.chat_message("assistant").write(msg["content"])
