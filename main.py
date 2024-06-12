# Integrate code with OPENAI API

import os 
from constants import openai_key
from langchain.llms import openAI

import streamlit as st

from langchain import PromptTemplate
from langchain.chains import LLMChain

from langchain.chains import SequentialChain

from langchain.memory import ConversationBufferMemory

os.environ['OPENAI_API_KEY'] = openai_key

#streamlit framework
st.title('Celebrity Search Application')
input_text = st.text_input("Search the celebrity")

# Prompt Templates
first_input_prompt = PromptTemplate(
    input_variables = ['name'],
    template = "Tell me about celebrity {name}"
)

# MEMORY
person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
desc_memory = ConversationBufferMemory(input_key='dob', memory_key='desc_history')


# OPENAI LLMS   
llm = openAI(temperature=0.8)
chain = LLMChain(llm = llm, prompt = first_input_prompt, verbose = True, output_key = 'person', memory = person_memory)

# Prompt Templates
second_input_prompt = PromptTemplate(
    input_variables = ['person'],
    template = "When was {person} born"
)
 
# OPENAI LLMS   
chain2 = LLMChain(llm = llm, prompt = second_input_prompt, verbose = True, output_key = 'dob', memory = dob_memory)

# Prompt Templates
third_input_prompt = PromptTemplate(
    input_variables = ['dob'],
    template = "Mention five major event happed around {dob}"
)

# OPENAI LLMS   
chain3 = LLMChain(llm = llm, prompt = third_input_prompt, verbose = True, output_key = 'description', memory = desc_memory)

parent_chain = SequentialChain(
    chains=[chain, chain2, chain3], 
    input_variables = ['name'],
    output_variables = ['person', 'dob', 'description'],
    verbose = True)

if input_text:
    st.write(parent_chain({'name': input_text}))

    with st.expander('Person Name'):
        st.info(person_memory.buffer)

    with st.expander('Major Events'):
        st.info(desc_memory.buffer)
