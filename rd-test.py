from langchain.llms import CTransformers
from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)

def load_llm():
    llm = CTransformers(
        model = "mistral-7b-instruct-v0.1.Q5_K_S.gguf",
        model_type="mistral",
        max_new_token = 2096,
        temperature = 0.9
    )
    return llm

domain_generate_systmp = "You are an expert marketer working at domain name registrar. Your customer has provided you information relating to their own business. Your goal is to generate new domain names that are relevant to their business and align with their industry. You should do it without any explanation."
dm_generate_tmp = """We're a domain registrar and most of our customers operate in Viet Nam. Please generate {number} additional domain names that cater to the customer's business interests in "{query}" industry relating to "{detail}". Domain name must include both SLD (Second Level Domain) and TLD (Top Level Domain) part. Domain name can be short, include compound or catchy word but unique and end with extensions .vn, .com.vn or .net.vn.
Output one object which bounded by curly bracket containing these domain names. Each domain name should have an unique key start with domain_name. Do not return any characters other than the JSON."""

system_message_prompt = SystemMessagePromptTemplate.from_template(domain_generate_systmp)

prompt_template = PromptTemplate(
        input_variables=["number","query", "detail"],
        template=dm_generate_tmp
)

human_message_prompt = HumanMessagePromptTemplate(
    prompt=prompt_template
)
chat_prompt_template = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

llm = load_llm()
chain = LLMChain(llm=llm, prompt=chat_prompt_template, verbose=True)

query = "Phòng khám đa khoa"
number = 5
detail = "chuyên nghiệp"


chain.run(number=number, query=query, detail=detail)
