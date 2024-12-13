from datasets import load_dataset
import re
from langchain.prompts import ChatPromptTemplate    
from langchain.schema.runnable import RunnablePassthrough   
from langchain.schema.output_parser import StrOutputParser     
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser 

def get_gsm8k_answer(ans):
    # 使用正则表达式查找'#### '后面的数字
    match = re.search(r'#### (\d+)', ans)
    if match:
        return int(match.group(1))
    else:
        return None  # 如果没有找到，返回 None
def get_last_number(s):
    # 查找所有数字，并返回最后一个数字
    numbers = re.findall(r'\d+', s)
    return int(numbers[-1]) if numbers else None

# Load GSM8k dataset
dataset = load_dataset("gsm8k",'main')
gsm8k_test = dataset["test"]
# print(gsm8k_test[0])
# for q in gsm8k_test:
#     query = q['question']
#     ground_truth = get_gsm8k_answer(q['answer'])
#     print(ground_truth)


# Define the prompt template
initial_template = """You are an assistant for question-answering tasks. Provide your answer as a plain number without any commas and put it at the end of your output. Question: {question} Answer: Let's think step by step"""   
initial_prompt = ChatPromptTemplate.from_template(initial_template) 

actor_template = """You are an assistant for question-answering tasks. You will be given a previous reasoning trial and your reflections on it. Previous failed trial: Question {question}, Answer: {answer}. Given the following reflections: {mem}, please answer the question again: {question}. Provide your answer as a plain number without any commas and put it at the end of your output. Answer: Let's think step by step"""   
actor_prompt = ChatPromptTemplate.from_template(actor_template)   

reflection_template = """You are an expert on question-answering tasks. You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question. Previous trial: Question: {question}, Answer: {answer}. Be severe that your reflection can maximize improvement. You have given the following reflections: {mem}. Your new reflection:""" 
reflection_prompt = ChatPromptTemplate.from_template(reflection_template) 

# Get the LLM
chat_model = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key='sk-e39450eb1e1d4a8e825df0a7e4f5f411',
    openai_api_base='https://api.deepseek.com',
    max_tokens=1024, # 最大输出长度
    temperature=0.0, # 生成温度
)

# Build the chain
initial_chain = (initial_prompt | chat_model | StrOutputParser())
actor_chain = (actor_prompt | chat_model | StrOutputParser())
reflection_chain = (reflection_prompt | chat_model | StrOutputParser())

def check_correctness(ground_ans, proposed_ans):
    ground_truth = get_gsm8k_answer(ground_ans)
    cur_ans = get_last_number(proposed_ans)
    # print(proposed_ans)
    # print(cur_ans)
    # print(ground_truth)
    # print(ground_truth == cur_ans)
    return ground_truth == cur_ans

reflection_time = 1
correct_cnt = 0
reflection_cnt = 0
all_test_cnt = len(gsm8k_test)
for ids, q in enumerate(gsm8k_test):
    mem = ""
    query = q['question']
    ground_truth = get_gsm8k_answer(q['answer'])
    answer = initial_chain.invoke(query)
    with open('./reflexion_evaluation.txt', 'a', encoding='utf-8') as file:
        file.write(f"\n\n\n{ids}: \n")
        file.write(f"Question: {query}\n")
        file.write(f"Prediction 0: {answer}\n")
        
    if check_correctness(q['answer'], answer):
        correct_cnt += 1
        with open('./reflexion_evaluation.txt', 'a', encoding='utf-8') as file:
            file.write(f"Gold: {q['answer']}\n")
            file.write(f"Correct!!!!!!!!!!!!!!!!!!!!!!!!\n")
    else:
        reflection_cnt += 1
        for t in range(reflection_time):
            new_reflection = reflection_chain.invoke({'question' : query, 'answer' :answer, 'mem': mem})
            mem += f"Reflection at time{t}: {new_reflection}\n"
            answer = actor_chain.invoke({'question' : query, 'answer' :answer, 'mem': mem})
            with open('./reflexion_evaluation.txt', 'a', encoding='utf-8') as file:
                file.write(f"Reflection {t}: {new_reflection}\n")
                file.write(f"Prediction {t+1}: {answer}\n")
                file.write(f"Gold: {q['answer']}\n")
            if check_correctness(q['answer'], answer):
                correct_cnt += 1
                with open('./reflexion_evaluation.txt', 'a', encoding='utf-8') as file:
                    file.write(f"Correct!!!!!!!!!!!!!!!!!!!!!!!!\n")
                break

with open('./reflexion_evaluation.txt', 'a', encoding='utf-8') as file:
        file.write(f"\n\n\correct_cnt: {correct_cnt}: \n")
        file.write(f"reflection_cnt: {reflection_cnt}: \n")
        file.write(f"all_test_cnt: {all_test_cnt}: \n")
        file.write(f"accuracy: {correct_cnt/all_test_cnt}: \n")
        
        


# Use QA Chain
# chain = load_qa_chain(chat_model, chain_type="stuff",verbose=True)
# query = "What are the emotional benefits of owning a pet?"
# answer = chain.run(input_documents = '',question=query)
# print(answer)

# Try PromptTemplate
# prompt = PromptTemplate(
#     input_variables=["product"],
#     template="What is a good name for a company that makes {product}?",
# )

# print(prompt.format(product="sockets"))

# chain = (prompt | chat_model|StrOutputParser())
# answer = chain.invoke("colorful socks")
# print(answer)
