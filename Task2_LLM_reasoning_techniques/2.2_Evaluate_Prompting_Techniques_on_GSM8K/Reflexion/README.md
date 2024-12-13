# Reflexion steps
1. CoT prompt 第一次生成解答，得到（Q，A）.
2. 对照正确答案，判定正误.
3. 将当前（Q，A）+ 正/误 + mem + self-reflection prompt 输入 LLM 得到 feedback，将 feedback 加入 mem.
   
# 运行方法
python reflexion.py 

# 测试结果
- all_correct_cnt: 1244
- all_test_cnt: 1319
- accuracy: **94.31%**
- reflexion_cnt: 104
- reflexion_correct: 29
- 经reflexion做对的比例: **27.9%**
  
- 测试过程的记录在 reflexion_evaluation.txt 
- 挑选了一些经过reflexion做对的样例在 reflexion_success_case.txt

# 参考

https://blog.csdn.net/beingstrong/article/details/139902804

https://github.com/noahshinn/reflexion

https://github.com/UmerHA/langchain/blob/2316-reflexion/langchain/agents/reflexion/alfworld_prompt.py

https://www.analyticsvidhya.com/blog/2023/07/generative-ai-applications-using-langchain-and-openai-api/#h-installing-the-project-dependencies

