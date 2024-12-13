# 文件结构:
- README.md
- Reflexion 实现reflection
  - README.md
  - reflexion.py 
  - reflexion_evaluation.txt 在GSM8K的测试输出
  - reflexion_success_case.txt 经过reflection成功的样例
- opencompass_CoT_ICL 用opencompass在GSM8K上测试CoT与ICL
  - README.md
  - note.md
  - opencompass

# Configuration

```python
models = [
    dict(
        # abbr='internlm2.5-latest',  # 可选的模型缩写
        type=OpenAISDK,  # 模型类型
        path='deepseek-chat',  # 请求服务时的 model name
        # 换成自己申请的APIkey
        key=internlm_api_key,  # API key
        openai_api_base=internlm_url,  # 服务地址
        rpm_verbose=True,  # 是否打印请求速率
        query_per_second=0.16,  # 服务请求速率（每秒请求次数）
        max_out_len=1024,  # 最大输出长度
        max_seq_len=4096,  # 最大输入长度
        temperature=0.0,  # 生成温度
        batch_size=1,  # 批处理大小
        retry=3,  # 重试次数
    )
]
```
# 测试结果:
- few-shot COT: **90.07**
- COT best prompt: **78.77**
- COT official prompt: **76.88**
- zero-shot: **79.23**
- Reflexion: **94.31**

# Case

## Zero-Shot: Wrong answer, but have chain of thought.
```
"901": {
        "origin_prompt": "Question: Calvin is making soup for his family for dinner. He has a pot with enough soup to fill four adult's bowls or eight child's bowls. He is an adult and will be eating with his adult wife and their two children. If everyone eats one bowl at a meal, how many times will each child be able to have a bowl of soup for lunch from the leftover soup?\nAnswer:",
        "prediction": "To determine how many times each child will be able to have a bowl of soup for lunch from the leftover soup, we need to follow these steps:\n\n1. **Calculate the total number of bowls needed for dinner:**\n   - Calvin (adult) needs 1 bowl.\n   - Calvin's wife (adult) needs 1 bowl.\n   - Their two children each need 1 bowl.\n   - Total bowls needed for dinner: \\(1 + 1 + 1 + 1 = 4\\) bowls.\n\n2. **Determine the total capacity of the pot:**\n   - The pot can fill 4 adult bowls or 8 child's bowls.\n   - Since 4 adult bowls = 8 child's bowls, the pot can fill 8 child's bowls.\n\n3. **Calculate the leftover soup after dinner:**\n   - Total bowls needed for dinner: 4 bowls.\n   - Total capacity of the pot: 8 child's bowls.\n   - Leftover bowls: \\(8 - 4 = 4\\) child's bowls.\n\n4. **Determine how many times each child can have a bowl of soup for lunch:**\n   - Leftover bowls: 4 child's bowls.\n   - Number of children: 2.\n   - Each child can have a bowl of soup for lunch: \\(\\frac{4 \\text{ bowls}}{2 \\text{ children}} = 2\\) times.\n\nTherefore, each child will be able to have a bowl of soup for lunch 2 times from the leftover soup.",
        "gold": "Four adult bowls equal eight children’s bowls, so each adult bowl is 8 / 4 = <<8/4=2>>2 children’s bowls.\nCalvin and his wife will eat two adult bowls, so they will eat the equivalent of 2 * 2 = <<2*2=4>>4 children’s bowls.\nThus, he, his wife, and their children will eat 4 + 2 = <<4+2=6>>6 children’s bowls at dinner.\nAfter dinner, there will be 8 - 6 = <<8-6=2>>2 children’s bowls left over.\nThere are 2 children, so each child will be able to eat a bowl of leftover soup for lunch 2 / 2 = <<2/2=1>>1 time.\n#### 1"
    },
```

## Few-Shot-CoT: Correct! With demonstrations LLM can generate correct deduction chain.
```
"901": {
        "origin_prompt": "Question: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?\nLet's think step by step\nAnswer:\nAngelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.\nFor the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.\nAngelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.\nHowever, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, so 12 total hours x 10 minutes = 120 extra minutes for breaks.\nThey also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.\nAnd they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.\nSo Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.\nThey want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75\nThey will need to plan to study 4 days to allow for all the time they need.\nThe answer is 4\n\nQuestion: Mark's basketball team scores 25 2 pointers, 8 3 pointers and 10 free throws.  Their opponents score double the 2 pointers but half the 3 pointers and free throws.  What's the total number of points scored by both teams added together?\nLet's think step by step\nAnswer:\nMark's team scores 25 2 pointers, meaning they scored 25*2= 50 points in 2 pointers.\nHis team also scores 6 3 pointers, meaning they scored 8*3= 24 points in 3 pointers\nThey scored 10 free throws, and free throws count as one point so they scored 10*1=10 points in free throws.\nAll together his team scored 50+24+10= 84 points\nMark's opponents scored double his team's number of 2 pointers, meaning they scored 50*2=100 points in 2 pointers.\nHis opponents scored half his team's number of 3 pointers, meaning they scored 24/2= 12 points in 3 pointers.\nThey also scored half Mark's team's points in free throws, meaning they scored 10/2=5 points in free throws.\nAll together Mark's opponents scored 100+12+5=117 points\nThe total score for the game is both team's scores added together, so it is 84+117=201 points\nThe answer is 201\n\nQuestion: Bella has two times as many marbles as frisbees. She also has 20 more frisbees than deck cards. If she buys 2/5 times more of each item, what would be the total number of the items she will have if she currently has 60 marbles?\nLet's think step by step\nAnswer:\nWhen Bella buys 2/5 times more marbles, she'll have increased the number of marbles by 2/5*60 = 24\nThe total number of marbles she'll have is 60+24 = 84\nIf Bella currently has 60 marbles, and she has two times as many marbles as frisbees, she has 60/2 = 30 frisbees.\nIf Bella buys 2/5 times more frisbees, she'll have 2/5*30 = 12 more frisbees.\nThe total number of frisbees she'll have will increase to 30+12 = 42\nBella also has 20 more frisbees than deck cards, meaning she has 30-20 = 10 deck cards\nIf she buys 2/5 times more deck cards, she'll have 2/5*10 = 4 more deck cards.\nThe total number of deck cards she'll have is 10+4 = 14\nTogether, Bella will have a total of 14+42+84 = 140 items\nThe answer is 140\n\nQuestion: A group of 4 fruit baskets contains 9 apples, 15 oranges, and 14 bananas in the first three baskets and 2 less of each fruit in the fourth basket. How many fruits are there?\nLet's think step by step\nAnswer:\nFor the first three baskets, the number of apples and oranges in one basket is 9+15=24\nIn total, together with bananas, the number of fruits in one basket is 24+14=38 for the first three baskets.\nSince there are three baskets each having 38 fruits, there are 3*38=114 fruits in the first three baskets.\nThe number of apples in the fourth basket is 9-2=7\nThere are also 15-2=13 oranges in the fourth basket\nThe combined number of oranges and apples in the fourth basket is 13+7=20\nThe fourth basket also contains 14-2=12 bananas.\nIn total, the fourth basket has 20+12=32 fruits.\nThe four baskets together have 32+114=146 fruits.\nThe answer is 146\n\nQuestion: Calvin is making soup for his family for dinner. He has a pot with enough soup to fill four adult's bowls or eight child's bowls. He is an adult and will be eating with his adult wife and their two children. If everyone eats one bowl at a meal, how many times will each child be able to have a bowl of soup for lunch from the leftover soup?\nLet's think step by step\nAnswer:",
        "prediction": "First, let's determine how much soup Calvin's family will consume for dinner. There are 2 adults (Calvin and his wife) and 2 children in the family. Since the pot can fill four adult's bowls or eight child's bowls, we can find the equivalent number of child's bowls for the adults.\n\nEach adult's bowl is equivalent to 2 child's bowls (since 4 adult's bowls = 8 child's bowls). So, Calvin and his wife will consume 2 adults * 2 child's bowls = 4 child's bowls.\n\nNow, let's add the children's bowls to the total: 4 child's bowls (adults) + 2 child's bowls (children) = 6 child's bowls.\n\nThe pot can fill 8 child's bowls, so there will be 8 - 6 = 2 child's bowls of soup leftover after dinner.\n\nSince each child can have one bowl of soup for lunch, the leftover soup will be enough for 2 bowls / 2 children = 1 bowl per child.\n\nThe answer is 1.",
        "gold": "Four adult bowls equal eight children’s bowls, so each adult bowl is 8 / 4 = <<8/4=2>>2 children’s bowls.\nCalvin and his wife will eat two adult bowls, so they will eat the equivalent of 2 * 2 = <<2*2=4>>4 children’s bowls.\nThus, he, his wife, and their children will eat 4 + 2 = <<4+2=6>>6 children’s bowls at dinner.\nAfter dinner, there will be 8 - 6 = <<8-6=2>>2 children’s bowls left over.\nThere are 2 children, so each child will be able to eat a bowl of leftover soup for lunch 2 / 2 = <<2/2=1>>1 time.\n#### 1"
    },
```
## Zero-Shot-CoT: Wrong answer. No much difference between zero shot.
```
"901": {
        "origin_prompt": "Question: Calvin is making soup for his family for dinner. He has a pot with enough soup to fill four adult's bowls or eight child's bowls. He is an adult and will be eating with his adult wife and their two children. If everyone eats one bowl at a meal, how many times will each child be able to have a bowl of soup for lunch from the leftover soup?\nLet's think step by step\nAnswer:",
        "prediction": "Step 1: Determine the total amount of soup in terms of child's bowls.\nCalvin has enough soup to fill four adult's bowls or eight child's bowls. So, the total amount of soup is equivalent to 8 child's bowls.\n\nStep 2: Calculate the amount of soup consumed by the family during dinner.\nCalvin and his wife are adults, so they will consume 2 adult's bowls of soup (1 bowl each). Their two children will consume 2 child's bowls of soup (1 bowl each).\n\nStep 3: Calculate the leftover soup after dinner.\nTotal soup consumed during dinner = 2 adult's bowls + 2 child's bowls = 4 child's bowls (since 1 adult's bowl = 2 child's bowls).\nLeftover soup = Total soup - Soup consumed during dinner = 8 child's bowls - 4 child's bowls = 4 child's bowls.\n\nStep 4: Determine how many times each child can have a bowl of soup for lunch from the leftover soup.\nThere are 2 children, and the leftover soup is equivalent to 4 child's bowls. Since each child eats 1 bowl at a meal, each child can have a bowl of soup for lunch 2 times (4 bowls / 2 children = 2 bowls per child).\n\nAnswer: Each child will be able to have a bowl of soup for lunch 2 times from the leftover soup.",
        "gold": "Four adult bowls equal eight children’s bowls, so each adult bowl is 8 / 4 = <<8/4=2>>2 children’s bowls.\nCalvin and his wife will eat two adult bowls, so they will eat the equivalent of 2 * 2 = <<2*2=4>>4 children’s bowls.\nThus, he, his wife, and their children will eat 4 + 2 = <<4+2=6>>6 children’s bowls at dinner.\nAfter dinner, there will be 8 - 6 = <<8-6=2>>2 children’s bowls left over.\nThere are 2 children, so each child will be able to eat a bowl of leftover soup for lunch 2 / 2 = <<2/2=1>>1 time.\n#### 1"
    },
```

## Reflexion: Correct
```
901: 
Question: Calvin is making soup for his family for dinner. He has a pot with enough soup to fill four adult's bowls or eight child's bowls. He is an adult and will be eating with his adult wife and their two children. If everyone eats one bowl at a meal, how many times will each child be able to have a bowl of soup for lunch from the leftover soup?

Prediction 0: Sure, let's break it down step by step:

1. **Total Soup Capacity**: The pot can fill either 4 adult bowls or 8 child bowls.

2. **Dinner Consumption**:
   - Calvin (adult): 1 bowl
   - Calvin's wife (adult): 1 bowl
   - Two children: 2 bowls (1 bowl each)

   Total bowls consumed at dinner: \(1 + 1 + 2 = 4\) bowls.

3. **Leftover Soup**:
   - The pot can fill 4 adult bowls, but only 4 bowls are consumed at dinner.
   - Therefore, there are \(4 - 4 = 0\) bowls left after dinner.

4. **Lunch for Children**:
   - Since there are no bowls left after dinner, the children cannot have any soup for lunch from the leftover soup.

**Answer**: 0

Reflection 0: Your new reflection:

In the previous attempt, the mistake was in assuming that the pot's capacity was fully utilized during dinner. Specifically, the calculation incorrectly treated the pot as if it could only serve 4 bowls in total, regardless of whether they were adult or child bowls. This led to the conclusion that there were no leftovers, which was incorrect.

To avoid this mistake in the future, the strategy should be revised as follows:

1. **Clarify the Pot's Capacity**: Recognize that the pot can fill either 4 adult bowls or 8 child bowls. This means the pot has a total capacity of 8 child bowls, not just 4 bowls regardless of size.

2. **Calculate Dinner Consumption in Child Bowls**:
   - Convert the adult bowls consumed at dinner into child bowls to maintain consistency in the calculation.
   - Calvin (adult): 1 bowl = 2 child bowls
   - Calvin's wife (adult): 1 bowl = 2 child bowls
   - Two children: 2 bowls (1 bowl each)

   Total child bowls consumed at dinner: \(2 + 2 + 2 = 6\) child bowls.

3. **Determine Leftover Soup**:
   - The pot can fill 8 child bowls.
   - After dinner, \(8 - 6 = 2\) child bowls are left.

4. **Calculate Lunch for Children**:
   - Each child can have 1 bowl of soup for lunch.
   - With 2 child bowls left, each child can have 1 bowl of soup for lunch.

**Answer**: 1.

By converting all measurements to child bowls and ensuring consistency in the calculation, the correct number of leftover bowls can be determined, leading to the accurate answer.

Prediction 1: 1. **Total Soup Capacity**: The pot can fill either 4 adult bowls or 8 child bowls.

2. **Dinner Consumption**:
   - Calvin (adult): 1 bowl = 2 child bowls
   - Calvin's wife (adult): 1 bowl = 2 child bowls
   - Two children: 2 bowls (1 bowl each)

   Total child bowls consumed at dinner: \(2 + 2 + 2 = 6\) child bowls.

3. **Leftover Soup**:
   - The pot can fill 8 child bowls.
   - After dinner, \(8 - 6 = 2\) child bowls are left.

4. **Lunch for Children**:
   - Each child can have 1 bowl of soup for lunch.
   - With 2 child bowls left, each child can have 1 bowl of soup for lunch.

**Answer**: 1
Gold: Four adult bowls equal eight children’s bowls, so each adult bowl is 8 / 4 = <<8/4=2>>2 children’s bowls.
Calvin and his wife will eat two adult bowls, so they will eat the equivalent of 2 * 2 = <<2*2=4>>4 children’s bowls.
Thus, he, his wife, and their children will eat 4 + 2 = <<4+2=6>>6 children’s bowls at dinner.
After dinner, there will be 8 - 6 = <<8-6=2>>2 children’s bowls left over.
There are 2 children, so each child will be able to eat a bowl of leftover soup for lunch 2 / 2 = <<2/2=1>>1 time.
#### 1
Correct!!!!!!!!!!!!!!!!!!!!!!!!
```

## Analysis
- Zero-Shot failed to differentiate adult's bowl and child's bowl.
- Zero-Shot-CoT understood the relationship between the adult's bowl and child's bowl, but failed in the calculation. 
"Total soup consumed during dinner = 2 adult's bowls + 2 child's bowls = 4 child's bowls (since 1 adult's bowl = 2 child's bowls)"

- In Reflexion, the self-reflection model captured the mistakes and gave correct feedback. 
"Specifically, the calculation incorrectly treated the pot as if it could only serve 4 bowls in total, regardless of whether they were adult or child bowls. This led to the conclusion that there were no leftovers, which was incorrect."
"Recognize that the pot can fill either 4 adult bowls or 8 child bowls. This means the pot has a total capacity of 8 child bowls, not just 4 bowls regardless of size.
"