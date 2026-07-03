# GSM8K GRPO Eval Preview

- source_file: `datasets\gsm8k_sft\eval_20.parquet`
- source_rows_previewed: 3
- output_rows_total: 20

## Sample 0

### Original Messages

```text
user: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? Let's think step by step and output the final answer after "####".

assistant: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.
#### 18

```

### Converted GRPO Row

```text
data_source: openai/gsm8k
ability: math
prompt: [{'role': 'user', 'content': 'Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers\' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers\' market? Let\'s think step by step and output the final answer after "####".'}]
reward_model: {'style': 'rule', 'ground_truth': '18'}
extra_info: {'split': 'eval_20', 'index': 0, 'question': "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?", 'answer': 'Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.\n#### 18'}
```

## Sample 1

### Original Messages

```text
user: A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take? Let's think step by step and output the final answer after "####".

assistant: It takes 2/2=<<2/2=1>>1 bolt of white fiber
So the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric
#### 3

```

### Converted GRPO Row

```text
data_source: openai/gsm8k
ability: math
prompt: [{'role': 'user', 'content': 'A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take? Let\'s think step by step and output the final answer after "####".'}]
reward_model: {'style': 'rule', 'ground_truth': '3'}
extra_info: {'split': 'eval_20', 'index': 1, 'question': 'A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?', 'answer': 'It takes 2/2=<<2/2=1>>1 bolt of white fiber\nSo the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric\n#### 3'}
```

## Sample 2

### Original Messages

```text
user: Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make? Let's think step by step and output the final answer after "####".

assistant: The cost of the house and repairs came out to 80,000+50,000=$<<80000+50000=130000>>130,000
He increased the value of the house by 80,000*1.5=<<80000*1.5=120000>>120,000
So the new value of the house is 120,000+80,000=$<<120000+80000=200000>>200,000
So he made a profit of 200,000-130,000=$<<200000-130000=70000>>70,000
#### 70000

```

### Converted GRPO Row

```text
data_source: openai/gsm8k
ability: math
prompt: [{'role': 'user', 'content': 'Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make? Let\'s think step by step and output the final answer after "####".'}]
reward_model: {'style': 'rule', 'ground_truth': '70000'}
extra_info: {'split': 'eval_20', 'index': 2, 'question': 'Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?', 'answer': 'The cost of the house and repairs came out to 80,000+50,000=$<<80000+50000=130000>>130,000\nHe increased the value of the house by 80,000*1.5=<<80000*1.5=120000>>120,000\nSo the new value of the house is 120,000+80,000=$<<120000+80000=200000>>200,000\nSo he made a profit of 200,000-130,000=$<<200000-130000=70000>>70,000\n#### 70000'}
```
