# GSM8K GRPO Train Preview

- source_file: `datasets\gsm8k_sft\train.parquet`
- source_rows_previewed: 3
- output_rows_total: 7473

## Sample 0

### Original Messages

```text
user: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Let's think step by step and output the final answer after "####".

assistant: Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
#### 72

```

### Converted GRPO Row

```text
data_source: openai/gsm8k
ability: math
prompt: [{'role': 'user', 'content': 'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Let\'s think step by step and output the final answer after "####".'}]
reward_model: {'style': 'rule', 'ground_truth': '72'}
extra_info: {'split': 'train', 'index': 0, 'question': 'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?', 'answer': 'Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72'}
```

## Sample 1

### Original Messages

```text
user: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn? Let's think step by step and output the final answer after "####".

assistant: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10

```

### Converted GRPO Row

```text
data_source: openai/gsm8k
ability: math
prompt: [{'role': 'user', 'content': 'Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn? Let\'s think step by step and output the final answer after "####".'}]
reward_model: {'style': 'rule', 'ground_truth': '10'}
extra_info: {'split': 'train', 'index': 1, 'question': 'Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?', 'answer': 'Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10'}
```

## Sample 2

### Original Messages

```text
user: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet? Let's think step by step and output the final answer after "####".

assistant: In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.
Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30.
This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.
#### 5

```

### Converted GRPO Row

```text
data_source: openai/gsm8k
ability: math
prompt: [{'role': 'user', 'content': 'Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet? Let\'s think step by step and output the final answer after "####".'}]
reward_model: {'style': 'rule', 'ground_truth': '5'}
extra_info: {'split': 'train', 'index': 2, 'question': 'Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?', 'answer': "In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.\nBetty's grandparents gave her 15 * 2 = $<<15*2=30>>30.\nThis means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.\n#### 5"}
```
