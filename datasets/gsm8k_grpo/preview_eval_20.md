# RL Prompt 数据预览

- source_file: `datasets\gsm8k_grpo\eval_20.parquet`
- preview_rows: 5

## Sample 0

- **原始问题**: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fre
- **user_content**: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? Let's think step by
- **assistant 回答**: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.
#### 18
- **抽取的 ground_truth**: `18`
- **可作为 GRPO/PPO prompt?**: 是

---

## Sample 1

- **原始问题**: A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?
- **user_content**: A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take? Let's think step by step and output the final answer after "####".
- **assistant 回答**: It takes 2/2=<<2/2=1>>1 bolt of white fiber
So the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric
#### 3
- **抽取的 ground_truth**: `3`
- **可作为 GRPO/PPO prompt?**: 是

---

## Sample 2

- **原始问题**: Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?
- **user_content**: Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make? Let's think step by step and output the final answer after "####".
- **assistant 回答**: The cost of the house and repairs came out to 80,000+50,000=$<<80000+50000=130000>>130,000
He increased the value of the house by 80,000*1.5=<<80000*1.5=120000>>120,000
So the new value of the house i
- **抽取的 ground_truth**: `70000`
- **可作为 GRPO/PPO prompt?**: 是

---

## Sample 3

- **原始问题**: James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week?
- **user_content**: James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week? Let's think step by step and output the final answer after "####".
- **assistant 回答**: He sprints 3*3=<<3*3=9>>9 times
So he runs 9*60=<<9*60=540>>540 meters
#### 540
- **抽取的 ground_truth**: `540`
- **可作为 GRPO/PPO prompt?**: 是

---

## Sample 4

- **原始问题**: Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy.  She gives the chickens their feed in three separate
- **user_content**: Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy.  She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed.  In the afternoon, she give
- **assistant 回答**: If each chicken eats 3 cups of feed per day, then for 20 chickens they would need 3*20=<<3*20=60>>60 cups of feed per day.
If she feeds the flock 15 cups of feed in the morning, and 25 cups in the aft
- **抽取的 ground_truth**: `20`
- **可作为 GRPO/PPO prompt?**: 是

---
