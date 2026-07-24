# RL Prompt 数据预览

- source_file: `datasets\gsm8k_grpo\train.parquet`
- preview_rows: 5

## Sample 0

- **原始问题**: Hugo's mountain has an elevation of 10,000 feet above sea level.  Boris' mountain has an elevation that is 2,500 feet shorter than  Hugo's mountain.  If Hugo climbed his mountain 3 times, how many tim
- **user_content**: Hugo's mountain has an elevation of 10,000 feet above sea level.  Boris' mountain has an elevation that is 2,500 feet shorter than  Hugo's mountain.  If Hugo climbed his mountain 3 times, how many times would Boris need to climb his mountain to have climbed the same number of feet as Hugo? Let's thi
- **assistant 回答**: Boris' mountain has an elevation 2,500 feet shorter than Hugo's mountain, or 10,000-2,500=<<10000-2500=7500>>7500 feet above sea level.
If Hugo climbed his mountain 3 times, his total climbing distanc
- **抽取的 ground_truth**: `4`
- **可作为 GRPO/PPO prompt?**: 是

---

## Sample 1

- **原始问题**: Mark has an egg farm. His farm supplies one store with 5 dozen eggs and another store with 30 eggs each day. How many eggs does he supply these two stores in a week?
- **user_content**: Mark has an egg farm. His farm supplies one store with 5 dozen eggs and another store with 30 eggs each day. How many eggs does he supply these two stores in a week? Let's think step by step and output the final answer after "####".
- **assistant 回答**: Since a dozen is equal to 12, then Mark's egg farm supplies one store with 5 x 12 = <<5*12=60>>60 eggs each day.
It supplies the two stores with a total of 60 + 30 = <<60+30=90>>90 eggs each day.
Ther
- **抽取的 ground_truth**: `630`
- **可作为 GRPO/PPO prompt?**: 是

---

## Sample 2

- **原始问题**: Happy Island has 60 turtles. This means there are 10 more than 2 times as many turtles on Happy Island as on Lonely Island. How many turtles are there on Lonely Island?
- **user_content**: Happy Island has 60 turtles. This means there are 10 more than 2 times as many turtles on Happy Island as on Lonely Island. How many turtles are there on Lonely Island? Let's think step by step and output the final answer after "####".
- **assistant 回答**: After subtracting 10 from the number of turtles on Happy Island, we get 60 - 10 = <<60-10=50>>50 turtles
Dividing this number by two, we find that the number of turtles on Lonely Island is 50/2=<<50/2
- **抽取的 ground_truth**: `25`
- **可作为 GRPO/PPO prompt?**: 是

---

## Sample 3

- **原始问题**: A toy car factory made 60 cars yesterday. Today, they made twice the number of cars than yesterday. How many toy cars did the factory make?
- **user_content**: A toy car factory made 60 cars yesterday. Today, they made twice the number of cars than yesterday. How many toy cars did the factory make? Let's think step by step and output the final answer after "####".
- **assistant 回答**: They made 60 x 2 = <<60*2=120>>120 cars today.
So, the factory made a total of 60 + 120 = <<60+120=180>>180 cars.
#### 180
- **抽取的 ground_truth**: `180`
- **可作为 GRPO/PPO prompt?**: 是

---

## Sample 4

- **原始问题**: A teacher has to order supplies for his class to do a science project. Each student needs a bow, a small bottle of vinegar and a box of baking soda. Bows are $5 each, a bottle of vinegar is $2 and a b
- **user_content**: A teacher has to order supplies for his class to do a science project. Each student needs a bow, a small bottle of vinegar and a box of baking soda. Bows are $5 each, a bottle of vinegar is $2 and a box of baking soda is $1. The teacher has 23 students in this class. How much will the supplies cost?
- **assistant 回答**: The supplies for every student will cost 5 + 2 + 1 = $<<5+2+1=8>>8.
The supplies for the whole class will cost 8 * 23 = $<<8*23=184>>184 in total.
#### 184
- **抽取的 ground_truth**: `184`
- **可作为 GRPO/PPO prompt?**: 是

---
