# SFT 训练样本 tokenize 后预览

以下是 `MessageSFTDataset` tokenize 后实际喂给模型的 `input_ids` 和 `labels`。

- `labels=-100` 的位置：**不计算 loss**（user / prompt 部分）
- `labels=token_id` 的位置：**计算 loss**（assistant 回答部分）

每个 token 用 `` `token_text`(token_id) `` 表示，labels 在 `[]` 内标注。

---

## 样本 0

- 总 token 数: 153
- prompt (user) token 数: 92 → labels=-100
- assistant token 数: 61 → labels=token_id

### user 部分 (labels=-100, 不计算 loss)

```text
   0  '<|im_start|>'  token_id=151644  label=-100
   1  'user'  token_id=872  label=-100
   2  '\\n'  token_id=198  label=-100
   3  'Jan'  token_id=18315  label=-100
   4  'et'  token_id=295  label=-100
   5  '’s'  token_id=748  label=-100
   6  ' ducks'  token_id=77778  label=-100
   7  ' lay'  token_id=10962  label=-100
   8  ' '  token_id=220  label=-100
   9  '1'  token_id=16  label=-100
  10  '6'  token_id=21  label=-100
  11  ' eggs'  token_id=18805  label=-100
  12  ' per'  token_id=817  label=-100
  13  ' day'  token_id=1899  label=-100
  14  '.'  token_id=13  label=-100
  15  ' She'  token_id=2932  label=-100
  16  ' eats'  token_id=49677  label=-100
  17  ' three'  token_id=2326  label=-100
  18  ' for'  token_id=369  label=-100
  19  ' breakfast'  token_id=17496  label=-100
  20  ' every'  token_id=1449  label=-100
  21  ' morning'  token_id=6556  label=-100
  22  ' and'  token_id=323  label=-100
  23  ' b'  token_id=293  label=-100
  24  'akes'  token_id=2050  label=-100
  25  ' muff'  token_id=54304  label=-100
  26  'ins'  token_id=1330  label=-100
  27  ' for'  token_id=369  label=-100
  28  ' her'  token_id=1059  label=-100
  29  ' friends'  token_id=4780  label=-100
  30  ' every'  token_id=1449  label=-100
  31  ' day'  token_id=1899  label=-100
  32  ' with'  token_id=448  label=-100
  33  ' four'  token_id=3040  label=-100
  34  '.'  token_id=13  label=-100
  35  ' She'  token_id=2932  label=-100
  36  ' sells'  token_id=30778  label=-100
  37  ' the'  token_id=279  label=-100
  38  ' remainder'  token_id=26313  label=-100
  39  ' at'  token_id=518  label=-100
  40  ' the'  token_id=279  label=-100
  41  ' farmers'  token_id=20336  label=-100
  42  "'"  token_id=6  label=-100
  43  ' market'  token_id=3081  label=-100
  44  ' daily'  token_id=7298  label=-100
  45  ' for'  token_id=369  label=-100
  46  ' $'  token_id=400  label=-100
  47  '2'  token_id=17  label=-100
  48  ' per'  token_id=817  label=-100
  49  ' fresh'  token_id=7722  label=-100
  50  ' duck'  token_id=35985  label=-100
  51  ' egg'  token_id=18636  label=-100
  52  '.'  token_id=13  label=-100
  53  ' How'  token_id=2585  label=-100
  54  ' much'  token_id=1753  label=-100
  55  ' in'  token_id=304  label=-100
  56  ' dollars'  token_id=11192  label=-100
  57  ' does'  token_id=1558  label=-100
  58  ' she'  token_id=1340  label=-100
  59  ' make'  token_id=1281  label=-100
  60  ' every'  token_id=1449  label=-100
  61  ' day'  token_id=1899  label=-100
  62  ' at'  token_id=518  label=-100
  63  ' the'  token_id=279  label=-100
  64  ' farmers'  token_id=20336  label=-100
  65  "'"  token_id=6  label=-100
  66  ' market'  token_id=3081  label=-100
  67  '?'  token_id=30  label=-100
  68  ' Let'  token_id=6771  label=-100
  69  "'s"  token_id=594  label=-100
  70  ' think'  token_id=1744  label=-100
  71  ' step'  token_id=3019  label=-100
  72  ' by'  token_id=553  label=-100
  73  ' step'  token_id=3019  label=-100
  74  ' and'  token_id=323  label=-100
  75  ' output'  token_id=2550  label=-100
  76  ' the'  token_id=279  label=-100
  77  ' final'  token_id=1590  label=-100
  78  ' answer'  token_id=4226  label=-100
  79  ' after'  token_id=1283  label=-100
  80  ' "'  token_id=330  label=-100
  81  '####'  token_id=820  label=-100
  82  '".'  token_id=3263  label=-100
  83  '<|im_end|>'  token_id=151645  label=-100
  84  '\\n'  token_id=198  label=-100
  85  '<|im_start|>'  token_id=151644  label=-100
  86  'assistant'  token_id=77091  label=-100
  87  '\\n'  token_id=198  label=-100
  88  '<think>'  token_id=151667  label=-100
  89  '\\n\\n'  token_id=271  label=-100
  90  '</think>'  token_id=151668  label=-100
  91  '\\n\\n'  token_id=271  label=-100
```

### assistant 部分 (labels=token_id, 计算 loss)

```text
  92  'Jan'  token_id=18315  label=18315
  93  'et'  token_id=295  label=295
  94  ' sells'  token_id=30778  label=30778
  95  ' '  token_id=220  label=220
  96  '1'  token_id=16  label=16
  97  '6'  token_id=21  label=21
  98  ' -'  token_id=481  label=481
  99  ' '  token_id=220  label=220
 100  '3'  token_id=18  label=18
 101  ' -'  token_id=481  label=481
 102  ' '  token_id=220  label=220
 103  '4'  token_id=19  label=19
 104  ' ='  token_id=284  label=284
 105  ' <<'  token_id=1115  label=1115
 106  '1'  token_id=16  label=16
 107  '6'  token_id=21  label=21
 108  '-'  token_id=12  label=12
 109  '3'  token_id=18  label=18
 110  '-'  token_id=12  label=12
 111  '4'  token_id=19  label=19
 112  '='  token_id=28  label=28
 113  '9'  token_id=24  label=24
 114  '>>'  token_id=2452  label=2452
 115  '9'  token_id=24  label=24
 116  ' duck'  token_id=35985  label=35985
 117  ' eggs'  token_id=18805  label=18805
 118  ' a'  token_id=264  label=264
 119  ' day'  token_id=1899  label=1899
 120  '.\\n'  token_id=624  label=624
 121  'She'  token_id=7941  label=7941
 122  ' makes'  token_id=3643  label=3643
 123  ' '  token_id=220  label=220
 124  '9'  token_id=24  label=24
 125  ' *'  token_id=353  label=353
 126  ' '  token_id=220  label=220
 127  '2'  token_id=17  label=17
 128  ' ='  token_id=284  label=284
 129  ' $'  token_id=400  label=400
 130  '<<'  token_id=2442  label=2442
 131  '9'  token_id=24  label=24
 132  '*'  token_id=9  label=9
 133  '2'  token_id=17  label=17
 134  '='  token_id=28  label=28
 135  '1'  token_id=16  label=16
 136  '8'  token_id=23  label=23
 137  '>>'  token_id=2452  label=2452
 138  '1'  token_id=16  label=16
 139  '8'  token_id=23  label=23
 140  ' every'  token_id=1449  label=1449
 141  ' day'  token_id=1899  label=1899
 142  ' at'  token_id=518  label=518
 143  ' the'  token_id=279  label=279
 144  ' farmer'  token_id=36400  label=36400
 145  '’s'  token_id=748  label=748
 146  ' market'  token_id=3081  label=3081
 147  '.\\n'  token_id=624  label=624
 148  '####'  token_id=820  label=820
 149  ' '  token_id=220  label=220
 150  '1'  token_id=16  label=16
 151  '8'  token_id=23  label=23
 152  '<|im_end|>'  token_id=151645  label=151645
```

### 完整 text 解码对照

```text
<|im_start|>user
Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? Let's think step by step and output the final answer after "####".<|im_end|>
<|im_start|>assistant
<think>

</think>

Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.
#### 18<|im_end|>
```

**prompt 部分解码:**

```text
<|im_start|>user
Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? Let's think step by step and output the final answer after "####".<|im_end|>
<|im_start|>assistant
<think>

</think>


```

**assistant 部分解码 (这部分计算 loss):**

```text
Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.
#### 18<|im_end|>
```

---

## 样本 1

- 总 token 数: 101
- prompt (user) token 数: 53 → labels=-100
- assistant token 数: 48 → labels=token_id

### user 部分 (labels=-100, 不计算 loss)

```text
   0  '<|im_start|>'  token_id=151644  label=-100
   1  'user'  token_id=872  label=-100
   2  '\\n'  token_id=198  label=-100
   3  'A'  token_id=32  label=-100
   4  ' robe'  token_id=62619  label=-100
   5  ' takes'  token_id=4990  label=-100
   6  ' '  token_id=220  label=-100
   7  '2'  token_id=17  label=-100
   8  ' bolts'  token_id=48839  label=-100
   9  ' of'  token_id=315  label=-100
  10  ' blue'  token_id=6303  label=-100
  11  ' fiber'  token_id=23788  label=-100
  12  ' and'  token_id=323  label=-100
  13  ' half'  token_id=4279  label=-100
  14  ' that'  token_id=429  label=-100
  15  ' much'  token_id=1753  label=-100
  16  ' white'  token_id=4158  label=-100
  17  ' fiber'  token_id=23788  label=-100
  18  '.'  token_id=13  label=-100
  19  ' '  token_id=220  label=-100
  20  ' How'  token_id=2585  label=-100
  21  ' many'  token_id=1657  label=-100
  22  ' bolts'  token_id=48839  label=-100
  23  ' in'  token_id=304  label=-100
  24  ' total'  token_id=2790  label=-100
  25  ' does'  token_id=1558  label=-100
  26  ' it'  token_id=432  label=-100
  27  ' take'  token_id=1896  label=-100
  28  '?'  token_id=30  label=-100
  29  ' Let'  token_id=6771  label=-100
  30  "'s"  token_id=594  label=-100
  31  ' think'  token_id=1744  label=-100
  32  ' step'  token_id=3019  label=-100
  33  ' by'  token_id=553  label=-100
  34  ' step'  token_id=3019  label=-100
  35  ' and'  token_id=323  label=-100
  36  ' output'  token_id=2550  label=-100
  37  ' the'  token_id=279  label=-100
  38  ' final'  token_id=1590  label=-100
  39  ' answer'  token_id=4226  label=-100
  40  ' after'  token_id=1283  label=-100
  41  ' "'  token_id=330  label=-100
  42  '####'  token_id=820  label=-100
  43  '".'  token_id=3263  label=-100
  44  '<|im_end|>'  token_id=151645  label=-100
  45  '\\n'  token_id=198  label=-100
  46  '<|im_start|>'  token_id=151644  label=-100
  47  'assistant'  token_id=77091  label=-100
  48  '\\n'  token_id=198  label=-100
  49  '<think>'  token_id=151667  label=-100
  50  '\\n\\n'  token_id=271  label=-100
  51  '</think>'  token_id=151668  label=-100
  52  '\\n\\n'  token_id=271  label=-100
```

### assistant 部分 (labels=token_id, 计算 loss)

```text
  53  'It'  token_id=2132  label=2132
  54  ' takes'  token_id=4990  label=4990
  55  ' '  token_id=220  label=220
  56  '2'  token_id=17  label=17
  57  '/'  token_id=14  label=14
  58  '2'  token_id=17  label=17
  59  '='  token_id=28  label=28
  60  '<<'  token_id=2442  label=2442
  61  '2'  token_id=17  label=17
  62  '/'  token_id=14  label=14
  63  '2'  token_id=17  label=17
  64  '='  token_id=28  label=28
  65  '1'  token_id=16  label=16
  66  '>>'  token_id=2452  label=2452
  67  '1'  token_id=16  label=16
  68  ' bolt'  token_id=31842  label=31842
  69  ' of'  token_id=315  label=315
  70  ' white'  token_id=4158  label=4158
  71  ' fiber'  token_id=23788  label=23788
  72  '\\n'  token_id=198  label=198
  73  'So'  token_id=4416  label=4416
  74  ' the'  token_id=279  label=279
  75  ' total'  token_id=2790  label=2790
  76  ' amount'  token_id=3311  label=3311
  77  ' of'  token_id=315  label=315
  78  ' fabric'  token_id=13055  label=13055
  79  ' is'  token_id=374  label=374
  80  ' '  token_id=220  label=220
  81  '2'  token_id=17  label=17
  82  '+'  token_id=10  label=10
  83  '1'  token_id=16  label=16
  84  '='  token_id=28  label=28
  85  '<<'  token_id=2442  label=2442
  86  '2'  token_id=17  label=17
  87  '+'  token_id=10  label=10
  88  '1'  token_id=16  label=16
  89  '='  token_id=28  label=28
  90  '3'  token_id=18  label=18
  91  '>>'  token_id=2452  label=2452
  92  '3'  token_id=18  label=18
  93  ' bolts'  token_id=48839  label=48839
  94  ' of'  token_id=315  label=315
  95  ' fabric'  token_id=13055  label=13055
  96  '\\n'  token_id=198  label=198
  97  '####'  token_id=820  label=820
  98  ' '  token_id=220  label=220
  99  '3'  token_id=18  label=18
 100  '<|im_end|>'  token_id=151645  label=151645
```

### 完整 text 解码对照

```text
<|im_start|>user
A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take? Let's think step by step and output the final answer after "####".<|im_end|>
<|im_start|>assistant
<think>

</think>

It takes 2/2=<<2/2=1>>1 bolt of white fiber
So the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric
#### 3<|im_end|>
```

**prompt 部分解码:**

```text
<|im_start|>user
A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take? Let's think step by step and output the final answer after "####".<|im_end|>
<|im_start|>assistant
<think>

</think>


```

**assistant 部分解码 (这部分计算 loss):**

```text
It takes 2/2=<<2/2=1>>1 bolt of white fiber
So the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric
#### 3<|im_end|>
```

---

## 样本 2

- 总 token 数: 295
- prompt (user) token 数: 84 → labels=-100
- assistant token 数: 211 → labels=token_id

### user 部分 (labels=-100, 不计算 loss)

```text
   0  '<|im_start|>'  token_id=151644  label=-100
   1  'user'  token_id=872  label=-100
   2  '\\n'  token_id=198  label=-100
   3  'Josh'  token_id=50744  label=-100
   4  ' decides'  token_id=27627  label=-100
   5  ' to'  token_id=311  label=-100
   6  ' try'  token_id=1430  label=-100
   7  ' flipping'  token_id=64661  label=-100
   8  ' a'  token_id=264  label=-100
   9  ' house'  token_id=3753  label=-100
  10  '.'  token_id=13  label=-100
  11  ' '  token_id=220  label=-100
  12  ' He'  token_id=1260  label=-100
  13  ' buys'  token_id=49531  label=-100
  14  ' a'  token_id=264  label=-100
  15  ' house'  token_id=3753  label=-100
  16  ' for'  token_id=369  label=-100
  17  ' $'  token_id=400  label=-100
  18  '8'  token_id=23  label=-100
  19  '0'  token_id=15  label=-100
  20  ','  token_id=11  label=-100
  21  '0'  token_id=15  label=-100
  22  '0'  token_id=15  label=-100
  23  '0'  token_id=15  label=-100
  24  ' and'  token_id=323  label=-100
  25  ' then'  token_id=1221  label=-100
  26  ' puts'  token_id=9521  label=-100
  27  ' in'  token_id=304  label=-100
  28  ' $'  token_id=400  label=-100
  29  '5'  token_id=20  label=-100
  30  '0'  token_id=15  label=-100
  31  ','  token_id=11  label=-100
  32  '0'  token_id=15  label=-100
  33  '0'  token_id=15  label=-100
  34  '0'  token_id=15  label=-100
  35  ' in'  token_id=304  label=-100
  36  ' repairs'  token_id=30186  label=-100
  37  '.'  token_id=13  label=-100
  38  ' '  token_id=220  label=-100
  39  ' This'  token_id=1096  label=-100
  40  ' increased'  token_id=7172  label=-100
  41  ' the'  token_id=279  label=-100
  42  ' value'  token_id=897  label=-100
  43  ' of'  token_id=315  label=-100
  44  ' the'  token_id=279  label=-100
  45  ' house'  token_id=3753  label=-100
  46  ' by'  token_id=553  label=-100
  47  ' '  token_id=220  label=-100
  48  '1'  token_id=16  label=-100
  49  '5'  token_id=20  label=-100
  50  '0'  token_id=15  label=-100
  51  '%.'  token_id=14360  label=-100
  52  ' '  token_id=220  label=-100
  53  ' How'  token_id=2585  label=-100
  54  ' much'  token_id=1753  label=-100
  55  ' profit'  token_id=11372  label=-100
  56  ' did'  token_id=1521  label=-100
  57  ' he'  token_id=566  label=-100
  58  ' make'  token_id=1281  label=-100
  59  '?'  token_id=30  label=-100
  60  ' Let'  token_id=6771  label=-100
  61  "'s"  token_id=594  label=-100
  62  ' think'  token_id=1744  label=-100
  63  ' step'  token_id=3019  label=-100
  64  ' by'  token_id=553  label=-100
  65  ' step'  token_id=3019  label=-100
  66  ' and'  token_id=323  label=-100
  67  ' output'  token_id=2550  label=-100
  68  ' the'  token_id=279  label=-100
  69  ' final'  token_id=1590  label=-100
  70  ' answer'  token_id=4226  label=-100
  71  ' after'  token_id=1283  label=-100
  72  ' "'  token_id=330  label=-100
  73  '####'  token_id=820  label=-100
  74  '".'  token_id=3263  label=-100
  75  '<|im_end|>'  token_id=151645  label=-100
  76  '\\n'  token_id=198  label=-100
  77  '<|im_start|>'  token_id=151644  label=-100
  78  'assistant'  token_id=77091  label=-100
  79  '\\n'  token_id=198  label=-100
  80  '<think>'  token_id=151667  label=-100
  81  '\\n\\n'  token_id=271  label=-100
  82  '</think>'  token_id=151668  label=-100
  83  '\\n\\n'  token_id=271  label=-100
```

### assistant 部分 (labels=token_id, 计算 loss)

```text
  84  'The'  token_id=785  label=785
  85  ' cost'  token_id=2783  label=2783
  86  ' of'  token_id=315  label=315
  87  ' the'  token_id=279  label=279
  88  ' house'  token_id=3753  label=3753
  89  ' and'  token_id=323  label=323
  90  ' repairs'  token_id=30186  label=30186
  91  ' came'  token_id=3697  label=3697
  92  ' out'  token_id=700  label=700
  93  ' to'  token_id=311  label=311
  94  ' '  token_id=220  label=220
  95  '8'  token_id=23  label=23
  96  '0'  token_id=15  label=15
  97  ','  token_id=11  label=11
  98  '0'  token_id=15  label=15
  99  '0'  token_id=15  label=15
 100  '0'  token_id=15  label=15
 101  '+'  token_id=10  label=10
 102  '5'  token_id=20  label=20
 103  '0'  token_id=15  label=15
 104  ','  token_id=11  label=11
 105  '0'  token_id=15  label=15
 106  '0'  token_id=15  label=15
 107  '0'  token_id=15  label=15
 108  '=$'  token_id=3186  label=3186
 109  '<<'  token_id=2442  label=2442
 110  '8'  token_id=23  label=23
 111  '0'  token_id=15  label=15
 112  '0'  token_id=15  label=15
 113  '0'  token_id=15  label=15
 114  '0'  token_id=15  label=15
 115  '+'  token_id=10  label=10
 116  '5'  token_id=20  label=20
 117  '0'  token_id=15  label=15
 118  '0'  token_id=15  label=15
 119  '0'  token_id=15  label=15
 120  '0'  token_id=15  label=15
 121  '='  token_id=28  label=28
 122  '1'  token_id=16  label=16
 123  '3'  token_id=18  label=18
 124  '0'  token_id=15  label=15
 125  '0'  token_id=15  label=15
 126  '0'  token_id=15  label=15
 127  '0'  token_id=15  label=15
 128  '>>'  token_id=2452  label=2452
 129  '1'  token_id=16  label=16
 130  '3'  token_id=18  label=18
 131  '0'  token_id=15  label=15
 132  ','  token_id=11  label=11
 133  '0'  token_id=15  label=15
 134  '0'  token_id=15  label=15
 135  '0'  token_id=15  label=15
 136  '\\n'  token_id=198  label=198
 137  'He'  token_id=1519  label=1519
 138  ' increased'  token_id=7172  label=7172
 139  ' the'  token_id=279  label=279
 140  ' value'  token_id=897  label=897
 141  ' of'  token_id=315  label=315
 142  ' the'  token_id=279  label=279
 143  ' house'  token_id=3753  label=3753
 144  ' by'  token_id=553  label=553
 145  ' '  token_id=220  label=220
 146  '8'  token_id=23  label=23
 147  '0'  token_id=15  label=15
 148  ','  token_id=11  label=11
 149  '0'  token_id=15  label=15
 150  '0'  token_id=15  label=15
 151  '0'  token_id=15  label=15
 152  '*'  token_id=9  label=9
 153  '1'  token_id=16  label=16
 154  '.'  token_id=13  label=13
 155  '5'  token_id=20  label=20
 156  '='  token_id=28  label=28
 157  '<<'  token_id=2442  label=2442
 158  '8'  token_id=23  label=23
 159  '0'  token_id=15  label=15
 160  '0'  token_id=15  label=15
 161  '0'  token_id=15  label=15
 162  '0'  token_id=15  label=15
 163  '*'  token_id=9  label=9
 164  '1'  token_id=16  label=16
 165  '.'  token_id=13  label=13
 166  '5'  token_id=20  label=20
 167  '='  token_id=28  label=28
 168  '1'  token_id=16  label=16
 169  '2'  token_id=17  label=17
 170  '0'  token_id=15  label=15
 171  '0'  token_id=15  label=15
 172  '0'  token_id=15  label=15
 173  '0'  token_id=15  label=15
 174  '>>'  token_id=2452  label=2452
 175  '1'  token_id=16  label=16
 176  '2'  token_id=17  label=17
 177  '0'  token_id=15  label=15
 178  ','  token_id=11  label=11
 179  '0'  token_id=15  label=15
 180  '0'  token_id=15  label=15
 181  '0'  token_id=15  label=15
 182  '\\n'  token_id=198  label=198
 183  'So'  token_id=4416  label=4416
 184  ' the'  token_id=279  label=279
 185  ' new'  token_id=501  label=501
 186  ' value'  token_id=897  label=897
 187  ' of'  token_id=315  label=315
 188  ' the'  token_id=279  label=279
 189  ' house'  token_id=3753  label=3753
 190  ' is'  token_id=374  label=374
 191  ' '  token_id=220  label=220
 192  '1'  token_id=16  label=16
 193  '2'  token_id=17  label=17
 194  '0'  token_id=15  label=15
 195  ','  token_id=11  label=11
 196  '0'  token_id=15  label=15
 197  '0'  token_id=15  label=15
 198  '0'  token_id=15  label=15
 199  '+'  token_id=10  label=10
 200  '8'  token_id=23  label=23
 201  '0'  token_id=15  label=15
 202  ','  token_id=11  label=11
 203  '0'  token_id=15  label=15
 204  '0'  token_id=15  label=15
 205  '0'  token_id=15  label=15
 206  '=$'  token_id=3186  label=3186
 207  '<<'  token_id=2442  label=2442
 208  '1'  token_id=16  label=16
 209  '2'  token_id=17  label=17
 210  '0'  token_id=15  label=15
 211  '0'  token_id=15  label=15
 212  '0'  token_id=15  label=15
 213  '0'  token_id=15  label=15
 214  '+'  token_id=10  label=10
 215  '8'  token_id=23  label=23
 216  '0'  token_id=15  label=15
 217  '0'  token_id=15  label=15
 218  '0'  token_id=15  label=15
 219  '0'  token_id=15  label=15
 220  '='  token_id=28  label=28
 221  '2'  token_id=17  label=17
 222  '0'  token_id=15  label=15
 223  '0'  token_id=15  label=15
 224  '0'  token_id=15  label=15
 225  '0'  token_id=15  label=15
 226  '0'  token_id=15  label=15
 227  '>>'  token_id=2452  label=2452
 228  '2'  token_id=17  label=17
 229  '0'  token_id=15  label=15
 230  '0'  token_id=15  label=15
 231  ','  token_id=11  label=11
 232  '0'  token_id=15  label=15
 233  '0'  token_id=15  label=15
 234  '0'  token_id=15  label=15
 235  '\\n'  token_id=198  label=198
 236  'So'  token_id=4416  label=4416
 237  ' he'  token_id=566  label=566
 238  ' made'  token_id=1865  label=1865
 239  ' a'  token_id=264  label=264
 240  ' profit'  token_id=11372  label=11372
 241  ' of'  token_id=315  label=315
 242  ' '  token_id=220  label=220
 243  '2'  token_id=17  label=17
 244  '0'  token_id=15  label=15
 245  '0'  token_id=15  label=15
 246  ','  token_id=11  label=11
 247  '0'  token_id=15  label=15
 248  '0'  token_id=15  label=15
 249  '0'  token_id=15  label=15
 250  '-'  token_id=12  label=12
 251  '1'  token_id=16  label=16
 252  '3'  token_id=18  label=18
 253  '0'  token_id=15  label=15
 254  ','  token_id=11  label=11
 255  '0'  token_id=15  label=15
 256  '0'  token_id=15  label=15
 257  '0'  token_id=15  label=15
 258  '=$'  token_id=3186  label=3186
 259  '<<'  token_id=2442  label=2442
 260  '2'  token_id=17  label=17
 261  '0'  token_id=15  label=15
 262  '0'  token_id=15  label=15
 263  '0'  token_id=15  label=15
 264  '0'  token_id=15  label=15
 265  '0'  token_id=15  label=15
 266  '-'  token_id=12  label=12
 267  '1'  token_id=16  label=16
 268  '3'  token_id=18  label=18
 269  '0'  token_id=15  label=15
 270  '0'  token_id=15  label=15
 271  '0'  token_id=15  label=15
 272  '0'  token_id=15  label=15
 273  '='  token_id=28  label=28
 274  '7'  token_id=22  label=22
 275  '0'  token_id=15  label=15
 276  '0'  token_id=15  label=15
 277  '0'  token_id=15  label=15
 278  '0'  token_id=15  label=15
 279  '>>'  token_id=2452  label=2452
 280  '7'  token_id=22  label=22
 281  '0'  token_id=15  label=15
 282  ','  token_id=11  label=11
 283  '0'  token_id=15  label=15
 284  '0'  token_id=15  label=15
 285  '0'  token_id=15  label=15
 286  '\\n'  token_id=198  label=198
 287  '####'  token_id=820  label=820
 288  ' '  token_id=220  label=220
 289  '7'  token_id=22  label=22
 290  '0'  token_id=15  label=15
 291  '0'  token_id=15  label=15
 292  '0'  token_id=15  label=15
 293  '0'  token_id=15  label=15
 294  '<|im_end|>'  token_id=151645  label=151645
```

### 完整 text 解码对照

```text
<|im_start|>user
Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make? Let's think step by step and output the final answer after "####".<|im_end|>
<|im_start|>assistant
<think>

</think>

The cost of the house and repairs came out to 80,000+50,000=$<<80000+50000=130000>>130,000
He increased the value of the house by 80,000*1.5=<<80000*1.5=120000>>120,000
So the new value of the house is 120,000+80,000=$<<120000+80000=200000>>200,000
So he made a profit of 200,000-130,000=$<<200000-130000=70000>>70,000
#### 70000<|im_end|>
```

**prompt 部分解码:**

```text
<|im_start|>user
Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make? Let's think step by step and output the final answer after "####".<|im_end|>
<|im_start|>assistant
<think>

</think>


```

**assistant 部分解码 (这部分计算 loss):**

```text
The cost of the house and repairs came out to 80,000+50,000=$<<80000+50000=130000>>130,000
He increased the value of the house by 80,000*1.5=<<80000*1.5=120000>>120,000
So the new value of the house is 120,000+80,000=$<<120000+80000=200000>>200,000
So he made a profit of 200,000-130,000=$<<200000-130000=70000>>70,000
#### 70000<|im_end|>
```

---

## 补充：特殊 token 速查

| token | id | 说明 |
|---|---|---|
| `<|im_start|>` | 151644 | |
| `<|im_end|>` | 151645 | |
| `<|endoftext|>` | 151643 | |