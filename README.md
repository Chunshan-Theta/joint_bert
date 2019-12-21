# joint_bert
中文多任務模型（Ner + intents）


基本上就是透過內容分析主題與辨析實體。

------------

# 快速啟動
``` python3 run_joint_bert.py ```

# 訓練部分
### 訓練參數

```
train_batch_size = 16
eval_batch_size = 16
predict_batch_size = 16
learning_rate = 5e-5
num_train_epochs = 15
warmup_proportion = 0.1
save_checkpoints_steps = 1000
iterations_per_loop = 1000
```


### 輸入資料

範例檔案 （使用SMP2018ECDTCorpus資料集製作,31種意圖種類）
```
  data/
      - dev-joint.tsv
      - test-joint.tsv
      - train-joint.tsv
```
範例如下 （標籤可能為複數）

| 標籤  | 內容  | ner-tag |
| ------------ | ------------ |------------ |
| [chat, tvchannel]	| 有体育方面的吗。CCTV四套	| O O O O O O O O O O O O O O
| [email, chat]	| 给我打开mail。丰田车在哪里生产的？	| O O O O B-ORG I-ORG I-ORG I-ORG O B-ORG I-ORG O O O O O O O O
| [flight, video]	| 查询广州到海南的航班。想唱就唱	| O O B-LOC I-LOC O B-LOC I-LOC O O O O O O O O
| [chat, cookbook]	| 什么时候去。炒鸡蛋，需要，哪些材料？	| O O O O O O O O O O O O O O O O O O
| [weather]	| 你好请问一下明天广州的天气如何。郑州天气怎么样	| O O O O O O O O B-LOC I-LOC O O O O O O B-LOC I-LOC O O O O O
| [chat, health]	| 买几瓶。宫外孕怎么办？	| O O O O O O O O O O O
| [chat]	| 你有什么梦想啊。你认识徐俊吗	| O O O O O O O O O O O B-PER I-PER O
| [chat, cookbook]	| 谁写的你。回锅肉。	| O O O O O O O O O



### 評估結果

總項
```
class_intent_f1 = 0.8505595
class_precision = 0.8934993
class_recall = 0.81155777
class_threshold_ = 0.2763819

loss = 2.4448438

slot_accuracy = 0.9720809
slot_f1 = 0.8959768
slot_precision = 0.9069778
slot_recall = 0.8878232

global_step = 1737
```

分項 對應的標籤可參考[output/label_map.txt](https://github.com/Chunshan-Theta/joint_bert/blob/master/output/label_map.txt)
```
class00_f1 = 0.6521738
class00_precision = 0.5555556
class00_recall = 0.7894737
class00_threshold = 0.16582915

class01_f1 = 1.0
class01_precision = 1.0
class01_recall = 1.0
class01_threshold = 0.2512563

class02_f1 = 1.0
class02_precision = 1.0
class02_recall = 1.0
class02_threshold = 0.2361809

class03_f1 = 0.8543688
class03_precision = 0.80981594
class03_recall = 0.9041096
class03_threshold = 0.085427135

class04_f1 = 0.9523809
class04_precision = 0.90909094
class04_recall = 1.0
class04_threshold = 0.28140703

class05_f1 = 0.9166666
class05_precision = 0.9166667
class05_recall = 0.9166667
class05_threshold = 0.5829146

class06_f1 = 0.98947364
class06_precision = 0.9894737
class06_recall = 0.9894737
class06_threshold = 0.16080402

class07_f1 = 0.2222222
class07_precision = 0.125
class07_recall = 0.9999999
class07_threshold = 0.1758794

class08_f1 = 0.8888889
class08_precision = 1.0
class08_recall = 0.8
class08_threshold = 0.1557789

class09_f1 = 0.8249999
class09_precision = 0.84615386
class09_recall = 0.80487806
class09_threshold = 0.21105528

class10_f1 = 0.98113203
class10_precision = 1.0
class10_recall = 0.962963
class10_threshold = 0.085427135

class11_f1 = 0.98113203
class11_precision = 1.0
class11_recall = 0.962963
class11_threshold = 0.12060302

class12_f1 = 0.87499994
class12_precision = 0.7777778
class12_recall = 1.0
class12_threshold = 0.110552765

class13_f1 = 0.8205128
class13_precision = 0.8888889
class13_recall = 0.7619048
class13_threshold = 0.34673366

class14_f1 = 1.0
class14_precision = 1.0
class14_recall = 1.0
class14_threshold = 0.65829146

class15_f1 = 1.0
class15_precision = 1.0
class15_recall = 1.0
class15_threshold = 0.30653265

class16_f1 = 0.8823529
class16_precision = 0.88235295
class16_recall = 0.88235295
class16_threshold = 0.1959799

class17_f1 = 0.9795917
class17_precision = 0.96
class17_recall = 1.0
class17_threshold = 0.13065326

class18_f1 = 0.57142854
class18_precision = 1.0
class18_recall = 0.4
class18_threshold = 0.2713568

class19_f1 = 0.9122806
class19_precision = 0.962963
class19_recall = 0.8666667
class19_threshold = 0.22613065

class20_f1 = 0.9285714
class20_precision = 1.0
class20_recall = 0.8666667
class20_threshold = 0.14070351

class21_f1 = 1.0
class21_precision = 1.0
class21_recall = 1.0
class21_threshold = 0.26633167

class22_f1 = 0.97435886
class22_precision = 1.0
class22_recall = 0.95
class22_threshold = 0.2713568

class23_f1 = 0.95833325
class23_precision = 1.0
class23_recall = 0.92
class23_threshold = 0.2562814

class24_f1 = 0.8852458
class24_precision = 0.8181818
class24_recall = 0.96428573
class24_threshold = 0.10552764

class25_f1 = 0.98181814
class25_precision = 1.0
class25_recall = 0.96428573
class25_threshold = 0.10552764

class26_f1 = 0.982456
class26_precision = 0.9655172
class26_recall = 1.0
class26_threshold = 0.09045226

class27_f1 = 0.68085104
class27_precision = 0.7619048
class27_recall = 0.61538464
class27_threshold = 0.28140703

class28_f1 = 0.64761895
class28_precision = 0.6296296
class28_recall = 0.6666667
class28_threshold = 0.18090452

class29_f1 = 0.8837209
class29_precision = 0.8636364
class29_recall = 0.9047619
class29_threshold = 0.52261305

class30_f1 = 0.7741935
class30_precision = 0.7058824
class30_recall = 0.85714287
class30_threshold = 0.4723618

```
------------
# 預測

```
class_intent_f1 = 0.8582755203171458
class_intent_precision = 0.8316261203585147
class_intent_recall = 0.8866894197952219


slot_accuracy = 0.9749567623659633
slot_f1 = 0.8981856267166586
slot_precision = 0.9212813221018604
slot_recall = 0.8775508948061814
```


範例如下 （標籤可能為複數）

| frame_O_X  | intent  | 意圖標籤（答案） |意圖標籤（預測） |原句 |實體（預測） |實體（答案） |
| ------------ | ------------ |------------ |------------ |------------ |------------ |------------ |
| X	| X	| email website	app | email website	| [CLS]替我回复这封邮件。打开网页猫扑	| [CLS] O O O O O O O O O O O O O O O	| [CLS] O O O O O O O O O O O O O O O	O
| X	| X	| lottery video	| lottery	| [CLS]查询大乐透的开奖公告。大耳朵图图二	| [CLS] O O O O O O O O O O O O O O O O O| 	[CLS] O O O O O O O O O O O O O O O O O	O
| O| 	O	| chat map	| chat map	| [CLS]是我说话太直接了么。查询合肥市植物园的位置	| [CLS] O O O O O O O O O O O O B-LOC I-LOC I-LOC I-LOC I-LOC I-LOC O O O| 	[CLS] O O O O O O O O O O O O B-LOC I-LOC I-LOC I-LOC I-LOC I-LOC O O O	O
| O| 	O	| bus novel	| bus novel	| [CLS]搜一搜三毛的小说。在合肥怎么坐去南京的汽车	| [CLS] O O O B-PER I-PER O O O O O B-LOC I-LOC O O O O B-LOC I-LOC O O O	| [CLS] O O O B-PER I-PER O O O O O B-LOC I-LOC O O O O B-LOC I-LOC O O O	O
| O	| O	| epg flight	| epg flight	| [CLS]明天安徽卫视演什么。后天去大连的航班	| [CLS] O O B-ORG I-ORG I-ORG I-ORG O O O O O O O B-LOC I-LOC O O O	| [CLS] O O B-ORG I-ORG I-ORG I-ORG O O O O O O O B-LOC I-LOC O O O	O
