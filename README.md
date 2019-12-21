# joint_bert
中文多任務模型（Ner + intents）


基本上就是透過內容分析主題。

------------

# 快速啟動
``` python3 mulit_label.py ```

# 訓練部分
### 訓練參數

```
train_batch_size = 16
eval_batch_size = 16
predict_batch_size = 16
learning_rate = 5e-5
num_train_epochs = 5
```


### 輸入資料

範例檔案 （使用SMP2018ECDTCorpus資料集製作,31種意圖種類）
```
  data/
      - dev-kashgari.tsv
      - test-kashgari.tsv
      - train-kashgari.tsv
```
範例如下 （標籤可能為複數）

| 標籤  | 內容  |
| ------------ | ------------ |
| [weather, news]	| 查看汕头天气 讯飞语音我想知道山东省东营市今天的新闻。
| [cookbook]	| 土豆饼怎么做啊？
| [map, chat]	| 到郑州火车站 唠什么
| [app, message]	| ﻿打开熊猫看书 发信息给董昊老婆
| [website, chat]	| 打开索尼官方网站 怎么不懂讲晚安呢
| [chat, video]	| 能想起我吗 换到米老鼠



### 評估結果

總項
```
class_intent_f1 = 0.9022081
class_precision = 0.9178434
class_recall = 0.88709676
class_threshold_ = 0.34673366
loss = 1.3868933
global_step = 587
```

分項 對應的標籤可參考[output/label_map.txt](https://github.com/Chunshan-Theta/bert_mulit_label_chinese/blob/master/output/label_map.txt)
```
class00_f1 = 0.7727272
class00_precision = 0.68
class00_recall = 0.8947368
class00_threshold = 0.03517588

class01_f1 = 1.0
class01_precision = 1.0
class01_recall = 1.0
class01_threshold = 0.7085427

class02_f1 = 1.0
class02_precision = 1.0
class02_recall = 1.0
class02_threshold = 0.30653265

class03_f1 = 0.9253731
class03_precision = 0.92537314
class03_recall = 0.92537314
class03_threshold = 0.31658292

class04_f1 = 0.96551716
class04_precision = 1.0
class04_recall = 0.93333334
class04_threshold = 0.61809045

class05_f1 = 0.88888884
class05_precision = 0.8888889
class05_recall = 0.8888889
class05_threshold = 0.55778897

class06_f1 = 0.9906541
class06_precision = 0.9814815
class06_recall = 1.0
class06_threshold = 0.07537688

class07_f1 = 0.10526314
class07_precision = 0.05882353
class07_recall = 0.5
class07_threshold = 0.03517588

class08_f1 = 0.9333333
class08_precision = 1.0
class08_recall = 0.875
class08_threshold = 0.06030151

class09_f1 = 0.84782594
class09_precision = 0.90697676
class09_recall = 0.79591835
class09_threshold = 0.3115578

class10_f1 = 0.9803921
class10_precision = 1.0
class10_recall = 0.96153843
class10_threshold = 0.14572865

class11_f1 = 0.98305076
class11_precision = 0.96666664
class11_recall = 1.0
class11_threshold = 0.055276383

class12_f1 = 0.9
class12_precision = 0.8181818
class12_recall = 1.0
class12_threshold = 0.030150754

class13_f1 = 0.9166666
class13_precision = 0.9166667
class13_recall = 0.9166667
class13_threshold = 0.16080402

class14_f1 = 0.8421052
class14_precision = 0.8
class14_recall = 0.8888889
class14_threshold = 0.08040201

class15_f1 = 1.0
class15_precision = 1.0
class15_recall = 1.0
class15_threshold = 0.6381909

class16_f1 = 0.9285714
class16_precision = 1.0
class16_recall = 0.8666667
class16_threshold = 0.6934673

class17_f1 = 0.98181814
class17_precision = 0.96428573
class17_recall = 1.0
class17_threshold = 0.8190955

class18_f1 = 0.74999994
class18_precision = 1.0
class18_recall = 0.6
class18_threshold = 0.040201005

class19_f1 = 0.98305076
class19_precision = 0.96666664
class19_recall = 1.0
class19_threshold = 0.16080402

class20_f1 = 0.96296287
class20_precision = 1.0
class20_recall = 0.9285714
class20_threshold = 0.085427135

class21_f1 = 1.0
class21_precision = 1.0
class21_recall = 1.0
class21_threshold = 0.04522613

class22_f1 = 1.0
class22_precision = 1.0
class22_recall = 1.0
class22_threshold = 0.06532663

class23_f1 = 0.98113203
class23_precision = 1.0
class23_recall = 0.962963
class23_threshold = 0.44221106

class24_f1 = 0.98181814
class24_precision = 0.96428573
class24_recall = 1.0
class24_threshold = 0.14070351

class25_f1 = 0.982456
class25_precision = 1.0
class25_recall = 0.9655172
class25_threshold = 0.06030151

class26_f1 = 1.0
class26_precision = 1.0
class26_recall = 1.0
class26_threshold = 0.63316584

class27_f1 = 0.70270264
class27_precision = 0.8125
class27_recall = 0.61904764
class27_threshold = 0.41708544

class28_f1 = 0.7368421
class28_precision = 0.7291667
class28_recall = 0.7446808
class28_threshold = 0.24120604

class29_f1 = 0.87499994
class29_precision = 0.9130435
class29_recall = 0.84
class29_threshold = 0.06532663

class30_f1 = 0.8999999
class30_precision = 0.9
class30_recall = 0.9
class30_threshold = 0.6683417

```
------------
# 預測

有輸出一份結果 在 [bert_mulit_label_chinese/output/test_results.tsv](https://github.com/Chunshan-Theta/bert_mulit_label_chinese/blob/master/output/test_results.tsv)裡面
