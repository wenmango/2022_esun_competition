2022 Esun competition
----

# 系統環境
* 系統平台：Linux
* 程式語言：python
* 函式庫：pandas, numpy, lightgbm, xgboost, sklearn


# 檔案目錄
```
├ Preprocess
│ └ final_feature.py  (產出訓練用資料)
├ Model
│ └ final_model.py    (訓練+推論+產出提交檔案)
├ requirements.txt
└ README
```

# 復現步驟
1. 在根目錄放入此次比賽的所有資料
2. 運行 final_feature.py 產出 raw_data_1225_submit.pkl 放入 Model 資料夾
3. 運行 final_model.py 產出預測結果 final_1225_v2.csv


# 模型超參數
* lightgbm 使用參數如下：
	```
	eta:0.005
	max_depth:15
	subsample:0.75
	colsample_bytree:0.22
	scale_pos_weight:20
	min_child_weight:5
	min_split_gain:0.10677364238023246
	num_leaves:72
	subsample_for_bin:3066
	subsample_freq:5
	objective:binary
	boosting_type:gbdt
	first_metric_only:True
	metric:None
	seed:20221225
	num_threads:35
	num_boost_round:5000
	early_stopping_rounds:200
	verbose:-100
	```
* xgboost 使用參數如下：
	```
	eta:0.005
	max_depth:15
	min_child_weight:5
	subsample:0.7
	colsample_bytree:0.2
	gamma:0.01
	scale_pos_weight:20
	objective:binary:logistic
	tree_method:gpu_hist
	disable_default_eval_metric:True
	num_boost_round:5000
	early_stopping_rounds:200
	seed:20221225
	```
* LogisticRegression 使用參數如下：
	```
	random_state:0
	```
