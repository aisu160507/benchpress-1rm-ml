# benchpress-1rm-ml
ベンチプレスのトレーニング記録（重量・回数・セット・休養日数・疲労度）から、機械学習で1RMを予測するツール

# Bench Press 1RM Prediction (with RPE)

個人のトレーニング記録（重量・回数・セット数・休養日数・疲労度等）から、1RM（ベンチプレス最大挙上重量）を予測する機械学習プロトタイプです。  
課題に合う手法選定を目的に、線形回帰 / RandomForest / XGBoost を比較しました。

## データ
- `bench_press_sample_with_rpe.csv` は公開用サンプルです（個人データではありません）。
- 列例：`日付, 重量(kg), 回数, セット数, 休養日数, 疲労度, 実測1RM(kg)`

## 使い方
```bash
pip install -r requirements.txt
python train.py
