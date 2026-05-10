# k8s_audit Execution Guide

![](/Users/liumengyao/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_sqtfbh92hfg621_e8b2/temp/InputTemp/f0d34f94-4e2a-4bc9-9700-9ea942b3c0ac.png)

<center>The Architecture of BeKube</center>

<img src="file:///Users/liumengyao/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_sqtfbh92hfg621_e8b2/temp/InputTemp/128c64c7-bd45-40e3-bbda-7eafd92e76c9.png" title="" alt="" width="302">

<center>graph construction</center>

<img src="file:///Users/liumengyao/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_sqtfbh92hfg621_e8b2/temp/InputTemp/7ecbecf6-6fd4-4be3-98d0-0ce0a04e8be6.png" title="" alt="" width="350">

<center>multi-task graph learning with derived masked views</center>

This document only covers the main execution pipeline implemented by these four scripts:

- `data_parser.py`
- `data_prepare.py`
- `method/graph_hete_train.py`
- `method/graph_hete_test.py`

The full pipeline is:

```text
Raw Kubernetes audit logs
  -> data_parser.py
  -> csv/*_filter.csv
  -> data_prepare.py
  -> traintest/*.txt
  -> method/graph_hete_train.py
  -> trained heterogeneous graph detector
  -> method/graph_hete_test.py
  -> metrics, window scores, and case studies
```

## 1. Parse and Filter Logs

`data_parser.py` converts raw Kubernetes audit JSONL logs into structured CSV files and then creates filtered `*_filter.csv` files. These filtered CSV files are used by the window-building step and the graph detector.

Normal logs do not require a label file. Their labels default to `0`:

```bash
python3 -c "import sys, data_parser; sys.argv=['data_parser.py','--input','dataset/normal/normal_log_new.log','--output','csv/normal_log_new.csv','--filter_output','csv/normal_log_new_filter.csv']; data_parser.main()"
```

Attack logs require label files. Each label file must align with the corresponding raw log file line by line:

```bash
python3 -c "import sys, data_parser; sys.argv=['data_parser.py','--input','dataset/attack/audit_abuse_dev.log','--output','csv/audit_abuse_dev.csv','--filter_output','csv/audit_abuse_dev_filter.csv','--label_input','dataset/attack/label_dev']; data_parser.main()"

python3 -c "import sys, data_parser; sys.argv=['data_parser.py','--input','dataset/attack/audit_abuse_ops.log','--output','csv/audit_abuse_ops.csv','--filter_output','csv/audit_abuse_ops_filter.csv','--label_input','dataset/attack/label_ops']; data_parser.main()"

python3 -c "import sys, data_parser; sys.argv=['data_parser.py','--input','dataset/attack/audit_abuse_sys.log','--output','csv/audit_abuse_sys.csv','--filter_output','csv/audit_abuse_sys_filter.csv','--label_input','dataset/attack/label_sys']; data_parser.main()"

python3 -c "import sys, data_parser; sys.argv=['data_parser.py','--input','dataset/attack/escape_dev.log','--output','csv/dev_escape.csv','--filter_output','csv/dev_escape_filter.csv','--label_input','dataset/attack/label_dev_escape']; data_parser.main()"

python3 -c "import sys, data_parser; sys.argv=['data_parser.py','--input','dataset/attack/audit_escape_ops.log','--output','csv/ops_escape.csv','--filter_output','csv/ops_escape_filter.csv','--label_input','dataset/attack/label_ops_escape']; data_parser.main()"

python3 -c "import sys, data_parser; sys.argv=['data_parser.py','--input','dataset/attack/kubeconfig_node2.log','--output','csv/kubeconfig_node2.csv','--filter_output','csv/kubeconfig_node2_filter.csv','--label_input','dataset/attack/kubeconfig_node2']; data_parser.main()"

python3 -c "import sys, data_parser; sys.argv=['data_parser.py','--input','dataset/attack/kubeconfig_node3.log','--output','csv/kubeconfig_node3.csv','--filter_output','csv/kubeconfig_node3_filter.csv','--label_input','dataset/attack/kubeconfig_node3']; data_parser.main()"
```

The generated files are stored in `csv/`, for example:

- `csv/normal_log_new_filter.csv`
- `csv/audit_abuse_dev_filter.csv`
- `csv/dev_escape_filter.csv`
- `csv/kubeconfig_node2_filter.csv`

## 2. Build Sliding Windows

`data_prepare.py` reads all `*_filter.csv` files under `csv/` and creates sliding-window index files.

The default setting is a 3-minute window with a 1-minute step:

```bash
python3 data_prepare.py --input-dir csv --output-dir traintest --window-min 3 --step-min 1
```

The output files are stored in `traintest/`. Each line has three columns:

```text
start_idx    end_idx    window_label
```

Column meanings:

- `start_idx`: row index of the first log in the window
- `end_idx`: row index of the last log in the window
- `window_label`: `1` if the window contains at least one anomalous log, otherwise `0`

## 3. Train the Graph Detector

`method/graph_hete_train.py` trains the heterogeneous graph anomaly detector on normal training windows and calibrates the anomaly threshold from normal training scores.

Run training:

```bash
PYTHONPATH=Arithmetic python3 method/graph_hete_train.py \
  --device cpu \
  --epochs 30 \
  --batch-size 16 \
  --window-score-reduction max
```

To use a GPU, replace `--device cpu` with the target device:

```bash
PYTHONPATH=Arithmetic python3 method/graph_hete_train.py --device cuda --epochs 30
```

Default training outputs are written to `Arithmetic/`:

- `Arithmetic/graph_hete_train_vocab.json`
- `Arithmetic/graph_hete_train_model.pt`
- `Arithmetic/graph_hete_train_threshold.json`
- `Arithmetic/graph_hete_train_scores.jsonl`
- `Arithmetic/graph_hete_train_report.txt`

## 4. Test the Graph Detector

`method/graph_hete_test.py` loads the saved vocabulary, model checkpoint, and threshold, then evaluates all test windows and exports metrics, window scores, and case studies.

Run testing:

```bash
PYTHONPATH=Arithmetic python3 method/graph_hete_test.py \
  --device cpu \
  --case-per-dataset
```

To use a GPU:

```bash
PYTHONPATH=Arithmetic python3 method/graph_hete_test.py \
--device cuda \
--case-per-dataset
```

Default testing outputs are written to `Arithmetic/`:

- `Arithmetic/graph_hete_test_window_scores.jsonl`
- `Arithmetic/graph_hete_test_metrics.csv`
- `Arithmetic/case_study/`

Output meanings:

- `graph_hete_test_window_scores.jsonl`: score and prediction for each test window
- `graph_hete_test_metrics.csv`: precision, recall, F1, FPR, and related metrics
- `case_study/`: local graphs and explanation reports for selected high-score anomalous windows

## End-to-End Command Sequence

After raw logs and label files are ready, run the following commands in order:

```bash
# 1. Parse logs into filtered CSV files.
python3 -c "import sys, data_parser; sys.argv=['data_parser.py','--input','dataset/normal/normal_log_new.log','--output','csv/normal_log_new.csv','--filter_output','csv/normal_log_new_filter.csv']; data_parser.main()"
python3 -c "import sys, data_parser; sys.argv=['data_parser.py','--input','dataset/attack/audit_abuse_dev.log','--output','csv/audit_abuse_dev.csv','--filter_output','csv/audit_abuse_dev_filter.csv','--label_input','dataset/attack/label_dev']; data_parser.main()"
python3 -c "import sys, data_parser; sys.argv=['data_parser.py','--input','dataset/attack/audit_abuse_ops.log','--output','csv/audit_abuse_ops.csv','--filter_output','csv/audit_abuse_ops_filter.csv','--label_input','dataset/attack/label_ops']; data_parser.main()"
python3 -c "import sys, data_parser; sys.argv=['data_parser.py','--input','dataset/attack/audit_abuse_sys.log','--output','csv/audit_abuse_sys.csv','--filter_output','csv/audit_abuse_sys_filter.csv','--label_input','dataset/attack/label_sys']; data_parser.main()"
python3 -c "import sys, data_parser; sys.argv=['data_parser.py','--input','dataset/attack/escape_dev.log','--output','csv/dev_escape.csv','--filter_output','csv/dev_escape_filter.csv','--label_input','dataset/attack/label_dev_escape']; data_parser.main()"
python3 -c "import sys, data_parser; sys.argv=['data_parser.py','--input','dataset/attack/audit_escape_ops.log','--output','csv/ops_escape.csv','--filter_output','csv/ops_escape_filter.csv','--label_input','dataset/attack/label_ops_escape']; data_parser.main()"
python3 -c "import sys, data_parser; sys.argv=['data_parser.py','--input','dataset/attack/kubeconfig_node2.log','--output','csv/kubeconfig_node2.csv','--filter_output','csv/kubeconfig_node2_filter.csv','--label_input','dataset/attack/kubeconfig_node2']; data_parser.main()"
python3 -c "import sys, data_parser; sys.argv=['data_parser.py','--input','dataset/attack/kubeconfig_node3.log','--output','csv/kubeconfig_node3.csv','--filter_output','csv/kubeconfig_node3_filter.csv','--label_input','dataset/attack/kubeconfig_node3']; data_parser.main()"

# 2. Build sliding-window train/test index files.
python3 data_prepare.py \
--input-dir csv \
--output-dir traintest \
--window-min 3 \
--step-min 1

# 3. Train the heterogeneous graph detector.
PYTHONPATH=Arithmetic python3 method/graph_hete_train.py \
--device cuda \
--epochs 30 \
--batch-size 16 \
--window-score-reduction max \
--learning-rate 5e-4 \
--rgcn-layers 2 \
--nu 0.1 \
--lambda-rel 0.1 \
--lambda-trans 0.05

# 4. Test and export metrics plus case studies.
PYTHONPATH=Arithmetic python3 method/graph_hete_test.py \
--device cuda \
--case-per-dataset
```
