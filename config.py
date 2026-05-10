"""Global configuration for data paths, model paths, and training hyperparameters."""

# -----------------
# data path
# -----------------

# Define log normal.
log_normal = "dataset/normal/normal_log_new.log"

# Define audit abuse dev.
audit_abuse_dev = "dataset/attack/audit_abuse_dev.log"
audit_abuse_ops = "dataset/attack/audit_abuse_ops.log"
audit_abuse_sys = "dataset/attack/audit_abuse_sys.log"

# Define abuse dev label.
abuse_dev_label = "dataset/attack/label_dev"
abuse_ops_label = "dataset/attack/label_ops"
abuse_sys_label = "dataset/attack/label_sys"

# Define audit kubeconfig node2.
audit_kubeconfig_node2 = "dataset/attack/kubeconfig_node2.log"
audit_kubeconfig_node3 = "dataset/attack/kubeconfig_node3.log"

# Define kubeconfig node2 label.
kubeconfig_node2_label = "dataset/attack/kubeconfig_node2"
kubeconfig_node3_label = "dataset/attack/kubeconfig_node3"

# Define dev escape.
dev_escape = "dataset/attack/escape_dev.log"

# Define dev escape label.
dev_escape_label = "dataset/attack/label_dev_escape"

# Define ops escape.
ops_escape = "dataset/attack/audit_escape_ops.log"

# Define ops escape label.
ops_escape_label = "dataset/attack/label_ops_escape"

# -----------------
# Define csv path.
# -----------------

csv_path = "csv"
normal_log_new = "normal_log_new.csv"
normal_log_new_filter = "normal_log_new_filter.csv"
audit_abuse_dev_csv = "audit_abuse_dev.csv"
audit_abuse_dev_filter = "audit_abuse_dev_filter.csv"
audit_abuse_ops_csv = "audit_abuse_ops.csv"
audit_abuse_ops_filter = "audit_abuse_ops_filter.csv"
audit_abuse_sys_csv = "audit_abuse_sys.csv"
audit_abuse_sys_filter = "audit_abuse_sys_filter.csv"
kubeconfig_node2_csv = "kubeconfig_node2.csv"
kubeconfig_node2_filter = "kubeconfig_node2_filter.csv"
kubeconfig_node3_csv = "kubeconfig_node3.csv"
kubeconfig_node3_filter = "kubeconfig_node3_filter.csv"
dev_escape_csv = "dev_escape.csv"
dev_escape_filter = "dev_escape_filter.csv"
ops_escape_csv = "ops_escape.csv"
ops_escape_filter = "ops_escape_filter.csv"

# -----------------
# Define traintest path.
# -----------------

traintest_path = "traintest"
audit_abuse_dev_window = "audit_abuse_dev.txt"
audit_abuse_ops_window = "audit_abuse_ops.txt"
audit_abuse_sys_window = "audit_abuse_sys.txt"
kubeconfig_node2_window = "kubeconfig_node2.txt"
kubeconfig_node3_window = "kubeconfig_node3.txt"
dev_escape_window = "dev_escape.txt"
ops_escape_window = "ops_escape.txt"
normal_log_new_window = "normal_log_new.txt"
window_count_file = "count.txt"

# -----------------
# Define split path.
# -----------------

split_path = "split"
normal_log_new_split_text = "normal_log_new_split.txt"
audit_abuse_dev_split_text = "audit_abuse_dev_split.txt"
audit_abuse_ops_split_text = "audit_abuse_ops_split.txt"
audit_abuse_sys_split_text = "audit_abuse_sys_split.txt"
kubeconfig_node2_split_text = "kubeconfig_node2_split.txt"
kubeconfig_node3_split_text = "kubeconfig_node3_split.txt"
dev_escape_split_text = "dev_escape_split.txt"
ops_escape_split_text = "ops_escape_split.txt"

normal_log_new_log_embedding = "normal_log_new_log_embedding.jsonl"
audit_abuse_dev_log_embedding = "audit_abuse_dev_log_embedding.jsonl"
audit_abuse_ops_log_embedding = "audit_abuse_ops_log_embedding.jsonl"
audit_abuse_sys_log_embedding = "audit_abuse_sys_log_embedding.jsonl"
kubeconfig_node2_log_embedding = "kubeconfig_node2_log_embedding.jsonl"
kubeconfig_node3_log_embedding = "kubeconfig_node3_log_embedding.jsonl"
dev_escape_log_embedding = "dev_escape_log_embedding.jsonl"
ops_escape_log_embedding = "ops_escape_log_embedding.jsonl"

normal_log_new_window_embedding = "normal_log_new_window_embedding.jsonl"
audit_abuse_dev_window_embedding = "audit_abuse_dev_window_embedding.jsonl"
audit_abuse_ops_window_embedding = "audit_abuse_ops_window_embedding.jsonl"
audit_abuse_sys_window_embedding = "audit_abuse_sys_window_embedding.jsonl"
kubeconfig_node2_window_embedding = "kubeconfig_node2_window_embedding.jsonl"
kubeconfig_node3_window_embedding = "kubeconfig_node3_window_embedding.jsonl"
dev_escape_window_embedding = "dev_escape_window_embedding.jsonl"
ops_escape_window_embedding = "ops_escape_window_embedding.jsonl"

# -----------------
# Define arithmetic path.
# -----------------

arithmetic_path = "Arithmetic"
sequence_transformer_metrics_file = "sequence_transformer_metrics.csv"
sequence_transformer_report_file = "sequence_transformer_report.txt"
sequence_transformer_vocab_file = "sequence_transformer_vocab.json"
sequence_transformer_log_encoder_model_file = "sequence_transformer_log_encoder.pt"
sequence_transformer_bilstm_model_file = "sequence_transformer_bilstm_encoder.pt"
sequence_transformer_svdd_model_file = "sequence_transformer_svdd_model.json"
sequence_char_metrics_file = "sequence_char_metrics.csv"
sequence_char_report_file = "sequence_char_report.txt"
sequence_char_vocab_file = "sequence_char_vocab.json"
sequence_char_model_file = "sequence_char_model.pt"
sequence_char_svdd_model_file = "sequence_char_svdd_model.json"
sequence_char_window_embedding_file = "sequence_char_window_embeddings.jsonl"
sequence_char_log_embedding_file = "sequence_char_log_embeddings.jsonl"
sequence_new_metrics_file = "sequence_new_metrics.csv"
sequence_new_report_file = "sequence_new_report.txt"
sequence_new_vocab_file = "sequence_new_vocab.json"
sequence_new_log_encoder_model_file = "sequence_new_log_encoder.pt"
sequence_new_window_encoder_model_file = "sequence_new_window_transformer.pt"
sequence_new_detector_model_file = "sequence_new_detector_model.json"
sequence_new_log_embedding_file = "sequence_new_log_embeddings.jsonl"
sequence_new_window_embedding_file = "sequence_new_window_embeddings.jsonl"
sequence_d2d_metrics_file = "sequence_d2d_metrics.csv"
sequence_d2d_report_file = "sequence_d2d_report.txt"
sequence_d2d_vocab_file = "sequence_d2d_vocab.json"
sequence_d2d_model_file = "sequence_d2d_model.pt"
sequence_d2d_threshold_file = "sequence_d2d_threshold.json"
sequence_d2d_window_score_file = "sequence_d2d_window_scores.jsonl"

# -----------------
# Define graph path.
# -----------------

graph_path = "graph"
normal_log_new_graph_file = "normal_log_new_hetero_graph.jsonl"
audit_abuse_dev_graph_file = "audit_abuse_dev_hetero_graph.jsonl"
audit_abuse_ops_graph_file = "audit_abuse_ops_hetero_graph.jsonl"
audit_abuse_sys_graph_file = "audit_abuse_sys_hetero_graph.jsonl"
kubeconfig_node2_graph_file = "kubeconfig_node2_hetero_graph.jsonl"
kubeconfig_node3_graph_file = "kubeconfig_node3_hetero_graph.jsonl"
dev_escape_graph_file = "dev_escape_hetero_graph.jsonl"
ops_escape_graph_file = "ops_escape_hetero_graph.jsonl"
hetero_graph_schema_file = "hetero_graph_schema.json"
sample_window_graph_dot_file = "sample_window_graph.dot"
sample_window_graph_svg_file = "sample_window_graph.svg"
sample_window_graph_summary_file = "sample_window_graph_summary.json"
