from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd


def generate_cicids2017_sample(
    n_samples: int = 5000,
    output_path: Optional[Union[str, Path]] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generates a realistic synthetic sample resembling CIC-IDS2017 network traffic.
    Includes anomalies like infinite values, missing values, high skewness,
    and imbalanced benign vs attack labels.
    """
    rng = np.random.RandomState(random_state)

    # 80% Benign, 20% Attacks (typical imbalance)
    label_choices = ["BENIGN", "DoS Hulk", "PortScan", "DDoS"]
    label_probs = [0.80, 0.10, 0.05, 0.05]
    labels = rng.choice(label_choices, size=n_samples, p=label_probs)

    # Generate realistic network flow features (with leading/trailing whitespace common in CIC-IDS2017)
    flow_duration = rng.exponential(scale=100000, size=n_samples).astype(float)
    total_fwd_packets = rng.poisson(lam=10, size=n_samples).astype(float)
    total_bwd_packets = rng.poisson(lam=8, size=n_samples).astype(float)
    fwd_packet_len_mean = rng.normal(loc=250, scale=80, size=n_samples)
    bwd_packet_len_mean = rng.normal(loc=400, scale=100, size=n_samples)
    dest_port = rng.choice([80, 443, 22, 21, 8080, 53, 3389], size=n_samples)
    flow_iat_mean = rng.exponential(scale=5000, size=n_samples)
    fin_flag_cnt = rng.binomial(1, 0.15, size=n_samples)
    syn_flag_cnt = rng.binomial(1, 0.20, size=n_samples)
    rst_flag_cnt = rng.binomial(1, 0.05, size=n_samples)
    ack_flag_cnt = rng.binomial(1, 0.70, size=n_samples)
    avg_packet_size = (fwd_packet_len_mean + bwd_packet_len_mean) / 2.0

    # Skewed features: Flow Bytes/s and Flow Packets/s
    flow_bytes_s = (total_fwd_packets * fwd_packet_len_mean) / np.maximum(flow_duration / 1000.0, 0.001)
    flow_packets_s = (total_fwd_packets + total_bwd_packets) / np.maximum(flow_duration / 1000.0, 0.001)

    # Introduce attack-specific signals
    is_attack = labels != "BENIGN"
    total_fwd_packets[is_attack] *= rng.uniform(2.0, 5.0, size=np.sum(is_attack))
    flow_bytes_s[is_attack] *= rng.uniform(3.0, 10.0, size=np.sum(is_attack))
    syn_flag_cnt[labels == "PortScan"] = 1

    # Inject real-world CIC-IDS2017 anomalies:
    # 1. Infinite values (division by 0 in packet capture)
    inf_indices_bytes = rng.choice(n_samples, size=max(1, int(n_samples * 0.01)), replace=False)
    flow_bytes_s[inf_indices_bytes] = np.inf
    inf_indices_pkts = rng.choice(n_samples, size=max(1, int(n_samples * 0.01)), replace=False)
    flow_packets_s[inf_indices_pkts] = np.inf

    # 2. Missing values (NaN)
    nan_indices = rng.choice(n_samples, size=max(1, int(n_samples * 0.02)), replace=False)
    flow_iat_mean[nan_indices] = np.nan

    data = {
        " Destination Port": dest_port,
        " Flow Duration": flow_duration,
        " Total Fwd Packets": total_fwd_packets,
        " Total Backward Packets": total_bwd_packets,
        " Fwd Packet Length Mean": np.maximum(fwd_packet_len_mean, 0),
        " Bwd Packet Length Mean": np.maximum(bwd_packet_len_mean, 0),
        " Flow Bytes/s": flow_bytes_s,
        " Flow Packets/s": flow_packets_s,
        " Flow IAT Mean": flow_iat_mean,
        " FIN Flag Count": fin_flag_cnt,
        " SYN Flag Count": syn_flag_cnt,
        " RST Flag Count": rst_flag_cnt,
        " ACK Flag Count": ack_flag_cnt,
        " Average Packet Size": np.maximum(avg_packet_size, 0),
        " Fwd Header Length": total_fwd_packets * 20,
        " Bwd Header Length": total_bwd_packets * 20,
        " Packet Length Mean": (fwd_packet_len_mean + bwd_packet_len_mean) / 2,
        " Subflow Fwd Packets": total_fwd_packets,
        " Subflow Bwd Packets": total_bwd_packets,
        " Init_Win_bytes_forward": rng.choice([8192, 29200, 65535], size=n_samples),
        " Init_Win_bytes_backward": rng.choice([0, 8192, 65535], size=n_samples),
        "Label": labels,
    }

    df = pd.DataFrame(data)

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_file, index=False)

    return df


if __name__ == "__main__":
    generate_cicids2017_sample(5000, "./dataset/cicids2017_sample.csv")
    print("Generated sample dataset at ./dataset/cicids2017_sample.csv")
