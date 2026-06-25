"""
运行效率评估脚本
================
统一测量所有模型的：参数量、FLOPs、推理延迟、吞吐量
输出一张对比表，可直接用于论文

用法：python efficiency_benchmark.py
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
import os
import sys
from collections import OrderedDict

# ============================================================
# 配置
# ============================================================
SEQ_LEN = 100
ENC_IN = 38         # CICIDS2017 特征数
BATCH_SIZE = 128
NUM_CLASSES = 2
WARMUP_RUNS = 50     # GPU 预热次数
BENCHMARK_RUNS = 200 # 正式测量次数
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SAVE_PATH = 'checkpoints/efficiency_results.json'


class MockConfig:
    """统一的模型配置"""
    def __init__(self):
        self.seq_len = SEQ_LEN
        self.enc_in = ENC_IN
        self.pred_len = ENC_IN
        self.num_classes = NUM_CLASSES
        self.individual = False

        # TreeMIL 专属参数
        self.d_model = 156
        self.ary_size = 2
        self.inner_size = 3
        self.d_k = 128
        self.d_v = 128
        self.d_inner_hid = 156
        self.n_head = 4
        self.n_layer = 4
        self.dropout = 0.1

        # FEDformer 专属参数
        self.version = 'Fourier'
        self.mode_select = 'random'
        self.modes = 32
        self.fed_d_model = 128
        self.n_heads = 8
        self.d_ff = 512
        self.e_layers = 4
        self.activation = 'gelu'
        self.moving_avg = 25
        self.embed = 'timeF'
        self.freq = 'h'


# ============================================================
# 模型定义区：把每个模型封装为统一接口
# ============================================================

def build_wstd(configs):
    """WSTD (你的模型)"""
    from models.mymodel.amd.Model_with_ddi import Model
    return Model(configs)


def build_lstm(configs):
    """LSTM baseline"""
    class LSTMModel(nn.Module):
        def __init__(self, configs):
            super().__init__()
            self.hidden_size = 256
            self.lstm = nn.LSTM(
                input_size=configs.enc_in, hidden_size=self.hidden_size,
                num_layers=2, batch_first=True,
                dropout=0.3, bidirectional=True
            )
            self.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(self.hidden_size * 2, configs.num_classes)
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    return LSTMModel(configs)


def build_mlp(configs):
    """MLP baseline"""
    class MLPModel(nn.Module):
        def __init__(self, configs):
            super().__init__()
            input_dim = configs.seq_len * configs.enc_in
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, configs.num_classes)
            )

        def forward(self, x):
            return self.mlp(x.reshape(x.size(0), -1))

    return MLPModel(configs)


def build_tcn(configs):
    """TCN baseline"""
    from torch.nn.utils import weight_norm

    class Chomp1d(nn.Module):
        def __init__(self, chomp_size):
            super().__init__()
            self.chomp_size = chomp_size
        def forward(self, x):
            return x[:, :, :-self.chomp_size].contiguous()

    class TemporalBlock(nn.Module):
        def __init__(self, n_in, n_out, k, stride, dilation, padding, dropout):
            super().__init__()
            self.net = nn.Sequential(
                weight_norm(nn.Conv1d(n_in, n_out, k, stride=stride, padding=padding, dilation=dilation)),
                Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),
                weight_norm(nn.Conv1d(n_out, n_out, k, stride=stride, padding=padding, dilation=dilation)),
                Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),
            )
            self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
            self.relu = nn.ReLU()
        def forward(self, x):
            out = self.net(x)
            res = x if self.downsample is None else self.downsample(x)
            return self.relu(out + res)

    class TCNModel(nn.Module):
        def __init__(self, configs):
            super().__init__()
            channels = [64, 128, 256, 512]
            k = 3
            layers = []
            for i, c in enumerate(channels):
                d = 2 ** i
                inp = configs.enc_in if i == 0 else channels[i-1]
                layers.append(TemporalBlock(inp, c, k, 1, d, (k-1)*d, 0.2))
            self.tcn = nn.Sequential(*layers)
            self.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(channels[-1], configs.num_classes))

        def forward(self, x):
            out = self.tcn(x.transpose(1, 2))
            return self.fc(out.mean(dim=2))

    return TCNModel(configs)


def build_lstm_ae(configs):
    """LSTM-AE baseline"""
    class LSTMAEModel(nn.Module):
        def __init__(self, configs):
            super().__init__()
            self.encoder_lstm = nn.LSTM(configs.enc_in, 128, 4, batch_first=True, dropout=0.2)
            self.encoder_fc = nn.Linear(128, 64)
            self.decoder_fc = nn.Linear(64, 128)
            self.decoder_lstm = nn.LSTM(128, 128, 4, batch_first=True, dropout=0.2)
            self.output_fc = nn.Linear(128, configs.enc_in)

        def forward(self, x):
            _, (h, _) = self.encoder_lstm(x)
            latent = self.encoder_fc(h[-1])
            hidden = self.decoder_fc(latent)
            dec_input = hidden.unsqueeze(1).repeat(1, x.size(1), 1)
            h_0 = hidden.unsqueeze(0).repeat(4, 1, 1)  # 2 → 4，匹配 num_layers
            c_0 = torch.zeros_like(h_0)
            out, _ = self.decoder_lstm(dec_input, (h_0, c_0))
            return self.output_fc(out), latent

    return LSTMAEModel(configs)


def build_transformer(configs):
    """Transformer baseline（论文中有但代码没给，按标准结构实现）"""
    class TransformerModel(nn.Module):
        def __init__(self, configs):
            super().__init__()
            self.input_proj = nn.Linear(configs.enc_in, 256)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=256, nhead=8, dim_feedforward=512,
                dropout=0.3, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
            self.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(256, configs.num_classes)
            )

        def forward(self, x):
            out = self.encoder(self.input_proj(x))
            return self.fc(out.mean(dim=1))

    return TransformerModel(configs)


# 尝试导入 TreeMIL（如果存在）
def build_treemil(configs):
    """TreeMIL baseline"""
    try:
        from scripts.othermodels.TreeMIL import TreeMIL

        class TreeMILWrapper(nn.Module):
            """包装TreeMIL，使forward输出与其他模型一致"""
            def __init__(self, configs):
                super().__init__()
                self.model = TreeMIL(configs)

            def forward(self, x):
                ret = self.model.get_scores(x)
                wscore = ret['wscore']  # [B]
                # 转为 [B, 2] 格式与其他模型对齐
                return torch.stack([1 - wscore, wscore], dim=1)

        return TreeMILWrapper(configs)
    except ImportError as e:
        print(f"  [WARNING] TreeMIL import failed: {e}")
        return None


def build_fedformer(configs):
    """FEDformer baseline"""
    from models.Fedformer_cls import Model

    class FedConfig:
        """Remap MockConfig fields to what FEDformer expects."""
        def __init__(self, c):
            self.seq_len = c.seq_len
            self.enc_in = c.enc_in
            self.num_classes = c.num_classes
            self.d_model = c.fed_d_model   # 避免与 TreeMIL 的 d_model 冲突
            self.n_heads = c.n_heads
            self.d_ff = c.d_ff
            self.e_layers = c.e_layers
            self.dropout = c.dropout
            self.activation = c.activation
            self.moving_avg = c.moving_avg
            self.embed = c.embed
            self.freq = c.freq
            self.version = c.version
            self.mode_select = c.mode_select
            self.modes = c.modes

    return Model(FedConfig(configs))


# ============================================================
# 测量工具函数
# ============================================================

def count_parameters(model):
    """计算可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model, input_shape, model_name=''):
    """计算 FLOPs"""
    try:
        from thop import profile, clever_format
        dummy = torch.randn(*input_shape).to(DEVICE)
        flops, params = profile(model, inputs=(dummy,), verbose=False)
        return flops
    except ImportError:
        print("  [INFO] thop not installed, estimating FLOPs from params")
        return None
    except Exception as e:
        print(f"  [WARNING] FLOPs calculation failed for {model_name}: {e}")
        return None


def measure_latency(model, input_shape, model_name='', is_ae=False):
    """
    测量推理延迟
    返回：每个 batch 的平均延迟 (ms)
    """
    model.eval()
    dummy = torch.randn(*input_shape).to(DEVICE)

    with torch.no_grad():
        # 预热
        for _ in range(WARMUP_RUNS):
            if is_ae:
                _ = model(dummy)
            else:
                _ = model(dummy)

        # 同步 GPU
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()

        # 正式测量
        times = []
        for _ in range(BENCHMARK_RUNS):
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            _ = model(dummy)

            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

    avg_ms = np.mean(times)
    std_ms = np.std(times)
    return avg_ms, std_ms


def measure_memory(model, input_shape):
    """测量 GPU 显存占用（推理时）"""
    if DEVICE.type != 'cuda':
        return None

    model.eval()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    dummy = torch.randn(*input_shape).to(DEVICE)
    with torch.no_grad():
        _ = model(dummy)

    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return peak_mb


# ============================================================
# 主流程
# ============================================================

def benchmark_pytorch_model(name, build_fn, configs, input_shape, is_ae=False):
    """对单个 PyTorch 模型进行全面基准测试"""
    print(f"\n{'='*50}")
    print(f"  Benchmarking: {name}")
    print(f"{'='*50}")

    # ---- 参数量（用第一个实例）----
    model = build_fn(configs)
    if model is None:
        return None
    model = model.to(DEVICE).eval()
    params = count_parameters(model)
    print(f"  Parameters: {params:,}")

    # ---- FLOPs（用这个实例，之后丢弃）----
    flops = count_flops(model, input_shape, name)
    if flops:
        print(f"  FLOPs: {flops:,.0f} ({flops/1e6:.2f}M)")
    del model
    torch.cuda.empty_cache() if DEVICE.type == 'cuda' else None

    # ---- 延迟和显存（用全新的第二个实例）----
    model2 = build_fn(configs)
    model2 = model2.to(DEVICE).eval()

    avg_ms, std_ms = measure_latency(model2, input_shape, name, is_ae)
    per_window_ms = avg_ms / input_shape[0]
    throughput = input_shape[0] * SEQ_LEN / (avg_ms / 1000)
    print(f"  Batch latency: {avg_ms:.2f} ± {std_ms:.2f} ms")
    print(f"  Per-window: {per_window_ms:.3f} ms")
    print(f"  Throughput: {throughput:,.0f} flows/s")

    memory = measure_memory(model2, input_shape)
    if memory:
        print(f"  GPU memory: {memory:.1f} MB")

    del model2
    torch.cuda.empty_cache() if DEVICE.type == 'cuda' else None

    return {
        'model': name,
        'params': params,
        'flops': flops,
        'flops_m': round(flops / 1e6, 2) if flops else None,
        'batch_latency_ms': round(avg_ms, 2),
        'per_window_ms': round(per_window_ms, 3),
        'throughput_flows_s': int(throughput),
        'gpu_memory_mb': round(memory, 1) if memory else None,
    }

def main():
    print("=" * 60)
    print("  Efficiency Benchmark")
    print(f"  Device: {DEVICE}")
    print(f"  Input shape: [{BATCH_SIZE}, {SEQ_LEN}, {ENC_IN}]")
    print(f"  Warmup: {WARMUP_RUNS}, Runs: {BENCHMARK_RUNS}")
    print("=" * 60)

    configs = MockConfig()
    input_shape = (BATCH_SIZE, SEQ_LEN, ENC_IN)

    # 定义要测试的模型
    models_to_test = [
        ('WSTD (Ours)', build_wstd, False),
        ('LSTM',        build_lstm, False),
        ('MLP',         build_mlp, False),
        ('TCN',         build_tcn, False),
        ('Transformer', build_transformer, False),
        ('LSTM-AE',     build_lstm_ae, True),
        ('TreeMIL',     build_treemil, False),
        ('FEDformer',   build_fedformer, False),
    ]

    results = []
    for name, build_fn, is_ae in models_to_test:
        try:
            result = benchmark_pytorch_model(name, build_fn, configs, input_shape, is_ae)
            if result:
                results.append(result)
        except Exception as e:
            print(f"\n  [ERROR] {name} failed: {e}")
            import traceback
            traceback.print_exc()

    # ---- 汇总表 ----
    print("\n\n" + "=" * 100)
    print("  SUMMARY TABLE (for paper)")
    print("=" * 100)
    print(f"{'Method':<20} {'Params':>12} {'FLOPs(M)':>10} "
          f"{'Latency(ms)':>14} {'Throughput':>15} {'Memory(MB)':>12}")
    print("-" * 100)

    for r in results:
        params_str = f"{r['params']:,}"
        flops_str = f"{r['flops_m']}" if r['flops_m'] else "N/A"
        latency_str = f"{r['per_window_ms']:.3f}"
        throughput_str = f"{r['throughput_flows_s']:,}"
        memory_str = f"{r['gpu_memory_mb']}" if r['gpu_memory_mb'] else "N/A"

        print(f"{r['model']:<20} {params_str:>12} {flops_str:>10} "
              f"{latency_str:>14} {throughput_str:>15} {memory_str:>12}")

    print("=" * 100)
    print(f"* Latency = per-window inference time (ms)")
    print(f"* Throughput = flows processed per second")
    print(f"* Measured on {DEVICE} with batch_size={BATCH_SIZE}")

    # ---- 保存 ----
    os.makedirs(os.path.dirname(SAVE_PATH) if os.path.dirname(SAVE_PATH) else '.', exist_ok=True)
    with open(SAVE_PATH, 'w') as f:
        json.dump({
            'config': {
                'device': str(DEVICE),
                'batch_size': BATCH_SIZE,
                'seq_len': SEQ_LEN,
                'enc_in': ENC_IN,
                'warmup_runs': WARMUP_RUNS,
                'benchmark_runs': BENCHMARK_RUNS,
            },
            'results': results
        }, f, indent=2)

    print(f"\nResults saved to: {SAVE_PATH}")


if __name__ == "__main__":
    main()