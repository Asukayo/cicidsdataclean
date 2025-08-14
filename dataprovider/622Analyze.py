import numpy as np
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from config import CICIDS_WINDOW_SIZE, CICIDS_WINDOW_STEP


def load_data(data_dir, window_size=100, step_size=20):
    """Load processed data"""
    X_file = os.path.join(data_dir, f'selected_X_w{window_size}_s{step_size}.npy')
    y_file = os.path.join(data_dir, f'selected_y_w{window_size}_s{step_size}.npy')
    metadata_file = os.path.join(data_dir, f'selected_metadata_w{window_size}_s{step_size}.pkl')

    print("Loading data...")
    X = np.load(X_file)
    y = np.load(y_file)

    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)

    print(f"Data shape: X{X.shape}, y{y.shape}")
    print(f"Number of features: {len(metadata['feature_names'])}")

    return X, y, metadata


def split_data_chronologically(X, y, train_ratio=0.6, val_ratio=0.2):
    """Split data chronologically"""
    total_samples = len(X)
    train_end = int(total_samples * train_ratio)
    val_end = int(total_samples * (train_ratio + val_ratio))

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]

    X_test = X[val_end:]
    y_test = y[val_end:]

    print(f"\nData split:")
    print(f"  Training set: {len(X_train)} windows ({len(X_train) / total_samples * 100:.1f}%)")
    print(f"  Validation set: {len(X_val)} windows ({len(X_val) / total_samples * 100:.1f}%)")
    print(f"  Test set: {len(X_test)} windows ({len(X_test) / total_samples * 100:.1f}%)")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def aggregate_window_labels(y_data):
    """Aggregate window labels using majority voting"""
    window_labels = []
    for window_y in y_data:
        # Calculate the ratio of malicious traffic in each window
        malicious_ratio = np.mean(window_y > 0)
        # Mark as malicious if malicious traffic ratio > 50%
        window_labels.append(1 if malicious_ratio > 0.0001 else 0)
    return np.array(window_labels)


def analyze_flow_level_distribution(y_data, label_mapping, set_name):
    """Analyze flow-level label distribution"""
    print(f"\n=== {set_name} Flow-Level Distribution ===")

    # Flatten all flows in the set
    all_flows = y_data.flatten()

    # Count each label
    flow_counts = Counter(all_flows)
    total_flows = len(all_flows)

    print(f"Total flows: {total_flows:,}")

    # Separate normal and malicious
    normal_count = flow_counts.get(0, 0)  # Assuming 0 is 'Benign'
    malicious_count = total_flows - normal_count

    print(f"Normal flows: {normal_count:,} ({normal_count / total_flows * 100:.2f}%)")
    print(f"Malicious flows: {malicious_count:,} ({malicious_count / total_flows * 100:.2f}%)")

    # Detailed breakdown by attack type
    print(f"\nDetailed breakdown:")
    sorted_labels = sorted(flow_counts.items(), key=lambda x: x[1], reverse=True)

    for label_id, count in sorted_labels:
        label_name = label_mapping.get(label_id, f"Unknown_{label_id}")
        percentage = count / total_flows * 100
        print(f"  {label_name}: {count:,} ({percentage:.2f}%)")

    return {
        'total_flows': total_flows,
        'normal_flows': normal_count,
        'malicious_flows': malicious_count,
        'detailed_counts': dict(sorted_labels),
        'label_mapping': {label_mapping.get(label_id, f"Unknown_{label_id}"): count
                          for label_id, count in sorted_labels}
    }


def analyze_window_level_distribution(y_data, set_name):
    """Analyze window-level label distribution"""
    print(f"\n=== {set_name} Window-Level Distribution ===")

    window_labels = aggregate_window_labels(y_data)
    total_windows = len(window_labels)

    normal_windows = np.sum(window_labels == 0)
    malicious_windows = np.sum(window_labels == 1)

    print(f"Total windows: {total_windows:,}")
    print(f"Normal windows: {normal_windows:,} ({normal_windows / total_windows * 100:.2f}%)")
    print(f"Malicious windows: {malicious_windows:,} ({malicious_windows / total_windows * 100:.2f}%)")

    # Analyze malicious ratio distribution in windows
    malicious_ratios = []
    for window_y in y_data:
        malicious_ratio = np.mean(window_y > 0)
        malicious_ratios.append(malicious_ratio)

    malicious_ratios = np.array(malicious_ratios)

    print(f"\nMalicious ratio statistics in windows:")
    print(f"  Mean: {np.mean(malicious_ratios):.4f}")
    print(f"  Std: {np.std(malicious_ratios):.4f}")
    print(f"  Min: {np.min(malicious_ratios):.4f}")
    print(f"  Max: {np.max(malicious_ratios):.4f}")
    print(f"  Windows with 0% malicious: {np.sum(malicious_ratios == 0):,}")
    print(f"  Windows with 100% malicious: {np.sum(malicious_ratios == 1):,}")
    print(f"  Windows with mixed traffic: {np.sum((malicious_ratios > 0) & (malicious_ratios < 1)):,}")

    return {
        'total_windows': total_windows,
        'normal_windows': normal_windows,
        'malicious_windows': malicious_windows,
        'malicious_ratios': malicious_ratios,
        'window_labels': window_labels
    }


def create_distribution_plots(train_stats, val_stats, test_stats, save_path=None):
    """Create visualization plots for data distribution"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Window-level distribution
    sets = ['Training', 'Validation', 'Test']
    window_stats = [train_stats['window'], val_stats['window'], test_stats['window']]

    for i, (set_name, stats) in enumerate(zip(sets, window_stats)):
        ax = axes[0, i]
        labels = ['Normal', 'Malicious']
        sizes = [stats['normal_windows'], stats['malicious_windows']]
        colors = ['lightblue', 'lightcoral']

        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title(f'{set_name} Set - Window Level\n({stats["total_windows"]:,} windows)')

        # Add count annotations
        for autotext, size in zip(autotexts, sizes):
            autotext.set_text(f'{size:,}\n({size / sum(sizes) * 100:.1f}%)')

    # Flow-level distribution
    flow_stats = [train_stats['flow'], val_stats['flow'], test_stats['flow']]

    for i, (set_name, stats) in enumerate(zip(sets, flow_stats)):
        ax = axes[1, i]
        labels = ['Normal', 'Malicious']
        sizes = [stats['normal_flows'], stats['malicious_flows']]
        colors = ['lightgreen', 'salmon']

        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title(f'{set_name} Set - Flow Level\n({stats["total_flows"]:,} flows)')

        # Add count annotations
        for autotext, size in zip(autotexts, sizes):
            autotext.set_text(f'{size:,}\n({size / sum(sizes) * 100:.1f}%)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distribution plots saved to: {save_path}")

    plt.show()


def create_malicious_ratio_histogram(train_stats, val_stats, test_stats, save_path=None):
    """Create histogram of malicious ratios in windows"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    sets = ['Training', 'Validation', 'Test']
    window_stats = [train_stats['window'], val_stats['window'], test_stats['window']]
    colors = ['blue', 'orange', 'green']

    for i, (set_name, stats, color) in enumerate(zip(sets, window_stats, colors)):
        ax = axes[i]
        ratios = stats['malicious_ratios']

        ax.hist(ratios, bins=50, alpha=0.7, color=color, edgecolor='black')
        ax.set_xlabel('Malicious Traffic Ratio in Window')
        ax.set_ylabel('Number of Windows')
        ax.set_title(f'{set_name} Set\nMalicious Ratio Distribution')
        ax.grid(True, alpha=0.3)

        # Add statistics text
        stats_text = f'Mean: {np.mean(ratios):.3f}\nStd: {np.std(ratios):.3f}'
        ax.text(0.7, 0.8, stats_text, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Malicious ratio histogram saved to: {save_path}")

    plt.show()


def create_summary_report(train_stats, val_stats, test_stats, output_dir):
    """Create a comprehensive summary report"""
    report_path = os.path.join(output_dir, 'data_distribution_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("CICIDS2017 Data Distribution Analysis Report\n")
        f.write("=" * 50 + "\n\n")

        f.write("OVERVIEW\n")
        f.write("-" * 20 + "\n")
        total_windows = (train_stats['window']['total_windows'] +
                         val_stats['window']['total_windows'] +
                         test_stats['window']['total_windows'])
        total_flows = (train_stats['flow']['total_flows'] +
                       val_stats['flow']['total_flows'] +
                       test_stats['flow']['total_flows'])

        f.write(f"Total windows: {total_windows:,}\n")
        f.write(f"Total flows: {total_flows:,}\n")
        f.write(f"Window size: {CICIDS_WINDOW_SIZE}\n")
        f.write(f"Step size: {CICIDS_WINDOW_STEP}\n\n")

        # Window-level analysis
        f.write("WINDOW-LEVEL DISTRIBUTION\n")
        f.write("-" * 30 + "\n")

        sets = [('Training', train_stats), ('Validation', val_stats), ('Test', test_stats)]

        for set_name, stats in sets:
            window_stats = stats['window']
            f.write(f"\n{set_name} Set:\n")
            f.write(f"  Total windows: {window_stats['total_windows']:,}\n")
            f.write(f"  Normal windows: {window_stats['normal_windows']:,} "
                    f"({window_stats['normal_windows'] / window_stats['total_windows'] * 100:.2f}%)\n")
            f.write(f"  Malicious windows: {window_stats['malicious_windows']:,} "
                    f"({window_stats['malicious_windows'] / window_stats['total_windows'] * 100:.2f}%)\n")

        # Flow-level analysis
        f.write(f"\n\nFLOW-LEVEL DISTRIBUTION\n")
        f.write("-" * 30 + "\n")

        for set_name, stats in sets:
            flow_stats = stats['flow']
            f.write(f"\n{set_name} Set:\n")
            f.write(f"  Total flows: {flow_stats['total_flows']:,}\n")
            f.write(f"  Normal flows: {flow_stats['normal_flows']:,} "
                    f"({flow_stats['normal_flows'] / flow_stats['total_flows'] * 100:.2f}%)\n")
            f.write(f"  Malicious flows: {flow_stats['malicious_flows']:,} "
                    f"({flow_stats['malicious_flows'] / flow_stats['total_flows'] * 100:.2f}%)\n")

        # Detailed attack type breakdown
        f.write(f"\n\nDETAILED ATTACK TYPE BREAKDOWN\n")
        f.write("-" * 40 + "\n")

        for set_name, stats in sets:
            f.write(f"\n{set_name} Set Attack Types:\n")
            label_mapping = stats['flow']['label_mapping']
            for label_name, count in sorted(label_mapping.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {label_name}: {count:,}\n")

    print(f"Summary report saved to: {report_path}")


def main():
    # Configuration
    data_dir = "../cicids2017/selected_features"
    output_dir = "./distribution_analysis"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    X, y, metadata = load_data(data_dir, CICIDS_WINDOW_SIZE, CICIDS_WINDOW_STEP)

    # Split data chronologically
    train_data, val_data, test_data = split_data_chronologically(X, y)

    # Analyze each set
    label_mapping = metadata['label_mapping']

    print("\n" + "=" * 60)
    print("ANALYZING DATA DISTRIBUTION")
    print("=" * 60)

    # Training set analysis
    train_flow_stats = analyze_flow_level_distribution(train_data[1], label_mapping, "TRAINING SET")
    train_window_stats = analyze_window_level_distribution(train_data[1], "TRAINING SET")

    # Validation set analysis
    val_flow_stats = analyze_flow_level_distribution(val_data[1], label_mapping, "VALIDATION SET")
    val_window_stats = analyze_window_level_distribution(val_data[1], "VALIDATION SET")

    # Test set analysis
    test_flow_stats = analyze_flow_level_distribution(test_data[1], label_mapping, "TEST SET")
    test_window_stats = analyze_window_level_distribution(test_data[1], "TEST SET")

    # Combine statistics
    train_stats = {'flow': train_flow_stats, 'window': train_window_stats}
    val_stats = {'flow': val_flow_stats, 'window': val_window_stats}
    test_stats = {'flow': test_flow_stats, 'window': test_window_stats}

    # Create visualizations
    print(f"\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)

    create_distribution_plots(
        train_stats, val_stats, test_stats,
        os.path.join(output_dir, 'data_distribution.png')
    )

    create_malicious_ratio_histogram(
        train_stats, val_stats, test_stats,
        os.path.join(output_dir, 'malicious_ratio_histogram.png')
    )

    # Create summary report
    create_summary_report(train_stats, val_stats, test_stats, output_dir)

    # Print final summary
    print(f"\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print(f"\nWindow-Level Summary:")
    print(
        f"  Training: {train_window_stats['normal_windows']:,} normal, {train_window_stats['malicious_windows']:,} malicious")
    print(
        f"  Validation: {val_window_stats['normal_windows']:,} normal, {val_window_stats['malicious_windows']:,} malicious")
    print(
        f"  Test: {test_window_stats['normal_windows']:,} normal, {test_window_stats['malicious_windows']:,} malicious")

    print(f"\nFlow-Level Summary:")
    print(f"  Training: {train_flow_stats['normal_flows']:,} normal, {train_flow_stats['malicious_flows']:,} malicious")
    print(f"  Validation: {val_flow_stats['normal_flows']:,} normal, {val_flow_stats['malicious_flows']:,} malicious")
    print(f"  Test: {test_flow_stats['normal_flows']:,} normal, {test_flow_stats['malicious_flows']:,} malicious")

    print(f"\nAnalysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()