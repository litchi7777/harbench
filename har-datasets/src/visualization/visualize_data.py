"""
Sensor Data Visualization Tool for Browser

Generates interactive HTML files using Plotly.

Examples:
    # Visualize a specific sensor/modality
    python tools/visualize_data.py --dataset dsads --user USER00001 --sensor LeftArm_ACC

    # Compare multiple modalities
    python tools/visualize_data.py --dataset dsads --user USER00001 --sensor LeftArm_ACC LeftArm_GYRO LeftArm_MAG

    # Visualize class distribution
    python tools/visualize_data.py --dataset dsads --user USER00001 --sensor LeftArm_ACC --plot-distribution

    # Compare multiple users
    python tools/visualize_data.py --dataset dsads --sensor LeftArm_ACC --compare-users USER00001 USER00002

    # Specify output filename
    python tools/visualize_data.py --dataset dsads --user USER00001 --sensor LeftArm_ACC --output my_viz.html
"""

import argparse
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Tuple

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("Error: plotly is not installed. Please install it:")
    print("  pip install plotly")
    exit(1)


# Activity names (for DSADS)
DSADS_ACTIVITIES = {
    0: "Sitting",
    1: "Standing",
    2: "Lying on back",
    3: "Lying on right side",
    4: "Ascending stairs",
    5: "Descending stairs",
    6: "Standing in elevator",
    7: "Moving in elevator",
    8: "Walking in parking lot",
    9: "Walking on treadmill (flat)",
    10: "Walking on treadmill (incline)",
    11: "Running on treadmill",
    12: "Exercising on stepper",
    13: "Exercising on cross trainer",
    14: "Cycling (horizontal)",
    15: "Cycling (vertical)",
    16: "Rowing",
    17: "Jumping",
    18: "Playing basketball"
}


def load_sensor_data(dataset_path: Path, user_id: str, sensor_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load sensor data"""
    sensor_path = dataset_path / user_id / sensor_name
    X_path = sensor_path / 'X.npy'
    Y_path = sensor_path / 'Y.npy'

    if not X_path.exists() or not Y_path.exists():
        raise FileNotFoundError(f"Data not found: {sensor_path}")

    X = np.load(X_path)
    Y = np.load(Y_path)

    return X, Y


def plot_time_series(X: np.ndarray, Y: np.ndarray, title: str,
                     num_samples: int = 10, activity_names: Dict = None) -> go.Figure:
    """
    Visualize time series data

    Args:
        X: Data (num_windows, channels, window_size)
        Y: Labels (num_windows,)
        title: Plot title
        num_samples: Number of windows to display
        activity_names: Dictionary of activity names
    """
    num_channels = X.shape[1]
    window_size = X.shape[2]

    # Create subplots (one per channel)
    fig = make_subplots(
        rows=num_channels,
        cols=1,
        subplot_titles=[f'Channel {i}' for i in range(num_channels)],
        vertical_spacing=0.05,
        shared_xaxes=True
    )

    # Color map (different color per class)
    colors = {}
    unique_labels = np.unique(Y[:num_samples])

    # Plot each sample
    for i in range(min(num_samples, len(X))):
        label = int(Y[i])
        if label not in colors:
            # Assign color to new label
            color_idx = len(colors) % 20
            colors[label] = f'hsl({color_idx * 360 / 20}, 70%, 50%)'

        # Activity name
        activity_name = activity_names.get(label, f'Class {label}') if activity_names else f'Class {label}'

        for ch in range(num_channels):
            x_vals = np.arange(i * window_size, (i + 1) * window_size)
            y_vals = X[i, ch, :]

            showlegend = (ch == 0)  # Show legend only for first channel

            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='lines',
                    name=activity_name,
                    line=dict(color=colors[label], width=1),
                    legendgroup=f'class_{label}',
                    showlegend=showlegend,
                    hovertemplate=f'{activity_name}<br>Window: {i}<br>Value: %{{y:.3f}}<extra></extra>'
                ),
                row=ch + 1,
                col=1
            )

    # Layout settings
    fig.update_layout(
        title=title,
        height=300 * num_channels,
        hovermode='closest',
        showlegend=True
    )

    fig.update_xaxes(title_text="Time (samples)", row=num_channels, col=1)

    return fig


def plot_class_distribution(Y: np.ndarray, title: str, activity_names: Dict = None) -> go.Figure:
    """Visualize class distribution"""
    unique_labels, counts = np.unique(Y, return_counts=True)

    # Label names
    labels = [activity_names.get(int(l), f'Class {l}') if activity_names else f'Class {l}'
              for l in unique_labels]

    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=counts,
            text=counts,
            textposition='auto',
            hovertemplate='%{x}<br>Count: %{y}<br>Percentage: %{customdata:.2f}%<extra></extra>',
            customdata=[(c / len(Y) * 100) for c in counts]
        )
    ])

    fig.update_layout(
        title=title,
        xaxis_title='Activity',
        yaxis_title='Number of Windows',
        height=500,
        xaxis_tickangle=-45
    )

    return fig


def plot_channel_statistics(X: np.ndarray, title: str) -> go.Figure:
    """Visualize statistics per channel"""
    num_channels = X.shape[1]

    # Calculate statistics for each channel
    channel_stats = []
    for ch in range(num_channels):
        ch_data = X[:, ch, :]
        channel_stats.append({
            'channel': f'Ch {ch}',
            'mean': ch_data.mean(),
            'std': ch_data.std(),
            'min': ch_data.min(),
            'max': ch_data.max(),
            'median': np.median(ch_data)
        })

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=['Mean & Std', 'Min & Max'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )

    # Mean & Std
    fig.add_trace(
        go.Bar(
            x=[s['channel'] for s in channel_stats],
            y=[s['mean'] for s in channel_stats],
            name='Mean',
            error_y=dict(
                type='data',
                array=[s['std'] for s in channel_stats],
                visible=True
            )
        ),
        row=1, col=1
    )

    # Min & Max
    channels = [s['channel'] for s in channel_stats]
    fig.add_trace(
        go.Scatter(
            x=channels,
            y=[s['min'] for s in channel_stats],
            mode='lines+markers',
            name='Min',
            line=dict(color='blue')
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=channels,
            y=[s['max'] for s in channel_stats],
            mode='lines+markers',
            name='Max',
            line=dict(color='red')
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=channels,
            y=[s['median'] for s in channel_stats],
            mode='lines+markers',
            name='Median',
            line=dict(color='green', dash='dash')
        ),
        row=1, col=2
    )

    fig.update_layout(
        title=title,
        height=400,
        showlegend=True
    )

    return fig


def plot_multi_sensor_comparison(dataset_path: Path, user_id: str,
                                 sensor_names: List[str], num_samples: int = 5) -> go.Figure:
    """Compare multiple sensors/modalities"""
    num_sensors = len(sensor_names)

    # Load data
    data_list = []
    for sensor_name in sensor_names:
        try:
            X, Y = load_sensor_data(dataset_path, user_id, sensor_name)
            data_list.append((sensor_name, X, Y))
        except Exception as e:
            print(f"Warning: Failed to load {sensor_name}: {e}")

    if not data_list:
        raise ValueError("No data loaded")

    # Compare first channel of each sensor
    fig = make_subplots(
        rows=num_sensors,
        cols=1,
        subplot_titles=[name for name, _, _ in data_list],
        vertical_spacing=0.05,
        shared_xaxes=True
    )

    for idx, (sensor_name, X, Y) in enumerate(data_list):
        window_size = X.shape[2]

        # Plot first few samples
        for i in range(min(num_samples, len(X))):
            x_vals = np.arange(i * window_size, (i + 1) * window_size)
            # Channel 0 data
            y_vals = X[i, 0, :]

            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='lines',
                    name=f'Window {i}',
                    legendgroup=f'window_{i}',
                    showlegend=(idx == 0),
                    hovertemplate=f'{sensor_name}<br>Window: {i}<br>Value: %{{y:.3f}}<extra></extra>'
                ),
                row=idx + 1,
                col=1
            )

    fig.update_layout(
        title=f'Multi-Sensor Comparison: {user_id}',
        height=300 * num_sensors,
        hovermode='closest'
    )

    fig.update_xaxes(title_text="Time (samples)", row=num_sensors, col=1)

    return fig


def create_dashboard(dataset_path: Path, user_id: str, sensor_name: str,
                     num_samples: int = 10) -> str:
    """Create comprehensive dashboard"""
    # Load data
    X, Y = load_sensor_data(dataset_path, user_id, sensor_name)

    # Create multiple plots
    figures = []

    # 1. Time series data
    fig1 = plot_time_series(X, Y, f'{user_id} / {sensor_name} - Time Series',
                            num_samples, DSADS_ACTIVITIES)
    figures.append(fig1)

    # 2. Class distribution
    fig2 = plot_class_distribution(Y, f'{user_id} / {sensor_name} - Class Distribution',
                                   DSADS_ACTIVITIES)
    figures.append(fig2)

    # 3. Channel statistics
    fig3 = plot_channel_statistics(X, f'{user_id} / {sensor_name} - Channel Statistics')
    figures.append(fig3)

    # Combine as HTML
    html_parts = ['<html><head><meta charset="utf-8"><title>Sensor Data Visualization</title></head><body>']
    html_parts.append(f'<h1>Sensor Data Dashboard</h1>')
    html_parts.append(f'<p>Dataset: {dataset_path.name} | User: {user_id} | Sensor: {sensor_name}</p>')
    html_parts.append(f'<p>Data shape: X{X.shape}, Y{Y.shape}</p>')

    for fig in figures:
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    html_parts.append('</body></html>')

    return '\n'.join(html_parts)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize HAR sensor data in browser',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., dsads)')
    parser.add_argument('--user', type=str, help='User ID (e.g., USER00001)')
    parser.add_argument('--sensor', type=str, nargs='+', help='Sensor/modality name(s)')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples to plot')
    parser.add_argument('--output', type=str, default='visualizations/sensor_viz.html', help='Output HTML file')
    parser.add_argument('--data-dir', type=str, default='data/processed', help='Processed data directory')
    parser.add_argument('--plot-distribution', action='store_true', help='Plot class distribution only')
    parser.add_argument('--compare-users', type=str, nargs='+', help='Compare multiple users')
    parser.add_argument('--dashboard', action='store_true', help='Create comprehensive dashboard')

    args = parser.parse_args()

    dataset_path = Path(args.data_dir) / args.dataset.lower()

    if not dataset_path.exists():
        print(f"Error: Dataset not found: {dataset_path}")
        return 1

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if args.dashboard and args.user and args.sensor:
            # Create dashboard
            html_content = create_dashboard(dataset_path, args.user, args.sensor[0], args.num_samples)
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(html_content)

        elif len(args.sensor or []) > 1 and args.user:
            # Multi-sensor comparison
            fig = plot_multi_sensor_comparison(dataset_path, args.user, args.sensor, args.num_samples)
            fig.write_html(args.output)

        elif args.sensor and args.user:
            # Single sensor visualization
            X, Y = load_sensor_data(dataset_path, args.user, args.sensor[0])

            if args.plot_distribution:
                fig = plot_class_distribution(Y, f'{args.user} / {args.sensor[0]}', DSADS_ACTIVITIES)
            else:
                fig = plot_time_series(X, Y, f'{args.user} / {args.sensor[0]}',
                                     args.num_samples, DSADS_ACTIVITIES)

            fig.write_html(args.output)

        else:
            print("Error: Please specify --user and --sensor, or use --dashboard")
            return 1

        print(f"✓ Visualization saved: {args.output}")
        print(f"  Open this file in your browser to view the interactive visualization")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
