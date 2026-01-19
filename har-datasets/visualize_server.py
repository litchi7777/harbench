#!/usr/bin/env python3
"""
Web Application for Sensor Data Visualization

Explore datasets with hierarchical navigation:
1. Top page: Dataset list
2. Dataset page: User x Sensor x Modality grid
3. Visualization page: Visualization of selected data

Examples:
    python serve.py
    python serve.py --port 8080
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from flask import Flask, render_template_string, request, jsonify

# Plotly import
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("Error: plotly is not installed. Please install it:")
    print("  pip install plotly")
    sys.exit(1)

# Dataset info import
sys.path.insert(0, str(Path(__file__).parent))
from src.dataset_info import DATASETS

app = Flask(__name__)

# Global settings
DATA_DIR = Path('data/processed')

# Helper function to get activity name from dataset name
def get_activity_name(dataset_name: str, class_id: int) -> str:
    """Get activity name from dataset name and class ID"""
    dataset_key = dataset_name.upper()
    if dataset_key in DATASETS and 'labels' in DATASETS[dataset_key]:
        return DATASETS[dataset_key]['labels'].get(class_id, f'Class {class_id}')
    return f'Class {class_id}'

# HTML template
NEW_UI_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HAR Data Visualization - Interactive</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        html {
            overflow: hidden;
            height: 100vh;
            width: 100vw;
        }
        body {
            font-family: 'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #f5f5f5;
            overflow: hidden;
            height: 100vh;
            width: 100vw;
        }
        .app-container {
            display: flex;
            height: 100vh;
            width: 100vw;
            overflow: hidden;
        }

        /* Leftmost navigation bar */
        .nav-bar {
            width: 80px;
            background: #202124;
            display: flex;
            flex-direction: column;
            border-right: 1px solid #000;
        }
        .nav-item {
            padding: 20px 0;
            text-align: center;
            cursor: pointer;
            color: #9aa0a6;
            border-left: 3px solid transparent;
            transition: all 0.2s;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 8px;
        }
        .nav-item:hover {
            background: #292a2d;
            color: #e8eaed;
        }
        .nav-item.active {
            background: #292a2d;
            color: #1a73e8;
            border-left-color: #1a73e8;
        }
        .nav-icon {
            font-size: 24px;
        }
        .nav-label {
            font-size: 11px;
            font-weight: 500;
        }

        /* Left sidebar */
        .sidebar {
            width: 300px;
            flex-shrink: 0;
            background: white;
            border-right: 1px solid #e0e0e0;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .sidebar-header {
            padding: 16px;
            border-bottom: 1px solid #e0e0e0;
            background: #1a73e8;
            color: white;
        }
        .sidebar-header h1 {
            font-size: 18px;
            font-weight: 500;
        }
        .tree-container {
            flex: 1;
            overflow-y: auto;
            overflow-x: hidden;
            padding: 8px;
        }
        .tree-node {
            padding: 8px;
            cursor: pointer;
            border-radius: 4px;
            margin: 2px 0;
            user-select: none;
            -webkit-tap-highlight-color: transparent;
        }
        .tree-node:hover {
            background: #f0f0f0;
        }
        .tree-node:focus {
            outline: none;
        }
        .tree-node.dataset {
            font-weight: 500;
            color: #202124;
        }
        .tree-node.user {
            margin-left: 8px;
            color: #5f6368;
        }
        .tree-node.position {
            margin-left: 16px;
            color: #1a73e8;
        }
        .tree-node.modality {
            margin-left: 24px;
            color: #fb8c00;
        }
        .tree-node.class {
            margin-left: 32px;
            color: #34a853;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 13px;
        }
        .add-btn {
            background: #1a73e8;
            color: white;
            border: none;
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 11px;
            cursor: pointer;
            flex-shrink: 0;
        }
        .add-btn:hover {
            background: #1557b0;
        }
        .collapsed > .tree-children {
            display: none;
        }

        /* Right main area */
        .main-area {
            flex: 1;
            min-width: 0;
            max-width: 100%;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .main-header {
            padding: 16px 24px;
            background: white;
            border-bottom: 1px solid #e0e0e0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .main-header h2 {
            font-size: 16px;
            font-weight: 500;
            color: #202124;
        }
        .header-controls {
            display: flex;
            gap: 12px;
            align-items: center;
        }
        .stats-btn {
            padding: 8px 16px;
            border: 1px solid #1a73e8;
            border-radius: 4px;
            background: white;
            color: #1a73e8;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: background 0.2s, color 0.2s;
        }
        .stats-btn:hover {
            background: #1a73e8;
            color: white;
        }
        .panels-container {
            flex: 1;
            overflow-y: auto;
            overflow-x: hidden;
            padding: 16px;
            width: 100%;
            min-width: 0;
        }

        /* Panels */
        .panel {
            background: white;
            border: 1px solid #dadce0;
            border-radius: 8px;
            margin-bottom: 16px;
            overflow: hidden;
            width: 100%;
            min-width: 0;
        }
        .panel-header {
            padding: 12px 16px;
            background: #f8f9fa;
            border-bottom: 1px solid #e0e0e0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
        }
        .panel-title {
            font-weight: 500;
            color: #202124;
            font-size: 14px;
            flex: 1;
            min-width: 0;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .panel-controls {
            display: flex;
            gap: 8px;
            align-items: center;
            flex-shrink: 0;
        }
        .panel-controls select,
        .panel-controls input {
            padding: 4px 8px;
            border: 1px solid #dadce0;
            border-radius: 4px;
            font-size: 12px;
        }
        .panel-controls button {
            padding: 4px 12px;
            border: 1px solid #dadce0;
            border-radius: 4px;
            background: white;
            cursor: pointer;
            font-size: 12px;
        }
        .panel-controls button:hover {
            background: #f8f9fa;
        }
        .remove-btn {
            color: #d93025;
            border-color: #d93025;
        }
        .remove-btn:hover {
            background: #fce8e6;
        }
        .panel-content {
            display: flex;
            width: 100%;
            min-width: 0;
            min-height: 280px;
            overflow: hidden;
        }
        .panel-metadata {
            width: 150px;
            padding: 12px;
            border-right: 1px solid #e0e0e0;
            background: #fafafa;
            flex-shrink: 0;
            overflow-y: auto;
        }
        .metadata-item {
            margin-bottom: 8px;
            font-size: 12px;
        }
        .metadata-label {
            color: #5f6368;
            font-size: 10px;
            text-transform: uppercase;
            margin-bottom: 2px;
        }
        .metadata-value {
            color: #202124;
            font-weight: 500;
        }
        .panel-plots {
            flex: 1;
            min-width: 0;
            display: flex;
            overflow-x: auto;
            overflow-y: hidden;
            padding: 12px;
            gap: 12px;
            flex-wrap: nowrap;
            -webkit-overflow-scrolling: touch;
        }
        .plot-item {
            width: 300px;
            min-width: 300px;
            max-width: 300px;
            flex: 0 0 300px;
            height: 250px;
            display: flex;
            flex-direction: column;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            background: white;
        }
        .plot-header {
            font-size: 12px;
            color: #5f6368;
            padding: 8px;
            background: #f8f9fa;
            border-bottom: 1px solid #e0e0e0;
            flex-shrink: 0;
            height: 36px;
            box-sizing: border-box;
        }
        .plot-item > div:last-child {
            flex: 1;
            width: 100%;
            height: calc(100% - 36px);
            overflow: hidden;
        }
        .empty-state {
            text-align: center;
            padding: 48px;
            color: #5f6368;
        }
        .empty-state h3 {
            margin-bottom: 8px;
            color: #202124;
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
        }
        ::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #a1a1a1;
        }

        .loading {
            text-align: center;
            padding: 24px;
            color: #5f6368;
        }

        /* Statistics modal */
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            z-index: 2000;
            align-items: center;
            justify-content: center;
        }
        .modal.show {
            display: flex;
        }
        .modal-content {
            background: white;
            border-radius: 8px;
            max-width: 900px;
            max-height: 80vh;
            width: 90%;
            display: flex;
            flex-direction: column;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        .modal-header {
            padding: 20px 24px;
            border-bottom: 1px solid #e0e0e0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .modal-title {
            font-size: 20px;
            font-weight: 500;
            color: #202124;
        }
        .modal-close {
            background: none;
            border: none;
            font-size: 24px;
            color: #5f6368;
            cursor: pointer;
            padding: 0;
            width: 32px;
            height: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
        }
        .modal-close:hover {
            background: #f0f0f0;
        }
        .modal-body {
            padding: 24px;
            overflow-y: auto;
            flex: 1;
        }
        .stats-loading {
            text-align: center;
            padding: 48px;
            color: #5f6368;
        }
        .stats-summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        .stats-card {
            background: #f8f9fa;
            padding: 16px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
        }
        .stats-card-label {
            font-size: 12px;
            color: #5f6368;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .stats-card-value {
            font-size: 32px;
            font-weight: 500;
            color: #1a73e8;
        }
        .stats-dataset {
            margin-bottom: 24px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
        }
        .stats-dataset-header {
            background: #1a73e8;
            color: white;
            padding: 12px 16px;
            font-weight: 500;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .stats-dataset-header:hover {
            background: #1557b0;
        }
        .stats-dataset-body {
            padding: 16px;
            background: white;
        }
        .stats-dataset.collapsed .stats-dataset-body {
            display: none;
        }
        .stats-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        .stats-table th {
            text-align: left;
            padding: 8px;
            background: #f8f9fa;
            color: #5f6368;
            font-weight: 500;
            border-bottom: 2px solid #e0e0e0;
        }
        .stats-table td {
            padding: 8px;
            border-bottom: 1px solid #f0f0f0;
            color: #202124;
        }
        .stats-table tr:hover {
            background: #f8f9fa;
        }
        .expand-icon {
            transition: transform 0.2s;
        }
        .stats-dataset.collapsed .expand-icon {
            transform: rotate(-90deg);
        }

        /* Statistics view specific styles */
        .stats-view-container {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
            background: #f5f5f5;
        }
        .stats-section {
            background: white;
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 24px;
            border: 1px solid #dadce0;
        }
        .stats-section-title {
            font-size: 18px;
            font-weight: 500;
            color: #202124;
            margin-bottom: 16px;
        }

        /* View switching */
        .view-mode {
            display: none !important;
            flex: 1;
            min-width: 0;
            max-width: 100%;
            overflow: hidden;
        }
        .view-mode.active {
            display: flex !important;
        }
        #statsView.active {
            flex-direction: column;
        }

        /* Responsive design */
        @media (max-width: 1200px) {
            .sidebar {
                width: 250px;
                min-width: 200px;
            }
            .plot-item {
                min-width: 250px;
            }
        }

        @media (max-width: 768px) {
            .nav-bar {
                width: 60px;
            }
            .nav-label {
                display: none;
            }
            .sidebar {
                width: 200px;
                min-width: 150px;
            }
            .panel-metadata {
                min-width: 100px;
                max-width: 120px;
            }
            .plot-item {
                min-width: 220px;
                max-width: 400px;
            }
            .main-header {
                padding: 12px 16px;
            }
            .panels-container {
                padding: 12px;
            }
        }

        @media (max-width: 480px) {
            .nav-bar {
                display: none;
            }
            .sidebar {
                width: 100%;
                max-width: 100%;
                border-right: none;
            }
            .main-area {
                display: none;
            }
            .app-container {
                flex-direction: column;
            }
        }

        /* Plotly chart responsive */
        .js-plotly-plot {
            width: 100%;
        }
    </style>
</head>
<body>
    <div class="app-container">
        <!-- Leftmost navigation bar -->
        <div class="nav-bar">
            <div class="nav-item active" onclick="switchView('data')" id="navData">
                <div class="nav-icon">[D]</div>
                <div class="nav-label">Data</div>
            </div>
            <div class="nav-item" onclick="switchView('stats')" id="navStats">
                <div class="nav-icon">[S]</div>
                <div class="nav-label">Statistics</div>
            </div>
        </div>

        <!-- Data View -->
        <div id="dataView" class="view-mode active" style="display: flex; flex: 1;">
            <!-- Left sidebar -->
            <div class="sidebar">
                <div class="sidebar-header">
                    <h1>HAR Datasets</h1>
                </div>
                <div class="tree-container" id="treeContainer">
                    <div class="loading">Loading...</div>
                </div>
            </div>

            <!-- Right main area -->
            <div class="main-area">
                <div class="main-header">
                    <h2>Sensor Data Panels</h2>
                    <div class="header-controls">
                        <label style="font-size: 14px; color: #5f6368;">Sampling:</label>
                        <select id="samplingMode" onchange="updateSamplingMode()" style="padding: 6px 12px; border: 1px solid #dadce0; border-radius: 4px; font-size: 14px;">
                            <option value="random">Random</option>
                            <option value="sequential">Sequential</option>
                        </select>
                        <button onclick="clearAllPanels()" style="padding: 8px 16px; border: 1px solid #dadce0; border-radius: 4px; background: white; cursor: pointer;">
                            Clear All
                        </button>
                    </div>
                </div>
                <div class="panels-container" id="panelsContainer">
                    <div class="empty-state">
                        <h3>No panels added</h3>
                        <p>Select a sensor from the left sidebar to get started</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Statistics View -->
        <div id="statsView" class="view-mode">
            <div class="main-header" style="border-bottom: 1px solid #e0e0e0;">
                <h2>Dataset Statistics</h2>
                <div class="header-controls">
                    <button onclick="loadStatistics()" style="padding: 8px 16px; border: 1px solid #1a73e8; border-radius: 4px; background: white; color: #1a73e8; cursor: pointer; font-weight: 500;">
                        Refresh
                    </button>
                </div>
            </div>
            <div class="stats-view-container" id="statsViewContainer">
                <div class="stats-loading">Loading statistics...</div>
            </div>
        </div>
    </div>

    <script>
        let panels = [];
        let panelIdCounter = 0;
        let globalSamplingMode = 'random';  // Global sampling mode
        let currentView = 'data';  // Current view
        let statsCache = null;  // Statistics cache

        // View switching
        function switchView(viewName) {
            // Update navigation items
            document.querySelectorAll('.nav-item').forEach(item => {
                item.classList.remove('active');
            });
            document.getElementById('nav' + viewName.charAt(0).toUpperCase() + viewName.slice(1)).classList.add('active');

            // Switch view
            document.querySelectorAll('.view-mode').forEach(view => {
                view.classList.remove('active');
            });
            document.getElementById(viewName + 'View').classList.add('active');

            currentView = viewName;

            // Load statistics when switching to stats view
            if (viewName === 'stats' && !statsCache) {
                loadStatistics();
            }
        }

        // Load statistics
        async function loadStatistics() {
            const container = document.getElementById('statsViewContainer');
            container.innerHTML = '<div class="stats-loading">Loading statistics...</div>';

            try {
                const response = await fetch('/api/statistics');
                const stats = await response.json();

                if (stats.error) {
                    container.innerHTML = `<div class="stats-loading">Error: ${stats.error}</div>`;
                    return;
                }

                statsCache = stats;
                renderStatisticsView(stats);
            } catch (error) {
                console.error('Failed to load statistics:', error);
                container.innerHTML = '<div class="stats-loading">Failed to load statistics</div>';
            }
        }

        // Render statistics view
        function renderStatisticsView(stats) {
            const container = document.getElementById('statsViewContainer');

            let html = '';

            // Summary section
            html += '<div class="stats-section">';
            html += '<div class="stats-section-title">Overview</div>';
            html += '<div class="stats-summary">';
            html += `
                <div class="stats-card">
                    <div class="stats-card-label">Total Datasets</div>
                    <div class="stats-card-value">${stats.total_datasets}</div>
                </div>
                <div class="stats-card">
                    <div class="stats-card-label">Total Windows</div>
                    <div class="stats-card-value">${stats.total_windows.toLocaleString()}</div>
                </div>
                <div class="stats-card">
                    <div class="stats-card-label">Total Users</div>
                    <div class="stats-card-value">${stats.total_users}</div>
                </div>
            `;
            html += '</div>';
            html += '</div>';

            // Details for each dataset
            stats.datasets.forEach((dataset, idx) => {
                html += '<div class="stats-section">';
                html += `<div class="stats-section-title">${dataset.name.toUpperCase()}</div>`;

                // Dataset summary
                html += '<div class="stats-summary" style="margin-bottom: 20px;">';
                html += `
                    <div class="stats-card">
                        <div class="stats-card-label">Total Windows</div>
                        <div class="stats-card-value">${dataset.total_windows.toLocaleString()}</div>
                    </div>
                    <div class="stats-card">
                        <div class="stats-card-label">Users</div>
                        <div class="stats-card-value">${dataset.num_users}</div>
                    </div>
                    <div class="stats-card">
                        <div class="stats-card-label">Sensors</div>
                        <div class="stats-card-value">${dataset.num_sensors}</div>
                    </div>
                    <div class="stats-card">
                        <div class="stats-card-label">Activity Classes</div>
                        <div class="stats-card-value">${dataset.num_classes}</div>
                    </div>
                `;
                html += '</div>';

                // Sensor list
                html += '<div style="margin-bottom: 20px;">';
                html += '<h3 style="font-size: 14px; color: #5f6368; margin-bottom: 8px;">Available Sensors:</h3>';
                html += '<div style="display: flex; flex-wrap: wrap; gap: 8px;">';
                dataset.sensors.forEach(sensor => {
                    html += `<span style="background: #e8f0fe; color: #1967d2; padding: 4px 12px; border-radius: 16px; font-size: 12px; font-weight: 500;">${sensor}</span>`;
                });
                html += '</div></div>';

                // Activity class summary
                html += '<h3 style="font-size: 14px; color: #5f6368; margin-bottom: 12px;">Activity Class Summary:</h3>';
                html += '<table class="stats-table" style="margin-bottom: 20px;">';
                html += `
                    <thead>
                        <tr>
                            <th>Class ID</th>
                            <th>Activity Name</th>
                            <th>Total Windows</th>
                            <th>Sensors</th>
                            <th>Windows/Sensor</th>
                        </tr>
                    </thead>
                    <tbody>
                `;

                dataset.class_summary.forEach(classInfo => {
                    const windowsPerSensor = (classInfo.total_windows / classInfo.num_sensors).toFixed(1);
                    html += `
                        <tr>
                            <td><strong>${classInfo.class_id}</strong></td>
                            <td>${classInfo.name}</td>
                            <td><strong>${classInfo.total_windows.toLocaleString()}</strong></td>
                            <td>${classInfo.num_sensors} sensors</td>
                            <td style="color: #1a73e8; font-weight: 500;">${windowsPerSensor}</td>
                        </tr>
                    `;
                });

                html += '</tbody></table>';

                // Detailed table (collapsible)
                html += `
                    <details style="margin-top: 16px;">
                        <summary style="cursor: pointer; padding: 8px; background: #f8f9fa; border-radius: 4px; font-weight: 500; color: #5f6368;">
                            Show detailed breakdown (${dataset.details.length} entries)
                        </summary>
                        <table class="stats-table" style="margin-top: 12px;">
                            <thead>
                                <tr>
                                    <th>User</th>
                                    <th>Sensor</th>
                                    <th>Activity Class</th>
                                    <th>Windows</th>
                                </tr>
                            </thead>
                            <tbody>
                `;

                dataset.details.forEach(detail => {
                    const activityName = detail.activity_name || `Class ${detail.class_id}`;
                    html += `
                        <tr>
                            <td>${detail.user}</td>
                            <td>${detail.sensor}</td>
                            <td>${activityName}</td>
                            <td><strong>${detail.count.toLocaleString()}</strong></td>
                        </tr>
                    `;
                });

                html += `
                            </tbody>
                        </table>
                    </details>
                `;

                html += '</div>';
            });

            container.innerHTML = html;
        }

        // Load tree data
        async function loadTree() {
            try {
                const response = await fetch('/api/tree');
                const tree = await response.json();
                renderTree(tree);
            } catch (error) {
                console.error('Failed to load tree:', error);
                document.getElementById('treeContainer').innerHTML = '<div class="loading">Error loading data</div>';
            }
        }

        // Render tree
        function renderTree(tree) {
            const container = document.getElementById('treeContainer');
            container.innerHTML = '';

            tree.forEach(dataset => {
                const datasetNode = createTreeNode(dataset, 'dataset');
                container.appendChild(datasetNode);
            });
        }

        // Create tree node
        function createTreeNode(node, type) {
            const div = document.createElement('div');

            if (type === 'dataset') {
                div.className = 'tree-node dataset collapsed';
                div.innerHTML = `${node.name}`;
                div.onclick = (e) => {
                    e.stopPropagation();
                    div.classList.toggle('collapsed');
                };

                const childrenDiv = document.createElement('div');
                childrenDiv.className = 'tree-children';
                node.children.forEach(user => {
                    childrenDiv.appendChild(createTreeNode(user, 'user'));
                });
                div.appendChild(childrenDiv);
            } else if (type === 'user') {
                div.className = 'tree-node user collapsed';
                div.innerHTML = `${node.name}`;
                div.onclick = (e) => {
                    e.stopPropagation();
                    div.classList.toggle('collapsed');
                };

                const childrenDiv = document.createElement('div');
                childrenDiv.className = 'tree-children';
                node.children.forEach(position => {
                    childrenDiv.appendChild(createTreeNode(position, 'position'));
                });
                div.appendChild(childrenDiv);
            } else if (type === 'position') {
                div.className = 'tree-node position collapsed';
                div.innerHTML = `${node.name}`;
                div.onclick = (e) => {
                    e.stopPropagation();
                    div.classList.toggle('collapsed');
                };

                const childrenDiv = document.createElement('div');
                childrenDiv.className = 'tree-children';
                node.children.forEach(modality => {
                    childrenDiv.appendChild(createTreeNode(modality, 'modality'));
                });
                div.appendChild(childrenDiv);
            } else if (type === 'modality') {
                div.className = 'tree-node modality collapsed';
                div.innerHTML = `${node.name}`;
                div.onclick = (e) => {
                    e.stopPropagation();
                    div.classList.toggle('collapsed');
                };

                const childrenDiv = document.createElement('div');
                childrenDiv.className = 'tree-children';
                node.children.forEach(cls => {
                    childrenDiv.appendChild(createTreeNode(cls, 'class'));
                });
                div.appendChild(childrenDiv);
            } else if (type === 'class') {
                div.className = 'tree-node class';
                div.innerHTML = `
                    <span>${node.name}</span>
                    <button class="add-btn" onclick="addPanel('${node.path}', event, ${node.class_id})">Add</button>
                `;
            }

            return div;
        }

        // Add panel
        async function addPanel(source, event, classId = null) {
            if (event) event.stopPropagation();

            const panelId = panelIdCounter++;
            const panel = {
                id: panelId,
                source: source,
                numSamples: 6,
                selectedClasses: classId !== null ? [classId] : null
            };

            panels.push(panel);
            await renderPanels();
        }

        // Remove panel
        function removePanel(panelId) {
            panels = panels.filter(p => p.id !== panelId);
            renderPanels();
        }

        // Clear all panels
        function clearAllPanels() {
            panels = [];
            renderPanels();
        }

        // Render panels
        async function renderPanels() {
            const container = document.getElementById('panelsContainer');

            if (panels.length === 0) {
                container.innerHTML = `
                    <div class="empty-state">
                        <h3>No panels added</h3>
                        <p>Select a sensor from the left sidebar to get started</p>
                    </div>
                `;
                return;
            }

            container.innerHTML = '';

            for (const panel of panels) {
                const panelDiv = await createPanelElement(panel);
                container.appendChild(panelDiv);
            }
        }

        // Create panel element
        async function createPanelElement(panel) {
            const div = document.createElement('div');
            div.className = 'panel';
            div.id = `panel-${panel.id}`;

            // Fetch data
            try {
                let url = `/api/panel_data?source=${encodeURIComponent(panel.source)}&num_samples=${panel.numSamples}&sampling=${globalSamplingMode}`;
                if (panel.selectedClasses) {
                    url += `&classes=${panel.selectedClasses.join(',')}`;
                }
                const response = await fetch(url);
                const data = await response.json();

                if (data.error) {
                    div.innerHTML = `<div class="loading">Error: ${data.error}</div>`;
                    return div;
                }

                const { metadata, samples } = data;

                // Panel header
                const headerHTML = `
                    <div class="panel-header">
                        <div class="panel-title">${metadata.dataset} / ${metadata.user} / ${metadata.sensor}</div>
                        <div class="panel-controls">
                            <button onclick="refreshPanel(${panel.id})">Refresh</button>
                            <button class="remove-btn" onclick="removePanel(${panel.id})">Remove</button>
                        </div>
                    </div>
                `;

                // Metadata section
                const metadataHTML = `
                    <div class="panel-metadata">
                        <div class="metadata-item">
                            <div class="metadata-label">Dataset</div>
                            <div class="metadata-value">${metadata.dataset}</div>
                        </div>
                        <div class="metadata-item">
                            <div class="metadata-label">User</div>
                            <div class="metadata-value">${metadata.user}</div>
                        </div>
                        <div class="metadata-item">
                            <div class="metadata-label">Sensor</div>
                            <div class="metadata-value">${metadata.sensor}</div>
                        </div>
                        <div class="metadata-item">
                            <div class="metadata-label">Channels</div>
                            <div class="metadata-value">${metadata.num_channels}</div>
                        </div>
                        <div class="metadata-item">
                            <div class="metadata-label">Window Size</div>
                            <div class="metadata-value">${metadata.window_size}</div>
                        </div>
                    </div>
                `;

                // Plots section
                let plotsHTML = '<div class="panel-plots">';
                samples.forEach((sample, idx) => {
                    const plotId = `plot-${panel.id}-${idx}`;
                    plotsHTML += `
                        <div class="plot-item">
                            <div class="plot-header">
                                <strong>${sample.activity}</strong> (Sample #${sample.index})
                            </div>
                            <div id="${plotId}" style="width: 100%; height: 100%;"></div>
                        </div>
                    `;
                });
                plotsHTML += '</div>';

                div.innerHTML = headerHTML + '<div class="panel-content">' + metadataHTML + plotsHTML + '</div>';

                // Render chart with Plotly
                setTimeout(() => {
                    samples.forEach((sample, idx) => {
                        const plotId = `plot-${panel.id}-${idx}`;
                        const isLastPlot = idx === samples.length - 1;
                        createPlot(plotId, sample.data, metadata.window_size, isLastPlot);
                    });
                }, 0);

            } catch (error) {
                console.error('Failed to load panel data:', error);
                div.innerHTML = '<div class="loading">Error loading data</div>';
            }

            return div;
        }

        // Create Plotly chart
        function createPlot(plotId, data, windowSize, showLegend = false) {
            const numChannels = data.length;
            const xValues = Array.from({length: windowSize}, (_, i) => i);

            const traces = [];

            // Generate colors and labels based on number of channels
            const colors = ['#ea4335', '#34a853', '#1a73e8', '#fbbc04', '#9c27b0'];
            const axisLabels = {
                2: ['Ch1', 'Ch2'],
                3: ['X', 'Y', 'Z'],
                4: ['X', 'Y', 'Z', 'W']
            };
            const names = axisLabels[numChannels] || Array.from({length: numChannels}, (_, i) => `Ch${i+1}`);

            // Each channel
            for (let ch = 0; ch < numChannels; ch++) {
                traces.push({
                    x: xValues,
                    y: data[ch],
                    mode: 'lines',
                    name: names[ch],
                    line: { color: colors[ch % colors.length], width: 1.5 }
                });
            }

            // Magnitude (only for 3 or more channels)
            if (numChannels >= 3) {
                let magnitudeSquared = data[0].map((_, i) =>
                    data.slice(0, Math.min(numChannels, 4)).reduce((sum, channel) => sum + channel[i]**2, 0)
                );
                const magnitude = magnitudeSquared.map(val => Math.sqrt(val));

                traces.push({
                    x: xValues,
                    y: magnitude,
                    mode: 'lines',
                    name: 'Magnitude',
                    line: { color: '#9e9e9e', width: 2, dash: 'dot' }
                });
            }

            const layout = {
                autosize: true,
                margin: { t: 20, r: showLegend ? 80 : 20, b: 40, l: 50 },
                xaxis: { title: 'Time (samples)', titlefont: { size: 10 } },
                yaxis: { title: 'Value', titlefont: { size: 10 } },
                showlegend: showLegend,
                legend: {
                    x: 1.02,
                    y: 1,
                    xanchor: 'left',
                    font: { size: 9 },
                    orientation: 'v'
                },
                hovermode: 'closest',
                font: { size: 10 }
            };

            const config = {
                responsive: true,
                displayModeBar: true,
                displaylogo: false
            };

            Plotly.newPlot(plotId, traces, layout, config);
        }


        // Refresh panel
        async function refreshPanel(panelId) {
            await renderPanels();
        }

        // Update sampling mode
        function updateSamplingMode() {
            globalSamplingMode = document.getElementById('samplingMode').value;
            // Re-render all panels
            renderPanels();
        }

        // Initialize
        loadTree();
    </script>
</body>
</html>
"""

INDEX_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HAR Data Visualization</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #f5f5f5;
            min-height: 100vh;
        }
        .header {
            background: white;
            border-bottom: 1px solid #e0e0e0;
            padding: 24px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 24px;
        }
        h1 {
            color: #202124;
            font-size: 28px;
            font-weight: 400;
            margin-bottom: 8px;
        }
        .subtitle {
            color: #5f6368;
            font-size: 14px;
        }
        .content {
            max-width: 1200px;
            margin: 32px auto;
            padding: 0 24px;
        }
        .dataset-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 24px;
        }
        .dataset-card {
            background: white;
            border-radius: 8px;
            padding: 24px;
            border: 1px solid #dadce0;
            transition: box-shadow 0.2s, transform 0.2s;
            cursor: pointer;
        }
        .dataset-card:hover {
            box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
            transform: translateY(-2px);
        }
        .dataset-name {
            font-size: 20px;
            font-weight: 500;
            color: #1a73e8;
            margin-bottom: 16px;
        }
        .dataset-info {
            color: #5f6368;
            line-height: 1.6;
            font-size: 14px;
        }
        .dataset-info div {
            margin: 8px 0;
        }
        .badge {
            display: inline-block;
            background: #e8f0fe;
            color: #1967d2;
            padding: 4px 12px;
            border-radius: 16px;
            font-size: 12px;
            margin: 4px 4px 4px 0;
            font-weight: 500;
        }
        .no-datasets {
            background: white;
            padding: 48px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #dadce0;
        }
        .no-datasets h2 {
            color: #202124;
            font-size: 20px;
            font-weight: 400;
            margin-bottom: 16px;
        }
        .no-datasets p {
            color: #5f6368;
            margin-bottom: 16px;
        }
        .code-block {
            background: #f1f3f4;
            color: #202124;
            padding: 16px;
            border-radius: 4px;
            margin: 16px 0;
            font-family: 'Roboto Mono', monospace;
            text-align: left;
            white-space: pre;
            border: 1px solid #dadce0;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <h1>HAR Data Visualization</h1>
            <div class="subtitle">Select a dataset to explore sensor data</div>
        </div>
    </div>

    <div class="content">
        {% if datasets %}
        <div class="dataset-grid">
            {% for dataset in datasets %}
            <div class="dataset-card" onclick="location.href='/dataset/{{ dataset.name }}'">
                <div class="dataset-name">{{ dataset.name.upper() }}</div>
                <div class="dataset-info">
                    <div><strong>Users:</strong> {{ dataset.num_users }}</div>
                    <div><strong>Activities:</strong> {{ dataset.num_activities }}</div>
                    <div><strong>Sensors:</strong> {{ dataset.num_sensors }}</div>
                    <div style="margin-top: 12px;">
                        {% for modality in dataset.modalities %}
                        <span class="badge">{{ modality }}</span>
                        {% endfor %}
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
        {% else %}
        <div class="no-datasets">
            <h2>No datasets found</h2>
            <p>Please preprocess a dataset first:</p>
            <div class="code-block">python preprocess.py --dataset dsads --download</div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

DATASET_TEMPLATE = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ dataset_name.upper() }} - Visualization</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #f5f5f5;
            min-height: 100vh;
        }
        .header {
            background: white;
            border-bottom: 1px solid #e0e0e0;
            padding: 24px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 24px;
        }
        .back-link {
            display: inline-block;
            color: #1a73e8;
            text-decoration: none;
            margin-bottom: 12px;
            font-size: 14px;
        }
        .back-link:hover {
            text-decoration: underline;
        }
        h1 {
            font-size: 28px;
            font-weight: 400;
            color: #202124;
            margin-bottom: 16px;
        }
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px;
            margin-top: 16px;
        }
        .info-item {
            background: #f1f3f4;
            padding: 12px;
            border-radius: 4px;
            font-size: 14px;
            color: #5f6368;
        }
        .info-item strong {
            color: #202124;
        }
        .content {
            max-width: 1200px;
            margin: 24px auto;
            padding: 0 24px;
        }
        .user-section {
            background: white;
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 24px;
            border: 1px solid #dadce0;
        }
        .user-title {
            font-size: 18px;
            font-weight: 500;
            color: #202124;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid #e0e0e0;
        }
        .sensor-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
            gap: 12px;
        }
        .sensor-btn {
            background: white;
            color: #1a73e8;
            border: 1px solid #dadce0;
            padding: 12px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: background 0.2s, border-color 0.2s, box-shadow 0.2s;
            font-family: 'Roboto', sans-serif;
        }
        .sensor-btn:hover {
            background: #f8f9fa;
            border-color: #1a73e8;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        }
        .sensor-btn:active {
            background: #e8f0fe;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <a href="/" class="back-link">&lt; Back to datasets</a>
            <h1>{{ dataset_name.upper() }}</h1>
            <div class="info-grid">
                <div class="info-item">
                    <strong>Users:</strong> {{ metadata.users|length }}
                </div>
                <div class="info-item">
                    <strong>Activities:</strong> {{ metadata.num_activities }}
                </div>
                <div class="info-item">
                    <strong>Window Size:</strong> {{ metadata.window_size }}
                </div>
                <div class="info-item">
                    <strong>Stride:</strong> {{ metadata.stride }}
                </div>
            </div>
        </div>
    </div>

    <div class="content">
        {% for user_id, user_data in metadata.users.items() %}
        <div class="user-section">
            <div class="user-title">{{ user_id }}</div>
            <div class="sensor-grid">
                {% for sensor_name in user_data.sensor_modalities.keys()|sort %}
                <button class="sensor-btn"
                        onclick="location.href='/visualize/{{ dataset_name }}/{{ user_id }}/{{ sensor_name }}'">
                    {{ sensor_name }}
                </button>
                {% endfor %}
            </div>
        </div>
        {% endfor %}
    </div>
</body>
</html>
"""


def get_available_datasets():
    """Get list of available datasets"""
    if not DATA_DIR.exists():
        return []

    datasets = []
    for dataset_path in DATA_DIR.iterdir():
        if dataset_path.is_dir():
            metadata_path = dataset_path / 'metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                datasets.append({
                    'name': dataset_path.name,
                    'num_users': len(metadata.get('users', {})),
                    'num_activities': metadata.get('num_activities', 'N/A'),
                    'num_sensors': metadata.get('num_sensors', 'N/A'),
                    'modalities': metadata.get('modalities', [])
                })

    return datasets


def load_metadata(dataset_name):
    """Load dataset metadata"""
    metadata_path = DATA_DIR / dataset_name / 'metadata.json'
    if not metadata_path.exists():
        return None

    with open(metadata_path, 'r') as f:
        return json.load(f)


def scan_users_lazy(dataset_name):
    """
    Scan directory to lazily retrieve user list

    Fallback when metadata.json does not have users key
    """
    dataset_path = DATA_DIR / dataset_name
    if not dataset_path.exists():
        return {}

    users = {}
    for user_dir in sorted(dataset_path.glob("USER*")):
        if not user_dir.is_dir():
            continue

        user_id = user_dir.name
        user_data = {"sensor_modalities": {}}

        # Find sensor/modality combinations
        for sensor_dir in user_dir.iterdir():
            if not sensor_dir.is_dir():
                continue

            for modality_dir in sensor_dir.iterdir():
                if not modality_dir.is_dir():
                    continue

                sensor_modality_key = f"{sensor_dir.name}/{modality_dir.name}"

                # Check if labels.npy exists (presence of data)
                labels_path = modality_dir / "labels.npy"
                if labels_path.exists():
                    # Lazy load: do not retrieve shape info (for faster performance)
                    user_data["sensor_modalities"][sensor_modality_key] = {
                        "lazy_loaded": True
                    }

        if user_data["sensor_modalities"]:
            users[user_id] = user_data

    return users


def load_sensor_data(dataset_name, user_id, sensor_name):
    """Load sensor data

    Data structure: USER00001/LeftArm/ACC/X.npy
    sensor_name: "LeftArm/ACC"
    """
    parts = sensor_name.split('/')
    data_path = DATA_DIR / dataset_name / user_id / parts[0] / parts[1]

    X_path = data_path / 'X.npy'
    Y_path = data_path / 'Y.npy'

    if not X_path.exists() or not Y_path.exists():
        return None, None

    X = np.load(X_path)
    Y = np.load(Y_path)
    return X, Y


def create_visualization(X, Y, dataset_name, user_id, sensor_name,
                        selected_classes=None, grid_layout='2x2'):
    """Generate visualization - sample individual windows from each class

    Args:
        X: Sensor data (samples, channels, window_size)
        Y: Labels
        dataset_name: Dataset name
        user_id: User ID
        sensor_name: Sensor name
        selected_classes: List of activity classes to display (None for all)
        grid_layout: Grid layout ('1x1', '2x2', '3x3', etc.)
    """
    num_channels = X.shape[1]
    window_size = X.shape[2]

    # Fixed axis colors (X-axis=red, Y-axis=green, Z-axis=blue)
    AXIS_COLORS = ['#ea4335', '#34a853', '#1a73e8']  # Red, Green, Blue
    AXIS_NAMES = ['X-axis', 'Y-axis', 'Z-axis']

    # Class filtering
    if selected_classes is not None:
        mask = np.isin(Y, selected_classes)
        X = X[mask]
        Y = Y[mask]

    if len(Y) == 0:
        # Return error message if no data available
        return "<html><body><h2>No data available for the selected classes</h2></body></html>"

    # Parse grid layout
    rows, cols = map(int, grid_layout.split('x'))
    total_subplots = rows * cols

    # Sample evenly from each class (adjusted to grid size)
    unique_labels = np.unique(Y)
    samples_per_class = max(1, total_subplots // len(unique_labels))

    # Collect samples
    sampled_data = []
    for label in unique_labels:
        indices = np.where(Y == label)[0]
        n_samples = min(samples_per_class, len(indices))
        selected = np.random.choice(indices, n_samples, replace=False)
        for idx in selected:
            sampled_data.append({
                'index': idx,
                'label': label,
                'data': X[idx]
            })

    # Sort by class
    sampled_data.sort(key=lambda x: x['label'])

    # Adjust to grid size
    sampled_data = sampled_data[:total_subplots]

    # Create subplots
    subplot_titles = []
    for s in sampled_data:
        label = int(s['label'])
        activity = get_activity_name(dataset_name, label)
        subplot_titles.append(f'Sample {s["index"]}: {activity}')

    # Adjust margins (larger grids need larger margins)
    vertical_spacing = max(0.12 / rows, 0.03)
    horizontal_spacing = max(0.10 / cols, 0.03)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing
    )

    # Plot each sample
    for plot_idx, sample in enumerate(sampled_data):
        row = plot_idx // cols + 1
        col = plot_idx % cols + 1

        label = int(sample['label'])
        activity_name = get_activity_name(dataset_name, label)
        data = sample['data']

        for ch in range(min(num_channels, 3)):  # Display up to 3 axes
            x_vals = np.arange(window_size)
            y_vals = data[ch, :]

            # Show legend only in first subplot
            showlegend = (plot_idx == 0)

            axis_name = AXIS_NAMES[ch] if ch < len(AXIS_NAMES) else f'Channel {ch}'
            axis_color = AXIS_COLORS[ch] if ch < len(AXIS_COLORS) else '#5f6368'

            fig.add_trace(
                go.Scatter(
                    x=x_vals, y=y_vals,
                    mode='lines',
                    name=axis_name,
                    line=dict(color=axis_color, width=1.5),
                    legendgroup=f'axis_{ch}',
                    showlegend=showlegend,
                    hovertemplate=f'{axis_name}<br>Time: %{{x}}<br>Value: %{{y:.3f}}<extra></extra>'
                ),
                row=row, col=col
            )

        # Calculate and add magnitude (norm)
        if num_channels >= 3:
            norm = np.sqrt(data[0, :]**2 + data[1, :]**2 + data[2, :]**2)
            fig.add_trace(
                go.Scatter(
                    x=x_vals, y=norm,
                    mode='lines',
                    name='Magnitude',
                    line=dict(color='#202124', width=2, dash='solid'),
                    legendgroup='magnitude',
                    showlegend=(plot_idx == 0),
                    hovertemplate='Magnitude<br>Time: %{x}<br>Value: %{y:.3f}<extra></extra>'
                ),
                row=row, col=col
            )

    # Layout (adjust height considering margins)
    subplot_height = 350
    total_height = subplot_height * rows + 100  # Add title and margins

    fig.update_layout(
        title=dict(
            text=f'{dataset_name.upper()} - {user_id} - {sensor_name}',
            font=dict(size=20, color='#202124')
        ),
        height=total_height,
        hovermode='closest',
        showlegend=True,
        template='plotly_white',
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family='Roboto, sans-serif', color='#5f6368'),
        margin=dict(t=100, b=50)  # Ensure top/bottom margins
    )

    # Axis labels
    for i in range(len(sampled_data)):
        row = i // cols + 1
        col = i % cols + 1
        fig.update_xaxes(title_text="Time (samples)", row=row, col=col,
                        gridcolor='#e0e0e0', linecolor='#dadce0')
        fig.update_yaxes(title_text="Value", row=row, col=col,
                        gridcolor='#e0e0e0', linecolor='#dadce0')

    return fig.to_html(full_html=True, include_plotlyjs='cdn')


@app.route('/')
def index():
    """Top page: new UI"""
    return render_template_string(NEW_UI_TEMPLATE)


@app.route('/api/tree')
def api_tree():
    """Return dataset hierarchy structure for tree view

    Hierarchy: Dataset > User > Position (LeftArm) > Modality (ACC) > Class
    Explores file system using glob
    """
    if not DATA_DIR.exists():
        return jsonify([])

    tree = []

    # Explore Y.npy files using glob
    y_files = sorted(DATA_DIR.glob('*/USER*/*/*/Y.npy'))

    # Group by dataset
    dataset_dict = {}

    for y_path in y_files:
        # Path parsing: data/processed/dsads/USER00001/LeftArm/ACC/Y.npy
        parts = y_path.parts
        if len(parts) < 6:
            continue

        dataset_name = parts[-5]
        user_name = parts[-4]
        position = parts[-3]
        modality = parts[-2]

        # Load Y.npy and get class information
        try:
            Y = np.load(y_path)
            # Process only 1D arrays (labels)
            if Y.ndim == 1:
                unique_classes = sorted(np.unique(Y).astype(int).tolist())
            else:
                # Skip 2D or higher (old PAAL format Y-axis data, etc.)
                continue
        except Exception as e:
            print(f"Error loading {y_path}: {e}")
            continue

        # Create dataset node
        if dataset_name not in dataset_dict:
            dataset_dict[dataset_name] = {}

        # Create user node
        if user_name not in dataset_dict[dataset_name]:
            dataset_dict[dataset_name][user_name] = {}

        # Create position node
        if position not in dataset_dict[dataset_name][user_name]:
            dataset_dict[dataset_name][user_name][position] = {}

        # Create modality node
        dataset_dict[dataset_name][user_name][position][modality] = unique_classes

    # Convert to tree structure
    for dataset_name in sorted(dataset_dict.keys()):
        dataset_node = {
            'name': dataset_name,
            'type': 'dataset',
            'children': []
        }

        for user_name in sorted(dataset_dict[dataset_name].keys()):
            user_node = {
                'name': user_name,
                'type': 'user',
                'children': []
            }

            for position in sorted(dataset_dict[dataset_name][user_name].keys()):
                position_node = {
                    'name': position,
                    'type': 'position',
                    'children': []
                }

                for modality in sorted(dataset_dict[dataset_name][user_name][position].keys()):
                    modality_node = {
                        'name': modality,
                        'type': 'modality',
                        'children': []
                    }

                    unique_classes = dataset_dict[dataset_name][user_name][position][modality]
                    sensor_name = f'{position}/{modality}'

                    for cls in unique_classes:
                        activity_name = get_activity_name(dataset_name, cls)
                        class_node = {
                            'name': f'{activity_name} (Class {cls})',
                            'type': 'class',
                            'path': f'{dataset_name}/{user_name}/{sensor_name}',
                            'class_id': cls
                        }
                        modality_node['children'].append(class_node)

                    position_node['children'].append(modality_node)

                user_node['children'].append(position_node)

            dataset_node['children'].append(user_node)

        tree.append(dataset_node)

    return jsonify(tree)


@app.route('/api/statistics')
def api_statistics():
    """Return statistics for all datasets

    Explores file system using glob
    """
    if not DATA_DIR.exists():
        return jsonify({'error': 'Data directory not found'}), 404

    stats = {
        'total_datasets': 0,
        'total_windows': 0,
        'total_users': 0,
        'datasets': []
    }

    all_users = set()

    # Explore Y.npy files using glob
    y_files = sorted(DATA_DIR.glob('*/USER*/*/*/Y.npy'))

    # Group by dataset
    dataset_stats_dict = {}

    for y_path in y_files:
        # Path parsing: data/processed/dsads/USER00001/LeftArm/ACC/Y.npy
        parts = y_path.parts
        if len(parts) < 6:
            continue

        dataset_name = parts[-5]
        user_name = parts[-4]
        position = parts[-3]
        modality = parts[-2]
        sensor_name = f'{position}/{modality}'

        # Initialize dataset statistics
        if dataset_name not in dataset_stats_dict:
            dataset_stats_dict[dataset_name] = {
                'name': dataset_name,
                'total_windows': 0,
                'users': set(),
                'sensors': set(),
                'classes': {},  # class_id -> {name, total_windows, sensors: {sensor_name: count}}
                'details': []
            }

        dataset_stats_dict[dataset_name]['users'].add(user_name)
        dataset_stats_dict[dataset_name]['sensors'].add(sensor_name)
        all_users.add(user_name)

        # Load Y.npy and get class distribution
        try:
            Y = np.load(y_path)
            # Process only 1D arrays (labels)
            if Y.ndim != 1:
                continue
            unique_classes = np.unique(Y)

            for cls in unique_classes:
                count = int(np.sum(Y == cls))
                class_id = int(cls)
                activity_name = get_activity_name(dataset_name, class_id)

                # Aggregate by class
                if class_id not in dataset_stats_dict[dataset_name]['classes']:
                    dataset_stats_dict[dataset_name]['classes'][class_id] = {
                        'name': activity_name,
                        'total_windows': 0,
                        'sensors': {}
                    }

                dataset_stats_dict[dataset_name]['classes'][class_id]['total_windows'] += count
                dataset_stats_dict[dataset_name]['classes'][class_id]['sensors'][sensor_name] = \
                    dataset_stats_dict[dataset_name]['classes'][class_id]['sensors'].get(sensor_name, 0) + count

                dataset_stats_dict[dataset_name]['details'].append({
                    'user': user_name,
                    'position': position,
                    'modality': modality,
                    'sensor': sensor_name,
                    'class_id': class_id,
                    'activity_name': activity_name,
                    'count': count
                })

                dataset_stats_dict[dataset_name]['total_windows'] += count

        except Exception as e:
            print(f"Error loading {y_path}: {e}")
            continue

    # Format statistics
    for dataset_name in sorted(dataset_stats_dict.keys()):
        dataset_stats = dataset_stats_dict[dataset_name]

        if dataset_stats['details']:
            dataset_stats['num_users'] = len(dataset_stats['users'])
            dataset_stats['num_sensors'] = len(dataset_stats['sensors'])
            dataset_stats['num_classes'] = len(dataset_stats['classes'])

            # Convert set to list (for JSON serialization)
            dataset_stats['sensors'] = sorted(list(dataset_stats['sensors']))
            del dataset_stats['users']  # Set not needed

            # Format class information
            dataset_stats['class_summary'] = []
            for class_id in sorted(dataset_stats['classes'].keys()):
                class_info = dataset_stats['classes'][class_id]
                dataset_stats['class_summary'].append({
                    'class_id': class_id,
                    'name': class_info['name'] or f'Class {class_id}',
                    'total_windows': class_info['total_windows'],
                    'num_sensors': len(class_info['sensors']),
                    'sensors': class_info['sensors']
                })

            del dataset_stats['classes']  # Remove original dictionary format

            stats['datasets'].append(dataset_stats)
            stats['total_windows'] += dataset_stats['total_windows']
            stats['total_datasets'] += 1

    stats['total_users'] = len(all_users)

    return jsonify(stats)


@app.route('/api/panel_data')
def api_panel_data():
    """Return data for panel"""
    source = request.args.get('source', '')  # dataset/user/position/modality
    num_samples = int(request.args.get('num_samples', 5))
    selected_classes_param = request.args.get('classes', None)
    sampling_mode = request.args.get('sampling', 'random')  # 'random' or 'sequential'

    parts = source.split('/')
    if len(parts) != 4:
        return jsonify({'error': f'Invalid source format: expected 4 parts, got {len(parts)}'}), 400

    dataset_name, user_id, position, modality = parts
    sensor_name = f'{position}/{modality}'

    # Load data
    X, Y = load_sensor_data(dataset_name, user_id, sensor_name)
    if X is None or Y is None:
        return jsonify({'error': 'Data not found'}), 404

    # Class filter
    selected_classes = None
    if selected_classes_param:
        try:
            selected_classes = [int(c) for c in selected_classes_param.split(',')]
            mask = np.isin(Y, selected_classes)
            X = X[mask]
            Y = Y[mask]
        except ValueError:
            pass

    if len(Y) == 0:
        return jsonify({'error': 'No data available for selected classes'}), 404

    # Sample evenly from each class
    unique_labels = np.unique(Y)
    samples_per_class = max(1, num_samples // len(unique_labels))

    sampled_data = []
    for label in unique_labels:
        indices = np.where(Y == label)[0]
        n_samples = min(samples_per_class, len(indices))

        if sampling_mode == 'sequential':
            # Get from beginning in order
            selected = indices[:n_samples]
        else:
            # Get randomly
            selected = np.random.choice(indices, n_samples, replace=False)

        for idx in selected:
            sampled_data.append({
                'index': int(idx),
                'label': int(label),
                'activity': get_activity_name(dataset_name, int(label)),
                'data': X[idx].tolist()
            })

    # Sort by class
    sampled_data.sort(key=lambda x: x['label'])
    sampled_data = sampled_data[:num_samples]

    # Metadata
    metadata = {
        'dataset': dataset_name,
        'user': user_id,
        'sensor': sensor_name,
        'num_channels': X.shape[1],
        'window_size': X.shape[2],
        'available_classes': [int(l) for l in np.unique(Y)]
    }

    return jsonify({
        'metadata': metadata,
        'samples': sampled_data
    })


@app.route('/old')
def old_index():
    """Old top page: dataset list"""
    datasets = get_available_datasets()
    return render_template_string(INDEX_TEMPLATE, datasets=datasets)


@app.route('/dataset/<dataset_name>')
def dataset_detail(dataset_name):
    """Dataset detail: user x sensor grid"""
    metadata = load_metadata(dataset_name)
    if metadata is None:
        return f"Dataset '{dataset_name}' not found", 404

    # Lazy load if users key not in metadata.json
    if 'users' not in metadata:
        users = scan_users_lazy(dataset_name)
        metadata['users'] = users

    return render_template_string(DATASET_TEMPLATE,
                                  dataset_name=dataset_name,
                                  metadata=metadata)


def create_comparison_visualization(all_data, selected_classes=None, grid_layout='2x2'):
    """Generate comparison visualization from multiple sources"""
    rows, cols = map(int, grid_layout.split('x'))

    # Fixed axis colors
    AXIS_COLORS = ['#ea4335', '#34a853', '#1a73e8']
    AXIS_NAMES = ['X-axis', 'Y-axis', 'Z-axis']

    # Subplot titles
    subplot_titles = []
    filtered_data = []

    for data_source in all_data[:rows * cols]:
        X, Y = data_source['X'], data_source['Y']

        # Class filtering
        if selected_classes is not None:
            mask = np.isin(Y, selected_classes)
            X = X[mask]
            Y = Y[mask]

        if len(Y) > 0:
            # Randomly select one sample
            idx = np.random.choice(len(Y))
            filtered_data.append({
                'X': X[idx],
                'Y': Y[idx],
                'idx': idx,
                'dataset': data_source['dataset'],
                'user': data_source['user'],
                'sensor': data_source['sensor']
            })

            activity_name = get_activity_name(dataset_name, int(Y[idx]))
            title = f"{data_source['dataset']}/{data_source['user']}<br>{data_source['sensor']}<br>{activity_name}"
            subplot_titles.append(title)

    if not filtered_data:
        return "<html><body><h2>No data available for the selected classes</h2></body></html>"

    # Adjust margins
    vertical_spacing = max(0.15 / rows, 0.05)
    horizontal_spacing = max(0.10 / cols, 0.03)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing
    )

    # Plot each source
    for plot_idx, sample in enumerate(filtered_data):
        row = plot_idx // cols + 1
        col = plot_idx % cols + 1

        data = sample['X']
        num_channels = data.shape[0]
        window_size = data.shape[1]

        # Plot each axis
        for ch in range(min(num_channels, 3)):
            x_vals = np.arange(window_size)
            y_vals = data[ch, :]

            showlegend = (plot_idx == 0)
            axis_name = AXIS_NAMES[ch] if ch < len(AXIS_NAMES) else f'Channel {ch}'
            axis_color = AXIS_COLORS[ch] if ch < len(AXIS_COLORS) else '#5f6368'

            fig.add_trace(
                go.Scatter(
                    x=x_vals, y=y_vals,
                    mode='lines',
                    name=axis_name,
                    line=dict(color=axis_color, width=1.5),
                    legendgroup=f'axis_{ch}',
                    showlegend=showlegend,
                    hovertemplate=f'{axis_name}<br>Time: %{{x}}<br>Value: %{{y:.3f}}<extra></extra>'
                ),
                row=row, col=col
            )

        # Magnitude (norm)
        if num_channels >= 3:
            norm = np.sqrt(data[0, :]**2 + data[1, :]**2 + data[2, :]**2)
            fig.add_trace(
                go.Scatter(
                    x=x_vals, y=norm,
                    mode='lines',
                    name='Magnitude',
                    line=dict(color='#202124', width=2),
                    legendgroup='magnitude',
                    showlegend=(plot_idx == 0),
                    hovertemplate='Magnitude<br>Time: %{x}<br>Value: %{y:.3f}<extra></extra>'
                ),
                row=row, col=col
            )

    # Layout
    subplot_height = 350
    total_height = subplot_height * rows + 150

    fig.update_layout(
        title=dict(
            text='Sensor Data Comparison',
            font=dict(size=20, color='#202124')
        ),
        height=total_height,
        hovermode='closest',
        showlegend=True,
        template='plotly_white',
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family='Roboto, sans-serif', color='#5f6368'),
        margin=dict(t=120, b=50)
    )

    # Axis labels
    for i in range(len(filtered_data)):
        row = i // cols + 1
        col = i % cols + 1
        fig.update_xaxes(title_text="Time (samples)", row=row, col=col,
                        gridcolor='#e0e0e0', linecolor='#dadce0')
        fig.update_yaxes(title_text="Value", row=row, col=col,
                        gridcolor='#e0e0e0', linecolor='#dadce0')

    return fig.to_html(full_html=True, include_plotlyjs='cdn')


def generate_comparison_page(plot_html, all_data, sources, selected_classes, grid_layout):
    """Generate comparison page HTML"""
    # Get available classes from all sources
    all_classes = set()
    for data_source in all_data:
        all_classes.update(np.unique(data_source['Y']).astype(int).tolist())

    unique_classes = sorted(all_classes)

    # Class selection checkbox HTML
    class_checkboxes = []
    for cls in unique_classes:
        activity_name = get_activity_name(dataset_name, cls)
        checked = 'checked' if (selected_classes is None or cls in selected_classes) else ''
        class_checkboxes.append(
            f'<label class="checkbox-label">'
            f'<input type="checkbox" name="class" value="{cls}" {checked}> '
            f'{activity_name}'
            f'</label>'
        )

    class_checkboxes_html = '\n'.join(class_checkboxes)

    # Grid layout options
    grid_options = ['1x1', '2x2', '3x3', '4x4', '2x3', '3x2', '4x3', '5x4']
    grid_select_html = []
    for option in grid_options:
        selected = 'selected' if option == grid_layout else ''
        grid_select_html.append(f'<option value="{option}" {selected}>{option}</option>')

    grid_select_html = '\n'.join(grid_select_html)

    # Sources list
    sources_list_html = []
    for i, source in enumerate(sources, 1):
        sources_list_html.append(
            f'<div class="source-item">{i}. {source["dataset"]}/{source["user"]}/{source["sensor"]}</div>'
        )
    sources_list_html = '\n'.join(sources_list_html)

    # Reconstruct source parameters
    sources_param = ','.join([f'{s["dataset"]}/{s["user"]}/{s["sensor"]}' for s in sources])

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sensor Data Comparison</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #f5f5f5;
            min-height: 100vh;
        }}
        .header {{
            background: white;
            border-bottom: 1px solid #e0e0e0;
            padding: 16px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            position: sticky;
            top: 0;
            z-index: 1000;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 24px;
        }}
        .back-link {{
            display: inline-block;
            color: #1a73e8;
            text-decoration: none;
            margin-bottom: 8px;
            font-size: 14px;
        }}
        .back-link:hover {{ text-decoration: underline; }}
        h1 {{
            font-size: 20px;
            font-weight: 500;
            color: #202124;
            margin-bottom: 12px;
        }}
        .controls {{
            background: white;
            border: 1px solid #dadce0;
            border-radius: 8px;
            padding: 16px;
            margin: 16px 24px;
            max-width: 1400px;
            margin-left: auto;
            margin-right: auto;
        }}
        .control-section {{
            margin-bottom: 16px;
        }}
        .control-section:last-child {{
            margin-bottom: 0;
        }}
        .control-label {{
            font-weight: 500;
            color: #202124;
            margin-bottom: 8px;
            display: block;
            font-size: 14px;
        }}
        .class-filters {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 8px;
        }}
        .checkbox-label {{
            display: flex;
            align-items: center;
            padding: 6px 8px;
            background: #f8f9fa;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            color: #5f6368;
            transition: background 0.2s;
        }}
        .checkbox-label:hover {{
            background: #e8f0fe;
        }}
        .checkbox-label input {{
            margin-right: 8px;
            cursor: pointer;
        }}
        .control-row {{
            display: flex;
            gap: 16px;
            align-items: end;
            flex-wrap: wrap;
        }}
        .control-group {{
            display: flex;
            flex-direction: column;
        }}
        select {{
            padding: 8px 12px;
            border: 1px solid #dadce0;
            border-radius: 4px;
            font-size: 14px;
            font-family: inherit;
            background: white;
            min-width: 120px;
        }}
        .btn {{
            background: #1a73e8;
            color: white;
            border: none;
            padding: 10px 24px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: background 0.2s;
        }}
        .btn:hover {{
            background: #1557b0;
        }}
        .btn-secondary {{
            background: white;
            color: #5f6368;
            border: 1px solid #dadce0;
        }}
        .btn-secondary:hover {{
            background: #f8f9fa;
        }}
        .plot-container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 24px 24px 24px;
        }}
        .legend-info {{
            background: #e8f0fe;
            padding: 12px;
            border-radius: 4px;
            margin-bottom: 12px;
            font-size: 13px;
            color: #1967d2;
        }}
        .legend-info strong {{
            color: #1557b0;
        }}
        .axis-legend {{
            display: flex;
            gap: 16px;
            margin-top: 8px;
        }}
        .axis-item {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .axis-color {{
            width: 20px;
            height: 3px;
            border-radius: 2px;
        }}
        .sources-list {{
            background: #f8f9fa;
            padding: 12px;
            border-radius: 4px;
            margin-top: 12px;
        }}
        .source-item {{
            padding: 4px 0;
            color: #5f6368;
            font-size: 13px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <a href="/" class="back-link">&lt; Back to datasets</a>
            <h1>Sensor Data Comparison</h1>
        </div>
    </div>

    <div class="controls">
        <form id="filterForm" method="get">
            <input type="hidden" name="sources" value="{sources_param}">

            <div class="control-section">
                <label class="control-label">Comparing sources:</label>
                <div class="sources-list">
                    {sources_list_html}
                </div>
            </div>

            <div class="control-section">
                <label class="control-label">Activity class filter:</label>
                <div class="class-filters">
                    {class_checkboxes_html}
                </div>
            </div>

            <div class="control-section">
                <div class="control-row">
                    <div class="control-group">
                        <label class="control-label" for="grid">Grid layout:</label>
                        <select id="grid" name="grid">
                            {grid_select_html}
                        </select>
                    </div>
                    <button type="submit" class="btn">Update</button>
                    <button type="button" class="btn btn-secondary" onclick="selectAllClasses()">Select All</button>
                    <button type="button" class="btn btn-secondary" onclick="deselectAllClasses()">Deselect All</button>
                </div>
            </div>
        </form>
    </div>

    <div class="plot-container">
        <div class="legend-info">
            <strong>Axis colors:</strong>
            <div class="axis-legend">
                <div class="axis-item">
                    <div class="axis-color" style="background: #ea4335;"></div>
                    <span>X-axis</span>
                </div>
                <div class="axis-item">
                    <div class="axis-color" style="background: #34a853;"></div>
                    <span>Y-axis</span>
                </div>
                <div class="axis-item">
                    <div class="axis-color" style="background: #1a73e8;"></div>
                    <span>Z-axis</span>
                </div>
                <div class="axis-item">
                    <div class="axis-color" style="background: #202124;"></div>
                    <span>Magnitude (sqrt(x^2+y^2+z^2))</span>
                </div>
            </div>
        </div>
        {plot_html}
    </div>

    <script>
        function selectAllClasses() {{
            document.querySelectorAll('input[name="class"]').forEach(cb => cb.checked = true);
        }}

        function deselectAllClasses() {{
            document.querySelectorAll('input[name="class"]').forEach(cb => cb.checked = false);
        }}

        document.getElementById('filterForm').addEventListener('submit', function(e) {{
            e.preventDefault();
            const checkedClasses = Array.from(document.querySelectorAll('input[name="class"]:checked'))
                                        .map(cb => cb.value)
                                        .join(',');
            const grid = document.getElementById('grid').value;
            const sources = document.querySelector('input[name="sources"]').value;

            let url = '/compare?sources=' + encodeURIComponent(sources) + '&grid=' + grid;
            if (checkedClasses) {{
                url += '&classes=' + checkedClasses;
            }}

            window.location.href = url;
        }});
    </script>
</body>
</html>
"""


def generate_visualization_page(plot_html, dataset_name, user_id, sensor_name,
                               unique_classes, selected_classes, grid_layout):
    """Generate visualization page HTML"""
    # Class selection checkbox HTML
    class_checkboxes = []
    for cls in unique_classes:
        activity_name = get_activity_name(dataset_name, cls)
        checked = 'checked' if (selected_classes is None or cls in selected_classes) else ''
        class_checkboxes.append(
            f'<label class="checkbox-label">'
            f'<input type="checkbox" name="class" value="{cls}" {checked}> '
            f'{activity_name}'
            f'</label>'
        )

    class_checkboxes_html = '\n'.join(class_checkboxes)

    # Grid layout options
    grid_options = ['1x1', '2x2', '3x3', '4x4', '2x3', '3x2', '4x3', '5x4']
    grid_select_html = []
    for option in grid_options:
        selected = 'selected' if option == grid_layout else ''
        grid_select_html.append(f'<option value="{option}" {selected}>{option}</option>')

    grid_select_html = '\n'.join(grid_select_html)

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{dataset_name.upper()} - {user_id} - {sensor_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #f5f5f5;
            min-height: 100vh;
        }}
        .header {{
            background: white;
            border-bottom: 1px solid #e0e0e0;
            padding: 16px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            position: sticky;
            top: 0;
            z-index: 1000;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 24px;
        }}
        .back-link {{
            display: inline-block;
            color: #1a73e8;
            text-decoration: none;
            margin-bottom: 8px;
            font-size: 14px;
        }}
        .back-link:hover {{ text-decoration: underline; }}
        h1 {{
            font-size: 20px;
            font-weight: 500;
            color: #202124;
            margin-bottom: 12px;
        }}
        .controls {{
            background: white;
            border: 1px solid #dadce0;
            border-radius: 8px;
            padding: 16px;
            margin: 16px 24px;
            max-width: 1400px;
            margin-left: auto;
            margin-right: auto;
        }}
        .control-section {{
            margin-bottom: 16px;
        }}
        .control-section:last-child {{
            margin-bottom: 0;
        }}
        .control-label {{
            font-weight: 500;
            color: #202124;
            margin-bottom: 8px;
            display: block;
            font-size: 14px;
        }}
        .class-filters {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 8px;
        }}
        .checkbox-label {{
            display: flex;
            align-items: center;
            padding: 6px 8px;
            background: #f8f9fa;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            color: #5f6368;
            transition: background 0.2s;
        }}
        .checkbox-label:hover {{
            background: #e8f0fe;
        }}
        .checkbox-label input {{
            margin-right: 8px;
            cursor: pointer;
        }}
        .control-row {{
            display: flex;
            gap: 16px;
            align-items: end;
            flex-wrap: wrap;
        }}
        .control-group {{
            display: flex;
            flex-direction: column;
        }}
        select, input[type="number"] {{
            padding: 8px 12px;
            border: 1px solid #dadce0;
            border-radius: 4px;
            font-size: 14px;
            font-family: inherit;
            background: white;
            min-width: 120px;
        }}
        .btn {{
            background: #1a73e8;
            color: white;
            border: none;
            padding: 10px 24px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: background 0.2s;
        }}
        .btn:hover {{
            background: #1557b0;
        }}
        .btn-secondary {{
            background: white;
            color: #5f6368;
            border: 1px solid #dadce0;
        }}
        .btn-secondary:hover {{
            background: #f8f9fa;
        }}
        .plot-container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 24px 24px 24px;
        }}
        .legend-info {{
            background: #e8f0fe;
            padding: 12px;
            border-radius: 4px;
            margin-bottom: 12px;
            font-size: 13px;
            color: #1967d2;
        }}
        .legend-info strong {{
            color: #1557b0;
        }}
        .axis-legend {{
            display: flex;
            gap: 16px;
            margin-top: 8px;
        }}
        .axis-item {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .axis-color {{
            width: 20px;
            height: 3px;
            border-radius: 2px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <a href="/dataset/{dataset_name}" class="back-link">&lt; Back to {dataset_name.upper()}</a>
            <h1>{dataset_name.upper()} - {user_id} - {sensor_name}</h1>
        </div>
    </div>

    <div class="controls">
        <form id="filterForm" method="get">
            <div class="control-section">
                <label class="control-label">Activity class filter:</label>
                <div class="class-filters">
                    {class_checkboxes_html}
                </div>
            </div>

            <div class="control-section">
                <div class="control-row">
                    <div class="control-group">
                        <label class="control-label" for="grid">Grid layout:</label>
                        <select id="grid" name="grid">
                            {grid_select_html}
                        </select>
                    </div>
                    <button type="submit" class="btn">Update</button>
                    <button type="button" class="btn btn-secondary" onclick="selectAllClasses()">Select All</button>
                    <button type="button" class="btn btn-secondary" onclick="deselectAllClasses()">Deselect All</button>
                </div>
            </div>
        </form>
    </div>

    <div class="plot-container">
        <div class="legend-info">
            <strong>Axis colors:</strong>
            <div class="axis-legend">
                <div class="axis-item">
                    <div class="axis-color" style="background: #ea4335;"></div>
                    <span>X-axis</span>
                </div>
                <div class="axis-item">
                    <div class="axis-color" style="background: #34a853;"></div>
                    <span>Y-axis</span>
                </div>
                <div class="axis-item">
                    <div class="axis-color" style="background: #1a73e8;"></div>
                    <span>Z-axis</span>
                </div>
                <div class="axis-item">
                    <div class="axis-color" style="background: #202124;"></div>
                    <span>Magnitude (sqrt(x^2+y^2+z^2))</span>
                </div>
            </div>
        </div>
        {plot_html}
    </div>

    <script>
        function selectAllClasses() {{
            document.querySelectorAll('input[name="class"]').forEach(cb => cb.checked = true);
        }}

        function deselectAllClasses() {{
            document.querySelectorAll('input[name="class"]').forEach(cb => cb.checked = false);
        }}

        // Send selected classes as comma-separated values on form submission
        document.getElementById('filterForm').addEventListener('submit', function(e) {{
            e.preventDefault();
            const checkedClasses = Array.from(document.querySelectorAll('input[name="class"]:checked'))
                                        .map(cb => cb.value)
                                        .join(',');
            const grid = document.getElementById('grid').value;

            let url = window.location.pathname + '?grid=' + grid;
            if (checkedClasses) {{
                url += '&classes=' + checkedClasses;
            }}

            window.location.href = url;
        }});
    </script>
</body>
</html>
"""


@app.route('/visualize/<dataset_name>/<user_id>/<sensor_name>')
def visualize(dataset_name, user_id, sensor_name):
    """Visualization page"""
    X, Y = load_sensor_data(dataset_name, user_id, sensor_name)

    if X is None or Y is None:
        return f"Data not found for {dataset_name}/{user_id}/{sensor_name}", 404

    # Get settings from query parameters
    grid_layout = request.args.get('grid', '2x2')
    selected_classes_param = request.args.get('classes', None)

    # Parse class filter
    selected_classes = None
    if selected_classes_param:
        try:
            selected_classes = [int(c) for c in selected_classes_param.split(',')]
        except ValueError:
            pass

    # Generate visualization
    plot_html = create_visualization(
        X, Y, dataset_name, user_id, sensor_name,
        selected_classes=selected_classes,
        grid_layout=grid_layout
    )

    # Get available classes
    unique_classes = sorted(np.unique(Y).astype(int).tolist())

    # Generate complete HTML page with UI
    html = generate_visualization_page(
        plot_html, dataset_name, user_id, sensor_name,
        unique_classes, selected_classes, grid_layout
    )

    return html


@app.route('/compare')
def compare():
    """Multiple sensor comparison page"""
    # Get comparison targets from query parameters
    # Format: sources=dataset1/user1/sensor1,dataset2/user2/sensor2
    sources_param = request.args.get('sources', '')
    grid_layout = request.args.get('grid', '2x2')
    selected_classes_param = request.args.get('classes', None)

    if not sources_param:
        return "No sources specified. Use ?sources=dataset/user/sensor,dataset/user/sensor", 400

    # Parse sources
    sources = []
    for source in sources_param.split(','):
        parts = source.strip().split('/')
        if len(parts) == 3:
            sources.append({
                'dataset': parts[0],
                'user': parts[1],
                'sensor': parts[2]
            })

    if not sources:
        return "Invalid sources format", 400

    # Parse class filter
    selected_classes = None
    if selected_classes_param:
        try:
            selected_classes = [int(c) for c in selected_classes_param.split(',')]
        except ValueError:
            pass

    # Parse grid layout
    rows, cols = map(int, grid_layout.split('x'))
    total_subplots = rows * cols

    # Load data from each source
    all_data = []
    for source in sources[:total_subplots]:  # Limit to grid size
        X, Y = load_sensor_data(source['dataset'], source['user'], source['sensor'])
        if X is not None and Y is not None:
            all_data.append({
                'X': X,
                'Y': Y,
                'dataset': source['dataset'],
                'user': source['user'],
                'sensor': source['sensor']
            })

    if not all_data:
        return "No valid data found for the specified sources", 404

    # Generate visualization
    plot_html = create_comparison_visualization(
        all_data, selected_classes, grid_layout
    )

    # Generate complete HTML page with UI
    html = generate_comparison_page(
        plot_html, all_data, sources, selected_classes, grid_layout
    )

    return html


def main():
    parser = argparse.ArgumentParser(description='HAR Data Visualization Web App')
    parser.add_argument('--port', type=int, default=5001, help='Port number (default: 5000)')
    parser.add_argument('--no-debug', action='store_true', help='Disable debug mode (default: enabled)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host address (default: 127.0.0.1)')

    args = parser.parse_args()

    # Debug mode enabled by default (hot reload supported)
    debug_mode = not args.no_debug

    print("=" * 80)
    print("HAR Data Visualization Web App")
    print("=" * 80)
    print(f"  URL: http://{args.host}:{args.port}")
    print(f"  Data directory: {DATA_DIR.absolute()}")
    print(f"  Debug mode: {'ON' if debug_mode else 'OFF'} (Hot reload: {'enabled' if debug_mode else 'disabled'})")
    print(f"  Press Ctrl+C to stop the server")
    print("=" * 80)

    # Start server (debug mode enabled by default)
    app.run(host=args.host, port=args.port, debug=debug_mode, use_reloader=debug_mode)


if __name__ == '__main__':
    main()
