import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
import warnings

class ControlChartPipeline:
    def __init__(self, data, subgroup_col=None, value_col=None, date_col=None, 
                 sample_size_col=None, opportunity_col=None):
        """
        Initialize the control chart pipeline
        
        Parameters:
        data: DataFrame or array-like
        subgroup_col: column name for subgroup identifiers
        value_col: column name for measurement values
        date_col: column name for time sequence
        sample_size_col: column name for sample sizes (for P/NP charts)
        opportunity_col: column name for opportunities/area of inspection (for C/U charts)
        """
        self.data = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
        self.subgroup_col = subgroup_col
        self.value_col = value_col
        self.date_col = date_col
        self.sample_size_col = sample_size_col
        self.opportunity_col = opportunity_col
        self.chart_type = None
        self.control_limits = {}
        self.report = {}
        self.data_quality_issues = []
    
    def validate_data_quality(self):
        """
        Validate data quality before analysis
        Returns list of issues found
        """
        issues = []
        
        # Auto-detect value column if not specified (SMART DETECTION)
        if self.value_col is None:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) == 0:
                issues.append("ERROR: No numeric columns found for analysis")
                return issues
            
            # Skip columns that are likely identifiers/grouping variables
            skip_patterns = ['id', 'subgroup', 'batch', 'sample', 'group', 'lot', 'serial', 'number']
            filtered_cols = [c for c in numeric_cols if not any(pattern in c.lower() for pattern in skip_patterns)]
            
            # If we have columns after filtering, use those
            if filtered_cols:
                numeric_cols = filtered_cols
            
            # Prefer columns with measurement-related names
            priority_names = ['measurement', 'value', 'measure', 'data', 'reading', 'result', 'defect', 'count']
            for priority in priority_names:
                matching = [c for c in numeric_cols if priority in c.lower()]
                if matching:
                    self.value_col = matching[0]
                    break
            else:
                # Fallback to first remaining numeric column
                self.value_col = numeric_cols[0]
        
        # Check if value column exists
        if self.value_col not in self.data.columns:
            issues.append(f"ERROR: Column '{self.value_col}' not found in data. Available columns: {list(self.data.columns)}")
            return issues
        
        values = self.data[self.value_col]
        
        # Check for missing values
        missing_count = values.isna().sum()
        if missing_count > 0:
            missing_rows = values[values.isna()].index.tolist()
            issues.append(f"WARNING: {missing_count} missing values found in rows: {missing_rows[:10]}{'...' if len(missing_rows) > 10 else ''}")
        
        # Check for non-numeric values
        try:
            numeric_values = pd.to_numeric(values, errors='coerce')
            non_numeric = numeric_values.isna().sum() - missing_count
            if non_numeric > 0:
                issues.append(f"ERROR: {non_numeric} non-numeric values found in '{self.value_col}' column")
        except:
            pass
        
        # Check for suspicious values
        values_clean = values.dropna()
        if len(values_clean) > 0:
            # Check for error codes (999, 9999, -999, etc.)
            error_codes = values_clean[values_clean.isin([999, 9999, -999, -9999])]
            if len(error_codes) > 0:
                issues.append(f"WARNING: Potential error codes detected (999, -999) in rows: {error_codes.index.tolist()}")
            
            # Check for unreasonable negative values in count data
            if values_clean.min() < 0:
                negative_rows = values_clean[values_clean < 0].index.tolist()
                issues.append(f"WARNING: Negative values found (rows: {negative_rows}). If measuring counts/dimensions, negative values are impossible")
        
        # Check sample size
        if len(values_clean) < 25:
            issues.append(f"WARNING: Only {len(values_clean)} data points. Minimum 25 recommended for reliable control limits")
        
        # Check if all values are identical
        if len(values_clean) > 0 and values_clean.nunique() == 1:
            issues.append(f"WARNING: All values are identical ({values_clean.iloc[0]}). No variation to analyze!")
        
        # Check if variation is suspiciously high
        if len(values_clean) > 0:
            std = values_clean.std()
            mean = values_clean.mean()
            if mean != 0 and (std / abs(mean)) > 1.0:  # CV > 100%
                issues.append(f"WARNING: Very high variation detected (CV = {std/abs(mean)*100:.1f}%). Could this be measurement error rather than process variation?")
        
        self.data_quality_issues = issues
        return issues
        
    def detect_data_type(self):
        """Detect if data is continuous or attribute"""
        if self.value_col is None:
            # Smart detection - prefer measurement columns, skip identifiers
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found for analysis")
            
            # Skip ID/grouping columns
            skip_patterns = ['id', 'subgroup', 'batch', 'sample', 'group', 'lot', 'serial', 'number']
            filtered_cols = [c for c in numeric_cols if not any(p in c.lower() for p in skip_patterns)]
            if filtered_cols:
                numeric_cols = filtered_cols
            
            # Prefer measurement columns
            priority_names = ['measurement', 'value', 'measure', 'data', 'reading', 'result', 'defect', 'count']
            for priority in priority_names:
                matching = [c for c in numeric_cols if priority in c.lower()]
                if matching:
                    self.value_col = matching[0]
                    break
            else:
                self.value_col = numeric_cols[0]
        
        values = self.data[self.value_col]
        
        # Strong indicators of attribute data
        has_sample_size = self.sample_size_col is not None
        has_opportunity = self.opportunity_col is not None
        is_binary = values.isin([0, 1]).all()
        is_integer = np.all(values == values.astype(int))
        is_non_negative = np.all(values >= 0)
        max_value = values.max()
        
        # Check if data is attribute (count/defect data) or continuous
        if has_sample_size or has_opportunity or is_binary:
            # If sample_size or opportunity columns exist, it's attribute data
            self.data_type = "attribute"
        elif is_integer and is_non_negative and max_value < 50:
            # Small non-negative integers are likely counts (attribute data)
            self.data_type = "attribute"
        else:
            # If data contains decimals/floats, it's continuous (measurements)
            if not is_integer:
                self.data_type = "continuous"
            else:
                # For integers, check unique ratio
                unique_ratio = len(values.unique()) / len(values)
                if unique_ratio > 0.3 and pd.api.types.is_numeric_dtype(values):
                    self.data_type = "continuous"
                else:
                    self.data_type = "attribute"
        
        # Set additional attributes based on data type
        if self.data_type == "continuous":
            # Detect subgroup size for continuous data
            if self.subgroup_col:
                subgroup_sizes = self.data.groupby(self.subgroup_col).size()
                self.subgroup_size = subgroup_sizes.iloc[0]
                if not all(subgroup_sizes == self.subgroup_size):
                    warnings.warn("Variable subgroup sizes detected")
            else:
                self.subgroup_size = 1
        else:
            # For attribute data, determine if defectives or defects
            if is_binary or has_sample_size:
                self.attribute_type = "defectives"  # Binary or proportion data
            else:
                self.attribute_type = "defects"     # Count data
                
        return self.data_type
    
    def select_control_chart(self):
        """Select appropriate control chart based on data characteristics"""
        self.detect_data_type()
        
        if self.data_type == "continuous":
            if self.subgroup_size == 1:
                self.chart_type = "I-MR"
            elif self.subgroup_size <= 9:
                self.chart_type = "Xbar-R"
            else:
                self.chart_type = "Xbar-S"
                
        else:  # attribute data
            if self.attribute_type == "defectives":
                # Check if constant sample size
                if hasattr(self, 'sample_size_col') and self.sample_size_col:
                    sample_sizes = self.data[self.sample_size_col]
                    if sample_sizes.nunique() == 1:
                        self.chart_type = "NP"  # Constant sample size → NP chart
                    else:
                        self.chart_type = "P"   # Variable sample size → P chart
                else:
                    # Assume constant sample size if not specified
                    self.chart_type = "NP"
            else:  # defects
                if hasattr(self, 'opportunity_col') and self.opportunity_col:
                    opportunities = self.data[self.opportunity_col]
                    if opportunities.nunique() == 1:
                        self.chart_type = "C"   # Constant opportunity → C chart
                    else:
                        self.chart_type = "U"   # Variable opportunity → U chart
                else:
                    # Assume constant opportunity if not specified
                    self.chart_type = "C"
                    
        print(f"Selected control chart: {self.chart_type}")
        return self.chart_type
    
    def calculate_control_limits(self):
        """Calculate control limits based on selected chart type"""
        if self.chart_type is None:
            self.select_control_chart()
            
        values = self.data[self.value_col]
        
        if self.chart_type == "I-MR":
            # Individuals and Moving Range chart
            individuals = values
            moving_ranges = np.abs(individuals.diff().dropna())
            
            MR_bar = moving_ranges.mean()
            X_bar = individuals.mean()
            
            self.control_limits = {
                'individuals': {
                    'UCL': X_bar + 2.66 * MR_bar,
                    'LCL': X_bar - 2.66 * MR_bar,
                    'center': X_bar
                },
                'moving_range': {
                    'UCL': 3.27 * MR_bar,
                    'LCL': 0,
                    'center': MR_bar
                }
            }
            
        elif self.chart_type == "Xbar-R":
            # Xbar and R chart
            subgroups = self.data.groupby(self.subgroup_col)[self.value_col]
            subgroup_means = subgroups.mean()
            subgroup_ranges = subgroups.apply(lambda x: x.max() - x.min())
            
            X_bar = subgroup_means.mean()
            R_bar = subgroup_ranges.mean()
            
            # Constants for Xbar-R chart (for subgroup size)
            A2 = {2: 1.88, 3: 1.02, 4: 0.73, 5: 0.58, 6: 0.48, 
                  7: 0.42, 8: 0.37, 9: 0.34}.get(self.subgroup_size, 0.31)
            
            D3 = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0.08, 8: 0.14, 9: 0.18}
            D4 = {2: 3.27, 3: 2.57, 4: 2.28, 5: 2.11, 6: 2.00, 
                  7: 1.92, 8: 1.86, 9: 1.82}
            
            self.control_limits = {
                'xbar': {
                    'UCL': X_bar + A2 * R_bar,
                    'LCL': max(X_bar - A2 * R_bar, 0),
                    'center': X_bar
                },
                'range': {
                    'UCL': D4.get(self.subgroup_size, 1.78) * R_bar,
                    'LCL': D3.get(self.subgroup_size, 0) * R_bar,
                    'center': R_bar
                }
            }
            
        elif self.chart_type == "P":
            # P chart for proportion defective (variable sample size)
            if self.sample_size_col and self.sample_size_col in self.data.columns:
                sample_sizes = self.data[self.sample_size_col]
            else:
                # Assume all samples have same size, use total count
                sample_sizes = pd.Series([len(self.data)] * len(values))
            
            p_bar = values.sum() / sample_sizes.sum()
            
            # Calculate control limits for each point
            self.control_limits = {
                'proportion': {
                    'UCL': [p_bar + 3 * np.sqrt(p_bar * (1 - p_bar) / n) for n in sample_sizes],
                    'LCL': [max(p_bar - 3 * np.sqrt(p_bar * (1 - p_bar) / n), 0) for n in sample_sizes],
                    'center': p_bar
                }
            }
            
        elif self.chart_type == "NP":
            # NP chart for number of defectives (constant sample size)
            if self.sample_size_col and self.sample_size_col in self.data.columns:
                n = self.data[self.sample_size_col].iloc[0]
            else:
                # Assume constant sample size
                n = len(self.data)
            
            np_bar = values.mean()
            p_bar = np_bar / n
            
            self.control_limits = {
                'np': {
                    'UCL': np_bar + 3 * np.sqrt(np_bar * (1 - p_bar)),
                    'LCL': max(np_bar - 3 * np.sqrt(np_bar * (1 - p_bar)), 0),
                    'center': np_bar
                }
            }
            
        elif self.chart_type == "C":
            # C chart for count of defects (constant opportunity)
            c_bar = values.mean()
            
            self.control_limits = {
                'defects': {
                    'UCL': c_bar + 3 * np.sqrt(c_bar),
                    'LCL': max(c_bar - 3 * np.sqrt(c_bar), 0),
                    'center': c_bar
                }
            }
            
        elif self.chart_type == "U":
            # U chart for defects per unit (variable opportunity)
            if self.opportunity_col and self.opportunity_col in self.data.columns:
                opportunities = self.data[self.opportunity_col]
            else:
                # Default to assuming opportunities = 1 for each sample
                opportunities = pd.Series([1] * len(values))
            
            u_bar = values.sum() / opportunities.sum()
            
            # Calculate control limits for each point
            self.control_limits = {
                'defects_per_unit': {
                    'UCL': [u_bar + 3 * np.sqrt(u_bar / n) for n in opportunities],
                    'LCL': [max(u_bar - 3 * np.sqrt(u_bar / n), 0) for n in opportunities],
                    'center': u_bar
                }
            }
            
        return self.control_limits
    
    def generate_plot(self):
        """Generate interactive control chart plot"""
        if not self.control_limits:
            self.calculate_control_limits()
            
        values = self.data[self.value_col].tolist()
        
        # Fix: Convert range to list for Plotly
        if self.date_col and self.date_col in self.data.columns:
            sequence = self.data[self.date_col].tolist()
        else:
            sequence = list(range(1, len(values) + 1))  # Convert range to list
        
        # Create appropriate subplot structure based on chart type
        if self.chart_type == "I-MR":
            fig = make_subplots(
                rows=2, cols=1, 
                subplot_titles=['Individuals Chart', 'Moving Range Chart'],
                vertical_spacing=0.15
            )
            
            # Individuals chart
            fig.add_trace(
                go.Scatter(x=sequence, y=values, mode='lines+markers',
                          name='Individuals', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Moving Range chart
            moving_ranges = np.abs(pd.Series(values).diff().dropna()).tolist()
            mr_sequence = sequence[1:]  # Adjust sequence for moving ranges
            
            fig.add_trace(
                go.Scatter(x=mr_sequence, y=moving_ranges, mode='lines+markers',
                          name='Moving Range', line=dict(color='green')),
                row=2, col=1
            )
            
            # Add control limits for individuals
            limits_ind = self.control_limits['individuals']
            fig.add_trace(
                go.Scatter(x=sequence, y=[limits_ind['UCL']] * len(sequence),
                          mode='lines', name='UCL', line=dict(color='red', dash='dash')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=sequence, y=[limits_ind['LCL']] * len(sequence),
                          mode='lines', name='LCL', line=dict(color='red', dash='dash')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=sequence, y=[limits_ind['center']] * len(sequence),
                          mode='lines', name='Center', line=dict(color='green')),
                row=1, col=1
            )
            
            # Add control limits for moving range
            limits_mr = self.control_limits['moving_range']
            fig.add_trace(
                go.Scatter(x=mr_sequence, y=[limits_mr['UCL']] * len(mr_sequence),
                          mode='lines', name='MR UCL', line=dict(color='red', dash='dash')),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=mr_sequence, y=[limits_mr['LCL']] * len(mr_sequence),
                          mode='lines', name='MR LCL', line=dict(color='red', dash='dash')),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=mr_sequence, y=[limits_mr['center']] * len(mr_sequence),
                          mode='lines', name='MR Center', line=dict(color='green')),
                row=2, col=1
            )
            
        else:
            # For other chart types, use single plot for simplicity
            fig = make_subplots(rows=1, cols=1)
            
            fig.add_trace(
                go.Scatter(x=sequence, y=values, mode='lines+markers',
                          name='Values', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Add control limits based on chart type
            if self.chart_type == "Xbar-R":
                limits = self.control_limits['xbar']
            elif self.chart_type == "P":
                limits = self.control_limits['proportion']
            elif self.chart_type == "NP":
                limits = self.control_limits['np']
            elif self.chart_type == "C":
                limits = self.control_limits['defects']
            elif self.chart_type == "U":
                limits = self.control_limits['defects_per_unit']
            else:
                # Default to simple limits for other chart types
                mean_val = np.mean(values)
                std_val = np.std(values)
                limits = {'UCL': mean_val + 3*std_val, 'LCL': mean_val - 3*std_val, 'center': mean_val}
            
            # Handle variable control limits for P chart
            if isinstance(limits['UCL'], list):
                fig.add_trace(
                    go.Scatter(x=sequence, y=limits['UCL'], mode='lines',
                              name='UCL', line=dict(color='red', dash='dash')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=sequence, y=limits['LCL'], mode='lines',
                              name='LCL', line=dict(color='red', dash='dash')),
                    row=1, col=1
                )
            else:
                fig.add_trace(
                    go.Scatter(x=sequence, y=[limits['UCL']] * len(sequence), mode='lines',
                              name='UCL', line=dict(color='red', dash='dash')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=sequence, y=[limits['LCL']] * len(sequence), mode='lines',
                              name='LCL', line=dict(color='red', dash='dash')),
                    row=1, col=1
                )
            
            fig.add_trace(
                go.Scatter(x=sequence, y=[limits['center']] * len(sequence), mode='lines',
                          name='Center', line=dict(color='green')),
                row=1, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=600, 
            title_text=f"Control Chart Analysis - {self.chart_type}",
            showlegend=True
        )
        
        return fig
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        if not self.control_limits:
            self.calculate_control_limits()
            
        values = self.data[self.value_col]
        
        # Basic statistics
        report = {
            'chart_type': self.chart_type,
            'data_type': self.data_type,
            'sample_size': len(values),
            'mean': float(values.mean()),
            'std_dev': float(values.std()),
            'control_limits': self.control_limits,
            'out_of_control_points': self._detect_out_of_control()
        }
        
        self.report = report
        return report
    
    def _detect_out_of_control(self):
        """Detect points outside control limits"""
        values = self.data[self.value_col].tolist()
        out_of_control = []
        
        if self.chart_type == "I-MR":
            limits = self.control_limits['individuals']
            for i, val in enumerate(values):
                if val > limits['UCL'] or val < limits['LCL']:
                    out_of_control.append({'point': i+1, 'value': val, 'reason': 'Outside control limits'})
                    
        elif self.chart_type == "Xbar-R":
            subgroups = self.data.groupby(self.subgroup_col)[self.value_col]
            subgroup_means = subgroups.mean()
            limits = self.control_limits['xbar']
            
            for i, (subgroup, mean_val) in enumerate(subgroup_means.items()):
                if mean_val > limits['UCL'] or mean_val < limits['LCL']:
                    out_of_control.append({'point': subgroup, 'value': float(mean_val), 'reason': 'Outside control limits'})
                    
        elif self.chart_type in ["P", "U"]:
            # Variable control limits - check each point against its own limits
            limits_key = 'proportion' if self.chart_type == "P" else 'defects_per_unit'
            limits = self.control_limits[limits_key]
            
            for i, val in enumerate(values):
                ucl = limits['UCL'][i] if isinstance(limits['UCL'], list) else limits['UCL']
                lcl = limits['LCL'][i] if isinstance(limits['LCL'], list) else limits['LCL']
                if val > ucl or val < lcl:
                    out_of_control.append({'point': i+1, 'value': val, 'reason': 'Outside control limits'})
                    
        elif self.chart_type in ["NP", "C"]:
            # Constant control limits
            limits_key = 'np' if self.chart_type == "NP" else 'defects'
            limits = self.control_limits[limits_key]
            
            for i, val in enumerate(values):
                if val > limits['UCL'] or val < limits['LCL']:
                    out_of_control.append({'point': i+1, 'value': val, 'reason': 'Outside control limits'})
        
        return out_of_control
    
    def save_report(self, filename="control_chart_report.html"):
        """Save interactive report as HTML file"""
        import plotly.io as pio
        
        fig = self.generate_plot()
        report = self.generate_report()
        
        # Create comprehensive HTML report
        html_content = f"""
        <html>
        <head>
            <title>Control Chart Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .section {{ margin-bottom: 30px; }}
                .stats-table {{ border-collapse: collapse; width: 100%; }}
                .stats-table th, .stats-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .stats-table th {{ background-color: #f2f2f2; }}
                .ok {{ color: green; }}
                .warning {{ color: orange; }}
                .error {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Control Chart Analysis Report</h1>
            
            <div class="section">
                <h2>Summary</h2>
                <table class="stats-table">
                    <tr><th>Chart Type</th><td>{report['chart_type']}</td></tr>
                    <tr><th>Data Type</th><td>{report['data_type']}</td></tr>
                    <tr><th>Sample Size</th><td>{report['sample_size']}</td></tr>
                    <tr><th>Mean</th><td>{report['mean']:.4f}</td></tr>
                    <tr><th>Standard Deviation</th><td>{report['std_dev']:.4f}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Control Chart</h2>
                {pio.to_html(fig, include_plotlyjs='cdn')}
            </div>
            
            <div class="section">
                <h2>Control Limits</h2>
                <pre>{str(report['control_limits'])}</pre>
            </div>
            
            <div class="section">
                <h2>Out of Control Points</h2>
                <p>Number of out-of-control points: {len(report['out_of_control_points'])}</p>
                {"".join([f'<p class="{"error" if report["out_of_control_points"] else "ok"}">Point {p["point"]}: Value {p["value"]:.4f} - {p["reason"]}</p>' for p in report['out_of_control_points']]) if report['out_of_control_points'] else '<p class="ok">No out-of-control points detected</p>'}
            </div>
        </body>
        </html>
        """
        
        with open(filename, 'w') as f:
            f.write(html_content)
        
        print(f"Report saved as: {filename}")
        return filename

