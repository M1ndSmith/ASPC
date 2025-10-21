"""
MSA (Measurement System Analysis) Pipeline
Comprehensive analysis of measurement system capability
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings


class MSAPipeline:
    def __init__(self, data, part_col=None, operator_col=None, measurement_col=None, 
                 trial_col=None, reference_col=None, date_col=None):
        """
        Initialize the MSA pipeline
        
        Parameters:
        data: DataFrame containing measurement data
        part_col: Column name for part/sample identifiers
        operator_col: Column name for operator/appraiser identifiers
        measurement_col: Column name for measurement values
        trial_col: Column name for trial/repeat number
        reference_col: Column name for reference/standard values (for bias/linearity)
        date_col: Column name for date/time (for stability)
        """
        self.data = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
        self.part_col = part_col
        self.operator_col = operator_col
        self.measurement_col = measurement_col
        self.trial_col = trial_col
        self.reference_col = reference_col
        self.date_col = date_col
        self.study_type = None
        self.results = {}
        self.data_quality_issues = []
    
    def validate_data_quality(self, study_type=None):
        """
        Validate data quality for MSA studies
        Returns list of issues found
        """
        issues = []
        
        # Auto-detect measurement column (SMART DETECTION)
        if self.measurement_col is None:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) == 0:
                issues.append("ERROR: No numeric columns found for measurements")
                return issues
            
            # Skip columns that are likely identifiers/grouping variables
            skip_patterns = ['id', 'part', 'operator', 'trial', 'batch', 'sample', 'group', 'appraiser']
            filtered_cols = [c for c in numeric_cols if not any(pattern in c.lower() for pattern in skip_patterns)]
            
            # If we have columns after filtering, use those
            if filtered_cols:
                numeric_cols = filtered_cols
            
            # Prefer columns with measurement-related names
            priority_names = ['measurement', 'value', 'measure', 'reading', 'result', 'data']
            for priority in priority_names:
                matching = [c for c in numeric_cols if priority in c.lower()]
                if matching:
                    self.measurement_col = matching[0]
                    break
            else:
                # Fallback to first remaining numeric column
                self.measurement_col = numeric_cols[0]
        
        # Check measurement column exists
        if self.measurement_col not in self.data.columns:
            issues.append(f"ERROR: Measurement column '{self.measurement_col}' not found. Available: {list(self.data.columns)}")
            return issues
        
        values = self.data[self.measurement_col]
        
        # Check for missing values
        missing_count = values.isna().sum()
        if missing_count > 0:
            issues.append(f"ERROR: {missing_count} missing measurements. MSA requires complete data (no missing values)")
        
        # Gage R&R specific checks
        if study_type == "Gage R&R" or (self.part_col and self.operator_col):
            if self.part_col not in self.data.columns:
                issues.append(f"ERROR: Part column '{self.part_col}' not found")
            if self.operator_col not in self.data.columns:
                issues.append(f"ERROR: Operator column '{self.operator_col}' not found")
            
            if self.part_col in self.data.columns and self.operator_col in self.data.columns:
                # Check balanced design
                n_parts = self.data[self.part_col].nunique()
                n_operators = self.data[self.operator_col].nunique()
                
                # Check each part-operator combination
                combo_counts = self.data.groupby([self.part_col, self.operator_col]).size()
                if combo_counts.nunique() > 1:
                    issues.append(f"WARNING: Unbalanced design detected. Some part-operator combinations have different number of trials")
                
                if n_parts < 10:
                    issues.append(f"WARNING: Only {n_parts} parts. Gage R&R typically requires 10+ parts for reliable results")
                
                if n_operators < 2:
                    issues.append(f"WARNING: Only {n_operators} operator. Gage R&R requires 2+ operators to assess reproducibility")
        
        # Bias/Linearity specific checks
        if self.reference_col and self.reference_col in self.data.columns:
            ref_missing = self.data[self.reference_col].isna().sum()
            if ref_missing > 0:
                issues.append(f"ERROR: {ref_missing} missing reference values. Bias/Linearity require reference values for all measurements")
        
        # Check for reasonable values
        values_clean = values.dropna()
        if len(values_clean) > 0 and values_clean.std() == 0:
            issues.append("ERROR: All measurement values are identical. Cannot perform MSA analysis")
        
        self.data_quality_issues = issues
        return issues
        
    def detect_study_type(self):
        """Detect which type of MSA study based on available columns"""
        has_part = self.part_col is not None and self.part_col in self.data.columns
        has_operator = self.operator_col is not None and self.operator_col in self.data.columns
        has_reference = self.reference_col is not None and self.reference_col in self.data.columns
        has_date = self.date_col is not None and self.date_col in self.data.columns
        has_trial = self.trial_col is not None and self.trial_col in self.data.columns
        
        # Determine study type based on data structure
        if has_part and has_operator and has_trial:
            self.study_type = "Gage R&R"
        elif has_reference and not has_operator:
            # Check if multiple reference values (linearity) or single (bias)
            if has_reference:
                n_references = self.data[self.reference_col].nunique()
                if n_references > 1:
                    self.study_type = "Linearity"
                else:
                    self.study_type = "Bias"
        elif has_date and not has_operator:
            self.study_type = "Stability"
        else:
            # Default to Gage R&R if unclear
            self.study_type = "Gage R&R"
            
        print(f"Detected MSA study type: {self.study_type}")
        return self.study_type
    
    def run_gage_rr(self, tolerance=None, method="anova"):
        """
        Perform Gage R&R study using ANOVA or Range method
        
        Parameters:
        tolerance: Process tolerance (for %Tolerance calculation)
        method: "anova" or "range"
        """
        if method == "anova":
            return self._gage_rr_anova(tolerance)
        else:
            return self._gage_rr_range(tolerance)
    
    def _gage_rr_anova(self, tolerance=None):
        """Gage R&R using ANOVA method (more accurate)"""
        # Prepare data
        parts = self.data[self.part_col].unique()
        operators = self.data[self.operator_col].unique()
        measurements = self.data[self.measurement_col].values
        
        n_parts = len(parts)
        n_operators = len(operators)
        n_trials = len(self.data) // (n_parts * n_operators)
        
        # Calculate means
        grand_mean = measurements.mean()
        part_means = self.data.groupby(self.part_col)[self.measurement_col].mean()
        operator_means = self.data.groupby(self.operator_col)[self.measurement_col].mean()
        
        # ANOVA calculations
        # Total Sum of Squares
        SS_total = np.sum((measurements - grand_mean) ** 2)
        
        # Part Sum of Squares
        SS_part = n_operators * n_trials * np.sum((part_means - grand_mean) ** 2)
        
        # Operator Sum of Squares
        SS_operator = n_parts * n_trials * np.sum((operator_means - grand_mean) ** 2)
        
        # Interaction (Part x Operator)
        interaction_means = self.data.groupby([self.part_col, self.operator_col])[self.measurement_col].mean()
        SS_interaction = n_trials * np.sum((interaction_means - grand_mean) ** 2) - SS_part - SS_operator
        
        # Equipment/Repeatability
        SS_equipment = SS_total - SS_part - SS_operator - SS_interaction
        
        # Degrees of freedom
        df_part = n_parts - 1
        df_operator = n_operators - 1
        df_interaction = df_part * df_operator
        df_equipment = n_parts * n_operators * (n_trials - 1)
        df_total = len(measurements) - 1
        
        # Mean Squares
        MS_part = SS_part / df_part if df_part > 0 else 0
        MS_operator = SS_operator / df_operator if df_operator > 0 else 0
        MS_interaction = SS_interaction / df_interaction if df_interaction > 0 else 0
        MS_equipment = SS_equipment / df_equipment if df_equipment > 0 else 0
        
        # Variance components
        var_equipment = MS_equipment  # Repeatability
        var_reproducibility = max((MS_operator - MS_interaction) / (n_parts * n_trials), 0)
        var_interaction = max((MS_interaction - MS_equipment) / n_trials, 0)
        var_part = max((MS_part - MS_interaction) / (n_operators * n_trials), 0)
        
        # Total Gage R&R
        var_repeatability = var_equipment
        var_reproducibility_total = var_reproducibility + var_interaction
        var_gage_rr = var_repeatability + var_reproducibility_total
        var_total = var_gage_rr + var_part
        
        # Standard deviations
        std_repeatability = np.sqrt(var_repeatability)
        std_reproducibility = np.sqrt(var_reproducibility_total)
        std_gage_rr = np.sqrt(var_gage_rr)
        std_part = np.sqrt(var_part)
        std_total = np.sqrt(var_total)
        
        # Study variation (6 sigma)
        sv_repeatability = 6 * std_repeatability
        sv_reproducibility = 6 * std_reproducibility
        sv_gage_rr = 6 * std_gage_rr
        sv_part = 6 * std_part
        sv_total = 6 * std_total
        
        # Percent contribution
        pct_repeatability = (var_repeatability / var_total * 100) if var_total > 0 else 0
        pct_reproducibility = (var_reproducibility_total / var_total * 100) if var_total > 0 else 0
        pct_gage_rr = (var_gage_rr / var_total * 100) if var_total > 0 else 0
        pct_part = (var_part / var_total * 100) if var_total > 0 else 0
        
        # Percent study variation (%SV)
        pct_sv_repeatability = (sv_repeatability / sv_total * 100) if sv_total > 0 else 0
        pct_sv_reproducibility = (sv_reproducibility / sv_total * 100) if sv_total > 0 else 0
        pct_sv_gage_rr = (sv_gage_rr / sv_total * 100) if sv_total > 0 else 0
        
        # Percent tolerance
        if tolerance:
            pct_tol_repeatability = (sv_repeatability / tolerance * 100)
            pct_tol_reproducibility = (sv_reproducibility / tolerance * 100)
            pct_tol_gage_rr = (sv_gage_rr / tolerance * 100)
        else:
            pct_tol_repeatability = None
            pct_tol_reproducibility = None
            pct_tol_gage_rr = None
        
        # Number of distinct categories (NDC)
        ndc = int(np.floor(np.sqrt(2) * std_part / std_gage_rr)) if std_gage_rr > 0 else 0
        
        # Interpretation
        if pct_sv_gage_rr < 10:
            acceptability = "Excellent"
        elif pct_sv_gage_rr < 30:
            acceptability = "Acceptable"
        else:
            acceptability = "Unacceptable"
        
        self.results = {
            "study_type": "Gage R&R (ANOVA)",
            "n_parts": n_parts,
            "n_operators": n_operators,
            "n_trials": n_trials,
            "variance_components": {
                "repeatability": float(var_repeatability),
                "reproducibility": float(var_reproducibility_total),
                "gage_rr": float(var_gage_rr),
                "part_to_part": float(var_part),
                "total_variation": float(var_total)
            },
            "standard_deviations": {
                "repeatability": float(std_repeatability),
                "reproducibility": float(std_reproducibility),
                "gage_rr": float(std_gage_rr),
                "part_to_part": float(std_part),
                "total": float(std_total)
            },
            "study_variation": {
                "repeatability": float(sv_repeatability),
                "reproducibility": float(sv_reproducibility),
                "gage_rr": float(sv_gage_rr),
                "part_to_part": float(sv_part),
                "total": float(sv_total)
            },
            "percent_contribution": {
                "repeatability": float(pct_repeatability),
                "reproducibility": float(pct_reproducibility),
                "gage_rr": float(pct_gage_rr),
                "part_to_part": float(pct_part)
            },
            "percent_study_variation": {
                "repeatability": float(pct_sv_repeatability),
                "reproducibility": float(pct_sv_reproducibility),
                "gage_rr": float(pct_sv_gage_rr)
            },
            "percent_tolerance": {
                "repeatability": float(pct_tol_repeatability) if pct_tol_repeatability else None,
                "reproducibility": float(pct_tol_reproducibility) if pct_tol_reproducibility else None,
                "gage_rr": float(pct_tol_gage_rr) if pct_tol_gage_rr else None
            } if tolerance else None,
            "ndc": ndc,
            "acceptability": acceptability,
            "interpretation": {
                "gage_rr": f"{acceptability} - {pct_sv_gage_rr:.1f}% of total variation",
                "ndc": f"{'Adequate' if ndc >= 5 else 'Inadequate'} - {ndc} distinct categories"
            }
        }
        
        return self.results
    
    def _gage_rr_range(self, tolerance=None):
        """Gage R&R using Range method (simpler, less accurate)"""
        # Calculate ranges for each part by each operator
        ranges_by_part = []
        for part in self.data[self.part_col].unique():
            part_data = self.data[self.data[self.part_col] == part]
            for operator in self.data[self.operator_col].unique():
                operator_data = part_data[part_data[self.operator_col] == operator]
                if len(operator_data) > 1:
                    r = operator_data[self.measurement_col].max() - operator_data[self.measurement_col].min()
                    ranges_by_part.append(r)
        
        R_bar = np.mean(ranges_by_part)
        
        # Constants for range method (d2 values)
        n_trials = len(self.data) // (len(self.data[self.part_col].unique()) * len(self.data[self.operator_col].unique()))
        d2_values = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326}
        d2 = d2_values.get(n_trials, 2.326)
        
        # Equipment Variation (Repeatability)
        EV = R_bar / d2
        
        # Appraiser Variation (Reproducibility) - simplified
        operator_avgs = self.data.groupby(self.operator_col)[self.measurement_col].mean()
        R_operators = operator_avgs.max() - operator_avgs.min()
        n_parts = len(self.data[self.part_col].unique())
        
        AV = np.sqrt(max((R_operators / d2) ** 2 - (EV ** 2 / (n_parts * n_trials)), 0))
        
        # Part Variation
        part_avgs = self.data.groupby(self.part_col)[self.measurement_col].mean()
        R_parts = part_avgs.max() - part_avgs.min()
        PV = R_parts / d2
        
        # Total Gage R&R
        GRR = np.sqrt(EV ** 2 + AV ** 2)
        
        # Total Variation
        TV = np.sqrt(GRR ** 2 + PV ** 2)
        
        # Percentages
        pct_ev = (EV / TV * 100) if TV > 0 else 0
        pct_av = (AV / TV * 100) if TV > 0 else 0
        pct_grr = (GRR / TV * 100) if TV > 0 else 0
        pct_pv = (PV / TV * 100) if TV > 0 else 0
        
        self.results = {
            "study_type": "Gage R&R (Range)",
            "equipment_variation": float(EV),
            "appraiser_variation": float(AV),
            "gage_rr": float(GRR),
            "part_variation": float(PV),
            "total_variation": float(TV),
            "percent_study_variation": {
                "repeatability": float(pct_ev),
                "reproducibility": float(pct_av),
                "gage_rr": float(pct_grr),
                "part_to_part": float(pct_pv)
            }
        }
        
        return self.results
    
    def run_bias_study(self):
        """Analyze measurement bias against reference values"""
        if self.reference_col not in self.data.columns:
            raise ValueError("Reference column required for bias study")
        
        measured = self.data[self.measurement_col]
        reference = self.data[self.reference_col]
        
        # Calculate bias
        bias_values = measured - reference
        mean_bias = bias_values.mean()
        std_bias = bias_values.std()
        
        # T-test for bias significance
        t_stat, p_value = stats.ttest_1samp(bias_values, 0)
        is_significant = p_value < 0.05
        
        # Percent bias
        mean_reference = reference.mean()
        pct_bias = (mean_bias / mean_reference * 100) if mean_reference != 0 else 0
        
        self.results = {
            "study_type": "Bias",
            "mean_bias": float(mean_bias),
            "std_bias": float(std_bias),
            "percent_bias": float(pct_bias),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "is_significant": bool(is_significant),
            "interpretation": "Significant bias detected" if is_significant else "No significant bias",
            "n_measurements": len(measured)
        }
        
        return self.results
    
    def run_linearity_study(self):
        """Analyze measurement linearity across reference range"""
        if self.reference_col not in self.data.columns:
            raise ValueError("Reference column required for linearity study")
        
        measured = self.data[self.measurement_col]
        reference = self.data[self.reference_col]
        
        # Calculate bias at each reference level
        bias_by_ref = self.data.groupby(self.reference_col).apply(
            lambda x: (x[self.measurement_col] - x[self.reference_col]).mean()
        )
        
        # Linear regression: Bias vs Reference
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            reference, measured - reference
        )
        
        # Is linearity acceptable? (slope should be close to 0)
        is_linear = abs(slope) < 0.1  # Threshold can be adjusted
        
        self.results = {
            "study_type": "Linearity",
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_value ** 2),
            "p_value": float(p_value),
            "std_error": float(std_err),
            "is_linear": bool(is_linear),
            "interpretation": "Acceptable linearity" if is_linear else "Linearity issue detected",
            "bias_by_reference": bias_by_ref.to_dict()
        }
        
        return self.results
    
    def run_stability_study(self):
        """Analyze measurement stability over time"""
        if self.date_col not in self.data.columns:
            raise ValueError("Date column required for stability study")
        
        # Sort by date
        data_sorted = self.data.sort_values(self.date_col)
        measurements = data_sorted[self.measurement_col]
        
        # Calculate statistics
        mean = measurements.mean()
        std = measurements.std()
        
        # Control limits (like I-chart)
        moving_ranges = np.abs(measurements.diff().dropna())
        mr_bar = moving_ranges.mean()
        ucl = mean + 2.66 * mr_bar
        lcl = mean - 2.66 * mr_bar
        
        # Check for out of control points
        out_of_control = []
        for i, val in enumerate(measurements):
            if val > ucl or val < lcl:
                out_of_control.append(i)
        
        # Trend test (Mann-Kendall)
        n = len(measurements)
        s = 0
        for i in range(n-1):
            for j in range(i+1, n):
                s += np.sign(measurements.iloc[j] - measurements.iloc[i])
        
        # Simplified trend detection
        has_trend = abs(s) > (n * (n - 1) / 4)  # Simplified threshold
        
        self.results = {
            "study_type": "Stability",
            "mean": float(mean),
            "std_dev": float(std),
            "ucl": float(ucl),
            "lcl": float(lcl),
            "out_of_control_points": len(out_of_control),
            "has_trend": bool(has_trend),
            "interpretation": "Unstable" if (len(out_of_control) > 0 or has_trend) else "Stable",
            "n_measurements": len(measurements)
        }
        
        return self.results
    
    def generate_plot(self):
        """Generate appropriate plot based on study type"""
        if self.study_type == "Gage R&R":
            return self._plot_gage_rr()
        elif self.study_type == "Bias":
            return self._plot_bias()
        elif self.study_type == "Linearity":
            return self._plot_linearity()
        elif self.study_type == "Stability":
            return self._plot_stability()
    
    def _plot_gage_rr(self):
        """Generate Gage R&R plots"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Measurement by Part', 'Measurement by Operator',
                          'Variance Components', 'R Chart by Operator'],
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Plot 1: Measurements by Part
        for operator in self.data[self.operator_col].unique():
            op_data = self.data[self.data[self.operator_col] == operator]
            fig.add_trace(
                go.Scatter(x=op_data[self.part_col], y=op_data[self.measurement_col],
                          mode='markers+lines', name=f'Operator {operator}'),
                row=1, col=1
            )
        
        # Plot 2: Measurements by Operator
        for part in self.data[self.part_col].unique():
            part_data = self.data[self.data[self.part_col] == part]
            fig.add_trace(
                go.Scatter(x=part_data[self.operator_col], y=part_data[self.measurement_col],
                          mode='markers', name=f'Part {part}', showlegend=False),
                row=1, col=2
            )
        
        # Plot 3: Variance Components
        if 'variance_components' in self.results:
            components = self.results['variance_components']
            fig.add_trace(
                go.Bar(x=['Repeatability', 'Reproducibility', 'Part-to-Part'],
                      y=[components['repeatability'], components['reproducibility'], 
                         components['part_to_part']]),
                row=2, col=1
            )
        
        # Plot 4: Range chart by operator
        for operator in self.data[self.operator_col].unique():
            op_data = self.data[self.data[self.operator_col] == operator]
            ranges = op_data.groupby(self.part_col)[self.measurement_col].apply(
                lambda x: x.max() - x.min() if len(x) > 1 else 0
            )
            fig.add_trace(
                go.Scatter(x=list(range(len(ranges))), y=ranges,
                          mode='markers+lines', name=f'Op {operator}', showlegend=False),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Gage R&R Analysis", showlegend=True)
        return fig
    
    def _plot_bias(self):
        """Generate bias study plots"""
        measured = self.data[self.measurement_col]
        reference = self.data[self.reference_col]
        bias = measured - reference
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Measured vs Reference', 'Bias Distribution']
        )
        
        # Scatter plot
        fig.add_trace(
            go.Scatter(x=reference, y=measured, mode='markers', name='Measurements'),
            row=1, col=1
        )
        # Ideal line
        min_val, max_val = reference.min(), reference.max()
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                      mode='lines', name='Ideal', line=dict(dash='dash')),
            row=1, col=1
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=bias, name='Bias'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, title_text="Bias Study")
        return fig
    
    def _plot_linearity(self):
        """Generate linearity study plots"""
        reference = self.data[self.reference_col]
        measured = self.data[self.measurement_col]
        bias = measured - reference
        
        fig = go.Figure()
        
        # Scatter plot of bias vs reference
        fig.add_trace(go.Scatter(x=reference, y=bias, mode='markers', name='Bias'))
        
        # Regression line
        slope = self.results['slope']
        intercept = self.results['intercept']
        x_range = np.array([reference.min(), reference.max()])
        y_line = slope * x_range + intercept
        fig.add_trace(go.Scatter(x=x_range, y=y_line, mode='lines', 
                                name='Regression Line', line=dict(color='red')))
        
        # Zero line
        fig.add_trace(go.Scatter(x=x_range, y=[0, 0], mode='lines',
                                name='Zero Bias', line=dict(dash='dash', color='green')))
        
        fig.update_layout(title="Linearity Study", xaxis_title="Reference Value",
                         yaxis_title="Bias", height=500)
        return fig
    
    def _plot_stability(self):
        """Generate stability study plots"""
        data_sorted = self.data.sort_values(self.date_col)
        
        fig = go.Figure()
        
        # Measurements over time
        fig.add_trace(go.Scatter(x=data_sorted[self.date_col], 
                                y=data_sorted[self.measurement_col],
                                mode='markers+lines', name='Measurements'))
        
        # Control limits
        mean = self.results['mean']
        ucl = self.results['ucl']
        lcl = self.results['lcl']
        
        fig.add_trace(go.Scatter(x=data_sorted[self.date_col], y=[mean] * len(data_sorted),
                                mode='lines', name='Mean', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=data_sorted[self.date_col], y=[ucl] * len(data_sorted),
                                mode='lines', name='UCL', line=dict(color='red', dash='dash')))
        fig.add_trace(go.Scatter(x=data_sorted[self.date_col], y=[lcl] * len(data_sorted),
                                mode='lines', name='LCL', line=dict(color='red', dash='dash')))
        
        fig.update_layout(title="Stability Study", xaxis_title="Date/Time",
                         yaxis_title="Measurement", height=500)
        return fig
    
    def generate_report(self):
        """Generate comprehensive MSA report"""
        if not self.results:
            self.detect_study_type()
            if self.study_type == "Gage R&R":
                self.run_gage_rr()
            elif self.study_type == "Bias":
                self.run_bias_study()
            elif self.study_type == "Linearity":
                self.run_linearity_study()
            elif self.study_type == "Stability":
                self.run_stability_study()
        
        return self.results
    
    def save_report(self, filename="msa_report.html"):
        """Save comprehensive HTML report"""
        import plotly.io as pio
        
        fig = self.generate_plot()
        report = self.generate_report()
        
        html_content = f"""
        <html>
        <head>
            <title>MSA Report - {report['study_type']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .section {{ margin-bottom: 30px; }}
                .stats-table {{ border-collapse: collapse; width: 100%; }}
                .stats-table th, .stats-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .stats-table th {{ background-color: #f2f2f2; }}
                .excellent {{ color: green; }}
                .acceptable {{ color: orange; }}
                .unacceptable {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>MSA Report: {report['study_type']}</h1>
            
            <div class="section">
                <h2>Summary</h2>
                <pre>{str(report)}</pre>
            </div>
            
            <div class="section">
                <h2>Visualization</h2>
                {pio.to_html(fig, include_plotlyjs='cdn')}
            </div>
        </body>
        </html>
        """
        
        with open(filename, 'w') as f:
            f.write(html_content)
        
        print(f"MSA report saved as: {filename}")
        return filename

