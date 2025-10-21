"""
Process Capability Analysis Pipeline
Comprehensive analysis of process performance vs specifications
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings


class ProcessCapabilityPipeline:
    def __init__(self, data, measurement_col=None, subgroup_col=None, 
                 usl=None, lsl=None, target=None, date_col=None):
        """
        Initialize the Process Capability Pipeline
        
        Parameters:
        data: DataFrame containing process data
        measurement_col: Column name for measurement values
        subgroup_col: Column name for subgroup/rational subgroup (optional)
        usl: Upper Specification Limit
        lsl: Lower Specification Limit  
        target: Target value (nominal, optional - defaults to midpoint)
        date_col: Column for time-series analysis (optional)
        """
        self.data = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
        self.measurement_col = measurement_col
        self.subgroup_col = subgroup_col
        self.usl = usl
        self.lsl = lsl
        self.target = target
        self.date_col = date_col
        self.results = {}
        
        # Auto-detect measurement column if not provided
        if self.measurement_col is None:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                self.measurement_col = numeric_cols[0]
            else:
                raise ValueError("No numeric columns found for analysis")
        
        # Set target to midpoint if not provided
        if self.target is None and self.usl is not None and self.lsl is not None:
            self.target = (self.usl + self.lsl) / 2
        
        self.data_quality_issues = []
    
    def validate_data_quality(self):
        """
        Validate data quality before capability analysis
        Returns list of issues found
        """
        issues = []
        
        # Auto-detect measurement column (SMART DETECTION)
        if self.measurement_col is None:
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
            priority_names = ['measurement', 'value', 'measure', 'reading', 'result', 'data']
            for priority in priority_names:
                matching = [c for c in numeric_cols if priority in c.lower()]
                if matching:
                    self.measurement_col = matching[0]
                    break
            else:
                # Fallback to first remaining numeric column
                self.measurement_col = numeric_cols[0]
        
        # Check column exists
        if self.measurement_col not in self.data.columns:
            issues.append(f"ERROR: Column '{self.measurement_col}' not found. Available: {list(self.data.columns)}")
            return issues
        
        values = self.data[self.measurement_col]
        
        # Check for missing values
        missing_count = values.isna().sum()
        if missing_count > 0:
            missing_rows = values[values.isna()].index.tolist()
            issues.append(f"WARNING: {missing_count} missing values in rows: {missing_rows[:10]}{'...' if len(missing_rows) > 10 else ''}. Remove before capability analysis")
        
        # Check for non-numeric values
        try:
            numeric_values = pd.to_numeric(values, errors='coerce')
            non_numeric = numeric_values.isna().sum() - missing_count
            if non_numeric > 0:
                issues.append(f"ERROR: {non_numeric} non-numeric values found in '{self.measurement_col}'")
        except:
            pass
        
        # Check sample size
        values_clean = values.dropna()
        if len(values_clean) < 30:
            issues.append(f"WARNING: Only {len(values_clean)} points. Minimum 30 recommended for reliable Cpk (50+ preferred)")
        
        # Check for error codes
        if len(values_clean) > 0:
            error_codes = values_clean[values_clean.isin([999, 9999, -999, -9999])]
            if len(error_codes) > 0:
                issues.append(f"WARNING: Potential error codes (999, -999) in rows: {error_codes.index.tolist()}")
            
            # Check for impossible negative values
            if values_clean.min() < 0:
                negative_rows = values_clean[values_clean < 0].index.tolist()
                issues.append(f"WARNING: Negative values in rows: {negative_rows}. Check if these are data errors")
        
        # Check if all identical
        if len(values_clean) > 0 and values_clean.nunique() == 1:
            issues.append(f"ERROR: All values are identical ({values_clean.iloc[0]}). Cannot calculate capability with no variation")
        
        # Check specifications
        if self.usl is None or self.lsl is None:
            issues.append("ERROR: Both USL and LSL are required for capability analysis. Please provide specification limits")
        elif self.usl <= self.lsl:
            issues.append(f"ERROR: USL ({self.usl}) must be greater than LSL ({self.lsl})")
        
        # Check if data is within specs range (sanity check)
        if self.usl is not None and self.lsl is not None and len(values_clean) > 0:
            outside_spec = ((values_clean < self.lsl) | (values_clean > self.usl)).sum()
            if outside_spec > 0.5 * len(values_clean):
                issues.append(f"WARNING: {outside_spec}/{len(values_clean)} points ({outside_spec/len(values_clean)*100:.1f}%) outside specifications. Process may be severely incapable")
        
        self.data_quality_issues = issues
        return issues
    
    def check_normality(self, alpha=0.05):
        """
        Check if process data follows normal distribution
        Uses multiple normality tests
        
        Parameters:
        alpha: Significance level (default 0.05)
        
        Returns:
        Dictionary with normality test results
        """
        values = self.data[self.measurement_col].dropna()
        
        if len(values) < 3:
            return {
                "is_normal": None,
                "warning": "Insufficient data for normality test (n < 3)"
            }
        
        results = {}
        
        # 1. Anderson-Darling Test (most common in SPC)
        try:
            ad_result = stats.anderson(values, dist='norm')
            # Critical value at 5% significance is typically index 2
            critical_value = ad_result.critical_values[2] if len(ad_result.critical_values) > 2 else ad_result.critical_values[-1]
            ad_pass = ad_result.statistic < critical_value
            
            results['anderson_darling'] = {
                'statistic': float(ad_result.statistic),
                'critical_value': float(critical_value),
                'passes': bool(ad_pass)
            }
        except Exception as e:
            results['anderson_darling'] = {'error': str(e)}
        
        # 2. Shapiro-Wilk Test (good for small samples)
        if len(values) >= 3 and len(values) <= 5000:
            try:
                sw_stat, sw_pvalue = stats.shapiro(values)
                results['shapiro_wilk'] = {
                    'statistic': float(sw_stat),
                    'p_value': float(sw_pvalue),
                    'passes': bool(sw_pvalue > alpha)
                }
            except Exception as e:
                results['shapiro_wilk'] = {'error': str(e)}
        
        # 3. Kolmogorov-Smirnov Test
        try:
            mean = values.mean()
            std = values.std()
            ks_stat, ks_pvalue = stats.kstest(values, 'norm', args=(mean, std))
            results['kolmogorov_smirnov'] = {
                'statistic': float(ks_stat),
                'p_value': float(ks_pvalue),
                'passes': bool(ks_pvalue > alpha)
            }
        except Exception as e:
            results['kolmogorov_smirnov'] = {'error': str(e)}
        
        # 4. Skewness and Kurtosis
        try:
            skew = stats.skew(values)
            kurt = stats.kurtosis(values)
            # Rules of thumb: skewness between -1 and 1, kurtosis between -1 and 1
            skew_ok = abs(skew) < 1
            kurt_ok = abs(kurt) < 1
            
            results['distribution_shape'] = {
                'skewness': float(skew),
                'kurtosis': float(kurt),
                'skewness_acceptable': bool(skew_ok),
                'kurtosis_acceptable': bool(kurt_ok)
            }
        except Exception as e:
            results['distribution_shape'] = {'error': str(e)}
        
        # Overall assessment
        tests_passed = sum([
            results.get('anderson_darling', {}).get('passes', False),
            results.get('shapiro_wilk', {}).get('passes', False),
            results.get('kolmogorov_smirnov', {}).get('passes', False),
            results.get('distribution_shape', {}).get('skewness_acceptable', False),
            results.get('distribution_shape', {}).get('kurtosis_acceptable', False)
        ])
        
        total_tests = 5
        
        # If majority of tests pass, consider normal
        is_normal = tests_passed >= (total_tests / 2)
        
        results['overall'] = {
            'is_normal': bool(is_normal),
            'tests_passed': int(tests_passed),
            'total_tests': int(total_tests),
            'confidence': 'High' if tests_passed >= 4 else 'Medium' if tests_passed >= 3 else 'Low',
            'recommendation': self._normality_recommendation(is_normal, results)
        }
        
        self.results['normality'] = results
        return results
    
    def _normality_recommendation(self, is_normal, test_results):
        """Generate recommendation based on normality test results"""
        if is_normal:
            return "Data appears normally distributed. Cp/Cpk calculations are valid."
        else:
            skew = test_results.get('distribution_shape', {}).get('skewness', 0)
            
            if abs(skew) > 2:
                return "Data is highly skewed. Consider Box-Cox or log transformation before capability analysis."
            elif abs(skew) > 1:
                return "Data shows moderate skewness. Capability indices may be approximate. Consider using percentile method or transformation."
            else:
                return "Data is slightly non-normal. Capability indices are reasonable but interpret with caution."
    
    def suggest_transformation(self):
        """
        Suggest appropriate transformation for non-normal data
        """
        values = self.data[self.measurement_col]
        
        # Check current distribution
        if 'normality' not in self.results:
            self.check_normality()
        
        if self.results['normality']['overall']['is_normal']:
            return {
                'needed': False,
                'message': 'Data is already normally distributed. No transformation needed.'
            }
        
        skew = self.results['normality']['distribution_shape']['skewness']
        
        # Determine best transformation
        if skew > 1:
            suggested = "Log transformation (for right-skewed data)"
            formula = "log(x) or log(x+1) if values include zero"
        elif skew < -1:
            suggested = "Square transformation (for left-skewed data)"
            formula = "x^2"
        else:
            suggested = "Box-Cox transformation (automatic optimization)"
            formula = "Optimizes lambda parameter"
        
        # Try Box-Cox if all values positive
        if np.all(values > 0):
            try:
                transformed, lambda_param = stats.boxcox(values)
                boxcox_suggestion = f"Box-Cox with lambda={lambda_param:.3f}"
            except:
                boxcox_suggestion = "Box-Cox failed (try manual transformation)"
        else:
            boxcox_suggestion = "Box-Cox requires all positive values"
        
        return {
            'needed': True,
            'current_skewness': float(skew),
            'suggested_transformation': suggested,
            'formula': formula,
            'boxcox': boxcox_suggestion,
            'alternative': 'Use non-parametric capability methods (percentile-based)'
        }
    
    def calculate_short_term_capability(self, check_normality=True):
        """
        Calculate short-term capability (Cp, Cpk)
        Uses within-subgroup variation (if subgroups present)
        
        Parameters:
        check_normality: If True, checks normality before calculation (default: True)
        """
        values = self.data[self.measurement_col]
        
        if self.usl is None or self.lsl is None:
            raise ValueError("Both USL and LSL are required for capability analysis")
        
        # Check normality if requested
        if check_normality:
            normality_result = self.check_normality()
            if not normality_result['overall']['is_normal']:
                warnings.warn(
                    f"Data may not be normally distributed. "
                    f"{normality_result['overall']['recommendation']}"
                )
        
        # Calculate process mean
        mean = values.mean()
        
        # Calculate short-term (within) standard deviation
        if self.subgroup_col and self.subgroup_col in self.data.columns:
            # Use within-subgroup variation (Rbar/d2 method)
            subgroups = self.data.groupby(self.subgroup_col)[self.measurement_col]
            ranges = subgroups.apply(lambda x: x.max() - x.min() if len(x) > 1 else 0)
            rbar = ranges.mean()
            
            # d2 constant based on subgroup size
            n = int(subgroups.size().mean())
            d2_values = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 
                        7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}
            d2 = d2_values.get(n, 3.0)  # Default to 3.0 for larger groups
            
            sigma_within = rbar / d2
        else:
            # No subgroups - use overall standard deviation as approximation
            # For true short-term, consecutive measurements should be used
            sigma_within = values.std(ddof=1)
        
        # Calculate Cp (potential capability)
        spec_width = self.usl - self.lsl
        process_width = 6 * sigma_within
        cp = spec_width / process_width if process_width > 0 else 0
        
        # Calculate Cpk (actual capability - accounts for centering)
        cpu = (self.usl - mean) / (3 * sigma_within) if sigma_within > 0 else 0
        cpl = (mean - self.lsl) / (3 * sigma_within) if sigma_within > 0 else 0
        cpk = min(cpu, cpl)
        
        # Calculate Cpm (capability relative to target)
        if self.target is not None:
            tau_squared = sigma_within ** 2 + (mean - self.target) ** 2
            cpm = spec_width / (6 * np.sqrt(tau_squared)) if tau_squared > 0 else 0
        else:
            cpm = None
        
        self.results['short_term'] = {
            'sigma_within': float(sigma_within),
            'Cp': float(cp),
            'Cpk': float(cpk),
            'Cpu': float(cpu),
            'Cpl': float(cpl),
            'Cpm': float(cpm) if cpm is not None else None
        }
        
        return self.results['short_term']
    
    def calculate_long_term_capability(self):
        """
        Calculate long-term capability (Pp, Ppk)
        Uses overall (total) variation
        """
        values = self.data[self.measurement_col]
        
        if self.usl is None or self.lsl is None:
            raise ValueError("Both USL and LSL are required for capability analysis")
        
        # Calculate process mean
        mean = values.mean()
        
        # Calculate long-term (overall) standard deviation
        sigma_overall = values.std(ddof=1)
        
        # Calculate Pp (potential performance)
        spec_width = self.usl - self.lsl
        process_width = 6 * sigma_overall
        pp = spec_width / process_width if process_width > 0 else 0
        
        # Calculate Ppk (actual performance - accounts for centering)
        ppu = (self.usl - mean) / (3 * sigma_overall) if sigma_overall > 0 else 0
        ppl = (mean - self.lsl) / (3 * sigma_overall) if sigma_overall > 0 else 0
        ppk = min(ppu, ppl)
        
        # Calculate Ppm (performance relative to target)
        if self.target is not None:
            tau_squared = sigma_overall ** 2 + (mean - self.target) ** 2
            ppm = spec_width / (6 * np.sqrt(tau_squared)) if tau_squared > 0 else 0
        else:
            ppm = None
        
        self.results['long_term'] = {
            'sigma_overall': float(sigma_overall),
            'Pp': float(pp),
            'Ppk': float(ppk),
            'Ppu': float(ppu),
            'Ppl': float(ppl),
            'Ppm': float(ppm) if ppm is not None else None
        }
        
        return self.results['long_term']
    
    def calculate_process_performance(self):
        """
        Calculate process performance metrics
        """
        values = self.data[self.measurement_col]
        mean = values.mean()
        std = values.std(ddof=1)
        
        # Count defects
        above_usl = np.sum(values > self.usl) if self.usl is not None else 0
        below_lsl = np.sum(values < self.lsl) if self.lsl is not None else 0
        total_defects = above_usl + below_lsl
        
        # Calculate yield
        total_count = len(values)
        yield_pct = ((total_count - total_defects) / total_count * 100) if total_count > 0 else 0
        
        # Calculate DPMO (Defects Per Million Opportunities)
        dpmo = (total_defects / total_count * 1_000_000) if total_count > 0 else 0
        
        # Estimate Z-score (sigma level)
        if self.usl is not None and self.lsl is not None:
            z_usl = (self.usl - mean) / std if std > 0 else 0
            z_lsl = (mean - self.lsl) / std if std > 0 else 0
            z_bench = min(z_usl, z_lsl)
        elif self.usl is not None:
            z_bench = (self.usl - mean) / std if std > 0 else 0
        elif self.lsl is not None:
            z_bench = (mean - self.lsl) / std if std > 0 else 0
        else:
            z_bench = None
        
        # Estimate expected DPMO from Z-score
        if z_bench is not None and z_bench > 0:
            expected_dpmo = (1 - stats.norm.cdf(z_bench)) * 1_000_000
        else:
            expected_dpmo = None
        
        self.results['performance'] = {
            'mean': float(mean),
            'std_dev': float(std),
            'above_usl': int(above_usl),
            'below_lsl': int(below_lsl),
            'total_defects': int(total_defects),
            'total_count': int(total_count),
            'yield_percent': float(yield_pct),
            'dpmo': float(dpmo),
            'z_bench': float(z_bench) if z_bench is not None else None,
            'expected_dpmo': float(expected_dpmo) if expected_dpmo is not None else None
        }
        
        return self.results['performance']
    
    def analyze_centering(self):
        """
        Analyze process centering relative to target and specifications
        """
        values = self.data[self.measurement_col]
        mean = values.mean()
        
        if self.usl is None or self.lsl is None:
            return None
        
        # Calculate spec midpoint
        spec_midpoint = (self.usl + self.lsl) / 2
        
        # Calculate offset from target and midpoint
        offset_from_target = mean - self.target if self.target is not None else None
        offset_from_midpoint = mean - spec_midpoint
        
        # Calculate % of tolerance used
        tolerance = self.usl - self.lsl
        pct_tolerance_used = (abs(offset_from_midpoint) / (tolerance / 2) * 100)
        
        # Determine if process is centered
        # Generally, within 25% of center is considered acceptably centered
        is_centered = pct_tolerance_used < 25
        
        self.results['centering'] = {
            'process_mean': float(mean),
            'spec_midpoint': float(spec_midpoint),
            'target': float(self.target) if self.target is not None else None,
            'offset_from_midpoint': float(offset_from_midpoint),
            'offset_from_target': float(offset_from_target) if offset_from_target is not None else None,
            'pct_tolerance_used': float(pct_tolerance_used),
            'is_centered': bool(is_centered),
            'recommendation': 'Process is well centered' if is_centered else 'Process should be re-centered'
        }
        
        return self.results['centering']
    
    def generate_full_analysis(self):
        """
        Run complete capability analysis
        """
        # Calculate all metrics
        self.calculate_short_term_capability()
        self.calculate_long_term_capability()
        self.calculate_process_performance()
        self.analyze_centering()
        
        # Generate interpretation
        cpk = self.results['short_term']['Cpk']
        ppk = self.results['long_term']['Ppk']
        
        if cpk >= 1.67:
            cpk_rating = "Excellent (Six Sigma capable)"
        elif cpk >= 1.33:
            cpk_rating = "Adequate (meets requirements)"
        elif cpk >= 1.0:
            cpk_rating = "Marginal (may need improvement)"
        else:
            cpk_rating = "Unacceptable (requires improvement)"
        
        if ppk >= 1.67:
            ppk_rating = "Excellent"
        elif ppk >= 1.33:
            ppk_rating = "Adequate"
        elif ppk >= 1.0:
            ppk_rating = "Marginal"
        else:
            ppk_rating = "Unacceptable"
        
        self.results['interpretation'] = {
            'cpk_rating': cpk_rating,
            'ppk_rating': ppk_rating,
            'overall_assessment': self._generate_assessment()
        }
        
        return self.results
    
    def _generate_assessment(self):
        """Generate overall process assessment"""
        cpk = self.results['short_term']['Cpk']
        ppk = self.results['long_term']['Ppk']
        yield_pct = self.results['performance']['yield_percent']
        
        assessment = []
        
        # Capability assessment
        if cpk >= 1.33:
            assessment.append(f"✓ Process is capable (Cpk={cpk:.2f})")
        else:
            assessment.append(f"⚠ Process capability needs improvement (Cpk={cpk:.2f})")
        
        # Performance assessment
        if ppk >= 1.33:
            assessment.append(f"✓ Process performance is good (Ppk={ppk:.2f})")
        else:
            assessment.append(f"⚠ Process performance needs improvement (Ppk={ppk:.2f})")
        
        # Yield assessment
        if yield_pct >= 99.73:
            assessment.append(f"✓ Excellent yield ({yield_pct:.2f}%)")
        elif yield_pct >= 99:
            assessment.append(f"✓ Good yield ({yield_pct:.2f}%)")
        else:
            assessment.append(f"⚠ Yield needs improvement ({yield_pct:.2f}%)")
        
        # Centering
        if self.results['centering']['is_centered']:
            assessment.append("✓ Process is well centered")
        else:
            assessment.append("⚠ Process should be re-centered")
        
        return " | ".join(assessment)
    
    def generate_normality_plot(self):
        """Generate comprehensive normality assessment plots"""
        values = self.data[self.measurement_col].dropna()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Histogram with Normal Curve', 'Normal Probability Plot (Q-Q)',
                          'Box Plot', 'Distribution Statistics'],
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "box"}, {"type": "table"}]]
        )
        
        mean = values.mean()
        std = values.std()
        
        # 1. Histogram with normal overlay
        fig.add_trace(
            go.Histogram(x=values, name='Data', nbinsx=30, histnorm='probability density'),
            row=1, col=1
        )
        
        # Normal curve overlay
        x_range = np.linspace(values.min(), values.max(), 200)
        y_normal = stats.norm.pdf(x_range, mean, std)
        fig.add_trace(
            go.Scatter(x=x_range, y=y_normal, mode='lines',
                      name='Normal Curve', line=dict(color='red', width=2)),
            row=1, col=1
        )
        
        # 2. Q-Q Plot (Normal Probability Plot)
        sorted_values = np.sort(values)
        n = len(sorted_values)
        theoretical_quantiles = stats.norm.ppf(np.arange(1, n + 1) / (n + 1))
        
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=sorted_values, mode='markers',
                      name='Data Points', marker=dict(color='blue')),
            row=1, col=2
        )
        
        # Reference line
        slope = std
        intercept = mean
        ref_line = slope * theoretical_quantiles + intercept
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=ref_line, mode='lines',
                      line=dict(color='red', dash='dash'), name='Perfect Normal'),
            row=1, col=2
        )
        
        # 3. Box Plot
        fig.add_trace(
            go.Box(y=values, name='Distribution', boxmean='sd'),
            row=2, col=1
        )
        
        # 4. Statistics table
        if 'normality' in self.results:
            norm_results = self.results['normality']
            
            table_data = [
                ['Test', 'Result', 'Status'],
                ['Anderson-Darling', 
                 f"{norm_results.get('anderson_darling', {}).get('statistic', 'N/A'):.4f}" if 'anderson_darling' in norm_results else 'N/A',
                 '✓ Pass' if norm_results.get('anderson_darling', {}).get('passes', False) else '✗ Fail'],
                ['Shapiro-Wilk', 
                 f"p={norm_results.get('shapiro_wilk', {}).get('p_value', 0):.4f}" if 'shapiro_wilk' in norm_results else 'N/A',
                 '✓ Pass' if norm_results.get('shapiro_wilk', {}).get('passes', False) else '✗ Fail'],
                ['Skewness', 
                 f"{norm_results.get('distribution_shape', {}).get('skewness', 0):.3f}",
                 '✓ OK' if norm_results.get('distribution_shape', {}).get('skewness_acceptable', False) else '✗ High'],
                ['Kurtosis', 
                 f"{norm_results.get('distribution_shape', {}).get('kurtosis', 0):.3f}",
                 '✓ OK' if norm_results.get('distribution_shape', {}).get('kurtosis_acceptable', False) else '✗ High'],
                ['Overall', 
                 f"{norm_results['overall']['tests_passed']}/{norm_results['overall']['total_tests']}",
                 '✓ Normal' if norm_results['overall']['is_normal'] else '✗ Non-Normal']
            ]
        else:
            # Run normality check
            self.check_normality()
            return self.generate_normality_plot()  # Recursive call with results
        
        fig.add_trace(
            go.Table(
                header=dict(values=table_data[0], fill_color='paleturquoise', align='left'),
                cells=dict(values=list(zip(*table_data[1:])), fill_color='lavender', align='left')
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Normality Assessment",
            showlegend=False
        )
        
        return fig
    
    def generate_histogram(self):
        """Generate process histogram with spec limits and normal curve"""
        values = self.data[self.measurement_col]
        
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=values,
            name='Process Data',
            nbinsx=30,
            histnorm='probability density',
            marker=dict(color='lightblue', line=dict(color='darkblue', width=1))
        ))
        
        # Normal distribution curve
        mean = values.mean()
        std = values.std()
        x_range = np.linspace(values.min(), values.max(), 200)
        y_normal = stats.norm.pdf(x_range, mean, std)
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_normal,
            name='Normal Distribution',
            line=dict(color='red', width=2)
        ))
        
        # Specification limits
        if self.lsl is not None:
            fig.add_vline(x=self.lsl, line=dict(color='red', width=2, dash='dash'),
                         annotation_text="LSL", annotation_position="top")
        
        if self.usl is not None:
            fig.add_vline(x=self.usl, line=dict(color='red', width=2, dash='dash'),
                         annotation_text="USL", annotation_position="top")
        
        # Target
        if self.target is not None:
            fig.add_vline(x=self.target, line=dict(color='green', width=2, dash='dot'),
                         annotation_text="Target", annotation_position="top")
        
        # Mean
        fig.add_vline(x=mean, line=dict(color='blue', width=2),
                     annotation_text="Mean", annotation_position="bottom")
        
        fig.update_layout(
            title="Process Capability Histogram",
            xaxis_title="Measurement",
            yaxis_title="Probability Density",
            height=500,
            showlegend=True
        )
        
        return fig
    
    def generate_capability_plot(self):
        """Generate multi-panel capability analysis plot"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Histogram with Spec Limits', 'Normal Probability Plot',
                          'Individual Values Chart', 'Capability Indices'],
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        values = self.data[self.measurement_col]
        mean = values.mean()
        std = values.std()
        
        # 1. Histogram
        fig.add_trace(
            go.Histogram(x=values, name='Data', nbinsx=30, showlegend=False),
            row=1, col=1
        )
        
        # Add spec limits to histogram
        if self.lsl is not None:
            fig.add_vline(x=self.lsl, line=dict(color='red', dash='dash'), row=1, col=1)
        if self.usl is not None:
            fig.add_vline(x=self.usl, line=dict(color='red', dash='dash'), row=1, col=1)
        fig.add_vline(x=mean, line=dict(color='blue'), row=1, col=1)
        
        # 2. Normal Probability Plot
        sorted_values = np.sort(values)
        n = len(sorted_values)
        theoretical_quantiles = stats.norm.ppf(np.arange(1, n + 1) / (n + 1))
        
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=sorted_values, mode='markers',
                      name='Data', showlegend=False),
            row=1, col=2
        )
        
        # Add reference line
        slope = std
        intercept = mean
        ref_line = slope * theoretical_quantiles + intercept
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=ref_line, mode='lines',
                      line=dict(color='red', dash='dash'), name='Normal', showlegend=False),
            row=1, col=2
        )
        
        # 3. Individual Values Chart
        sequence = list(range(1, len(values) + 1))
        fig.add_trace(
            go.Scatter(x=sequence, y=values, mode='markers+lines',
                      name='Measurements', showlegend=False),
            row=2, col=1
        )
        
        if self.lsl is not None:
            fig.add_hline(y=self.lsl, line=dict(color='red', dash='dash'), row=2, col=1)
        if self.usl is not None:
            fig.add_hline(y=self.usl, line=dict(color='red', dash='dash'), row=2, col=1)
        fig.add_hline(y=mean, line=dict(color='blue'), row=2, col=1)
        
        # 4. Capability Indices Bar Chart
        if 'short_term' in self.results and 'long_term' in self.results:
            indices = ['Cp', 'Cpk', 'Pp', 'Ppk']
            values_bar = [
                self.results['short_term']['Cp'],
                self.results['short_term']['Cpk'],
                self.results['long_term']['Pp'],
                self.results['long_term']['Ppk']
            ]
            
            colors = ['green' if v >= 1.33 else 'orange' if v >= 1.0 else 'red' for v in values_bar]
            
            fig.add_trace(
                go.Bar(x=indices, y=values_bar, marker=dict(color=colors),
                      name='Indices', showlegend=False),
                row=2, col=2
            )
            
            # Add reference lines
            fig.add_hline(y=1.33, line=dict(color='green', dash='dash'), row=2, col=2)
            fig.add_hline(y=1.0, line=dict(color='orange', dash='dash'), row=2, col=2)
        
        fig.update_layout(height=800, title_text="Process Capability Analysis", showlegend=False)
        
        return fig
    
    def save_report(self, filename="capability_report.html"):
        """Save comprehensive capability report"""
        import plotly.io as pio
        
        if not self.results:
            self.generate_full_analysis()
        
        fig = self.generate_capability_plot()
        
        html_content = f"""
        <html>
        <head>
            <title>Process Capability Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .section {{ margin-bottom: 30px; }}
                .metrics-table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                .metrics-table th {{ background-color: #4CAF50; color: white; }}
                .excellent {{ color: green; font-weight: bold; }}
                .adequate {{ color: orange; font-weight: bold; }}
                .poor {{ color: red; font-weight: bold; }}
                .spec-box {{ background-color: #f0f0f0; padding: 15px; border-left: 4px solid #4CAF50; }}
            </style>
        </head>
        <body>
            <h1>Process Capability Analysis Report</h1>
            
            <div class="section spec-box">
                <h2>Specifications</h2>
                <p><strong>USL:</strong> {self.usl}</p>
                <p><strong>LSL:</strong> {self.lsl}</p>
                <p><strong>Target:</strong> {self.target if self.target else 'Not specified'}</p>
                <p><strong>Tolerance:</strong> {self.usl - self.lsl if self.usl and self.lsl else 'N/A'}</p>
            </div>
            
            {f'''<div class="section">
                <h2>⚠️ Normality Test Results</h2>
                <p><strong>Is Normal:</strong> <span class="{'excellent' if self.results.get('normality', {}).get('overall', {}).get('is_normal', False) else 'poor'}">
                    {'✓ YES' if self.results.get('normality', {}).get('overall', {}).get('is_normal', False) else '✗ NO'}
                </span></p>
                <p><strong>Confidence:</strong> {self.results.get('normality', {}).get('overall', {}).get('confidence', 'N/A')}</p>
                <p><strong>Tests Passed:</strong> {self.results.get('normality', {}).get('overall', {}).get('tests_passed', 0)}/{self.results.get('normality', {}).get('overall', {}).get('total_tests', 0)}</p>
                <p><strong>Recommendation:</strong> {self.results.get('normality', {}).get('overall', {}).get('recommendation', 'Check normality first')}</p>
                <details>
                    <summary>Click for detailed test results</summary>
                    <ul>
                        <li><strong>Anderson-Darling:</strong> {'✓ Pass' if self.results.get('normality', {}).get('anderson_darling', {}).get('passes', False) else '✗ Fail'} 
                            (Stat: {self.results.get('normality', {}).get('anderson_darling', {}).get('statistic', 'N/A')})</li>
                        <li><strong>Shapiro-Wilk:</strong> {'✓ Pass' if self.results.get('normality', {}).get('shapiro_wilk', {}).get('passes', False) else '✗ Fail'} 
                            (p-value: {self.results.get('normality', {}).get('shapiro_wilk', {}).get('p_value', 'N/A')})</li>
                        <li><strong>Skewness:</strong> {self.results.get('normality', {}).get('distribution_shape', {}).get('skewness', 'N/A'):.3f} 
                            ({'✓ Acceptable' if self.results.get('normality', {}).get('distribution_shape', {}).get('skewness_acceptable', False) else '✗ High'})</li>
                        <li><strong>Kurtosis:</strong> {self.results.get('normality', {}).get('distribution_shape', {}).get('kurtosis', 'N/A'):.3f} 
                            ({'✓ Acceptable' if self.results.get('normality', {}).get('distribution_shape', {}).get('kurtosis_acceptable', False) else '✗ High'})</li>
                    </ul>
                </details>
            </div>''' if 'normality' in self.results else ''}
            
            <div class="section">
                <h2>Capability Indices</h2>
                <table class="metrics-table">
                    <tr>
                        <th>Index</th>
                        <th>Value</th>
                        <th>Interpretation</th>
                    </tr>
                    <tr>
                        <td><strong>Cp</strong> (Potential Capability)</td>
                        <td>{self.results['short_term']['Cp']:.3f}</td>
                        <td class="{'excellent' if self.results['short_term']['Cp'] >= 1.33 else 'adequate' if self.results['short_term']['Cp'] >= 1.0 else 'poor'}">
                            {'Excellent' if self.results['short_term']['Cp'] >= 1.33 else 'Adequate' if self.results['short_term']['Cp'] >= 1.0 else 'Poor'}
                        </td>
                    </tr>
                    <tr>
                        <td><strong>Cpk</strong> (Actual Capability)</td>
                        <td>{self.results['short_term']['Cpk']:.3f}</td>
                        <td class="{'excellent' if self.results['short_term']['Cpk'] >= 1.33 else 'adequate' if self.results['short_term']['Cpk'] >= 1.0 else 'poor'}">
                            {'Excellent' if self.results['short_term']['Cpk'] >= 1.33 else 'Adequate' if self.results['short_term']['Cpk'] >= 1.0 else 'Poor'}
                        </td>
                    </tr>
                    <tr>
                        <td><strong>Pp</strong> (Potential Performance)</td>
                        <td>{self.results['long_term']['Pp']:.3f}</td>
                        <td class="{'excellent' if self.results['long_term']['Pp'] >= 1.33 else 'adequate' if self.results['long_term']['Pp'] >= 1.0 else 'poor'}">
                            {'Excellent' if self.results['long_term']['Pp'] >= 1.33 else 'Adequate' if self.results['long_term']['Pp'] >= 1.0 else 'Poor'}
                        </td>
                    </tr>
                    <tr>
                        <td><strong>Ppk</strong> (Actual Performance)</td>
                        <td>{self.results['long_term']['Ppk']:.3f}</td>
                        <td class="{'excellent' if self.results['long_term']['Ppk'] >= 1.33 else 'adequate' if self.results['long_term']['Ppk'] >= 1.0 else 'poor'}">
                            {'Excellent' if self.results['long_term']['Ppk'] >= 1.33 else 'Adequate' if self.results['long_term']['Ppk'] >= 1.0 else 'Poor'}
                        </td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Process Performance</h2>
                <p><strong>Mean:</strong> {self.results['performance']['mean']:.4f}</p>
                <p><strong>Std Dev:</strong> {self.results['performance']['std_dev']:.4f}</p>
                <p><strong>Yield:</strong> {self.results['performance']['yield_percent']:.2f}%</p>
                <p><strong>DPMO:</strong> {self.results['performance']['dpmo']:.0f}</p>
                <p><strong>Defects:</strong> {self.results['performance']['total_defects']} out of {self.results['performance']['total_count']}</p>
            </div>
            
            <div class="section">
                <h2>Process Centering</h2>
                <p><strong>Process Mean:</strong> {self.results['centering']['process_mean']:.4f}</p>
                <p><strong>Spec Midpoint:</strong> {self.results['centering']['spec_midpoint']:.4f}</p>
                <p><strong>Offset:</strong> {self.results['centering']['offset_from_midpoint']:.4f}</p>
                <p><strong>Status:</strong> <span class="{'excellent' if self.results['centering']['is_centered'] else 'poor'}">{self.results['centering']['recommendation']}</span></p>
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                {pio.to_html(fig, include_plotlyjs='cdn')}
            </div>
            
            <div class="section">
                <h2>Overall Assessment</h2>
                <p>{self.results['interpretation']['overall_assessment']}</p>
            </div>
            
            <div class="section">
                <h3>Capability Criteria Guide</h3>
                <ul>
                    <li><strong class="excellent">Cp/Cpk ≥ 1.33:</strong> Excellent - Six Sigma capable</li>
                    <li><strong class="adequate">Cp/Cpk ≥ 1.00:</strong> Adequate - Meets minimum requirements</li>
                    <li><strong class="poor">Cp/Cpk < 1.00:</strong> Poor - Process needs improvement</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(filename, 'w') as f:
            f.write(html_content)
        
        print(f"Capability report saved as: {filename}")
        return filename

