"""
Process Capability Tool for LangGraph Agent
Single comprehensive tool that wraps ProcessCapabilityPipeline functionality.
"""
import pandas as pd
import numpy as np
import json
from typing import Optional
from langchain.tools import tool
from .capability_pipeline import ProcessCapabilityPipeline


@tool
def run_capability_analysis(
    data_path: str,
    usl: float,
    lsl: float,
    target: Optional[float] = None,
    measurement_col: Optional[str] = None,
    subgroup_col: Optional[str] = None,
    date_col: Optional[str] = None
) -> str:
    """
    Complete process capability analysis with automatic validation and comprehensive metrics.
    
    This tool performs the entire capability workflow:
    1. Validates data file and schema
    2. Auto-detects measurement column if not specified
    3. Checks data normality (required for Cp/Cpk validity)
    4. Calculates short-term capability (Cp, Cpk, CPU, CPL)
    5. Calculates long-term performance (Pp, Ppk, PPU, PPL)
    6. Analyzes process yield (DPMO, Sigma level, defect rate)
    7. Checks process centering vs target
    8. Generates comprehensive HTML report
    
    Args:
        data_path: Path to CSV file containing process data
        usl: Upper Specification Limit (REQUIRED)
        lsl: Lower Specification Limit (REQUIRED)
        target: Target/nominal value (defaults to midpoint between USL and LSL)
        measurement_col: Column with measurement values (auto-detected if None)
        subgroup_col: Column for rational subgroups (optional, for Cp vs Pp distinction)
        date_col: Column for time sequence (optional)
    
    Returns:
        JSON string with complete capability results including:
        - Normality test results
        - Capability indices (Cp, Cpk, CPU, CPL)
        - Performance indices (Pp, Ppk, PPU, PPL)
        - Process yield (DPMO, Sigma level, defects)
        - Centering assessment
        - Rating and acceptance status
        - HTML report path
        - Recommendations
    """
    try:
        # ===== STEP 1: LOAD AND VALIDATE DATA =====
        try:
            data = pd.read_csv(data_path)
        except FileNotFoundError:
            return json.dumps({
                "status": "error",
                "error_type": "file_not_found",
                "message": f"File not found: {data_path}",
                "suggestion": "Please check the file path and try again."
            })
        except pd.errors.EmptyDataData:
            return json.dumps({
                "status": "error",
                "error_type": "empty_file",
                "message": "File is empty",
                "suggestion": "Please provide a CSV file with process data."
            })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "error_type": "read_error",
                "message": f"Error reading file: {str(e)}",
                "suggestion": "Make sure it's a valid CSV file."
            })
        
        # ===== STEP 2: SCHEMA VALIDATION =====
        if data.empty:
            return json.dumps({
                "status": "error",
                "error_type": "no_data",
                "message": "CSV file contains no data rows",
                "suggestion": "Ensure the file has data rows, not just headers."
            })
        
        available_columns = list(data.columns)
        
        # Auto-detect measurement column if not specified
        if measurement_col is None:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) == 0:
                return json.dumps({
                    "status": "error",
                    "error_type": "no_numeric_columns",
                    "message": "No numeric columns found for measurements",
                    "available_columns": available_columns,
                    "suggestion": "Ensure your CSV has at least one numeric column for measurements."
                })
            measurement_col = numeric_cols[0]
        
        # Validate measurement column exists
        if measurement_col not in data.columns:
            return json.dumps({
                "status": "error",
                "error_type": "column_not_found",
                "message": f"Measurement column '{measurement_col}' not found",
                "available_columns": available_columns,
                "suggestion": "Specify measurement_col or ensure data has a numeric column."
            })
        
        # ===== STEP 3: VALIDATE SPECIFICATIONS =====
        if usl <= lsl:
            return json.dumps({
                "status": "error",
                "error_type": "invalid_specifications",
                "message": f"USL ({usl}) must be greater than LSL ({lsl})",
                "suggestion": "Check your specification limits and provide correct values."
            })
        
        # Set target to midpoint if not provided
        if target is None:
            target = (usl + lsl) / 2
        
        # ===== STEP 4: CREATE PIPELINE =====
        pipeline = ProcessCapabilityPipeline(
            data=data,
            measurement_col=measurement_col,
            subgroup_col=subgroup_col,
            usl=usl,
            lsl=lsl,
            target=target,
            date_col=date_col
        )
        
        # ===== STEP 5: VALIDATE DATA QUALITY =====
        quality_issues = pipeline.validate_data_quality()
        
        # Check for critical errors
        has_critical_errors = any('ERROR' in issue for issue in quality_issues)
        if has_critical_errors:
            error_messages = [issue for issue in quality_issues if 'ERROR' in issue]
            warning_messages = [issue for issue in quality_issues if 'WARNING' in issue]
            return json.dumps({
                "status": "error",
                "error_type": "data_quality_error",
                "message": "Data quality issues detected",
                "errors": error_messages,
                "warnings": warning_messages,
                "suggestion": "Fix the errors in your data and try again."
            })
        
        # ===== STEP 6: CHECK NORMALITY =====
        try:
            normality_results = pipeline.check_normality()
            is_normal = normality_results.get('is_normal', False)
            
            if not is_normal:
                normality_warning = "Data is not normally distributed. Cp/Cpk results may not be valid. Consider data transformation or non-parametric methods."
            else:
                normality_warning = None
        except Exception as e:
            normality_results = {"error": str(e)}
            is_normal = False
            normality_warning = f"Normality check failed: {str(e)}"
        
        # ===== STEP 7: CALCULATE CAPABILITY INDICES =====
        try:
            capability_results = pipeline.calculate_short_term_capability(check_normality=False)
            cp = capability_results.get('Cp', 0)
            cpk = capability_results.get('Cpk', 0)
            cpu = capability_results.get('CPU', 0)
            cpl = capability_results.get('CPL', 0)
        except Exception as e:
            return json.dumps({
                "status": "error",
                "error_type": "calculation_error",
                "message": f"Error calculating capability: {str(e)}",
                "suggestion": "Check if your data is appropriate for capability analysis."
            })
        
        # ===== STEP 8: CALCULATE PERFORMANCE INDICES =====
        try:
            performance_results = pipeline.calculate_long_term_capability()
            pp = performance_results.get('Pp', 0)
            ppk = performance_results.get('Ppk', 0)
            ppu = performance_results.get('PPU', 0)
            ppl = performance_results.get('PPL', 0)
        except Exception as e:
            pp = ppk = ppu = ppl = 0
            performance_error = str(e)
        
        # ===== STEP 9: ANALYZE YIELD =====
        try:
            values = data[measurement_col].dropna()
            above_usl = (values > usl).sum()
            below_lsl = (values < lsl).sum()
            total_defects = above_usl + below_lsl
            dpmo = (total_defects / len(values)) * 1_000_000
            yield_pct = ((len(values) - total_defects) / len(values)) * 100
            
            # Estimate sigma level from DPMO
            if dpmo == 0:
                sigma_level = 6.0
            elif dpmo >= 691462:
                sigma_level = 1.0
            elif dpmo >= 308538:
                sigma_level = 2.0
            elif dpmo >= 66807:
                sigma_level = 3.0
            elif dpmo >= 6210:
                sigma_level = 4.0
            elif dpmo >= 233:
                sigma_level = 5.0
            else:
                sigma_level = 6.0
        except Exception as e:
            total_defects = dpmo = sigma_level = yield_pct = 0
            yield_error = str(e)
        
        # ===== STEP 10: CHECK CENTERING =====
        mean = values.mean()
        offset_from_target = mean - target
        is_centered = abs(cp - cpk) < 0.1  # If Cp ≈ Cpk, process is centered
        
        # ===== STEP 11: DETERMINE RATING AND ACCEPTANCE =====
        if cpk >= 1.67:
            rating = "World-class (5σ)"
            acceptance = "Excellent"
            recommendation = f"Process capability is world-class (Cpk={cpk:.3f}). Less than 1 DPMO expected. Maintain current performance."
        elif cpk >= 1.33:
            rating = "Excellent (4σ)"
            acceptance = "Excellent"
            recommendation = f"Process capability is excellent (Cpk={cpk:.3f}). Approximately {dpmo:.0f} DPMO. Continue monitoring."
        elif cpk >= 1.0:
            rating = "Adequate (3σ)"
            acceptance = "Acceptable"
            recommendation = f"Process capability is adequate (Cpk={cpk:.3f}). Approximately {dpmo:.0f} DPMO. Consider improvement to reduce defects."
        else:
            rating = "Unacceptable"
            acceptance = "Unacceptable"
            recommendation = f"Process capability is unacceptable (Cpk={cpk:.3f}). High defect rate ({dpmo:.0f} DPMO). URGENT: Reduce variation or adjust centering."
        
        # Add centering recommendation if needed
        if not is_centered:
            recommendation += f" Process is off-center by {offset_from_target:.4f} from target. Adjust process mean to improve Cpk."
        
        # Add normality warning if needed
        if normality_warning:
            recommendation += f" WARNING: {normality_warning}"
        
        # ===== STEP 12: GENERATE HTML REPORT =====
        try:
            # Run full analysis to populate pipeline.results for report generation
            pipeline.generate_full_analysis()
            html_report_path = pipeline.save_report(filename=data_path.replace('.csv', '_capability_report.html'))
        except Exception as e:
            html_report_path = None
            html_error = str(e)
        
        # ===== STEP 13: GENERATE PLOT =====
        try:
            plot_fig = pipeline.generate_plot()
            plot_generated = True
        except Exception as e:
            plot_generated = False
            plot_error = str(e)
        
        # ===== STEP 14: PREPARE RESPONSE =====
        warnings = [issue for issue in quality_issues if 'WARNING' in issue]
        if normality_warning:
            warnings.append(normality_warning)
        
        response = {
            "status": "success",
            "analysis_complete": True,
            "data_info": {
                "file": data_path,
                "sample_size": len(values),
                "mean": round(float(mean), 4),
                "std_dev": round(float(values.std()), 4),
                "columns_used": {
                    "measurement_col": measurement_col,
                    "subgroup_col": subgroup_col,
                    "date_col": date_col
                }
            },
            "specifications": {
                "USL": float(usl),
                "LSL": float(lsl),
                "Target": float(target),
                "tolerance": float(usl - lsl)
            },
            "normality": {
                "is_normal": is_normal,
                "shapiro_p_value": round(float(normality_results.get('shapiro_p', 0)), 4) if 'shapiro_p' in normality_results else None,
                "anderson_darling_stat": round(float(normality_results.get('anderson_stat', 0)), 4) if 'anderson_stat' in normality_results else None
            },
            "capability_indices": {
                "Cp": round(float(cp), 3),
                "Cpk": round(float(cpk), 3),
                "CPU": round(float(cpu), 3),
                "CPL": round(float(cpl), 3),
                "description": "Short-term capability (within-subgroup variation)"
            },
            "performance_indices": {
                "Pp": round(float(pp), 3),
                "Ppk": round(float(ppk), 3),
                "PPU": round(float(ppu), 3),
                "PPL": round(float(ppl), 3),
                "description": "Long-term performance (overall variation including shifts)"
            },
            "process_yield": {
                "defects": int(total_defects),
                "above_USL": int(above_usl),
                "below_LSL": int(below_lsl),
                "DPMO": round(float(dpmo), 2),
                "sigma_level": round(float(sigma_level), 2),
                "yield_percent": round(float(yield_pct), 2)
            },
            "centering": {
                "is_centered": is_centered,
                "offset_from_target": round(float(offset_from_target), 4),
                "description": "Process centered" if is_centered else "Process off-center"
            },
            "rating": rating,
            "acceptance": acceptance,
            "recommendations": recommendation,
            "html_report": html_report_path if html_report_path else "Report generation failed",
            "plot_generated": plot_generated
        }
        
        # Add warnings if any
        if warnings:
            response["warnings"] = warnings
        
        return json.dumps(response, indent=2)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error_type": "unexpected_error",
            "message": f"Unexpected error during capability analysis: {str(e)}",
            "suggestion": "Please check your data format and specification limits, then try again."
        })
