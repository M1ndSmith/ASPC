"""
MSA Tool for LangGraph Agent
Single comprehensive tool that wraps MSAPipeline functionality.
"""
import pandas as pd
import numpy as np
import json
from typing import Optional
from langchain_core.tools import tool
from .msa_pipeline import MSAPipeline


@tool
def run_msa_analysis(
    data_path: str,
    part_col: Optional[str] = None,
    operator_col: Optional[str] = None,
    measurement_col: Optional[str] = None,
    trial_col: Optional[str] = None,
    reference_col: Optional[str] = None,
    date_col: Optional[str] = None,
    tolerance: Optional[float] = None,
    method: str = "anova",
    study_type: Optional[str] = None
) -> str:
    """
    Complete Measurement System Analysis with automatic study type detection and validation.
    
    This tool performs comprehensive MSA workflow:
    1. Validates data file and schema
    2. Auto-detects columns if not specified
    3. Detects study type (Gage R&R, Bias, Linearity, Stability)
    4. Validates data structure for the detected study
    5. Runs appropriate MSA study
    6. Calculates all relevant metrics
    7. Generates comprehensive HTML report
    
    Args:
        data_path: Path to CSV file containing measurement data
        part_col: Column for part/sample identifiers (auto-detected if None)
        operator_col: Column for operator/appraiser identifiers (auto-detected if None)
        measurement_col: Column for measurement values (auto-detected if None)
        trial_col: Column for trial/repeat number (auto-detected if None)
        reference_col: Column for reference/standard values (auto-detected if None)
        date_col: Column for date/time (auto-detected if None)
        tolerance: Process tolerance for %Tolerance calculation (optional)
        method: Gage R&R method - 'anova' (default) or 'range'
        study_type: Override auto-detection with specific study type (optional)
                   Options: 'Gage R&R', 'Bias', 'Linearity', 'Stability'
    
    Returns:
        JSON string with complete MSA results including:
        - Study type identified
        - Validation status
        - All MSA metrics (%GRR, bias, linearity, stability)
        - Acceptance criteria evaluation
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
        except pd.errors.EmptyDataError:
            return json.dumps({
                "status": "error",
                "error_type": "empty_file",
                "message": "File is empty",
                "suggestion": "Please provide a CSV file with measurement data."
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
        
        # Auto-detect other columns based on common naming patterns
        cols_lower = {col.lower(): col for col in data.columns}
        
        if part_col is None:
            part_col = next((cols_lower[c] for c in cols_lower if 'part' in c), None)
        
        if operator_col is None:
            operator_col = next((cols_lower[c] for c in cols_lower if 'operator' in c or 'appraiser' in c), None)
        
        if trial_col is None:
            trial_col = next((cols_lower[c] for c in cols_lower if 'trial' in c or 'repeat' in c), None)
        
        if reference_col is None:
            reference_col = next((cols_lower[c] for c in cols_lower if 'reference' in c or 'standard' in c or 'master' in c), None)
        
        if date_col is None:
            date_col = next((cols_lower[c] for c in cols_lower if 'date' in c or 'time' in c), None)
        
        # Validate measurement column exists
        if measurement_col not in data.columns:
            return json.dumps({
                "status": "error",
                "error_type": "column_not_found",
                "message": f"Measurement column '{measurement_col}' not found",
                "available_columns": available_columns,
                "suggestion": "Specify measurement_col or ensure data has a numeric column."
            })
        
        # ===== STEP 3: CREATE PIPELINE =====
        pipeline = MSAPipeline(
            data=data,
            part_col=part_col,
            operator_col=operator_col,
            measurement_col=measurement_col,
            trial_col=trial_col,
            reference_col=reference_col,
            date_col=date_col
        )
        
        # ===== STEP 4: DETECT OR USE SPECIFIED STUDY TYPE =====
        if study_type:
            detected_study_type = study_type
            pipeline.study_type = study_type
        else:
            detected_study_type = pipeline.detect_study_type()
        
        # ===== STEP 5: VALIDATE DATA QUALITY FOR STUDY TYPE =====
        quality_issues = pipeline.validate_data_quality(study_type=detected_study_type)
        
        # Check for critical errors
        has_critical_errors = any('ERROR' in issue for issue in quality_issues)
        if has_critical_errors:
            error_messages = [issue for issue in quality_issues if 'ERROR' in issue]
            warning_messages = [issue for issue in quality_issues if 'WARNING' in issue]
            return json.dumps({
                "status": "error",
                "error_type": "data_quality_error",
                "message": "Data quality issues detected",
                "study_type": detected_study_type,
                "errors": error_messages,
                "warnings": warning_messages,
                "suggestion": "Fix the errors in your data structure and try again."
            })
        
        # ===== STEP 6: RUN APPROPRIATE MSA STUDY =====
        try:
            if detected_study_type == "Gage R&R":
                if not part_col or not operator_col:
                    return json.dumps({
                        "status": "error",
                        "error_type": "missing_columns",
                        "message": "Gage R&R requires Part and Operator columns",
                        "available_columns": available_columns,
                        "detected_columns": {
                            "part_col": part_col,
                            "operator_col": operator_col,
                            "measurement_col": measurement_col
                        },
                        "suggestion": "Specify part_col and operator_col explicitly or ensure columns are named 'Part' and 'Operator'."
                    })
                
                results = pipeline.run_gage_rr(tolerance=tolerance, method=method)
                
                # Determine acceptance
                grr_percent = results.get('grr_percent', 100)
                if grr_percent < 10:
                    acceptance = "Excellent"
                    recommendation = "Measurement system is excellent. %GRR < 10% - ready for process control."
                elif grr_percent < 30:
                    acceptance = "Acceptable"
                    recommendation = "Measurement system is acceptable. %GRR 10-30% - suitable for most applications."
                else:
                    acceptance = "Unacceptable"
                    recommendation = "Measurement system is unacceptable. %GRR > 30% - requires improvement: calibrate equipment, train operators, improve measurement procedure."
                
            elif detected_study_type == "Bias":
                if not reference_col:
                    return json.dumps({
                        "status": "error",
                        "error_type": "missing_columns",
                        "message": "Bias study requires Reference column",
                        "available_columns": available_columns,
                        "suggestion": "Specify reference_col or ensure column is named 'Reference' or 'Standard'."
                    })
                
                results = pipeline.run_bias_study()
                
                # Determine significance
                is_significant = results.get('p_value', 1.0) < 0.05
                if is_significant:
                    acceptance = "Significant Bias Detected"
                    recommendation = f"Bias is statistically significant (p={results.get('p_value', 0):.4f}). Recalibrate equipment or adjust measurement procedure."
                else:
                    acceptance = "No Significant Bias"
                    recommendation = "Bias is not statistically significant. Measurement system is unbiased."
                
            elif detected_study_type == "Linearity":
                if not reference_col:
                    return json.dumps({
                        "status": "error",
                        "error_type": "missing_columns",
                        "message": "Linearity study requires Reference column",
                        "available_columns": available_columns,
                        "suggestion": "Specify reference_col or ensure column is named 'Reference'."
                    })
                
                results = pipeline.run_linearity_study()
                
                # Determine linearity issue
                has_linearity_issue = results.get('p_value', 1.0) < 0.05
                if has_linearity_issue:
                    acceptance = "Linearity Issue Detected"
                    recommendation = "Bias changes across measurement range. Check calibration at all levels."
                else:
                    acceptance = "Linearity Acceptable"
                    recommendation = "Bias is consistent across measurement range. No linearity issues."
                
            elif detected_study_type == "Stability":
                if not date_col:
                    return json.dumps({
                        "status": "error",
                        "error_type": "missing_columns",
                        "message": "Stability study requires Date/Time column",
                        "available_columns": available_columns,
                        "suggestion": "Specify date_col or ensure column is named 'Date' or 'Time'."
                    })
                
                results = pipeline.run_stability_study()
                
                # Determine stability
                is_stable = results.get('out_of_control_points', 0) == 0
                if is_stable:
                    acceptance = "Stable"
                    recommendation = "Measurement system is stable over time. No drift detected."
                else:
                    acceptance = "Unstable"
                    recommendation = f"Instability detected: {results.get('out_of_control_points', 0)} out-of-control points. Investigate drift, wear, or environmental changes."
            
            else:
                return json.dumps({
                    "status": "error",
                    "error_type": "unknown_study_type",
                    "message": f"Unknown study type: {detected_study_type}",
                    "suggestion": "Specify study_type as 'Gage R&R', 'Bias', 'Linearity', or 'Stability'."
                })
                
        except Exception as e:
            return json.dumps({
                "status": "error",
                "error_type": "calculation_error",
                "message": f"Error running {detected_study_type} study: {str(e)}",
                "suggestion": "Check if your data structure matches the study type requirements."
            })
        
        # ===== STEP 7: GENERATE HTML REPORT =====
        try:
            html_report_path = pipeline.save_report(filename=data_path.replace('.csv', '_msa_report.html'))
        except Exception as e:
            html_report_path = None
            html_error = str(e)
        
        # ===== STEP 8: GENERATE PLOT =====
        try:
            plot_fig = pipeline.generate_plot()
            plot_generated = True
        except Exception as e:
            plot_generated = False
            plot_error = str(e)
        
        # ===== STEP 9: PREPARE RESPONSE =====
        warnings = [issue for issue in quality_issues if 'WARNING' in issue]
        
        response = {
            "status": "success",
            "analysis_complete": True,
            "study_info": {
                "study_type": detected_study_type,
                "file": data_path,
                "sample_size": len(data),
                "columns_used": {
                    "measurement_col": measurement_col,
                    "part_col": part_col,
                    "operator_col": operator_col,
                    "trial_col": trial_col,
                    "reference_col": reference_col,
                    "date_col": date_col
                }
            },
            "results": results,
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
            "message": f"Unexpected error during MSA analysis: {str(e)}",
            "suggestion": "Please check your data format and try again."
        })
