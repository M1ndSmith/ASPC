"""
Control Chart Tool for LangGraph Agent
Single comprehensive tool that wraps ControlChartPipeline functionality.
"""
import pandas as pd
import numpy as np
import json
from typing import Optional
from langchain_core.tools import tool
from .control_chart_pipeline import ControlChartPipeline


@tool
def run_control_chart_analysis(
    data_path: str,
    value_col: Optional[str] = None,
    subgroup_col: Optional[str] = None,
    date_col: Optional[str] = None,
    sample_size_col: Optional[str] = None,
    opportunity_col: Optional[str] = None,
    chart_type: Optional[str] = None
) -> str:
    """
    Complete control chart analysis with automatic validation, chart selection, and report generation.
    
    This tool performs the entire SPC workflow:
    1. Validates data file and schema
    2. Auto-detects columns if not specified
    3. Detects data type (variable/attribute)
    4. Selects appropriate chart type (I-MR, Xbar-R, P, NP, C, U)
    5. Calculates control limits (UCL, CL, LCL)
    6. Detects out-of-control points
    7. Generates comprehensive HTML report
    
    Args:
        data_path: Path to CSV file containing process data
        value_col: Column with measurement values (auto-detected if None)
        subgroup_col: Column for subgroup identifiers (optional)
        date_col: Column for time sequence (optional)
        sample_size_col: Column for sample sizes - for P/NP charts (optional)
        opportunity_col: Column for opportunities/area - for U charts (optional)
        chart_type: Override auto-selection with specific chart type (optional)
                   Options: 'I-MR', 'Xbar-R', 'P', 'NP', 'C', 'U'
    
    Returns:
        JSON string with complete analysis results including:
        - Data validation status
        - Chart type selected
        - Control limits (UCL, CL, LCL)
        - Out-of-control points
        - Statistical summary
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
                "suggestion": "Please provide a CSV file with data."
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
        
        # List available columns for troubleshooting
        available_columns = list(data.columns)
        
        # Auto-detect value column if not specified (SMART DETECTION)
        if value_col is None:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) == 0:
                return json.dumps({
                    "status": "error",
                    "error_type": "no_numeric_columns",
                    "message": "No numeric columns found in data",
                    "available_columns": available_columns,
                    "suggestion": "Ensure your CSV has at least one numeric column for measurements."
                })
            
            # Skip ID/grouping columns
            skip_patterns = ['id', 'subgroup', 'batch', 'sample', 'group', 'lot', 'serial', 'number']
            filtered_cols = [c for c in numeric_cols if not any(p in c.lower() for p in skip_patterns)]
            if filtered_cols:
                numeric_cols = filtered_cols
            
            # Prefer measurement-related names
            priority_names = ['measurement', 'value', 'measure', 'data', 'reading', 'result', 'defect', 'count']
            for priority in priority_names:
                matching = [c for c in numeric_cols if priority in c.lower()]
                if matching:
                    value_col = matching[0]
                    break
            else:
                value_col = numeric_cols[0]
        
        # Validate specified columns exist
        if value_col not in data.columns:
            return json.dumps({
                "status": "error",
                "error_type": "column_not_found",
                "message": f"Column '{value_col}' not found in data",
                "available_columns": available_columns,
                "suggestion": f"Use one of the available columns or let the tool auto-detect (value_col=None)."
            })
        
        if subgroup_col and subgroup_col not in data.columns:
            return json.dumps({
                "status": "error",
                "error_type": "column_not_found",
                "message": f"Subgroup column '{subgroup_col}' not found in data",
                "available_columns": available_columns,
                "suggestion": "Check column name or set subgroup_col=None for individual measurements."
            })
        
        # ===== STEP 3: CREATE PIPELINE AND VALIDATE DATA QUALITY =====
        pipeline = ControlChartPipeline(
            data=data,
            value_col=value_col,
            subgroup_col=subgroup_col,
            date_col=date_col,
            sample_size_col=sample_size_col,
            opportunity_col=opportunity_col
        )
        
        # Validate data quality
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
        
        # ===== STEP 4: DETECT DATA TYPE =====
        data_type = pipeline.detect_data_type()
        
        # ===== STEP 5: SELECT CHART TYPE =====
        if chart_type:
            # User override
            pipeline.chart_type = chart_type
            selected_chart = chart_type
        else:
            # Auto-select
            selected_chart = pipeline.select_control_chart()
        
        # ===== STEP 6: CALCULATE CONTROL LIMITS =====
        try:
            limits = pipeline.calculate_control_limits()
        except Exception as e:
            return json.dumps({
                "status": "error",
                "error_type": "calculation_error",
                "message": f"Error calculating control limits: {str(e)}",
                "suggestion": "Check if your data is appropriate for the selected chart type."
            })
        
        # ===== STEP 7: GENERATE REPORT =====
        try:
            report = pipeline.generate_report()
        except Exception as e:
            return json.dumps({
                "status": "error",
                "error_type": "report_generation_error",
                "message": f"Error generating report: {str(e)}",
                "suggestion": "Analysis completed but report generation failed."
            })
        
        # ===== STEP 8: GENERATE HTML REPORT =====
        try:
            html_report_path = pipeline.save_report(filename=data_path.replace('.csv', '_control_chart_report.html'))
        except Exception as e:
            html_report_path = None
            html_error = str(e)
        
        # ===== STEP 9: GENERATE PLOT =====
        try:
            plot_fig = pipeline.generate_plot()
            plot_generated = True
        except Exception as e:
            plot_generated = False
            plot_error = str(e)
        
        # ===== STEP 10: PREPARE RESPONSE =====
        # Determine process status
        out_of_control_count = len(report.get('out_of_control_points', []))
        if out_of_control_count == 0:
            status = "IN CONTROL"
            recommendation = "Process is stable and predictable. Continue monitoring. Natural variation only."
        else:
            status = "OUT OF CONTROL"
            recommendation = f"Process has {out_of_control_count} out-of-control points. Investigate special causes using 6M analysis (Man, Machine, Material, Method, Measurement, Environment)."
        
        # Compile warnings
        warnings = [issue for issue in quality_issues if 'WARNING' in issue]
        
        # Build comprehensive response
        response = {
            "status": "success",
            "analysis_complete": True,
            "data_info": {
                "file": data_path,
                "data_type": data_type,
                "sample_size": report.get('sample_size', len(data)),
                "columns_used": {
                    "value_col": value_col,
                    "subgroup_col": subgroup_col,
                    "date_col": date_col
                }
            },
            "chart_info": {
                "chart_type": selected_chart,
                "description": f"{selected_chart} chart selected for {data_type} data"
            },
            "control_limits": {
                "UCL": round(float(limits.get('UCL', 0)), 4),
                "CL": round(float(limits.get('CL', 0)), 4),
                "LCL": round(float(limits.get('LCL', 0)), 4)
            },
            "statistical_summary": {
                "mean": round(float(report.get('mean', 0)), 4),
                "std_dev": round(float(report.get('std_dev', 0)), 4),
                "range": round(float(report.get('range', 0)), 4) if 'range' in report else None
            },
            "process_status": {
                "status": status,
                "out_of_control_points": out_of_control_count,
                "out_of_control_indices": report.get('out_of_control_points', [])
            },
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
            "message": f"Unexpected error during analysis: {str(e)}",
            "suggestion": "Please check your data format and try again."
        })
