import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.s3_handler import (
    check_connection,
    upload_model,
    upload_dataframe,
    upload_file,
    list_bucket_contents
)

def main():

    print("=" * 55)
    print("     UPLOADING PROJECT ARTIFACTS TO S3")
    print("=" * 55)

    # Step 1 - Verify connection
    print("\nStep 1 - Checking storage connection...")
    if not check_connection():
        print("Connection failed. Check your setup and retry.")
        return

    # Step 2 - Upload models
    print("\nStep 2 - Uploading trained models...")
    upload_model('../models/xgb_model.pkl', 'models/xgb_model.pkl')
    upload_model('../models/lgb_model.pkl', 'models/lgb_model.pkl')

    # Step 3 - Upload processed data
    print("\nStep 3 - Uploading processed data...")
    upload_file('../data/cleaned_data.csv',   'data/cleaned_data.csv')
    upload_file('../data/X_test.csv',         'data/X_test.csv')
    upload_file('../data/y_test.csv',         'data/y_test.csv')
    upload_file('../data/shap_values.csv',    'data/shap_values.csv')
    upload_file('../data/feature_names.csv',  'data/feature_names.csv')

    # Step 4 - Upload charts
    print("\nStep 4 - Uploading generated charts...")
    charts = [
        'target_distribution.png',
        'missing_values.png',
        'class_imbalance.png',
        'income_vs_default.png',
        'credit_income_ratio.png',
        'categorical_default_rates.png',
        'feature_correlations.png',
        'model_comparison_curves.png',
        'confusion_matrices.png',
        'feature_importance.png',
        'shap_summary_plot.png',
        'shap_bar_plot.png',
        'shap_dependence_plot.png',
        'shap_waterfall_defaulter.png',
        'shap_force_plot.png',
        'shap_comparison.png'
    ]

    for chart in charts:
        local = f'../data/{chart}'
        if os.path.exists(local):
            upload_file(local, f'charts/{chart}')
        else:
            print(f"  Skipped (not found): {chart}")

    # Step 5 - List everything uploaded
    print("\nStep 5 - Verifying uploads...")
    list_bucket_contents()

    print("\n" + "=" * 55)
    print("     ALL ARTIFACTS UPLOADED SUCCESSFULLY")
    print("=" * 55)


if __name__ == "__main__":
    main()