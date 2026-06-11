import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from datetime import datetime

def generate_pdf_report(company_name, dataset_summary, features, anomalies_count, anomalies_percentage, output_dir):
    """
    Generates a PDF report using matplotlib.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"{company_name}_Report_{timestamp}.pdf")
    
    with PdfPages(report_path) as pdf:
        # Title Page
        fig = plt.figure(figsize=(8.5, 11))
        fig.clf()
        plt.axis('off')
        plt.text(0.5, 0.9, f"Anomaly Detection Report", transform=fig.transFigure, size=24, ha="center")
        plt.text(0.5, 0.85, f"Company: {company_name}", transform=fig.transFigure, size=18, ha="center")
        plt.text(0.5, 0.8, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", transform=fig.transFigure, size=14, ha="center")
        
        plt.text(0.1, 0.65, "Dataset Summary:", transform=fig.transFigure, size=16, weight='bold')
        y_pos = 0.6
        for k, v in dataset_summary.items():
            plt.text(0.15, y_pos, f"{k}: {v}", transform=fig.transFigure, size=12)
            y_pos -= 0.04
            
        plt.text(0.1, y_pos - 0.05, "Selected Features:", transform=fig.transFigure, size=16, weight='bold')
        y_pos -= 0.1
        plt.text(0.15, y_pos, ", ".join(features), transform=fig.transFigure, size=12)
        
        plt.text(0.1, y_pos - 0.1, "Anomaly Statistics:", transform=fig.transFigure, size=16, weight='bold')
        y_pos -= 0.15
        plt.text(0.15, y_pos, f"Total Anomalies Detected: {anomalies_count}", transform=fig.transFigure, size=12)
        plt.text(0.15, y_pos - 0.04, f"Anomaly Percentage: {anomalies_percentage:.2f}%", transform=fig.transFigure, size=12)
        
        pdf.savefig()
        plt.close()
        
    return report_path
