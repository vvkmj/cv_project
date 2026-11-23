import os
import subprocess
import sys

html_path = os.path.abspath("research/final/optimization_report.html")
pdf_path = os.path.abspath("research/final/Model_Optimization_and_Hyperparameter_Report.pdf")

if not os.path.exists(html_path):
    print(f"Error: HTML template not found at {html_path}")
    sys.exit(1)

chrome_bin = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
if not os.path.exists(chrome_bin):
    print(f"Error: Google Chrome binary not found at {chrome_bin}")
    sys.exit(1)

cmd = [
    chrome_bin,
    "--headless",
    "--disable-gpu",
    "--no-pdf-header-footer",
    "--user-data-dir=/tmp/chrome_opt_report_profile",
    f"--print-to-pdf={pdf_path}",
    f"file://{html_path}"
]

print("Rendering Model Optimization PDF via Chrome...")
try:
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
        print(f"Successfully generated Optimization Report PDF: {pdf_path} ({os.path.getsize(pdf_path):,} bytes)")
    else:
        print("Error: PDF was not created or is empty.")
        sys.exit(1)
except subprocess.CalledProcessError as e:
    print(f"Chrome PDF execution failed: {e.stderr}")
    sys.exit(1)
