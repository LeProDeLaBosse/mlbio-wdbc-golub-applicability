"""Single entry point: run the analysis, then compile the LaTeX report."""
import shutil
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
subprocess.run([sys.executable, str(root / "scripts" / "pipeline.py")], check=True)

for _ in range(2):
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
                   cwd=root / "report", check=True, stdout=subprocess.DEVNULL)
shutil.copy(root / "report" / "report.pdf", root / "report.pdf")
print("report.pdf written")
