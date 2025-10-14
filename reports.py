from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from io import BytesIO
import pandas as pd

def generate_pdf_report_bytes(df):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, height - 40, "CHV Bridge — Summary Report ")
    c.setFont("Helvetica", 10)
    y = height - 80
    total_visits = len(df)
    total_inc = df["Incentive"].sum() if not df.empty else 0
    c.drawString(40, y, f"Total visits: {total_visits}")
    c.drawString(200, y, f"Total incentives (KES): {total_inc}")
    y -= 30
    # show top rows
    if not df.empty:
        sample = df.head(20)
        for _, row in sample.iterrows():
            line = f"{row['Date']} | {row['CHV']} | {row['VisitType']} | KES {row['Incentive']}"
            c.drawString(40, y, line[:100])
            y -= 12
            if y < 60:
                c.showPage()
                y = height - 40
    c.showPage()
    c.save()
    pdf = buffer.getvalue()
    buffer.close()
    return pdf
