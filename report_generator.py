"""
AI-POWERED DEEPFAKE VOICE DETECTION
PDF Forensic Report Generator (report_generator.py)

Generates an academic, cyber-security styled PDF detection report containing:
- Project title & AIDL metadata
- Timestamp of analysis
- Audio filename and duration
- Classification verdict (REAL vs FAKE) with probability scores and confidence
- Spectrogram and waveform analysis
- Formal academic disclaimer
"""

import os
import io
import base64
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def generate_pdf_report(prediction_data, output_path=None):
    """
    Generate a forensic inspection PDF report for a deepfake voice prediction.

    Args:
        prediction_data (dict): Prediction dictionary returned by predict_audio().
        output_path (str, optional): Target file path. If None, saves in outputs/reports/.

    Returns:
        str: Absolute path to the generated PDF.
    """
    if output_path is None:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"deepfake_voice_audit_{timestamp_str}.pdf"
        output_path = os.path.join(REPORTS_DIR, filename)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        textColor=colors.HexColor('#0f172a'),
        leading=24,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#64748b'),
        leading=14,
        spaceAfter=15
    )

    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        textColor=colors.HexColor('#334155'),
        leading=13
    )

    disclaimer_style = ParagraphStyle(
        'DocDisclaimer',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        textColor=colors.HexColor('#64748b'),
        leading=11
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("AI-POWERED DEEPFAKE VOICE DETECTION", title_style))
    story.append(Paragraph("Acoustic Forensics & Deep Learning Voice Integrity Audit Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#8b5cf6'), spaceAfter=15))

    # 2. Key Metadata & Audio Details Table
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    filename = prediction_data.get('audio_filename', 'N/A')
    duration = f"{prediction_data.get('audio_duration', 0.0)} seconds"
    predicted_class = prediction_data.get('predicted_class', 'UNKNOWN')
    confidence = f"{prediction_data.get('confidence', 0.0)}%"
    real_prob = f"{prediction_data.get('real_percentage', 0.0)}% ({prediction_data.get('real_probability', 0.0)})"
    fake_prob = f"{prediction_data.get('fake_percentage', 0.0)}% ({prediction_data.get('fake_probability', 0.0)})"

    is_real = (predicted_class == "REAL VOICE")
    verdict_color = colors.HexColor('#059669') if is_real else colors.HexColor('#dc2626')
    verdict_bg = colors.HexColor('#ecfdf5') if is_real else colors.HexColor('#fef2f2')

    meta_data = [
        [Paragraph("<b>Inspection Date:</b>", body_style), Paragraph(now_str, body_style),
         Paragraph("<b>Target Audio File:</b>", body_style), Paragraph(str(filename), body_style)],
        [Paragraph("<b>Audio Duration:</b>", body_style), Paragraph(duration, body_style),
         Paragraph("<b>Sampling Rate:</b>", body_style), Paragraph("16,000 Hz (Resampled Mono)", body_style)],
        [Paragraph("<b>Deep Learning Model:</b>", body_style), Paragraph("PyTorch 4-Block 2D-CNN", body_style),
         Paragraph("<b>Feature Extractor:</b>", body_style), Paragraph("Log Mel-Spectrogram (128x128)", body_style)]
    ]

    meta_table = Table(meta_data, colWidths=[110, 155, 110, 155])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # 3. Detection Verdict Box
    verdict_text = f"<b>CLASSIFICATION VERDICT: {predicted_class}</b>"
    verdict_detail = f"Confidence Score: <b>{confidence}</b> &nbsp;&nbsp;|&nbsp;&nbsp; Genuine Speech: <b>{real_prob}</b> &nbsp;&nbsp;|&nbsp;&nbsp; Synthetic Spoof: <b>{fake_prob}</b>"

    verdict_p1 = Paragraph(f"<font color='{verdict_color.hexval()}' size='14'>{verdict_text}</font>", body_style)
    verdict_p2 = Paragraph(verdict_detail, body_style)
    verdict_desc = Paragraph(prediction_data.get('verdict_description', ''), body_style)

    verdict_table = Table([[verdict_p1], [verdict_p2], [Spacer(1, 4)], [verdict_desc]], colWidths=[530])
    verdict_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), verdict_bg),
        ('BOX', (0, 0), (-1, -1), 1.5, verdict_color),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
    ]))
    story.append(verdict_table)
    story.append(Spacer(1, 14))

    # 4. Embedded Acoustic Visualizations (Spectrogram & Waveform)
    story.append(Paragraph("Acoustic Time-Frequency Representations", section_heading))

    # Extract base64 image data and embed in flowable
    images_table_cells = []

    def b64_to_flowable_image(data_uri, target_w=250, target_h=100):
        try:
            if data_uri and data_uri.startswith('data:image'):
                b64_str = data_uri.split(',', 1)[1]
                img_data = base64.b64decode(b64_str)
                img_stream = io.BytesIO(img_data)
                return Image(img_stream, width=target_w, height=target_h)
        except Exception as e:
            print(f"ReportLab image decode error: {e}")
        return None

    wave_img = b64_to_flowable_image(prediction_data.get('waveform_plot'), 255, 95)
    spec_img = b64_to_flowable_image(prediction_data.get('spectrogram_plot'), 255, 95)

    if wave_img and spec_img:
        img_table = Table([[
            Paragraph("<b>Waveform Amplitude Analysis</b>", body_style),
            Paragraph("<b>Log Mel-Spectrogram (128 Bins)</b>", body_style)
        ], [wave_img, spec_img]], colWidths=[265, 265])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(img_table)
        story.append(Spacer(1, 10))

    # 5. Technical Architecture Summary
    story.append(Paragraph("Technical Pipeline & Methodology", section_heading))
    methodology_text = (
        "Audio signals undergo 16 kHz resampled mono normalization and fixed temporal framing (4.0s). "
        "A 128-channel Mel filterbank computes power spectrograms subsequently transformed into log decibel scale. "
        "The PyTorch Convolutional Neural Network extracts hierarchical spatio-temporal acoustic features across "
        "4 convolutional blocks featuring Batch Normalization, Max Pooling, and Dropout regularization. "
        "Logits are mapped to genuine vs. synthetic probability distributions via Softmax activation."
    )
    story.append(Paragraph(methodology_text, body_style))
    story.append(Spacer(1, 12))

    # 6. Disclaimer
    story.append(Paragraph("Academic & Forensic Disclaimer", section_heading))
    disclaimer_text = (
        "This verification report is produced by an Artificial Intelligence Deep Learning (AIDL) model "
        "trained on speech synthesis benchmarks (ASVspoof / WaveFake acoustic representations). Detection results "
        "are probabilistic confidence estimates and should be used in conjunction with secondary forensic "
        "and contextual verification methods. Noise, extreme clipping, or non-speech background audio can influence "
        "model confidence."
    )
    story.append(Paragraph(disclaimer_text, disclaimer_style))

    # Build PDF
    doc.build(story)
    return output_path
