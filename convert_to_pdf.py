import os
import sys
import io
from pathlib import Path
from PIL import Image
from playwright.sync_api import sync_playwright

def convert_html_presentation_to_pdf(html_path: str, output_pdf_path: str):
    html_file = Path(html_path).resolve()
    if not html_file.exists():
        raise FileNotFoundError(f"Cannot find HTML file: {html_file}")

    print(f"[*] Loading presentation: {html_file}")
    file_url = html_file.as_uri()

    with sync_playwright() as p:
        # Launch Chrome with 1920x1080 viewport and 2x retina scale for high sharpness
        browser = p.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            device_scale_factor=2
        )
        page = context.new_page()

        print(f"[*] Navigating to {file_url}...")
        page.goto(file_url, wait_until="networkidle")

        # Wait for web fonts (e.g. Outfit from Google Fonts) to be loaded
        page.evaluate("document.fonts.ready")

        # Hide UI controls and progress bar
        page.evaluate("""
            const controls = document.querySelector('.controls');
            if (controls) controls.style.display = 'none';
            const progress = document.querySelector('.progress-bar');
            if (progress) progress.style.display = 'none';
        """)

        # Get total slides count
        total_slides = page.evaluate("document.querySelectorAll('.slide').length")
        print(f"[+] Found {total_slides} slides in presentation.")

        slide_images = []

        for i in range(total_slides):
            print(f"[*] Capturing slide {i + 1}/{total_slides}...")
            # Trigger slide navigation
            page.evaluate(f"""
                currentSlide = {i};
                updateUI();
            """)
            # Wait for transition animation to complete (transition is 0.6s)
            page.wait_for_timeout(800)

            # Capture screenshot
            screenshot_bytes = page.screenshot()
            img = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
            slide_images.append(img)

        browser.close()

    print(f"[*] Assembling {len(slide_images)} slides into PDF: {output_pdf_path}...")
    os.makedirs(os.path.dirname(os.path.abspath(output_pdf_path)), exist_ok=True)
    slide_images[0].save(
        output_pdf_path,
        "PDF",
        resolution=150.0,
        save_all=True,
        append_images=slide_images[1:]
    )
    print(f"[OK] Successfully generated PDF: {output_pdf_path}")

if __name__ == "__main__":
    html_input = r"E:\Cac du an visual\C2\C2-Domain-Shift-Radar\presentation.html"
    pdf_output = r"E:\Cac du an visual\C2\C2-Domain-Shift-Radar\presentation.pdf"
    convert_html_presentation_to_pdf(html_input, pdf_output)
