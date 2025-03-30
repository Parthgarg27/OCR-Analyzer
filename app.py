import streamlit as st
import easyocr
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import os
import img2pdf
from skimage.filters import threshold_local
import imutils
import time

# Set page configuration
st.set_page_config(
    page_title="Image Scanner & OCR",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with better contrast and modern styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
        padding: 20px;
    }
    .stButton>button {
        background-color: #0066cc;
        color: #ffffff;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0052a3;
        transform: translateY(-2px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .stFileUploader {
        border: 2px dashed #0066cc;
        border-radius: 12px;
        padding: 20px;
        background-color: #e6f0fa;
        transition: all 0.3s ease;
    }
    .stFileUploader:hover {
        border-color: #0052a3;
        background-color: #d9e8f6;
    }
    .stFileUploader label {
        color: #003087 !important;
        font-weight: 600;
        font-size: 18px;
    }
    .header {
        font-size: 40px;
        font-weight: 700;
        color: #003087;
        text-align: center;
        margin-bottom: 20px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .subheader {
        font-size: 24px;
        font-weight: 600;
        color: #003087;
        margin: 20px 0 10px 0;
    }
    .sidebar .sidebar-content {
        background-color: #003087;
        color: #ffffff;
        padding: 20px;
        border-radius: 0 12px 12px 0;
    }
    .sidebar .sidebar-content h2 {
        color: #ffffff;
        font-size: 24px;
    }
    .sidebar .sidebar-content p {
        color: #e6f0fa;
        line-height: 1.6;
    }
    .stDataFrame {
        border: 1px solid #0066cc;
        border-radius: 8px;
        background-color: #ffffff;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTextArea textarea {
        background-color: #ffffff;
        border: 1px solid #0066cc;
        border-radius: 8px;
        color: #1a1a1a;
        font-size: 16px;
        padding: 10px;
    }
    .stRadio > label {
        color: #003087;
        font-weight: 600;
        font-size: 16px;
    }
    .card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .download-section {
        display: flex;
        gap: 15px;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

class ImageScanner:
    """Scanner that applies edge detection to scan an image into a grayscale scan
    while positioning the point of view accordingly if needed."""

    def __init__(self, image):
        """
        :param image: PIL Image object to scan
        """
        self.image = image
        self.user_defined_contours = []

    def scan(self):
        """Searches for a rectangular object in the given image and returns the scanned result."""
        cv2_image = np.array(self.image)
        screenContours = self.__analyze_contours(cv2_image)
        scan_img = self.__transform_and_scan(cv2_image, screenContours)
        return Image.fromarray(scan_img)

    def __analyze_contours(self, cv2_image):
        """Transforms the image to detect edges and find the document contour."""
        cv2_image = imutils.resize(cv2_image, height=500)

        # Convert to grayscale and enhance contrast
        if len(cv2_image.shape) == 3:
            grayscaled = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2GRAY)
        else:
            grayscaled = cv2_image

        # Enhance contrast using histogram equalization
        grayscaled = cv2.equalizeHist(grayscaled)

        # Apply Gaussian blur and edge detection
        blurred = cv2.GaussianBlur(grayscaled, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)

        # Dilate edges to connect broken lines
        kernel = np.ones((5, 5), np.uint8)
        edged = cv2.dilate(edged, kernel, iterations=1)

        # Find contours
        contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        sortedContours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        screenCnt = None
        for contour in sortedContours:
            peri = cv2.arcLength(contour, True)
            approximation = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approximation) == 4:
                screenCnt = approximation
                break

        if screenCnt is None:
            # Fallback: Use the largest contour or image boundaries
            if contours:
                largest_contour = contours[0]
                peri = cv2.arcLength(largest_contour, True)
                approximation = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
                if len(approximation) == 4:
                    screenCnt = approximation
                else:
                    h, w = grayscaled.shape
                    screenCnt = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
            else:
                raise ValueError("Could not detect document edges")

        return screenCnt

    def __transform_and_scan(self, cv2_image, screenCnt):
        """Transforms the perspective to a top-down view and creates the scan."""
        ratio = cv2_image.shape[0] / 500.0
        transformed = self.__four_point_transform(cv2_image, screenCnt.reshape(4, 2) * ratio)

        if len(transformed.shape) == 3:
            transformed_grayscaled = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
        else:
            transformed_grayscaled = transformed

        threshold = threshold_local(transformed_grayscaled, 11, offset=10, method="gaussian")
        transformed_grayscaled = (transformed_grayscaled > threshold).astype("uint8") * 255

        return transformed_grayscaled

    def __order_points(self, pts):
        """Orders the points for perspective transform."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def __four_point_transform(self, image, pts):
        """Performs four-point perspective transform."""
        rect = self.__order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

class ImageProcessor:
    @staticmethod
    def preprocess_image(image):
        """Preprocess image for OCR"""
        img_array = np.array(image)
        if len(img_array.shape) == 2:
            gray = img_array
        elif len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError("Unsupported image format")
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    @staticmethod
    def extract_text(image):
        """Extract text using EasyOCR"""
        try:
            processed_image = ImageProcessor.preprocess_image(image)
            results = reader.readtext(processed_image, detail=1)
            if not results:
                return None, "No text detected."
            structured_data = []
            for i, (bbox, text, confidence) in enumerate(results, 1):
                structured_data.append({
                    "Line": i,
                    "Text": text,
                    "Confidence": f"{confidence:.2%}"
                })
            raw_text = "\n".join([item[1] for item in results])
            return structured_data, raw_text
        except Exception as e:
            return None, f"Error during OCR: {str(e)}"

    @staticmethod
    def auto_rotate(image):
        """Auto-rotate image based on text orientation"""
        img_array = np.array(image)
        results = reader.readtext(img_array, detail=1)
        
        if not results:
            return image
            
        angles = []
        for (bbox, _, _) in results:
            (top_left, top_right, _, _) = bbox
            angle = np.arctan2(top_right[1] - top_left[1], top_right[0] - top_left[0]) * 180 / np.pi
            angles.append(angle)
        
        if not angles:
            return image
            
        avg_angle = np.mean(angles)
        
        if -135 <= avg_angle <= -45:
            return image.rotate(90, expand=True)
        elif 45 <= avg_angle <= 135:
            return image.rotate(-90, expand=True)
        elif avg_angle < -135 or avg_angle > 135:
            return image.rotate(180, expand=True)
        return image

def main():
    st.markdown('<div class="header">📝 Image Scanner & OCR</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #4a4a4a;">Process your images with background removal and text extraction</p>', 
                unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown('<div class="subheader">Upload Image</div>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Drop your image here or click to upload",
                type=["jpg", "jpeg", "png"],
                help="Supported formats: JPG, JPEG, PNG"
            )
            if uploaded_file:
                original_image = Image.open(uploaded_file)
                st.image(original_image, caption="Original Image")  # Removed use_container_width

        with col2:
            if uploaded_file:
                st.markdown('<div class="subheader">Processed Results</div>', unsafe_allow_html=True)
                if st.button("Process Image 🚀"):
                    with st.spinner("Processing image..."):
                        try:
                            # Use ImageScanner to scan the image
                            scanner = ImageScanner(original_image)
                            scanned_image = scanner.scan()
                            
                            # Auto-rotate the scanned image
                            scanned_image = ImageProcessor.auto_rotate(scanned_image)
                            
                            st.image(scanned_image, caption="Scanned Image")  # Removed use_container_width
                            
                            # Extract text
                            structured_data, raw_text = ImageProcessor.extract_text(scanned_image)
                            
                            st.markdown('<div class="subheader">Extracted Text</div>', unsafe_allow_html=True)
                            view_mode = st.radio("View as:", ["Formatted Table", "Raw Text"], 
                                               horizontal=True, label_visibility="collapsed")
                            
                            if view_mode == "Formatted Table" and structured_data:
                                df = pd.DataFrame(structured_data)
                                st.dataframe(df, use_container_width=True)
                            elif raw_text:
                                st.text_area("Text Output", raw_text, height=200)
                            
                            if raw_text and "Error" not in raw_text:
                                st.markdown('<div class="download-section">', unsafe_allow_html=True)
                                st.download_button(
                                    label="Download Text 📝",
                                    data=raw_text,
                                    file_name="extracted_text.txt",
                                    mime="text/plain"
                                )
                                cv2.imwrite("temp.jpg", np.array(scanned_image))
                                with open("temp.jpg", "rb") as img_file:
                                    pdf_bytes = img2pdf.convert(img_file.read())
                                st.download_button(
                                    label="Download PDF 📄",
                                    data=pdf_bytes,
                                    file_name="scanned_document.pdf",
                                    mime="application/pdf"
                                )
                                st.markdown('</div>', unsafe_allow_html=True)
                                os.remove("temp.jpg")
                        except Exception as e:
                            st.error(f"Error during processing: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("About This Tool")
        st.write("""
            Transform your images with:
            - Automatic background removal
            - Accurate text extraction
            - Multiple output formats
            - High-contrast scanning
            
            Perfect for digitizing documents, receipts, and more!
        """)
        st.markdown("---")
        st.subheader("Tips")
        st.write("""
            - Use clear, well-lit images
            - Ensure text is legible
            - Best results with rectangular documents
        """)
        st.markdown("---")
        st.write("Built with Streamlit & EasyOCR")

if __name__ == "__main__":
    main()