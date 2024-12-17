import pytesseract
import cv2

class TextExtraction:
    def __init__(self):
        pass
    def extract_image_info(self,image_path):

        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(thresh)
        
        return " ".join(text.split())
extract = TextExtraction()
path = "paracetamol-tablet.jpg"
print(extract.extract_image_info(path))
