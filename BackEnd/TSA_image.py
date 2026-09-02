import cv2 as cv
import pytesseract

def extract(image_path: str) -> str:
    img = cv.imread(image_path)

    if img is None:
        return ""

    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(grey, (5, 5), 0)
    _, thres = cv.threshold(
        blur,
        0,
        255,
        cv.THRESH_BINARY + cv.THRESH_OTSU
    )

    text = pytesseract.image_to_string(thres, config="--psm 6")
    return text.strip()