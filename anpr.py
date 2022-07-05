import cv2
import imutils
import numpy as np
import pytesseract


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img = cv2.imread(r'E:\master_thesis\project//numberplate_reco\data//15.jpg', cv2.IMREAD_COLOR)
img = cv2.resize(img, (600, 400))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 13, 15, 15)

edged = cv2.Canny(gray, 30, 200)
contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
screenCnt = None

for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    if len(approx) == 4:
        screenCnt = approx
        break
if screenCnt is None:
    detected = 0
    print("No contour detected")
else:
    detected = 1
if detected == 1:
    mask = np.zeros(img.shape, np.uint8)
    cv2.drawContours(mask, contours, -1, (255), 1)
    x, y, w, h = cv2.boundingRect(screenCnt)
    start = (x, y)
    end = (x+w, y + h)
    mask = np.zeros(img.shape[:2], dtype="uint8")
    cv2.rectangle(mask, start, end, 255, -1)
    masked = cv2.bitwise_and(img, img, mask=mask)
    m_gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    m_gray = cv2.bilateralFilter(masked, 13, 15, 15)
    (thresh, mm_gray) = cv2.threshold(m_gray, 130, 255, cv2.THRESH_BINARY)
    mm_gray[m_gray < 130] = 0
    mm_gray[m_gray > 130] = 255

    text = pytesseract.image_to_string(mm_gray)
    print(text)
    cv2.imshow("orignal", img)
    cv2.imshow("masked", mm_gray)
    cv2.waitKey(0)

    cv2.destroyAllWindows()