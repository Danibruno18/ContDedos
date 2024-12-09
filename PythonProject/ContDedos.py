import cv2
import numpy as np
import math

def contar_dedos(frame, contorno):
    hull = cv2.convexHull(contorno, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(contorno, hull)

        if defects is not None:
            dedos_erguidos = 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contorno[s][0])
                end = tuple(contorno[e][0])
                far = tuple(contorno[f][0])

                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((far[0] - end[0]) ** 2 + (far[1] - end[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))

                if angle <= math.pi / 2 and d > 20:
                    dedos_erguidos += 1

            return dedos_erguidos + 1
    return 0

def main():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        roi = frame[100:400, 100:400]

        cv2.rectangle(frame, (100, 100), (400, 400), (255, 0, 0), 2)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(max_contour) > 1000:
                cv2.drawContours(roi, [max_contour], -1, (0, 255, 0), 2)
                dedos = contar_dedos(frame, max_contour)
                cv2.putText(frame, f"Dedos erguidos: {dedos}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)
        cv2.imshow("Thresh", thresh)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
