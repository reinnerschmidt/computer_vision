# importar bibliotecas necessárias
import dlib
import cv2
import imutils
from imutils import face_utils
import time

vs = cv2.VideoCapture(0)
time.sleep(2.0)

if not vs.isOpened():
    print("Erro ao abrir a câmera.")
    exit()

while True:
    ret, frame = vs.read()
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detector de face
    # carregar o modelo do predictor (68 pontos)
    detector_face = dlib.get_frontal_face_detector()
    predictor_points = dlib.shape_predictor(
        "shape_predictor_68_face_landmarks.dat")
    rects = detector_face(cinza, 1)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for (i, rect) in enumerate(rects):
        # dentro dessa área (face detectada) faça previsão das landmarks
        pontos_referencia = predictor_points(cinza, rect)
        pontos_referencia = face_utils.shape_to_np(pontos_referencia)

    # coordenadas do retângulo no formato que o openCV reconhece
    # desenhar o retângulo para cada face
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        for (x, y) in pontos_referencia:
            cv2.circle(frame, (x, y), 1, [0, 255, 0], -1)

        cv2.putText(frame, f"Face {i+1}", (x, y-220), font, 1, (0, 255, 0), 1)
    cv2.imshow('Resultado', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
vs.stop()
cv2.destroyAllWindows()
