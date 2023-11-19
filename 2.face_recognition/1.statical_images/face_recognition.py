# importar bibliotecas necessárias
import dlib
import cv2
import imutils
from imutils import face_utils

# detector de face
# carregar o modelo do predictor (68 pontos)
detector_face = dlib.get_frontal_face_detector()
predictor_points = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat")
imagem_entrada = cv2.imread("teste_1.jpg")
imagem_entrada = cv2.resize(imagem_entrada, (810, 1080))
imagem_cinza = cv2.cvtColor(imagem_entrada, cv2.COLOR_BGR2GRAY)
rects = detector_face(imagem_cinza, 1)
font = cv2.FONT_HERSHEY_SIMPLEX
print(rects)

for (i, rect) in enumerate(rects):
    # dentro dessa área (face detectada) faça previsão das landmarks
    pontos_referencia = predictor_points(imagem_cinza, rect)
    pontos_referencia = face_utils.shape_to_np(pontos_referencia)

    # coordenadas do retângulo no formato que o openCV reconhece
    # desenhar o retângulo para cada face
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(imagem_entrada, (x, y), (x+w, y+h), (0, 255, 0), 2)

    for (x, y) in pontos_referencia:
        cv2.circle(imagem_entrada, (x, y), 1, [0, 255, 0], -1)

    cv2.putText(imagem_entrada, f"Face {i+1}",
                (x, y-220), font, 1, (0, 255, 0), 1)


cv2.imshow('Resultado', imagem_entrada)

cv2.waitKey(0)
cv2.destroyAllWindows()
