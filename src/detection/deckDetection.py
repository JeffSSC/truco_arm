import cv2
from ultralytics import YOLO

class DeckDetection:
    def __init__(self):
        pass

    def seeDeck(self):
        model = YOLO("best.pt")
        cap = cv2.VideoCapture(1)

        if not cap.isOpened():
            return "Erro: Não foi possível abrir a webcam."

        print("Aquecendo a câmera...")
        # 1. SOLUÇÃO DE AQUECIMENTO: Lê e descarta os 10 primeiros frames
        for _ in range(10):
            cap.read()

        print("Capturando imagem...")
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return "Erro: Falha ao capturar a foto."

        # 2. SALVANDO A IMAGEM PARA DEBUG
        cv2.imwrite("imagem_capturada_debug.jpg", frame)
        print("Imagem de debug 'imagem_capturada_debug.jpg' salva.")

        print("Analisando a imagem com o modelo YOLO...")
        # 3. SOLUÇÃO DE CONFIANÇA: Adicionado conf=0.1 para detectar mais coisas
        results = model(frame, verbose=False)
        
        cartas_detectadas = []

        # O 'results' é uma lista, então precisamos pegar o primeiro item
        # que contém os resultados da nossa imagem.
        result = results[0] 

        for box in result.boxes:
            class_name = result.names[int(box.cls[0].item())]
            xywh = box.xywh[0].tolist()
            
            cartas_detectadas.append({
                "numero_str": class_name,
                "x": xywh[0]
            })

        cartas_detectadas.sort(key=lambda c: c['x'])
        
        if not cartas_detectadas:
            print("AVISO: Nenhuma carta foi detectada na imagem.")
        else:
            print(f"Detectado {len(cartas_detectadas)} cartas.")
            
        return cartas_detectadas

# --- Execução do Código ---
if __name__ == '__main__':
    detector = DeckDetection()
    resultado = detector.seeDeck()
    print(f"Resultado final: {resultado}")