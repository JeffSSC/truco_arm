import cv2
from ultralytics import YOLO

# --- CARREGUE SEU MODELO ---
try:
    model = YOLO('best.pt')
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    print("Verifique se o arquivo 'best.pt' está no local correto.")
    exit()

# --- ABRA A WEBCAM ---
# Tente alterar o índice se 0 não funcionar. Comuns são 0, 1, ou -1.
camera_index = 1
cap = cv2.VideoCapture(camera_index)

# VERIFICAÇÃO 1: A câmera foi aberta com sucesso?
if not cap.isOpened():
    print(f"Erro: Não foi possível abrir a webcam com índice {camera_index}.")
    print("Tente alterar o valor de 'camera_index' para 1, 2, etc.")
    exit()

print("Webcam aberta. Pressione 'q' na janela de vídeo para sair.")

# --- LOOP DE DETECÇÃO ---
while True:
    # Captura um quadro (imagem) da webcam
    ret, frame = cap.read()

    # VERIFICAÇÃO 2 (A MAIS IMPORTANTE): O quadro foi capturado?
    # Se 'ret' for False, significa que a captura falhou.
    if not ret:
        print("Erro: Não foi possível capturar o quadro da webcam. Encerrando.")
        break # Sai do loop se não conseguir ler o quadro

    # Se o quadro foi lido com sucesso, 'frame' é uma imagem válida.
    # Agora podemos fazer a inferência com segurança.
    results = model(frame, conf=0.5)
    annotated_frame = results[0].plot()

    # Exibe o quadro (agora sabemos que 'annotated_frame' é uma imagem válida)
    cv2.imshow('Detecção em Tempo Real', annotated_frame)

    # Condição de parada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- LIBERA OS RECURSOS ---
print("Fechando...")
cap.release()
cv2.destroyAllWindows()