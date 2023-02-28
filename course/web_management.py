from loader import cv2, torch, np, Image



# Класс детектирования и обработки лица с веб-камеры
class FaceDetector(object):

    def __init__(self, mtcnn, mp, resnet, channels=1):
        # Создаем объект для считывания потока с веб-камеры(обычно вебкамера идет под номером 0. иногда 1)
        self.cap = cv2.VideoCapture(0)
        self.mtcnn = mtcnn
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.emodel = resnet
        self.channels = channels
        self.mp = mp

    # Функция рисования найденных параметров на кадре
    def _draw(self, frame, boxes, probs, landmarks):
        try:
            for box, prob, ld in zip(boxes, probs, landmarks):
                # Рисуем обрамляющий прямоугольник лица на кадре
                cv2.rectangle(frame,
                              (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])),
                              (0, 0, 255),
                              thickness=2)

                # пишем на кадре какая эмоция распознана
        #                 cv2.putText(frame,
        #                     emotion, (int(box[2]), int(box[3])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Рисуем особенные точки
        #                 cv2.circle(frame, (int(ld[0][0]),int(ld[0][1])), 5, (0, 0, 255), -1)
        #                 cv2.circle(frame, (int(ld[1][0]),int(ld[1][1])), 5, (0, 0, 255), -1)
        #                 cv2.circle(frame, (int(ld[2][0]),int(ld[2][1])), 5, (0, 0, 255), -1)
        #                 cv2.circle(frame, (int(ld[3][0]),int(ld[3][1])), 5, (0, 0, 255), -1)
        #                 cv2.circle(frame, (int(ld[4][0]),int(ld[4][1])), 5, (0, 0, 255), -1)
        except Exception as e:
            print('Something wrong im draw function!')
            print(f'error : {e}')

        return frame

    # Функция для вырезания лиц с кадра
    @staticmethod
    def crop_faces(frame, boxes):
        faces = []
        for i, box in enumerate(boxes):
            faces.append(frame[int(box[1] - 40):int(box[3] + 40),
                         int(box[0] - 40):int(box[2] + 40)])
        #             print(box)
        return faces

    @staticmethod
    def crop_hands(frame, hand_boxes):
        hands = []
        for i, hand_box in enumerate(hand_boxes):
            hands.append(frame[int(hand_box[1] - 60):int(hand_box[3] + 60),
                         int(hand_box[0] - 60):int(hand_box[2] + 60)])
        return hands

    # {0:'down', 1:'fist', 2:'ok', 3:'palm', 4:'thumb'}
    @staticmethod
    def digit_to_classname(digit):
        if digit == 0:
            return 'down'
        elif digit == 1:
            return 'fist'
        elif digit == 2:
            return 'ok'
        elif digit == 3:
            return 'palm'
        elif digit == 4:
            return 'thumb'

    @staticmethod
    def remove_background(frame):
        bgModel = cv2.createBackgroundSubtractorMOG2(0, 50)
        fgmask = bgModel.apply(frame, learningRate=0)
        kernel = np.ones((3, 3), np.uint8)
        fgmask = cv2.erode(fgmask, kernel, iterations=1)
        res = cv2.bitwise_and(frame, frame, mask=fgmask)
        return res

        # Функция в которой будет происходить процесс считывания и обработки каждого кадра

    def run(self):

        blurValue = 5  # GaussianBlur parameter

        mp_drawing = self.mp.solutions.drawing_utils
        mp_hands = self.mp.solutions.hands
        with mp_hands.Hands(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
            # Заходим в бесконечный цикл
            while True:
                # Считываем каждый новый кадр - frame
                # ret - логическая переменая. Смысл - считали ли мы кадр с потока или нет
                ret, frame = self.cap.read()
                h, w, c = frame.shape
                try:
                    # детектируем расположение лица на кадре, вероятности на сколько это лицо
                    # и особенные точки лица
                    boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)

                    #                 # Вырезаем лицо из кадра
                    face = self.crop_faces(frame, boxes)[0]

                    # Рисуем на кадре
                    self._draw(frame, boxes, probs, landmarks)

                    #                     img = self.remove_background(frame)
                    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                    # To improve performance, optionally mark the image as not writeable to
                    # pass by reference.
                    frame.flags.writeable = False
                    results = hands.process(frame)

                    # Draw the hand annotations on the image.
                    frame.flags.writeable = True
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    hand_landmarks = results.multi_hand_landmarks
                    if hand_landmarks:
                        hand_boxes = []
                        for handLMs in hand_landmarks:
                            #                             hand_box = []
                            x_max = 0
                            y_max = 0
                            x_min = w
                            y_min = h

                            for lm in handLMs.landmark:
                                #
                                x, y = int(lm.x * w), int(lm.y * h)
                                if x > x_max:
                                    x_max = x
                                if x < x_min:
                                    x_min = x
                                if y > y_max:
                                    y_max = y
                                if y < y_min:
                                    y_min = y
                        #                             hand_box.append(x_min, y_min, x_max, y_max)
                        #                             hand_boxes.append(hand_box)
                        #                         print(handLMs)
                        #                         hand_cv = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        hand_box = [x_min, y_min, x_max, y_max]
                        hand_boxes.append(hand_box)

                        #                         mp_drawing.draw_landmarks(frame, handLMs, mp_hands.HAND_CONNECTIONS)
                        # Вырезаем руку с кадра
                        hand = self.crop_hands(frame, hand_boxes)[0]

                        # Меняем размер изображения лица для входа в нейронную сеть
                        hand_img = cv2.resize(hand, (71, 64))

                        hand = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
                        # Превращаем в 1-канальное серое изображение
                        hand = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
                        hand = cv2.GaussianBlur(hand, (blurValue, blurValue), 0)
                        # Превращаем в 1-канальное черно-белое изображение
                        (thresh, hand) = cv2.threshold(hand, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                        cv2.imshow('bwhand', hand)

                        # Далее мы подготавливаем наш кадр для считывания нс
                        # Для этого перегоним его в формат pil_image
                        hand = Image.fromarray(hand)
                        # face = face.resize((48,48))
                        hand = np.asarray(hand).astype('float')
                        hand = torch.as_tensor(hand)

                        # Превращаем numpy-картинку вырезанного лица в pytorch-тензор
                        torch_hand = hand.unsqueeze(0).to(self.device).float()
                        # Загужаем наш тензор лица в нейронную сеть и получаем предсказание
                        emotion = self.emodel(torch_hand[None, ...])
                        print(emotion[0])
                        # Интерпретируем предсказание как строку нашей эмоции
                        emotion[0][3] = emotion[0][3] / 1000
                        emotion = self.digit_to_classname(emotion[0].argmax().item())
                        hand_cv = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv2.putText(frame, emotion,
                                    (x_max, y_max), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)




                except Exception as e:
                    print('Something wrong im main cycle!')
                    print(f'error : {e}')

                # Показываем кадр в окне, и назвываем его(окно) - 'Face Detection'
                cv2.imshow('Hands Detection', frame)

                # Функция, которая проверяет нажатие на клавишу 'q'
                # Если нажатие произошло - выход из цикла. Конец работы приложения
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Очищаем все объекты opencv, что мы создали
        self.cap.release()
        cv2.destroyAllWindows()