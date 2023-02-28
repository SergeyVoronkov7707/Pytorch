from loader import MTCNN, torch, mp
from ResNet import ResNet
from web_management import FaceDetector


mtcnn = MTCNN()
device = torch
model = ResNet(1,5)
model.load_state_dict(torch.load('./models/gesture_detection_model_state_20_epochs.pth'))

# ourResNet = FERModel(1, 7).to(device)
# ourResNet.load_state_dict(torch.load('./models/model2_50_epochs.pth'))


model.eval()
# Создаем объект нашего класса приложения
fcd = FaceDetector(mtcnn, mp, model)

# Запускаем
fcd.run()