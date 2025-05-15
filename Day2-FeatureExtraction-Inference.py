from img2vec_pytorch import Img2Vec
from PIL import Image
import pickle

with open(r"C:\Users\dell\PycharmProjects\ComputerVision_30Days\Classifier.pkl", "rb") as f:
    model = pickle.load(f)
img2vec = Img2Vec()

img_path = r"C:\Users\dell\PycharmProjects\ComputerVision_30Days\Resources\Weather-Analysis-main_split\val\rain\rain_10.jpeg"

img = Image.open(img_path)

feature = img2vec.get_vec(img)

pred = model.predict([feature])

print(pred)