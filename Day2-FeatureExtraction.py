from img2vec_pytorch import Img2Vec
import os
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

################# Prepare Data #################

img2vec = Img2Vec()

data_dir = r"C:\Users\dell\PycharmProjects\ComputerVision_30Days\Resources\Weather-Analysis-main_split"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")

data = {}

for j, dir in enumerate([train_dir, val_dir]):
    features = []
    labels = []
    for category in os.listdir(dir):
        for img_path in os.listdir(os.path.join(dir, category)):
            img_path = os.path.join(dir, category, img_path)
            img = Image.open(img_path)

            img_features = img2vec.get_vec(img)
            features.append(img_features)
            labels.append(category)
        data[['training_data', 'validation_data'][j]] = features
        data[['training_label', 'validation_label'][j]] = labels

################# Train Model #################

model = RandomForestClassifier()
model.fit(data['training_data'], data['training_label'])

################# Test Performance #################

y_pred = model.predict(data['validation_data'])
accuracy = accuracy_score(y_pred, data['validation_label'])

print(accuracy)

################# Save the Model #################

pickle.dump(model, open("C:\\Users\\dell\\PycharmProjects\\ComputerVision_30Days\\Classifier.pkl", 'wb'))