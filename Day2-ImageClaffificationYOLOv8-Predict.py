from ultralytics import YOLO

model = YOLO('C:\\Users\\dell\\PycharmProjects\\ComputerVision_30Days\\runs\\classify\\train3\\weights\\last.pt')
result = model(r"C:\Users\dell\PycharmProjects\ComputerVision_30Days\Resources\Weather-Analysis-main_split\train\cloud\cloud_0.jpeg")

names_dict = result[0].names

probs = result[0].probs.data.tolist()

print(probs)
print(names_dict)