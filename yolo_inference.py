from ultralytics import YOLO

model = YOLO('models/x1/best.pt')

results = model.predict('input_videos/08fd33_4.mp4', save=True)
print(results[0])
print('======================================================')
for boxes in results[0].boxes:
    print(boxes)
