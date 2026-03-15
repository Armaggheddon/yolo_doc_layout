import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train YOLO on DocLayNet 1.2")
    parser.add_argument("--model", type=str, default="yolo26n.pt", help="YOLO version (e.g. yolo11n.pt, yolo26n.pt)")
    parser.add_argument("--data", type=str, default="/home/user/doc_layout/doclaynet_v1.2.yaml", help="Path to yaml config")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=1280,
        batch=128,
        save=True,
        device=[0, 1, 2, 3],
        close_mosaic=20,
        patience=5,
    )

if __name__ == "__main__":
    main()

# YOLO26n summary (fused): 122 layers, 2,376,981 parameters, 0 gradients, 5.2 GFLOPs
#                  Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 3.3it/s 8.0s
#                    all       6489      99816      0.901      0.848      0.921      0.761
#                Caption       1067       1763      0.944       0.87      0.953      0.873
#               Footnote        179        312      0.897      0.718      0.805      0.659
#                Formula        548       1894       0.89      0.823      0.901      0.692
#              List-item       1687      13320      0.921      0.906      0.948      0.848
#            Page-footer       5134       5571      0.903      0.919      0.963      0.641
#            Page-header       3612       6683      0.932      0.839      0.948      0.688
#                Picture       1479       2775      0.878      0.852      0.919      0.835
#         Section-header       4506      15744      0.925      0.874      0.953      0.656
#                  Table       1478       2269       0.86      0.839      0.906      0.845
#                   Text       5762      49186      0.935      0.919       0.97       0.85
#                  Title        155        299      0.827      0.766      0.863      0.786

# YOLO26s summary (fused): 122 layers, 9,469,437 parameters, 0 gradients, 20.5 GFLOPs
#                  Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 26/26 2.4it/s 11.0s
#                    all       6489      99816      0.903      0.866      0.934      0.786
#                Caption       1067       1763       0.96      0.879      0.957      0.888
#               Footnote        179        312      0.901      0.731      0.859      0.704
#                Formula        548       1894      0.845       0.86      0.902       0.74
#              List-item       1687      13320      0.934      0.921      0.954      0.834
#            Page-footer       5134       5571      0.925      0.943      0.976      0.704
#            Page-header       3612       6683       0.94      0.816      0.944      0.686
#                Picture       1479       2775      0.906      0.881      0.946      0.875
#         Section-header       4506      15744      0.926      0.886      0.953       0.66
#                  Table       1478       2269      0.852      0.855      0.912       0.86
#                   Text       5762      49186      0.931       0.93      0.971      0.856
#                  Title        155        299      0.813      0.823      0.901      0.842

# YOLO26m summary (fused): 132 layers, 20,357,933 parameters, 0 gradients, 67.9 GFLOPs
#                  Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 51/51 2.8it/s 18.4s
#                    all       6489      99816      0.921      0.885      0.944      0.824
#                Caption       1067       1763      0.949      0.899      0.964        0.9
#               Footnote        179        312      0.918      0.786      0.882      0.731
#                Formula        548       1894      0.882      0.877      0.916      0.755
#              List-item       1687      13320      0.941      0.923      0.958       0.89
#            Page-footer       5134       5571      0.909       0.95      0.975      0.728
#            Page-header       3612       6683      0.924      0.875      0.959      0.786
#                Picture       1479       2775       0.91      0.893      0.945       0.87
#         Section-header       4506      15744      0.932      0.901      0.961      0.753
#                  Table       1478       2269      0.876       0.86      0.923      0.873
#                   Text       5762      49186      0.946      0.943      0.976      0.905
#                  Title        155        299      0.939      0.827      0.924      0.868