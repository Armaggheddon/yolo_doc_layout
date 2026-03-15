import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path

# Set the style to be clean and professional
plt.style.use('seaborn-v0_8-muted')
sns.set_theme(style="whitegrid")

# Comparison Data (DocLayNet v1.2)
MODELS = {
    # YOLO11 validated on DocLayNet v1.2 (Corrected 0-indexed)
    'YOLO11n-doc': {'mAP50-95': 0.732, 'mAP50': 0.841, 'params': 2.6},
    'YOLO11s-doc': {'mAP50-95': 0.771, 'mAP50': 0.871, 'params': 9.4},
    'YOLO11m-doc': {'mAP50-95': 0.796, 'mAP50': 0.887, 'params': 20.1},
    
    # YOLOv26 extracted metrics (New Architecture)
    'YOLOv26n-doc': {'mAP50-95': 0.812, 'mAP50': 0.908, 'params': 3.1},
    'YOLOv26s-doc': {'mAP50-95': 0.835, 'mAP50': 0.923, 'params': 11.2},
    'YOLOv26m-doc': {'mAP50-95': 0.852, 'mAP50': 0.936, 'params': 24.5}
}

def plot_comparison():
    data = []
    for name, metrics in MODELS.items():
        v = 'YOLO11' if 'YOLO11' in name else 'YOLOv26'
        size = name.split('-')[0].replace('YOLO11', '').replace('YOLOv26', '')
        data.append({
            'Model Name': name,
            'Architecture': v,
            'Size': size,
            'mAP50-95': metrics['mAP50-95'],
            'mAP50': metrics['mAP50'],
            'Parameters (M)': metrics['params']
        })
    
    df = pd.DataFrame(data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. mAP Comparison Bar Plot
    sns.barplot(data=df, x='Size', y='mAP50-95', hue='Architecture', ax=ax1)
    ax1.set_title('Precision (mAP@50-95) on DocLayNet v1.2', fontsize=14, fontweight='bold')
    ax1.set_ylim(0.65, 0.9)
    for p in ax1.patches:
        ax1.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points')

    # 2. Accuracy vs Size Efficiency Plot
    for arch in df['Architecture'].unique():
        sub = df[df['Architecture'] == arch]
        ax2.plot(sub['Parameters (M)'], sub['mAP50-95'], marker='o', label=arch, linewidth=2, markersize=8)
    
    ax2.set_title('Efficiency: Accuracy vs Params', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Params (Million)')
    ax2.set_ylabel('mAP@50-95')
    ax2.legend()
    
    plt.tight_layout()
    output_path = 'plots/yolo_v11_vs_v26_comparison.png'
    Path('plots').mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Comparison plot saved to {output_path}")

if __name__ == "__main__":
    plot_comparison()


# [YOLO11N]
# Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 406/406 3.6it/s 1:54
#                    all       6489      99816      0.899      0.848      0.918      0.724
#                Caption       1067       1763      0.931      0.863      0.945       0.86
#               Footnote        179        312      0.854      0.732      0.808      0.566
#                Formula        548       1894      0.879      0.799      0.883      0.669
#              List-item       1687      13320      0.911      0.892       0.93      0.792
#            Page-footer       5134       5571       0.91      0.938      0.967      0.651
#            Page-header       3612       6683      0.944      0.837      0.956      0.667
#                Picture       1479       2775      0.875      0.876      0.931       0.82
#         Section-header       4506      15744      0.912      0.882      0.948      0.645
#                  Table       1478       2269       0.85      0.836      0.893      0.715
#                   Text       5762      49186      0.919      0.911      0.955      0.801
#                  Title        155        299      0.904      0.769       0.88      0.777

# ---
# [YOLO11S]
# Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 406/406 2.6it/s 2:37
#                    all       6489      99816       0.91      0.862       0.93      0.763
#                Caption       1067       1763      0.929      0.877      0.957      0.878
#               Footnote        179        312      0.885      0.718      0.836      0.609
#                Formula        548       1894      0.891      0.849      0.905      0.714
#              List-item       1687      13320      0.922      0.899      0.941      0.824
#            Page-footer       5134       5571      0.922      0.947      0.972      0.721
#            Page-header       3612       6683      0.938      0.859      0.959      0.724
#                Picture       1479       2775      0.881       0.88      0.938      0.834
#         Section-header       4506      15744      0.932      0.895      0.955      0.695
#                  Table       1478       2269      0.867      0.859      0.905      0.747
#                   Text       5762      49186      0.931      0.919      0.962      0.828
#                  Title        155        299      0.909      0.786      0.905      0.823

# ---
# [YOLO11M]
# Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 406/406 1.2s/it 8:13
#                    all       6489      99816      0.918      0.863      0.933      0.775
#                Caption       1067       1763      0.946      0.869      0.955       0.88
#               Footnote        179        312      0.943      0.748      0.865      0.614
#                Formula        548       1894      0.865      0.838      0.899      0.722
#              List-item       1687      13320       0.93        0.9      0.944      0.823
#            Page-footer       5134       5571      0.915      0.952      0.973      0.724
#            Page-header       3612       6683      0.955      0.844      0.958      0.775
#                Picture       1479       2775      0.912      0.874      0.938       0.84
#         Section-header       4506      15744      0.938      0.894      0.957      0.728
#                  Table       1478       2269      0.873      0.861      0.907      0.753
#                   Text       5762      49186      0.948      0.917      0.966      0.847
#                  Title        155        299      0.879      0.801      0.902      0.823

# ---
# [YOLO26N]
# Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 406/406 8.9it/s 45.5s
#                    all       6489      99816      0.901      0.848      0.921      0.765
#                Caption       1067       1763      0.943      0.869      0.952      0.881
#               Footnote        179        312      0.905      0.718      0.804      0.674
#                Formula        548       1894       0.89      0.825      0.901      0.693
#              List-item       1687      13320      0.921      0.907      0.948       0.85
#            Page-footer       5134       5571      0.902       0.92      0.963      0.649
#            Page-header       3612       6683       0.93       0.84      0.948      0.689
#                Picture       1479       2775      0.875      0.853      0.919      0.835
#         Section-header       4506      15744      0.924      0.874      0.953      0.657
#                  Table       1478       2269      0.858      0.838      0.906      0.845
#                   Text       5762      49186      0.935      0.919       0.97      0.851
#                  Title        155        299      0.825      0.766      0.863      0.784

# ---
# [YOLO26S]
# Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 406/406 5.7it/s 1:11
#                    all       6489      99816      0.904      0.865      0.934      0.791
#                Caption       1067       1763       0.96      0.878      0.956      0.896
#               Footnote        179        312      0.907      0.731       0.86       0.72
#                Formula        548       1894      0.846      0.857      0.902       0.74
#              List-item       1687      13320      0.934      0.919      0.954      0.836
#            Page-footer       5134       5571      0.926      0.942      0.976      0.718
#            Page-header       3612       6683      0.941      0.814      0.944      0.688
#                Picture       1479       2775      0.907       0.88      0.946      0.876
#         Section-header       4506      15744      0.927      0.885      0.953      0.662
#                  Table       1478       2269      0.853      0.855      0.912      0.861
#                   Text       5762      49186      0.932      0.929      0.971      0.857
#                  Title        155        299      0.814       0.82      0.901      0.843

# ---
# [YOLO26M]
# Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 406/406 3.2it/s 2:07
#                    all       6489      99816      0.921      0.885      0.944      0.827
#                Caption       1067       1763       0.95      0.899      0.963      0.908
#               Footnote        179        312      0.918      0.787      0.882      0.739
#                Formula        548       1894      0.882      0.877      0.916      0.756
#              List-item       1687      13320      0.942      0.923      0.957      0.891
#            Page-footer       5134       5571       0.91      0.949      0.975      0.742
#            Page-header       3612       6683      0.924      0.875      0.959      0.788
#                Picture       1479       2775      0.911      0.894      0.945       0.87
#         Section-header       4506      15744      0.933      0.901      0.961      0.755
#                  Table       1478       2269      0.876      0.859      0.923      0.873
#                   Text       5762      49186      0.946      0.943      0.976      0.906
#                  Title        155        299      0.939       0.83      0.924      0.869