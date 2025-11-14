import os
import time
from dataclasses import dataclass
from itertools import product
from typing import Dict
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import clear_output
from pandas import DataFrame
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from sovia.data_preparation.utils import get_path_to_data
from sovia.model.image_loader import ImageLoader


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_net):
        super().__init__()
        self.embedding_net = embedding_net  # For example, a simple CNN

    def forward(self, x1, x2):
        out1 = self.embedding_net(x1)
        out2 = self.embedding_net(x2)
        return out1, out2


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss = (label) * torch.pow(euclidean_distance, 2) + \
               (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        return loss.mean()


@dataclass
class TrainingConfig:
    num_epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-4
    distance_threshold: float = 0.5
    rotate = True
    flip = True
    start_from_checkpoint: bool = True
    num_workers: int = max(1, min(8, (os.cpu_count() or 4) - 1))
    prefetch_factor: int = 4  # prefetch queues of images


class SiameseDataset(Dataset):
    def __init__(self, data: DataFrame, config: TrainingConfig, is_training=False):
        """
        image_pairs: list of (img_path_1, img_path_2)
        labels: list of int (1=similar, 0=dissimilar)
        """
        self.config = config
        if is_training and config.rotate:
            rotations = [0, 90, 180, 270]
        else:
            rotations = [0]
        if is_training and config.flip:
            flips = [False, True]
        else:
            flips = [False]
        kombinationen = pd.DataFrame((product(rotations, flips)), columns=["rotation", "flip"])
        image_data = data[["oi", "year_1", "year_2", "link_1", "link_2", "geom", "label"]]
        self.image_data = image_data.join(kombinationen, how='cross')
        self.labels = self.image_data["label"].tolist()
        self.polygons = data["geom"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_loader = ImageLoader()
        data_row = self.image_data.iloc[idx]
        img1_4ch, img2_4ch = img_loader.load(data_row["oi"], data_row["year_1"], data_row["link_1"], data_row["year_2"],
                                             data_row["link_2"], data_row["geom"])
        flip = data_row['flip']
        rotate = data_row['rotation']
        if bool(flip):
            img1_4ch = torch.flip(img1_4ch, dims=[-1])
            img2_4ch = torch.flip(img2_4ch, dims=[-1])
        img1_4ch = torch.rot90(img1_4ch, int(rotate / 90), dims=[-2, -1])
        img2_4ch = torch.rot90(img2_4ch, int(rotate / 90), dims=[-2, -1])
        label = torch.tensor(float(self.labels[idx]), dtype=torch.float32)
        return img1_4ch, img2_4ch, label


class MetricsCalculator:
    @staticmethod
    def compute_metrics(preds: List[int], targets: List[int]) -> Dict[str, float]:
        tp = sum((p == 1 and t == 1) for p, t in zip(preds, targets))
        fp = sum((p == 1 and t == 0) for p, t in zip(preds, targets))
        fn = sum((p == 0 and t == 1) for p, t in zip(preds, targets))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


class TrainingMetricsTracker:
    def __init__(self):
        self.train_loss_history: List[float] = []
        self.valid_loss_history: List[float] = []
        self.train_precision_history: List[float] = []
        self.valid_precision_history: List[float] = []
        self.train_recall_history: List[float] = []
        self.valid_recall_history: List[float] = []
        self.train_f1_history: List[float] = []
        self.valid_f1_history: List[float] = []

    def update_metrics(self, train_metrics: dict, valid_metrics: dict) -> None:
        self.train_loss_history.append(train_metrics['loss'])
        self.valid_loss_history.append(valid_metrics['loss'])
        self.train_precision_history.append(train_metrics['precision'])
        self.valid_precision_history.append(valid_metrics['precision'])
        self.train_recall_history.append(train_metrics['recall'])
        self.valid_recall_history.append(valid_metrics['recall'])
        self.train_f1_history.append(train_metrics['f1'])
        self.valid_f1_history.append(valid_metrics['f1'])


class TrainingVisualizer:
    @staticmethod
    def plot_metrics(epoch: int, metrics_tracker: TrainingMetricsTracker) -> None:
        plt.style.use('dark_background')
        clear_output(wait=True)

        fig = plt.figure(figsize=(14, 10))
        axes = fig.subplots(2, 2)

        metrics_config = [
            ('Loss', 'loss', metrics_tracker.train_loss_history, metrics_tracker.valid_loss_history),
            ('Precision', 'precision', metrics_tracker.train_precision_history,
             metrics_tracker.valid_precision_history),
            ('Recall', 'recall', metrics_tracker.train_recall_history, metrics_tracker.valid_recall_history),
            ('F1 Score', 'f1', metrics_tracker.train_f1_history, metrics_tracker.valid_f1_history)
        ]
        for (title, metric, train_hist, valid_hist), ax in zip(metrics_config, axes.flat):
            ax.plot(range(1, epoch + 1), train_hist, label=f"Train {title}", color="cyan")
            ax.plot(range(1, epoch + 1), valid_hist, label=f"Valid {title}", color="yellow")
            ax.set_title(f"{title} Over Epochs", color="white")
            ax.set_xlabel("Epochs", color="white")
            ax.set_ylabel(title, color="white")
            ax.legend()
            ax.grid(color="gray")

        plt.tight_layout()
        plt.show()
        plt.close(fig)  # Explicitly close the figure


class SimpleEmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 64, 5)  # Input: 3 channels (RGB)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(self.conv1.out_channels, 128, 5)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        adaptive_avg_pool_output_size = self.adaptive_pool.output_size[0] * self.adaptive_pool.output_size[1]
        self.fc1 = nn.Linear(self.conv2.out_channels * adaptive_avg_pool_output_size, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, 128)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.adaptive_pool(x)  # Ensure output spatial size is 4x4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def load_labeled_data(labelstudio_csv_path: str) -> Tuple[DataFrame, DataFrame]:
    df = pd.read_csv(labelstudio_csv_path)
    label_1 = df.loc[df['label'] == 1]
    haus_im_bau = df.loc[(df['haus_im_bau'] & (df['label'] == 0))]
    hat_sloar = df.loc[(df['hat_solar'] & (df['label'] == 0))]
    dach_gereinigt = df.loc[(df['dach_gereinigt'] & (df['label'] == 0))]
    dach_nicht_erkennbar = df.loc[(~df['dach_erkennbar'] & (df['label'] == 0))]
    rest = df.loc[
        ((df['label'] == 0) & df['dach_erkennbar'] & ~df['dach_gereinigt']) & ~df['hat_solar'] & ~df['haus_im_bau']]
    trainings_dfs = []
    validation_dfs = []
    for sub_df in [label_1, haus_im_bau, hat_sloar, dach_gereinigt, dach_nicht_erkennbar, rest]:
        df_shuffled = sub_df.sample(frac=1)
        df_splits = np.array_split(df_shuffled, 4)
        trainings_dfs.append(pd.DataFrame(df_splits[0]))
        trainings_dfs.append(pd.DataFrame(df_splits[1]))
        trainings_dfs.append(pd.DataFrame(df_splits[2]))
        validation_dfs.append(pd.DataFrame(df_splits[3]))
    train_set = pd.concat(trainings_dfs)
    validate_set = pd.concat(validation_dfs)
    return train_set, validate_set


class SiameseTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.metrics_tracker = TrainingMetricsTracker()
        self.visualizer = TrainingVisualizer()
        self.metrics_calculator = MetricsCalculator()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.checkpoint_path = get_path_to_data(__file__) / f"input/trained_models/checkpoint.pth"
        print(torch.__version__)

        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Current device: {torch.cuda.current_device()}" if torch.cuda.is_available() else "No CUDA device")
        print(f"Device count: {torch.cuda.device_count()}" if torch.cuda.is_available() else "No CUDA devices")
        print(torch.cuda.get_device_name(0))  # Should print your GPU name
        embedding_net = SimpleEmbeddingNet()  # Your own architecture here
        self.model = SiameseNetwork(embedding_net).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=1e-5)

    def train(self, train_data: DataFrame, validate_data: DataFrame) -> None:
        train_dataset = SiameseDataset(train_data, self.config, True)
        valid_dataset = SiameseDataset(validate_data, self.config)

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.config.batch_size,
                                  shuffle=False) if valid_dataset else None
        last_saved_f1 = 0
        criterion = ContrastiveLoss()
        start_epoch = self._load_checkpoint()
        for epoch in range(start_epoch, self.config.num_epochs):
            train_metrics = self._train_epoch(train_loader, criterion, self.optimizer)
            if valid_loader:
                valid_metrics = self._eval_epoch(valid_loader, criterion)
            else:
                valid_metrics = {'loss': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            self.metrics_tracker.update_metrics(train_metrics, valid_metrics)
            self.visualizer.plot_metrics(epoch + 1, self.metrics_tracker)
            self._log_progress(epoch, train_metrics, valid_metrics)
            if valid_metrics["f1"] > last_saved_f1:
                self._save_checkpoint(epoch + 1)
                last_saved_f1 = valid_metrics["f1"]

    def _save_checkpoint(self, epoch):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'metrics_tracker': {
                'train_loss_history': self.metrics_tracker.train_loss_history,
                'valid_loss_history': self.metrics_tracker.valid_loss_history,
                'train_precision_history': self.metrics_tracker.train_precision_history,
                'valid_precision_history': self.metrics_tracker.valid_precision_history,
                'train_recall_history': self.metrics_tracker.train_recall_history,
                'valid_recall_history': self.metrics_tracker.valid_recall_history,
                'train_f1_history': self.metrics_tracker.train_f1_history,
                'valid_f1_history': self.metrics_tracker.valid_f1_history,
            },
            'config': self.config  # falls nötig, speichere die Konfiguration
        }
        # Speicherpfad und Name des Modells
        torch.save(checkpoint, self.checkpoint_path)
        print(f"Zwischenspeicherung abgeschlossen: {self.checkpoint_path}")

    def _load_checkpoint(self):
        if self.checkpoint_path.exists() and self.config.start_from_checkpoint:
            # Lade den Checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

            # Wiederherstellen des Modellzustands
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # Wiederherstellen des Optimierungszustands
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Wiederherstellen der Metriken
            self.metrics_tracker.train_loss_history = checkpoint['metrics_tracker']['train_loss_history']
            self.metrics_tracker.valid_loss_history = checkpoint['metrics_tracker']['valid_loss_history']
            self.metrics_tracker.train_precision_history = checkpoint['metrics_tracker']['train_precision_history']
            self.metrics_tracker.valid_precision_history = checkpoint['metrics_tracker']['valid_precision_history']
            self.metrics_tracker.train_recall_history = checkpoint['metrics_tracker']['train_recall_history']
            self.metrics_tracker.valid_recall_history = checkpoint['metrics_tracker']['valid_recall_history']
            self.metrics_tracker.train_f1_history = checkpoint['metrics_tracker']['train_f1_history']
            self.metrics_tracker.valid_f1_history = checkpoint['metrics_tracker']['valid_f1_history']

            # Setzen der Epoche
            start_epoch = checkpoint['epoch']

            print(f"Checkpoint erfolgreich geladen. Fortsetzung ab Epoche {start_epoch}.")
            return start_epoch
        else:
            print(f"Checkpoint {self.checkpoint_path} nicht gefunden.")
            return 0

    def _train_epoch(self, loader, criterion, optimizer):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        for img1_batch, img2_batch, label in loader:
            img1_batch, img2_batch, label = img1_batch.to(self.device), img2_batch.to(self.device), label.to(
                self.device)
            output1, output2 = self.model(img1_batch, img2_batch)
            loss = criterion(output1, output2, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * img1_batch.size(0)
            dist = torch.nn.functional.pairwise_distance(output1, output2)
            pred = (dist < self.config.distance_threshold).long()
            all_preds.extend(pred.cpu().numpy().tolist())
            all_targets.extend(label.cpu().numpy().tolist())
        avg_loss = running_loss / len(loader.dataset)
        metrics = self.metrics_calculator.compute_metrics(all_preds, all_targets)
        metrics['loss'] = avg_loss
        return metrics

    def _eval_epoch(self, loader, criterion):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for img1, img2, label in loader:
                img1, img2, label = img1.to(self.device), img2.to(self.device), label.to(self.device)
                output1, output2 = self.model(img1, img2)
                loss = criterion(output1, output2, label)
                running_loss += loss.item() * img1.size(0)
                pred = (torch.nn.functional.pairwise_distance(output1, output2) < self.config.distance_threshold).long()
                all_preds.extend(pred.cpu().numpy().tolist())
                all_targets.extend(label.cpu().numpy().tolist())
        avg_loss = running_loss / len(loader.dataset)
        metrics = self.metrics_calculator.compute_metrics(all_preds, all_targets)
        metrics['loss'] = avg_loss
        return metrics

    @staticmethod
    def _log_progress(epoch: int, train_metrics: dict, valid_metrics: dict) -> None:
        print(
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} | "
            f"Epoch {epoch + 1} | "
            f"Train - Loss: {train_metrics['loss']:.4f} | Precision: {train_metrics['precision']:.4f} | "
            f"Recall: {train_metrics['recall']:.4f} | F1: {train_metrics['f1']:.4f} || "
            f"Valid - Loss: {valid_metrics['loss']:.4f} | Precision: {valid_metrics['precision']:.4f} | "
            f"Recall: {valid_metrics['recall']:.4f} | F1: {valid_metrics['f1']:.4f}"
        )


def main():
    load_weights = False
    save_weights = True
    weight_name = "second_training"

    config = TrainingConfig()
    trainer = SiameseTrainer(config)
    if load_weights:
        trainer.model.load_state_dict(
            torch.load(get_path_to_data(__file__) / f"input/trained_models/{weight_name}.pth"))
    train_df, validate_df = load_labeled_data(
        get_path_to_data(__file__) / f"input/training_data/{weight_name}.csv")

    trainer.train(train_df, validate_df)
    if save_weights:
        torch.save(trainer.model.state_dict(),
                   get_path_to_data(__file__) / f"input/trained_models/{weight_name}_weights.pth")


if __name__ == "__main__":
    main()
