import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# import optuna
# from optuna.trial import TrialState
import time

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_and_prepare_excel(file_path):
    df = pd.read_excel(file_path, engine="openpyxl")

    df.rename(columns={
        df.columns[0]: "Datum",
        df.columns[1]: "Intraday"
    }, inplace=True)

    df["Datum"] = pd.to_datetime(df["Datum"], errors='coerce')
    df.dropna(subset=["Datum"], inplace=True)

    df.set_index("Datum", inplace=True)
    df.sort_index(inplace=True)
    df.dropna(inplace=True)

    return df

def create_sequences_multifeature(X, y, window_size=24, horizon=24):
    X_seqs, y_seqs = [], []
    for i in range(len(X) - window_size - horizon + 1):
        X_window = X[i: i + window_size]
        y_future = y[i + window_size + horizon - 1]
        X_seqs.append(X_window)
        y_seqs.append(y_future)
    return np.array(X_seqs), np.array(y_seqs)

class TimeSeriesDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = torch.tensor(X_data, dtype=torch.float32)
        self.y_data = torch.tensor(y_data, dtype=torch.float32)

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        return self.X_data[idx], self.y_data[idx]

# def objective(trial):
#     start_time = time.time()

#     # Hyperparametern mit erweiterten Bereichen
#     hidden_size = trial.suggest_int('hidden_size', 32, 512)
#     num_layers = trial.suggest_int('num_layers', 1, 5)
#     dropout = trial.suggest_float('dropout', 0.1, 0.7, step=0.05)
#     learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)
#     batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024, 2048])

#     # Definiere das Modell mit den vorgeschlagenen Hyperparametern
#     class LSTMModel(nn.Module):
#         def __init__(self, input_dim, hidden_size, num_layers, dropout):
#             super(LSTMModel, self).__init__()
#             self.lstm = nn.LSTM(
#                 input_size=input_dim,
#                 hidden_size=hidden_size,
#                 num_layers=num_layers,
#                 batch_first=True,
#                 dropout=dropout if num_layers > 1 else 0
#             )
#             self.fc = nn.Linear(hidden_size, 1)

#         def forward(self, x):
#             out, _ = self.lstm(x)
#             last_out = out[:, -1, :]
#             out = self.fc(last_out)
#             return out

#     # Initialisiere das Modell
#     input_dim = X_train_scaled.shape[2]
#     model = LSTMModel(input_dim, hidden_size, num_layers, dropout)

#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     # Erstelle Datasets und DataLoader mit dem vorgeschlagenen Batch Size
#     train_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled)
#     val_dataset = TimeSeriesDataset(X_val_scaled, y_val_scaled)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

#     EPOCHS = 50
#     PATIENCE = 5

#     best_val_loss = float('inf')
#     early_stop_counter = 0

#     for epoch in range(EPOCHS):
#         # ------ TRAIN ------
#         model.train()
#         running_train_loss = 0.0

#         for X_batch, y_batch in train_loader:
#             X_batch = X_batch.to(device)
#             y_batch = y_batch.to(device)

#             optimizer.zero_grad()
#             preds = model(X_batch).squeeze()
#             loss = criterion(preds, y_batch)
#             loss.backward()
#             optimizer.step()

#             running_train_loss += loss.item()

#         epoch_train_loss = running_train_loss / len(train_loader)

#         # ------ VAL ------
#         model.eval()
#         running_val_loss = 0.0
#         with torch.no_grad():
#             for X_val_batch, y_val_batch in val_loader:
#                 X_val_batch = X_val_batch.to(device)
#                 y_val_batch = y_val_batch.to(device)

#                 val_preds = model(X_val_batch).squeeze()
#                 val_loss = criterion(val_preds, y_val_batch)

#                 running_val_loss += val_loss.item()

#         epoch_val_loss = running_val_loss / len(val_loader)

#         # Early Stopping Logik
#         if epoch_val_loss < best_val_loss:
#             best_val_loss = epoch_val_loss
#             early_stop_counter = 0
#         else:
#             early_stop_counter += 1
#             if early_stop_counter > PATIENCE:
#                 break

#         # Pruning
#         trial.report(epoch_val_loss, epoch)
#         if trial.should_prune():
#             raise optuna.exceptions.TrialPruned()

#     duration = time.time() - start_time
#     print(f"Trial {trial.number} abgeschlossen in {duration:.2f} Sekunden mit Val Loss: {best_val_loss:.4f}")

#     return best_val_loss


if __name__ == "__main__":
    set_seed(42)

    file_path = "1Daten 2020 - 2024.xlsx"

    df = read_and_prepare_excel(file_path)
    print("DF shape:", df.shape)
    print("Datum min:", df.index.min(), "Datum max:", df.index.max())

    if 2024 in df.index.year.unique():
        df_2024 = df.loc["2024-01-01":"2024-12-31"]
        print("Daten fÃ¼r 2024 gefunden.")
    else:
        df_2024 = pd.DataFrame()
        print("Keine Daten fÃ¼r 2024 gefunden.")

    df_2020 = df.loc["2020-01-01":"2020-12-31"]
    df_2021 = df.loc["2021-01-01":"2021-12-31"]
    df_2022 = df.loc["2022-01-01":"2022-12-31"]
    df_2023 = df.loc["2023-01-01":"2023-12-31"]

    if not df_2024.empty:
        df_train = pd.concat([df_2020, df_2021, df_2022])
        df_val = df_2023
        df_test = df_2024
    else:
        df_train = pd.concat([df_2020, df_2021])
        df_val = df_2022
        df_test = df_2023

    def split_Xy(df_in):
        y_ = df_in["Intraday"].values
        X_ = df_in.drop(columns=["Intraday"]).values
        return X_, y_

    X_train_raw, y_train_raw = split_Xy(df_train)
    X_val_raw, y_val_raw = split_Xy(df_val)
    if not df_test.empty:
        X_test_raw, y_test_raw = split_Xy(df_test)
    else:
        X_test_raw, y_test_raw = np.array([]), np.array([])

    WINDOW_SIZE = 24
    HORIZON = 24

    X_train_raw, y_train_raw = create_sequences_multifeature(X_train_raw, y_train_raw, WINDOW_SIZE, HORIZON)
    X_val_raw, y_val_raw = create_sequences_multifeature(X_val_raw, y_val_raw, WINDOW_SIZE, HORIZON)
    if len(X_test_raw) > 0 and len(y_test_raw) > 0:
        X_test_raw, y_test_raw = create_sequences_multifeature(X_test_raw, y_test_raw, WINDOW_SIZE, HORIZON)
    else:
        X_test_raw, y_test_raw = np.array([]), np.array([])

    if not df_test.empty:
        test_dates = df_test.index[WINDOW_SIZE + HORIZON - 1:]
    else:
        test_dates = pd.DatetimeIndex([])

    print("Test dates shape:", test_dates.shape)

    print("Train shape:", X_train_raw.shape, y_train_raw.shape)
    print("Val shape:  ", X_val_raw.shape, y_val_raw.shape)
    print("Test shape: ", X_test_raw.shape, y_test_raw.shape)

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_train_2d = X_train_raw.reshape(-1, X_train_raw.shape[2])
    scaler_X.fit(X_train_2d)

    X_train_scaled = scaler_X.transform(X_train_2d).reshape(X_train_raw.shape)
    X_val_scaled = scaler_X.transform(X_val_raw.reshape(-1, X_val_raw.shape[2])).reshape(X_val_raw.shape)
    if len(X_test_raw) > 0:
        X_test_scaled = scaler_X.transform(X_test_raw.reshape(-1, X_test_raw.shape[2])).reshape(X_test_raw.shape)
    else:
        X_test_scaled = np.array([]).reshape(0, WINDOW_SIZE, X_train_raw.shape[2])

    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_train_2d = y_train_raw.reshape(-1, 1)
    scaler_y.fit(y_train_2d)

    y_train_scaled = scaler_y.transform(y_train_2d).flatten()
    y_val_scaled = scaler_y.transform(y_val_raw.reshape(-1, 1)).flatten()
    if len(y_test_raw) > 0:
        y_test_scaled = scaler_y.transform(y_test_raw.reshape(-1, 1)).flatten()
    else:
        y_test_scaled = np.array([])

    # 6) Optuna-Studie erstellen und optimieren
    # study = optuna.create_study(
    #     direction='minimize',
    #     pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    # )
    # study.optimize(objective, n_trials=100, timeout=7200)  # ErhÃ¶hte n_trials und timeout

    # Ergebnisse anzeigen
    # print("Number of finished trials: ", len(study.trials))
    # print("Best trial:")
    # trial = study.best_trial

    # print("  Value (Best Validation Loss): ", trial.value)
    # print("  Best hyperparameters: ")
    # for key, value in trial.params.items():
    #     print(f"    {key}: {value}")

    # Optional: Visualisierungen
    # try:
    #     fig1 = optuna.visualization.plot_optimization_history(study)
    #     fig1.show()

    #     fig2 = optuna.visualization.plot_param_importances(study)
    #     fig2.show()
    # except Exception as e:
    #     print(f"Fehler bei der Visualisierung: {e}")

    # 7) Training des finalen Modells mit festen Hyperparametern

    # Feste Hyperparameter
    fixed_hidden_size = 511
    fixed_num_layers = 1
    fixed_dropout = 0.55
    fixed_learning_rate = 0.00830
    fixed_batch_size = 256

    print("Verwende feste Hyperparameter:")
    print(f"  hidden_size: {fixed_hidden_size}")
    print(f"  num_layers: {fixed_num_layers}")
    print(f"  dropout: {fixed_dropout}")
    print(f"  learning_rate: {fixed_learning_rate}")
    print(f"  batch_size: {fixed_batch_size}")

    class FinalLSTMModel(nn.Module):
        def __init__(self, input_dim, hidden_size, num_layers, dropout):
            super(FinalLSTMModel, self).__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            last_out = out[:, -1, :]
            out = self.fc(last_out)
            return out

    final_model = FinalLSTMModel(
        input_dim=X_train_scaled.shape[2],
        hidden_size=fixed_hidden_size,
        num_layers=fixed_num_layers,
        dropout=fixed_dropout
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_model.to(device)

    final_criterion = nn.MSELoss()
    final_optimizer = torch.optim.Adam(
        final_model.parameters(),
        lr=fixed_learning_rate,
        weight_decay=1e-5
    )

    final_train_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled)
    final_val_dataset = TimeSeriesDataset(X_val_scaled, y_val_scaled)
    if len(X_test_scaled) > 0:
        final_test_dataset = TimeSeriesDataset(X_test_scaled, y_test_scaled)
    else:
        final_test_dataset = TimeSeriesDataset(
            np.array([]).reshape(0, WINDOW_SIZE, X_train_scaled.shape[2]),
            np.array([])
        )

    final_train_loader = DataLoader(
        final_train_dataset,
        batch_size=fixed_batch_size,
        shuffle=True,
        drop_last=True
    )
    final_val_loader = DataLoader(
        final_val_dataset,
        batch_size=fixed_batch_size,
        shuffle=False,
        drop_last=True
    )
    if len(X_test_scaled) > 0:
        final_test_loader = DataLoader(
            final_test_dataset,
            batch_size=fixed_batch_size,
            shuffle=False,
            drop_last=True
        )
    else:
        final_test_loader = DataLoader(
            final_test_dataset,
            batch_size=fixed_batch_size,
            shuffle=False,
            drop_last=True
        )

    EPOCHS = 100
    PATIENCE = 10
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_path = "best_final_model.pth"

    for epoch in range(EPOCHS):
        final_model.train()
        running_train_loss = 0.0

        for X_batch, y_batch in final_train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            final_optimizer.zero_grad()
            preds = final_model(X_batch).squeeze()
            loss = final_criterion(preds, y_batch)
            loss.backward()
            final_optimizer.step()

            running_train_loss += loss.item()

        epoch_train_loss = running_train_loss / len(final_train_loader)

        final_model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X_val_batch, y_val_batch in final_val_loader:
                X_val_batch = X_val_batch.to(device)
                y_val_batch = y_val_batch.to(device)

                val_preds = final_model(X_val_batch).squeeze()
                val_loss = final_criterion(val_preds, y_val_batch)

                running_val_loss += val_loss.item()

        epoch_val_loss = running_val_loss / len(final_val_loader)

        print(f"Epoch [{epoch + 1}/{EPOCHS}] Train MSE: {epoch_train_loss:.4f}, Val MSE: {epoch_val_loss:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            early_stop_counter = 0
            torch.save(final_model.state_dict(), best_model_path)
        else:
            early_stop_counter += 1
            if early_stop_counter > PATIENCE:
                print("Early stopping triggered!")
                break

    final_model.load_state_dict(torch.load(best_model_path))
    final_model.eval()

    if not final_test_loader.dataset.X_data.size(0) == 0:
        test_loss = 0.0
        test_preds = []
        test_targets = []

        with torch.no_grad():
            for X_test_batch, y_test_batch in final_test_loader:
                X_test_batch = X_test_batch.to(device)
                y_test_batch = y_test_batch.to(device)

                preds_test = final_model(X_test_batch).squeeze()
                test_loss += final_criterion(preds_test, y_test_batch).item()

                test_preds.append(preds_test.cpu().numpy())
                test_targets.append(y_test_batch.cpu().numpy())

        test_loss /= len(final_test_loader)
        print(f"\nFinal Test MSE: {test_loss:.4f}")

        test_preds = np.concatenate(test_preds)
        test_targets = np.concatenate(test_targets)

        # Inverse Transform der Testdaten & Plot der ersten 200
        test_preds_2d = test_preds.reshape(-1, 1)
        test_targets_2d = test_targets.reshape(-1, 1)

        preds_rescaled = scaler_y.inverse_transform(test_preds_2d).flatten()
        targets_rescaled = scaler_y.inverse_transform(test_targets_2d).flatten()

        plt.figure(figsize=(10, 4))
        plt.plot(targets_rescaled[:200], label='True Intraday (Test 200h)')
        plt.plot(preds_rescaled[:200], label='Predicted Intraday (Test 200h)')
        if not df_test.empty:
            plt.title("Erste 200 Stunden im Januar (2024)")
        else:
            plt.title("Erste 200 Stunden im Januar (2023)")
        plt.xlabel("Zeitschritt (Stunden)")
        plt.ylabel("Intraday-Wert (ent-skaliert)")
        plt.legend()
        plt.show()

        if not df_test.empty:
            start_date = pd.Timestamp('2024-04-01')
        else:
            start_date = pd.Timestamp('2023-04-01')

        start_idx = np.where(test_dates >= start_date)[0]
        if len(start_idx) == 0:
            print(f"Kein Datenpunkt ab dem {start_date.date()} gefunden.")
        else:
            start_idx = start_idx[0]
            end_idx = start_idx + 200

            if end_idx > len(preds_rescaled):
                end_idx = len(preds_rescaled)
                print(f"Es gibt nur {end_idx - start_idx} Datenpunkte ab dem {start_date.date()}.")

            preds_april = preds_rescaled[start_idx:end_idx]
            targets_april = targets_rescaled[start_idx:end_idx]

            april_dates = test_dates[start_idx:end_idx]

            plt.figure(figsize=(10, 4))
            plt.plot(targets_april[:200], label='True Intraday (200h ab 01.04.)')
            plt.plot(preds_april[:200], label='Predicted Intraday (200h ab 01.04.)')
            if not df_test.empty:
                plt.title("Erste 200 Stunden im April (2024)")
            else:
                plt.title("Erste 200 Stunden im April (2023)")
            plt.xlabel("Zeitschritt (Stunden)")
            plt.ylabel("Intraday-Wert (ent-skaliert)")
            plt.legend()
            plt.show()
    else:
        print("Keine Testdaten vorhanden. Der Plot wird Ã¼bersprungen.")
