import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

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

    df["Datum"] = pd.to_datetime(df["Datum"])
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

class LSTMModel(nn.Module):
    def __init__(self, input_dim):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=128,
            num_layers=2,          # Erhöht die Anzahl der Schichten
            batch_first=True,
            dropout=0.3            # Dropout zwischen den LSTM-Schichten
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        out, _ = self.lstm(x)       # => [batch, seq_len, 128]
        last_out = out[:, -1, :]    # => [batch, 128]
        out = self.fc(last_out)     # => [batch, 1]
        return out

def train_and_validate(
        X_train_scaled, y_train_scaled,
        X_val_scaled, y_val_scaled,
        X_test_scaled, y_test_scaled,
        EPOCHS=50, LEARNING_RATE=0.0001, PATIENCE=5, BATCH_SIZE=512
):

    train_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled)
    val_dataset = TimeSeriesDataset(X_val_scaled, y_val_scaled)
    test_dataset = TimeSeriesDataset(X_test_scaled, y_test_scaled)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    input_dim = X_train_scaled.shape[2]
    model = LSTMModel(input_dim=input_dim)

    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []

    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        running_train_loss = 0.0
        running_train_mae = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch).squeeze()  # => [batch]
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            running_train_mae += mae_metric(preds, y_batch).item()

        epoch_train_loss = running_train_loss / len(train_loader)
        epoch_train_mae = running_train_mae / len(train_loader)

        model.eval()
        running_val_loss = 0.0
        running_val_mae = 0.0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch = X_val_batch.to(device)
                y_val_batch = y_val_batch.to(device)

                val_preds = model(X_val_batch).squeeze()
                val_loss = criterion(val_preds, y_val_batch)

                running_val_loss += val_loss.item()
                running_val_mae += mae_metric(val_preds, y_val_batch).item()

        epoch_val_loss = running_val_loss / len(val_loader)
        epoch_val_mae = running_val_mae / len(val_loader)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_maes.append(epoch_train_mae)
        val_maes.append(epoch_val_mae)

        print(f"Epoch [{epoch + 1}/{EPOCHS}] "
              f"Train MSE: {epoch_train_loss:.4f}, "
              f"Val MSE: {epoch_val_loss:.4f}, "
              f"Train MAE: {epoch_train_mae:.4f}, "
              f"Val MAE: {epoch_val_mae:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            early_stop_counter += 1
            if early_stop_counter > PATIENCE:
                print("Early stopping triggered!")
                break

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train MSE')
    plt.plot(val_losses, label='Val MSE')
    plt.title("MSE Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_maes, label='Train MAE')
    plt.plot(val_maes, label='Val MAE')
    plt.title("MAE over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()

    plt.tight_layout()
    plt.show()

    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    test_loss = 0.0
    test_mae = 0.0
    test_preds = []
    test_targets = []

    with torch.no_grad():
        for X_test_batch, y_test_batch in test_loader:
            X_test_batch = X_test_batch.to(device)
            y_test_batch = y_test_batch.to(device)

            preds_test = model(X_test_batch).squeeze()
            test_loss_mse = criterion(preds_test, y_test_batch).item()
            test_loss_mae = mae_metric(preds_test, y_test_batch).item()

            test_loss += test_loss_mse
            test_mae += test_loss_mae

            test_preds.append(preds_test.cpu().numpy())
            test_targets.append(y_test_batch.cpu().numpy())

    test_loss /= len(test_loader)
    test_mae /= len(test_loader)

    print(f"\nFinal Test MSE: {test_loss:.4f}")
    print(f"Final Test MAE: {test_mae:.4f}")

    test_preds = np.concatenate(test_preds)
    test_targets = np.concatenate(test_targets)

    return model, test_preds, test_targets

if __name__ == "__main__":

    set_seed(42)

    file_path = "10Daten 2020 - 2023.xlsx"

    df = read_and_prepare_excel(file_path)
    print("DF shape:", df.shape)
    print("Datum min:", df.index.min(), "Datum max:", df.index.max())

    df_2020 = df.loc["2020-01-01":"2020-12-31"]
    df_2021 = df.loc["2021-01-01":"2021-12-31"]
    df_2022 = df.loc["2022-01-01":"2022-12-31"]
    df_2023 = df.loc["2023-01-01":"2023-12-31"]

    df_train = pd.concat([df_2020, df_2021])
    df_val = df_2022
    df_test = df_2023


    def split_Xy(df_in):
        y_ = df_in["Intraday"].values
        X_ = df_in.drop(columns=["Intraday"]).values
        return X_, y_


    X_train_raw, y_train_raw = split_Xy(df_train)
    X_val_raw, y_val_raw = split_Xy(df_val)
    X_test_raw, y_test_raw = split_Xy(df_test)

    WINDOW_SIZE = 24
    HORIZON = 24

    X_train_raw, y_train_raw = create_sequences_multifeature(X_train_raw, y_train_raw, WINDOW_SIZE, HORIZON)
    X_val_raw, y_val_raw = create_sequences_multifeature(X_val_raw, y_val_raw, WINDOW_SIZE, HORIZON)
    X_test_raw, y_test_raw = create_sequences_multifeature(X_test_raw, y_test_raw, WINDOW_SIZE, HORIZON)

    test_dates = df_test.index[WINDOW_SIZE + HORIZON - 1:]
    print("Test dates shape:", test_dates.shape)

    print("Train shape:", X_train_raw.shape, y_train_raw.shape)
    print("Val shape:  ", X_val_raw.shape, y_val_raw.shape)
    print("Test shape: ", X_test_raw.shape, y_test_raw.shape)

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_train_2d = X_train_raw.reshape(-1, X_train_raw.shape[2])
    scaler_X.fit(X_train_2d)

    X_train_scaled = scaler_X.transform(X_train_2d).reshape(X_train_raw.shape)
    X_val_scaled = scaler_X.transform(X_val_raw.reshape(-1, X_val_raw.shape[2])).reshape(X_val_raw.shape)
    X_test_scaled = scaler_X.transform(X_test_raw.reshape(-1, X_test_raw.shape[2])).reshape(X_test_raw.shape)

    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_train_2d = y_train_raw.reshape(-1, 1)
    scaler_y.fit(y_train_2d)

    y_train_scaled = scaler_y.transform(y_train_2d).flatten()
    y_val_scaled = scaler_y.transform(y_val_raw.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test_raw.reshape(-1, 1)).flatten()

    model, test_preds, test_targets = train_and_validate(
        X_train_scaled, y_train_scaled,
        X_val_scaled, y_val_scaled,
        X_test_scaled, y_test_scaled,
        EPOCHS=50, LEARNING_RATE=0.0005, PATIENCE=5, BATCH_SIZE=1024
    )

    test_preds_2d = test_preds.reshape(-1, 1)
    test_targets_2d = test_targets.reshape(-1, 1)

    preds_rescaled = scaler_y.inverse_transform(test_preds_2d).flatten()
    targets_rescaled = scaler_y.inverse_transform(test_targets_2d).flatten()

    plt.figure(figsize=(10, 4))
    plt.plot(targets_rescaled[:200], label='True Intraday (Test 200h)')
    plt.plot(preds_rescaled[:200], label='Predicted Intraday (Test 200h)')
    plt.title("Erste 200 Stunden im Januar (2023)")
    plt.xlabel("Zeitschritt (Stunden)")
    plt.ylabel("Intraday-Wert (ent-skaliert)")
    plt.legend()
    plt.show()

    start_date = pd.Timestamp('2023-04-01')

    start_idx = np.where(test_dates >= start_date)[0]
    if len(start_idx) == 0:
        print("Kein Datenpunkt ab dem 01.04.2023 gefunden.")
    else:
        start_idx = start_idx[0]
        end_idx = start_idx + 200

        if end_idx > len(preds_rescaled):
            end_idx = len(preds_rescaled)
            print(f"Es gibt nur {end_idx - start_idx} Datenpunkte ab dem 01.04.2023.")

        preds_april = preds_rescaled[start_idx:end_idx]
        targets_april = targets_rescaled[start_idx:end_idx]

        april_dates = test_dates[start_idx:end_idx]

        plt.figure(figsize=(10, 4))
        plt.plot(targets_april[:200], label='True Intraday (200h ab 01.04.2023)')
        plt.plot(preds_april[:200], label='Predicted Intraday (200h ab 01.04.2023)')
        plt.title("Erste 200 Stunden im April (2023)")
        plt.xlabel("Zeitschritt (Stunden)")
        plt.ylabel("Intraday-Wert (ent-skaliert)")
        plt.legend()
        plt.show()


