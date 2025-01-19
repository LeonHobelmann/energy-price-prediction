import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import GridSearchCV  # Optional für Hyperparameter-Tuning
import warnings

import time

warnings.filterwarnings("ignore")

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

    df = df.asfreq('H')

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

def split_train_val_test_sarima(df_train, df_val, df_test):

    sarima_train = pd.concat([df_train, df_val])
    sarima_test = df_test
    return sarima_train, sarima_test


def fit_sarima(train_series, order=(1, 0, 1), seasonal_order=(1, 0, 1, 48)):

    model = SARIMAX(train_series,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    sarima_result = model.fit(disp=False)
    return sarima_result


def forecast_sarima(sarima_result, steps):

    forecast = sarima_result.get_forecast(steps=steps)
    forecast_values = forecast.predicted_mean
    return forecast_values


def evaluate_forecast(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    return mse, mae


def plot_forecast(test_dates, actual, sarima_pred, lstm_pred=None, rf_pred=None, title="Intraday Price Prediction"):
    plt.figure(figsize=(15, 6))
    plt.plot(test_dates[:200], actual[:200], label='True Intraday (First 200h)', color='blue')
    plt.plot(test_dates[:200], sarima_pred[:200], label='SARIMA Prediction (First 200h)', color='green')

    if lstm_pred is not None:
        plt.plot(test_dates[:200], lstm_pred[:200], label='LSTM Prediction (First 200h)', color='orange')

    if rf_pred is not None:
        plt.plot(test_dates[:200], rf_pred[:200], label='Random Forest Prediction (First 200h)', color='red')

    plt.title(title)
    plt.xlabel("Datum und Uhrzeit")
    plt.ylabel("Intraday-Preis")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    start_time = time.time()
    set_seed(42)

    file_path = "1Daten 2020 - 2024.xlsx"  # Aktualisiert

    df = read_and_prepare_excel(file_path)
    print("DF shape:", df.shape)
    print("Datum min:", df.index.min(), "Datum max:", df.index.max())

    if 2024 in df.index.year.unique():
        df_2024 = df.loc["2024-01-01":"2024-12-31"]
        print("Daten für 2024 gefunden.")
    else:
        df_2024 = pd.DataFrame()
        print("Keine Daten für 2024 gefunden.")

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

    print("\nImplementierung des Random Forest-Modells...")

    X_train_rf = X_train_scaled.reshape(X_train_scaled.shape[0], -1)
    X_val_rf = X_val_scaled.reshape(X_val_scaled.shape[0], -1)
    if len(X_test_scaled) > 0:
        X_test_rf = X_test_scaled.reshape(X_test_scaled.shape[0], -1)
    else:
        X_test_rf = np.array([]).reshape(0, WINDOW_SIZE * X_train_scaled.shape[2])

    y_train_rf = y_train_scaled
    y_val_rf = y_val_scaled
    y_test_rf = y_test_scaled

    rf = RandomForestRegressor(
        random_state=42,
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        bootstrap=True,
        n_jobs=-1
    )

    # Optional: Hyperparameter-Tuning mit GridSearchCV
    """
    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [10, 15],
        'min_samples_split': [10, 15],
        'min_samples_leaf': [4, 6],
        'bootstrap': [True]
    }

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_rf, y_train_rf)
    rf_best = grid_search.best_estimator_
    print(f"Beste Random Forest Hyperparameter: {grid_search.best_params_}")
    rf = rf_best
    """


    rf.fit(X_train_rf, y_train_rf)
    print("Random Forest-Modell erfolgreich trainiert.")


    if len(X_test_rf) > 0:
        rf_preds_scaled = rf.predict(X_test_rf)
        mse_rf, mae_rf = evaluate_forecast(y_test_rf, rf_preds_scaled)
        print(f"\nRandom Forest Evaluierung:")
        print(f"Mean Squared Error (MSE): {mse_rf:.4f}")
        print(f"Mean Absolute Error (MAE): {mae_rf:.4f}")

        rf_preds_2d = rf_preds_scaled.reshape(-1, 1)
        rf_preds_rescaled = scaler_y.inverse_transform(rf_preds_2d).flatten()
    else:
        rf_preds_rescaled = np.array([])


    print("\nFitting des SARIMA-Modells...")

    sarima_train, sarima_test = split_train_val_test_sarima(df_train, df_val, df_test)

    sarima_order = (1, 0, 1)
    sarima_seasonal_order = (1, 0, 1, 48)

    sarima_result = fit_sarima(sarima_train['Intraday'], order=sarima_order, seasonal_order=sarima_seasonal_order)
    print("SARIMA-Modell erfolgreich gefittet.")
    print(sarima_result.summary())

    steps = len(sarima_test)
    try:
        sarima_forecast = forecast_sarima(sarima_result, steps)
        print(f"SARIMA-Vorhersagen für {steps} Stunden erzeugt.")
    except ValueError as e:
        print(f"Fehler bei der Vorhersage mit SARIMA: {e}")
        sarima_forecast = pd.Series([np.nan]*steps, index=sarima_test.index)

    mse_sarima, mae_sarima = evaluate_forecast(sarima_test['Intraday'], sarima_forecast)
    print(f"\nSARIMA Evaluierung:")
    print(f"Mean Squared Error (MSE): {mse_sarima:.4f}")
    print(f"Mean Absolute Error (MAE): {mae_sarima:.4f}")


    sarima_preds_rescaled = sarima_forecast.values

    fixed_hidden_size = 511
    fixed_num_layers = 1
    fixed_dropout = 0.55
    fixed_learning_rate = 0.00830
    fixed_batch_size = 256

    print("\nVerwende feste Hyperparameter für LSTM:")
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
            drop_last=False
        )
    else:
        final_test_loader = DataLoader(
            final_test_dataset,
            batch_size=fixed_batch_size,
            shuffle=False,
            drop_last=False
        )

    EPOCHS = 100
    PATIENCE = 10
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_path = "best_final_model.pth"

    print("\nBeginne mit dem Training des LSTM-Modells...")

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
                loss = final_criterion(preds_test, y_test_batch)
                test_loss += loss.item()

                test_preds.append(preds_test.cpu().numpy())
                test_targets.append(y_test_batch.cpu().numpy())

        test_loss /= len(final_test_loader)
        print(f"\nFinal Test MSE (LSTM): {test_loss:.4f}")

        test_preds = np.concatenate(test_preds)
        test_targets = np.concatenate(test_targets)

        test_preds_2d = test_preds.reshape(-1, 1)
        test_targets_2d = test_targets.reshape(-1, 1)

        preds_rescaled = scaler_y.inverse_transform(test_preds_2d).flatten()
        targets_rescaled = scaler_y.inverse_transform(test_targets_2d).flatten()

        mae_lstm = mean_absolute_error(targets_rescaled, preds_rescaled)
        print(f"Final Test MAE (LSTM): {mae_lstm:.4f}")

        mae_lstm_scaled = mean_absolute_error(y_test_scaled, test_preds)
        print(f"Final Test MAE (LSTM, skaliert): {mae_lstm_scaled:.4f}")

        print(f"Beispielhafte Vorhersagen (unskaliert): {preds_rescaled[:5]}")
        print(f"Beispielhafte Ziele (unskaliert): {targets_rescaled[:5]}")

        plt.figure(figsize=(10, 4))
        plt.plot(preds_rescaled[:100], label='Predicted LSTM', color='orange')
        plt.plot(targets_rescaled[:100], label='True Values', color='blue')
        plt.legend()
        plt.title("Vergleich der LSTM-Vorhersagen und tatsächlichen Werte (erste 100 Stunden)")
        plt.show()

        errors = np.abs(targets_rescaled - preds_rescaled)
        plt.hist(errors, bins=50, color='purple')
        plt.title("Verteilung der MAE-Fehler")
        plt.xlabel("Absoluter Fehler")
        plt.ylabel("Häufigkeit")
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(targets_rescaled[:200], label='True Intraday (Test 200h)', color='blue')
        plt.plot(preds_rescaled[:200], label='Predicted Intraday (Test 200h) - LSTM', color='orange')
        plt.plot(sarima_preds_rescaled[:200], label='Predicted Intraday (Test 200h) - SARIMA', color='green')
        if len(rf_preds_rescaled) > 0:
            plt.plot(rf_preds_rescaled[:200], label='Predicted Intraday (Test 200h) - Random Forest', color='red')
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

        start_idx = np.where(sarima_test.index >= start_date)[0]
        if len(start_idx) == 0:
            print(f"Kein Datenpunkt ab dem {start_date.date()} gefunden.")
        else:
            start_idx = start_idx[0]
            end_idx = start_idx + 200

            if end_idx > len(preds_rescaled):
                end_idx = len(preds_rescaled)
                print(f"Es gibt nur {end_idx - start_idx} Datenpunkte ab dem {start_date.date()}.")
            if end_idx > len(sarima_preds_rescaled):
                end_idx = len(sarima_preds_rescaled)
                print(f"Es gibt nur {end_idx - start_idx} SARIMA-Datenpunkte ab dem {start_date.date()}.")
            if len(rf_preds_rescaled) > 0 and end_idx > len(rf_preds_rescaled):
                end_idx = len(rf_preds_rescaled)
                print(f"Es gibt nur {end_idx - start_idx} Random Forest-Datenpunkte ab dem {start_date.date()}.")

            preds_april = preds_rescaled[start_idx:end_idx]
            sarima_preds_april = sarima_preds_rescaled[start_idx:end_idx]
            targets_april = targets_rescaled[start_idx:end_idx]
            if len(rf_preds_rescaled) > 0:
                rf_preds_april = rf_preds_rescaled[start_idx:end_idx]
            else:
                rf_preds_april = np.array([])

            april_dates = test_dates[start_idx:end_idx]

            plt.figure(figsize=(10, 4))
            plt.plot(targets_april[:200], label='True Intraday (200h ab 01.04.)', color='blue')
            plt.plot(preds_april[:200], label='Predicted Intraday (200h ab 01.04.) - LSTM', color='orange')
            plt.plot(sarima_preds_april[:200], label='Predicted Intraday (200h ab 01.04.) - SARIMA', color='green')
            if len(rf_preds_april) > 0:
                plt.plot(rf_preds_april[:200], label='Predicted Intraday (200h ab 01.04.) - Random Forest', color='red')
            if not df_test.empty:
                plt.title("Erste 200 Stunden im April (2024)")
            else:
                plt.title("Erste 200 Stunden im April (2023)")
            plt.xlabel("Zeitschritt (Stunden)")
            plt.ylabel("Intraday-Wert (ent-skaliert)")
            plt.legend()
            plt.show()
    else:
        print("Keine Testdaten vorhanden. Der Plot wird übersprungen.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nGesamte Laufzeit: {elapsed_time/60:.2f} Minuten")


