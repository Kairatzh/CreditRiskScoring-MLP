import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

df = []

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

le = LabelEncoder()
df['cb_person_default_on_file'] = le.fit_transform(df['cb_person_default_on_file'])
df['person_home_ownership'] = le.fit_transform(df['person_home_ownership'])
df['loan_intent'] = le.fit_transform(df['loan_intent'])
df['loan_grade'] = le.fit_transform(df['loan_grade'])

numerical_cols = df.select_dtypes(include=np.number).columns

for col in numerical_cols:
    if df[col].isnull().any():
        current_col_median_val = df[col].median()
        df[col] = df[col].fillna(current_col_median_val)

print("Проверка на пропущенные значения после заполнения:")
print(df.isnull().sum())

from sklearn.model_selection import train_test_split
X = df.drop('cb_person_default_on_file', axis=1)
y = df['cb_person_default_on_file']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(11, 32)
        self.init_weights()

        self.model = nn.Sequential(
            self.fc1,
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),


            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),

            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.01),

            nn.Linear(8, 1),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


model = MLP()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
sheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

epochs = 1000
for epoch in range(epochs):
    preds = model(X_train)
    loss = criterion(preds, y_train)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        val_preds = model(X_test)
        val_loss = criterion(val_preds, y_test)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: train_loss={loss.item():.4f}, val_loss={val_loss.item():.4f}")


from sklearn.metrics import accuracy_score
with torch.no_grad():
  preds = model.predict(X_test)
  binary_preds = (preds >= 0.5).float()
  print(accuracy_score(y_test, binary_preds))


class AdvancedMLP(nn.Module):
  def __init__(self,
               input_dim=11,
               hidden_dim=[128, 64, 32, 16, 8],
               output_dim=1,
               activation="LeakyReLU",
               dropout=0.2,
               use_batchnorm=True):
    super().__init__()
    self.activation_name = activation
    layers = []
    pred_dim = input_dim

    for h_dim in hidden_dim:
      layers.append(nn.Linear(pred_dim, h_dim))
      if use_batchnorm:
        layers.append(nn.BatchNorm1d(h_dim))
      layers.append(self._get_activation())
      if dropout > 0:
        layers.append(nn.Dropout(dropout))
      pred_dim = h_dim
    layers.append(nn.Linear(pred_dim, output_dim))
    self.model = nn.Sequential(*layers)
    self._init_weights()

  def _get_activation(self):
    if self.activation_name == "ReLU":
      return nn.ReLU()
    elif self.activation_name == "LeakyReLU":
      return nn.LeakyReLU()
    elif self.activation_name == "GELU":
      return nn.GELU()
    elif self.activation_name == "SiLU":
      return nn.SiLU()
    else:
      raise ValueError(f"Unknown activation function: {self.activation_name}")

  def _init_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Linear):
        if self.activation_name == "GELU":
          nn.init.normal_(m.weight, mean=0.0, std=1.0)
        if self.activation_name in ["ReLU", "LeakyReLU", "SiLU"]:
          nn.init.xavier_uniform_(m.weight)
        else:
          nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
          nn.init.zeros_(m.bias)

  def forward(self, x):
    return self.model(x)

  def predict(self, x):
    with torch.no_grad():
      return torch.sigmoid(self.forward(x))


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, epochs=1000, lr=0.001, weight_decay=0,
                             patience=None, step_size=None, gamma=None):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None
    if patience is not None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=patience, factor=0.5)
    elif step_size is not None and gamma is not None:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    best_val_loss = float('inf')

    print(f"\n--- Training {model.__class__.__name__} ---")

    for epoch in range(epochs):
        model.train()
        preds = model(X_train)
        loss = criterion(preds, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_test)
            val_loss = criterion(val_preds, y_test)

        if scheduler:
            if patience is not None:
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1}/{epochs}: train_loss={loss.item():.4f}, val_loss={val_loss.item():.4f}")

    with torch.no_grad():
        model.eval()
        preds_test = model.predict(X_test)
        binary_preds = (preds_test >= 0.5).float()
        accuracy = accuracy_score(y_test, binary_preds)

    print(f"Final Accuracy: {accuracy:.4f}")
    return {'accuracy': accuracy, 'final_val_loss': val_loss.item()}


print("Evaluating initial MLP model...")
model_mlp = MLP()
mlp_results = train_and_evaluate_model(model_mlp, X_train, y_train, X_test, y_test,
                                       epochs=1000, lr=0.001, step_size=10, gamma=0.9)

print(f"Initial MLP Model Accuracy: {mlp_results['accuracy']:.4f}")


benchmark_results = {}

def print_metrics(model_name, metrics):
    print(f"\n--- Результаты для модели: {model_name} ---")
    print(f"Точность (Accuracy): {metrics['accuracy']:.4f}")
    print(f"Итоговая функция потерь на валидации (Final Validation Loss): {metrics['final_val_loss']:.4f}")



print("Бенчмаркинг: AdvancedMLP с LeakyReLU, Dropout и BatchNorm...")
model_leaky_dropout_bn = AdvancedMLP(
    input_dim=X_train.shape[1],
    activation="LeakyReLU",
    dropout=0.2,
    use_batchnorm=True
)
results_leaky_dropout_bn = train_and_evaluate_model(
    model_leaky_dropout_bn, X_train, y_train, X_test, y_test,
    epochs=100, lr=0.001, weight_decay=1e-4, patience=5 # Use patience for ReduceLROnPlateau
)
benchmark_results['LeakyReLU_Dropout_BatchNorm'] = results_leaky_dropout_bn
print_metrics('LeakyReLU_Dropout_BatchNorm', results_leaky_dropout_bn)

print("Бенчмаркинг: AdvancedMLP с ReLU, Dropout и BatchNorm...")
model_relu_dropout_bn = AdvancedMLP(
    input_dim=X_train.shape[1],
    activation="ReLU",
    dropout=0.2,
    use_batchnorm=True
)
results_relu_dropout_bn = train_and_evaluate_model(
    model_relu_dropout_bn, X_train, y_train, X_test, y_test,
    epochs=100, lr=0.001, weight_decay=1e-4, patience=5
)
benchmark_results['ReLU_Dropout_BatchNorm'] = results_relu_dropout_bn
print_metrics('ReLU_Dropout_BatchNorm', results_relu_dropout_bn)



print("Бенчмаркинг: AdvancedMLP с GELU, Dropout и BatchNorm...")
model_gelu_dropout_bn = AdvancedMLP(
    input_dim=X_train.shape[1],
    activation="GELU",
    dropout=0.2,
    use_batchnorm=True
)
results_gelu_dropout_bn = train_and_evaluate_model(
    model_gelu_dropout_bn, X_train, y_train, X_test, y_test,
    epochs=100, lr=0.001, weight_decay=1e-4, patience=5
)
benchmark_results['GELU_Dropout_BatchNorm'] = results_gelu_dropout_bn
print_metrics('GELU_Dropout_BatchNorm', results_gelu_dropout_bn)


print("Бенчмаркинг: AdvancedMLP с LeakyReLU, без Dropout, с BatchNorm...")
model_no_dropout = AdvancedMLP(
    input_dim=X_train.shape[1],
    activation="SiLU",
    dropout=0,
    use_batchnorm=True
)
results_no_dropout = train_and_evaluate_model(
    model_no_dropout, X_train, y_train, X_test, y_test,
    epochs=100, lr=0.001, weight_decay=1e-4, patience=5
)
benchmark_results['LeakyReLU_NoDropout_BatchNorm'] = results_no_dropout
print_metrics('LeakyReLU_NoDropout_BatchNorm', results_no_dropout)


print("Бенчмаркинг: AdvancedMLP с LeakyReLU, с Dropout, без BatchNorm...")
model_no_bn = AdvancedMLP(
    input_dim=X_train.shape[1],
    activation="ReLU",
    dropout=0.2,
    use_batchnorm=False
)
results_no_bn = train_and_evaluate_model(
    model_no_bn, X_train, y_train, X_test, y_test,
    epochs=100, lr=0.001, weight_decay=1e-4, patience=5
)
benchmark_results['LeakyReLU_Dropout_NoBatchNorm'] = results_no_bn
print_metrics('LeakyReLU_Dropout_NoBatchNorm', results_no_bn)


import pandas as pd

results_df = pd.DataFrame.from_dict(benchmark_results, orient='index')
results_df.index.name = 'Model Configuration'
display(results_df.sort_values(by='accuracy', ascending=False))

