from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

SCORES = (
    mean_absolute_percentage_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

MULTIREG_MODELS = (
    DummyRegressor(strategy="mean"),
    LinearRegression(n_jobs=-1),

    KNeighborsRegressor(n_jobs=-1, n_neighbors=10, weights="distance"),

    RandomForestRegressor(random_state=1, n_jobs=-1,
                          n_estimators=600, max_samples=2500, min_impurity_decrease=.5),

    XGBRegressor(random_state=1, n_estimators=300, n_jobs=-1,
                 max_depth=2, min_child_weight=1100, learning_rate=.1),

    MLPRegressor(random_state=1, hidden_layer_sizes=(25, 50),
                 max_iter=2000, activation="relu"),
)

AUTOREG_MODELS = (
    DummyRegressor(strategy="mean"),
    LinearRegression(n_jobs=-1),

    KNeighborsRegressor(n_jobs=-1, n_neighbors=10, weights="distance"),

    RandomForestRegressor(random_state=1, n_jobs=-1,
                          n_estimators=600, min_impurity_decrease=.5),

    XGBRegressor(random_state=1, n_estimators=400, n_jobs=-1,
                 max_depth=4, learning_rate=.01),

    MLPRegressor(random_state=1, hidden_layer_sizes=(25, 50),
                 max_iter=1000, activation="relu"),
)
