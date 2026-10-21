"""Step 2 · 3주차 — 논문 설정 그대로의 세 모델 (강의 3주차 단계 1, 본 강의 32~34쪽)."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

SEED = 42
MODEL_NAMES = ["SVM", "RF", "XGBoost"]


def make_model(name):
    if name == "SVM":      # 기본값 그대로: RBF, C=1, gamma='scale'
        return SVC(kernel="rbf", C=1.0, gamma="scale", decision_function_shape="ovr")
    if name == "RF":       # 트리 100개, 나머지 기본값
        return RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    if name == "XGBoost":  # 논문에서 유일하게 예비 튜닝을 거친 설정
        return XGBClassifier(objective="multi:softprob", n_estimators=400, max_depth=6, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist",
                             random_state=SEED, n_jobs=-1)
    raise ValueError(name)


def scores(model, X):
    """윈도우별 클래스 점수 (n, 8). 세그먼트 단위 판정에 쓰는 '연속 점수' — SVM 은 ovr 결정함수, RF·XGBoost 는 확률."""
    return model.decision_function(X) if isinstance(model, SVC) else model.predict_proba(X)
