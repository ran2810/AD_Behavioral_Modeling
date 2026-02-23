from data.preprocess import preprocess_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def main():
    # Load dataset
    df = preprocess_file("data/01_tracks.csv")

    # Features & labels
    X = df[["x", "y", "xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]]
    y = df["lane_change"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

    # Train RF model
    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced")
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
