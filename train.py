from __future__ import print_function

import sys
import numpy as np
from sklearn.linear_model import LogisticRegression

import mlflow
import mlflow.sklearn

if __name__ == "__main__":

    penalty = sys.argv[1] # either 'l1', 'l2', 'elasticnet', or 'none'
    C = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4

    X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
    y = np.array([0, 0, 1, 1, 1, 0])
    lr = LogisticRegression(penalty=penalty, C=C, tol=tol)
    lr.fit(X, y)
    score = lr.score(X, y)
    mlflow.log_metric("score", score)
    mlflow.sklearn.log_model(lr, "model")
    print("Score: %s" % score)
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
