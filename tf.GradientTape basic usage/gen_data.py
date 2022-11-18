import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def gen_data(
    feature_num=10, data_size=100000, test_size=0.2, random_state=42, preproess=True
):
    random_data = np.random.uniform(low=-5, high=5, size=(data_size, feature_num))

    random_data_exp = np.exp(random_data)
    random_data_sin = np.sin(random_data)
    random_data_square = np.square(random_data)

    X = np.vstack((random_data_exp, random_data_sin))
    X = np.vstack((X, random_data_square))

    y0 = np.zeros((data_size))
    y1 = np.ones((data_size))
    y2 = np.ones((data_size)) * 2
    y = np.append(y0, y1)
    y = np.append(y, y2)

    indices = np.random.permutation(3 * data_size)
    X = X[indices]
    y = y[indices]

    if preproess:
        sc = StandardScaler()
        sc.fit(X)
        X = sc.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test
