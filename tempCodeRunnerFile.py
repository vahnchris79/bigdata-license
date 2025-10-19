for d in range(3, 16):
    model3 = RandomForestRegressor(max_depth=d, random_state=1234).fit(x_train, y_train)
    print(d, get_scores(model3, x_train, x_test, y_train, y_test)) # 15 r2: 0.8737, -0.0438, MAE: 46.6147, 130.7703