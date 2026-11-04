# Ridge / Lasso / ElasticNet: her fold'da tüm alpha yolu TEK seferde çözülür; her alpha ayrı aday olarak
# AYNI boşluklu walk-forward CV ile ölçülür. Standartlaştırma her fold'un SADECE eğitim kısmıyla yapılır.
# Çözümler sklearn Ridge / Lasso / ElasticNet(+StandardScaler) ile birebir aynıdır (alt kısımda test).
def _std(X, y):
    mx, sx = X.mean(axis=0), X.std(axis=0)
    sx = np.where(sx > 0, sx, 1.0)
    return (X - mx) / sx, y - y.mean(), mx, sx, y.mean()


def alpha_yolu(Xs, yc, aile, alphalar, l1=None):
    """Standart X, merkezlenmiş y -> (n_alpha x p) katsayı matrisi."""
    if aile == "Ridge":
        U, s_, Vt = np.linalg.svd(Xs, full_matrices=False)
        Uy = U.T @ yc
        return np.stack([Vt.T @ (s_ / (s_ ** 2 + a) * Uy) for a in alphalar])
    Xf = np.asfortranarray(Xs)
    _, coefs, _ = enet_path(Xf, yc, l1_ratio=1.0 if aile == "Lasso" else l1, alphas=alphalar,
                            max_iter=CONFIG["MAX_ITER"], tol=CONFIG["MODEL_TOL"],
                            check_input=False, precompute=Xf.T @ Xf, Xy=Xf.T @ yc)
    return coefs.T


def yol_varyantlari(aile):
    if aile == "Ridge":
        return [(np.asarray(CONFIG["ALPHA_RIDGE"], float), None)]
    if aile == "Lasso":
        return [(np.asarray(CONFIG["ALPHA_LASSO"], float), None)]
    return [(np.asarray(CONFIG["ALPHA_LASSO"], float), l1) for l1 in CONFIG["L1_ORANLARI"]]


def cv_yol(kolonlar, aile, foldlar=None):
    """Bir aday + bir aile için tüm alpha'ların CV sonuçları: [(params, metrikler), ...]."""
    X, hazir = fold_hazirla(kolonlar, foldlar)
    sonuc = []
    for alphalar, l1 in yol_varyantlari(aile):
        alphalar = np.sort(alphalar)[::-1]
        fold_P = []
        for tr, val in hazir:
            Xs, yc, mx, sx, my = _std(X[tr], Y_CV[tr])
            B = alpha_yolu(Xs, yc, aile, alphalar, l1)                 # n_alpha x p
            fold_P.append((tr, val, ((X[val] - mx) / sx) @ B.T + my, Xs @ B.T + my))
        for j, a in enumerate(alphalar):
            kay = [(tr, val, P[:, j], Ptr[:, j]) for tr, val, P, Ptr in fold_P]
            p = {"alpha": float(a)} if l1 is None else {"alpha": float(a), "l1": float(l1)}
            sonuc.append((p, metrik_ozet(kay)))
    return sonuc


def model_fabrika(aile, p=None):
    p = p or {}
    return {
        "Dummy":       lambda: DummyRegressor(strategy="mean"),
        "Linear":      lambda: LinearRegression(),
        "Ridge":       lambda: Ridge(alpha=p["alpha"]),
        "Lasso":       lambda: Lasso(alpha=p["alpha"], max_iter=CONFIG["MAX_ITER"], tol=CONFIG["MODEL_TOL"]),
        "ElasticNet":  lambda: ElasticNet(alpha=p["alpha"], l1_ratio=p["l1"], max_iter=CONFIG["MAX_ITER"],
                                          tol=CONFIG["MODEL_TOL"]),
        "Ridge*":      lambda: Ridge(alpha=p["alpha"]),
        "Lasso*":      lambda: Lasso(alpha=p["alpha"], max_iter=100000),
        "ElasticNet*": lambda: ElasticNet(alpha=p["alpha"], l1_ratio=p["l1"], max_iter=100000),
        "Huber":       lambda: HuberRegressor(epsilon=p["eps"], alpha=p["alpha"], max_iter=2000),
        "SVR":         lambda: SVR(kernel=p["kernel"], C=p["C"], epsilon=p["eps"]),
        "ExtraTrees":  lambda: ExtraTreesRegressor(n_estimators=300, max_depth=p["depth"],
                                                   min_samples_leaf=p["leaf"], random_state=42),
        "GBoost":      lambda: GradientBoostingRegressor(n_estimators=p["n"], learning_rate=p["lr"],
                                                         max_depth=p["depth"], random_state=42),
    }[aile]()


# Doğrulama: alpha yolu = sklearn (aynı alpha, aynı aday, aynı fold'lar)
_kol = [ONCULLER[1], KKB_KOL]
_kontrol = []
for aile in ["Ridge", "Lasso", "ElasticNet"]:
    p_, m_yol = cv_yol(_kol, aile)[3]
    m_sk = cv_degerlendir(_kol, model_fabrika(aile, p_))
    _kontrol.append({"AILE": aile, "PARAMS": p_, "CV_WMAPE_YOL": m_yol["CV_WMAPE"], "CV_WMAPE_SKLEARN": m_sk["CV_WMAPE"],
                     "FARK": abs(m_yol["CV_WMAPE"] - m_sk["CV_WMAPE"])})
display(pd.DataFrame(_kontrol).style.format({"CV_WMAPE_YOL": "{:.6f}", "CV_WMAPE_SKLEARN": "{:.6f}", "FARK": "{:.1e}"}))
if max(k["FARK"] for k in _kontrol) > 1e-3:
    raise AssertionError("Alpha yolu sklearn ile uyuşmuyor.")
print("Alpha ızgarası | Ridge:", np.round(CONFIG["ALPHA_RIDGE"], 4).tolist(),
      "| Lasso/EN:", [f"{a:.1e}" for a in CONFIG["ALPHA_LASSO"]], "| EN L1:", CONFIG["L1_ORANLARI"])
