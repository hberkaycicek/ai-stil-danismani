class HizliDogrusalCV(BaseEstimator, RegressorMixin):
    """Ridge / Lasso / ElasticNet: iç zamansal CV ile alpha (ve L1 oranı) seçimi, sonra tüm eğitime fit.
    Ridge: SVD ile kapalı çözüm (tüm alpha yolu tek seferde). Lasso/EN: enet_path.
    Her iç fold'da standartlaştırma sadece o fold'un eğitim kısmıyla yapılır."""

    def __init__(self, aile="Ridge", alphalar=None, l1_oranlari=(1.0,), ic_cv=3, max_iter=100000, tol=1e-6):
        self.aile, self.alphalar, self.l1_oranlari = aile, alphalar, l1_oranlari
        self.ic_cv, self.max_iter, self.tol = ic_cv, max_iter, tol

    @staticmethod
    def _std(X, y):
        mx, sx = X.mean(axis=0), X.std(axis=0)
        sx = np.where(sx > 0, sx, 1.0)
        return (X - mx) / sx, y - y.mean(), mx, sx, y.mean()

    def _yol(self, Xs, yc, l1, alphalar):
        if self.aile == "Ridge":
            U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
            Uy = U.T @ yc
            return np.stack([Vt.T @ (s / (s ** 2 + a) * Uy) for a in alphalar])
        Xf = np.asfortranarray(Xs)                     # Gram matrisi önceden: küçük p için ~6 kat hızlı
        _, coefs, _ = enet_path(Xf, yc, l1_ratio=l1, alphas=alphalar, max_iter=self.max_iter, tol=self.tol,
                                check_input=False, precompute=Xf.T @ Xf, Xy=Xf.T @ yc)
        return coefs.T

    def fit(self, X, y, sample_weight=None):
        X, y = np.asarray(X, float), np.asarray(y, float)
        alphalar = np.sort(np.asarray(self.alphalar, float))[::-1]
        l1s = [None] if self.aile == "Ridge" else list(self.l1_oranlari)
        en_iyi = (np.inf, None, None)
        for l1 in l1s:
            mse = np.zeros(len(alphalar))
            for tr, te in TimeSeriesSplit(n_splits=self.ic_cv).split(X):
                Xs, yc, mx, sx, my = self._std(X[tr], y[tr])
                B = self._yol(Xs, yc, l1, alphalar)
                P = ((X[te] - mx) / sx) @ B.T + my
                mse += ((P - y[te][:, None]) ** 2).mean(axis=0)
            j = int(np.argmin(mse))
            if mse[j] < en_iyi[0]:
                en_iyi = (mse[j], alphalar[j], l1)
        self.alpha_, self.l1_ratio_ = float(en_iyi[1]), en_iyi[2]
        Xs, yc, mx, sx, my = self._std(X, y)
        b = self._yol(Xs, yc, self.l1_ratio_, np.array([self.alpha_]))[0]
        self.coef_ = b / sx
        self.intercept_ = float(my - mx @ self.coef_)
        return self

    def predict(self, X):
        return np.asarray(X, float) @ self.coef_ + self.intercept_


def model_fabrika(aile, p=None):
    p = p or {}
    ortak = dict(alphalar=CONFIG["ALPHALAR"], ic_cv=CONFIG["IC_CV_SPLITS"],
                 max_iter=CONFIG["MAX_ITER"], tol=CONFIG["MODEL_TOL"])
    return {
        "Dummy":       lambda: DummyRegressor(strategy="mean"),
        "Linear":      lambda: LinearRegression(),
        "Ridge":       lambda: HizliDogrusalCV("Ridge", **ortak),
        "Lasso":       lambda: HizliDogrusalCV("Lasso", l1_oranlari=(1.0,), **ortak),
        "ElasticNet":  lambda: HizliDogrusalCV("ElasticNet", l1_oranlari=tuple(CONFIG["L1_ORANLARI"]), **ortak),
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


# Doğrulama: aynı alpha'da motor = sklearn
_X = StandardScaler().fit_transform(cv_df[[ONCULLER[1], KKB_KOL]].to_numpy(float))
_kontrol = []
for aile, sk, a, l1 in [("Ridge", Ridge(alpha=1.0), 1.0, None), ("Lasso", Lasso(alpha=1e-4, max_iter=100000), 1e-4, 1.0),
                        ("ElasticNet", ElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=100000), 1e-4, 0.5)]:
    hm = HizliDogrusalCV(aile, alphalar=[a], l1_oranlari=(l1 or 1.0,), tol=1e-10).fit(_X, Y_CV)
    _kontrol.append({"AILE": aile, "MAKS_KATSAYI_FARKI": float(np.max(np.abs(hm.coef_ - sk.fit(_X, Y_CV).coef_)))})
display(pd.DataFrame(_kontrol).style.format({"MAKS_KATSAYI_FARKI": "{:.2e}"}))
