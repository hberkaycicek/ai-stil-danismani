if CONFIG["OPTUNA_CALISTIR"]:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    setler = (grid[grid["GECERLI"] & grid["SECILEBILIR"]].drop_duplicates("KOLONLAR")["KOLONLAR"]
              .head(CONFIG["OPTUNA_SET_SAYISI"]).tolist())
    opt_kayit = []

    def amac(trial):
        kolonlar = list(setler[trial.suggest_int("set", 0, len(setler) - 1)])
        aile = trial.suggest_categorical("aile", ["Ridge*", "Lasso*", "ElasticNet*", "Huber",
                                                  "SVR", "ExtraTrees", "GBoost"])
        p = {}
        if aile in ("Ridge*", "Huber"):                       # standart X'te ceza
            p["alpha"] = trial.suggest_float(f"{aile}_alpha", 1e-3, 1e3, log=True)
        if aile in ("Lasso*", "ElasticNet*"):                 # DÜZELTME: hedef ölçeğinde ceza (alpha_max ~1e-3)
            p["alpha"] = trial.suggest_float(f"{aile}_alpha", 1e-7, 1e-2, log=True)
        if aile == "ElasticNet*":
            p["l1"] = trial.suggest_float("l1", 0.05, 0.95)
        if aile == "Huber":
            p["eps"] = trial.suggest_float("h_eps", 1.1, 2.5)
        if aile == "SVR":
            p.update(kernel=trial.suggest_categorical("kernel", ["linear", "rbf"]),
                     C=trial.suggest_float("C", 1e-2, 1e2, log=True),
                     eps=trial.suggest_float("svr_eps", 1e-5, 1e-3, log=True))
        if aile == "ExtraTrees":
            p.update(depth=trial.suggest_int("et_depth", 2, 4), leaf=trial.suggest_int("leaf", 2, 6))
        if aile == "GBoost":
            p.update(n=trial.suggest_int("n", 50, 300), lr=trial.suggest_float("lr", 0.01, 0.2, log=True),
                     depth=trial.suggest_int("gb_depth", 1, 3))
        s = cv_degerlendir(kolonlar, model_fabrika(aile, p))
        opt_kayit.append(kayit_ekle(s, kolonlar, aile, p))
        return s["CV_WMAPE"], (s["DELTA_MAE"] if np.isfinite(s["DELTA_MAE"]) else 1.0)

    study = optuna.create_study(directions=["minimize", "minimize"], sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(amac, n_trials=CONFIG["OPTUNA_TRIALS"])
    grid = rank_ekle(pd.concat([grid_ham, pd.DataFrame(opt_kayit)], ignore_index=True))
    grid["VARYANT"] = varyant_etiketi(grid)
    OZET_AILE = grup_ozet(grid, "MODEL")
    print(f"Optuna sonrası: {len(grid):,} değerlendirme | geçerli {int(grid['GECERLI'].sum()):,}")
    display(OZET_AILE.sort_values("EN_IYI_CV_WMAPE").style.format(
        {"EN_IYI_CV_WMAPE": "{:.2f}", "MEDYAN_CV_WMAPE": "{:.2f}"}, decimal=",", na_rep="-"))
else:
    print("Optuna atlandı.")
