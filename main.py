# Sunum görselleri (tahminler Gelecek Tahmini hücresinden). Köprü tahminleri tabloda * ve sarı, grafikte turuncu elmas.
def tahmin_tablosu():
    """Sunum için: olgunlaşmamış kohortların tahmin tablosu (görsel). * = köprü (eksik öncül oranla tahmin edildi)."""
    if not len(TAHMIN):
        print("Olgunlaşmamış kohort yok -> tahmin tablosu yok.")
        return
    kolonlar = [("DONEM_1", "Kohort"), ("GOZLENEN_AY", "Gözlenen ay")]
    for rol in MODELLER:
        kolonlar += [(f"{rol}_HAM", f"{rol.title()} ham"), (f"{rol}_DUZ", f"{rol.title()} overlay")]
    if "CHAIN_LADDER" in TAHMIN:
        kolonlar += [("CHAIN_LADDER", "Chain-ladder"), ("CL_KAYNAK", "CL kaynak")]
    hucre, kopru_hucre = [], []
    for i, (_, r) in enumerate(TAHMIN.iterrows(), 1):
        satir = []
        for j, (k, _a) in enumerate(kolonlar):
            v = r[k]
            if k in ("DONEM_1", "GOZLENEN_AY"):
                satir.append(str(int(v)))
            elif k == "CL_KAYNAK":
                satir.append(v if isinstance(v, str) else "-")
            else:
                rol_ = k.rsplit("_", 1)[0]
                kopru = str(r.get(f"{rol_}_GIRDI", "")).startswith("köprü") and pd.notna(v)
                satir.append(yuzde(v) + ("*" if kopru else ""))
                if kopru:
                    kopru_hucre.append((i, j))
        hucre.append(satir)
    fig, ax = plt.subplots(figsize=(16, 0.42 * len(hucre) + 1.8)); ax.axis("off")
    tbl = ax.table(cellText=hucre, colLabels=[a for _, a in kolonlar], cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.5)
    for (i, j), c in tbl.get_celld().items():
        c.set_edgecolor("#e5e7eb")
        if i == 0:
            c.set_facecolor("#1f2937"); c.set_text_props(color="white", fontweight="bold")
        else:
            c.set_facecolor(ARKA["TAHMIN"] if i % 2 else "#ffffff")
        if (i, j) in kopru_hucre:
            c.set_facecolor("#fef3c7"); c.set_text_props(fontstyle="italic")
    ov_ = " | ".join(f"{r}: overlay {o['deger'] * 100:+.3f} yp ({'uygulanıyor' if o['uygula'] else 'yok'})"
                     for r, o in OVERLAY.items())
    ax.set_title(f"{HK} tahminleri | bilgi ayı {BILGI} | {ov_}", fontweight="bold", loc="left")
    kopru_ = sorted({f"{r['DONEM_1']} {rol.title()}: {r[f'{rol}_GIRDI'][7:]}" for rol in MODELLER
                     for _, r in TAHMIN.iterrows() if str(r.get(f"{rol}_GIRDI", "")).startswith("köprü")})
    if kopru_:
        ax.text(0, -0.02, "* Köprü: öncül bilgi ayında henüz tamamlanmadı, aynı kohortun bilinen öncülünden oranla tahmin "
                          "edildi (daha belirsiz). " + " | ".join(kopru_), transform=ax.transAxes, fontsize=8.5,
                color="#92400e", va="top")
    plt.show()


def bolge_boya(ax, alanlar):
    for ad, (p0, p1) in alanlar.items():
        x0, x1 = ts(p0) - pd.Timedelta(days=15), ts(p1) + pd.Timedelta(days=15)
        ax.axvspan(x0, x1, color=ARKA[ad], zorder=0)
        ax.text(x0 + (x1 - x0) / 2, 1.01, {"EGITIM": "EĞİTİM", "VALIDASYON": "VALİDASYON (CV)", "OOT": "OOT",
                                           "TAHMIN": "TAHMİN"}[ad], transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=RENK[ad] if ad != "EGITIM" else "#475569")


def ana_grafik(rol):
    spec, w = MODELLER[rol], WF[rol]
    kol = spec["kolonlar"]
    egit = olgun_df[olgun_df["AY"] < VAL_BAS].dropna(subset=kol)
    m_eg = pipe_fit(model_fabrika(spec["aile"], spec["params"]), egit[kol], egit[HEDEF])
    seg = {"EGITIM": pd.DataFrame({"AY": egit["AY"], "GERCEK": egit[HEDEF], "TAHMIN": m_eg.predict(egit[kol])}),
           "VALIDASYON": w[(w["BOLGE"] == "GELISTIRME") & (w["AY"] >= VAL_BAS)],
           "OOT": w[w["BOLGE"] == "OOT"]}
    alanlar = {k: (v["AY"].min(), v["AY"].max()) for k, v in seg.items()}
    if len(TAHMIN):
        alanlar["TAHMIN"] = (TAHMIN["AY"].min(), TAHMIN["AY"].max())

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.25], hspace=0.12)
    ax = fig.add_subplot(gs[0])
    bolge_boya(ax, alanlar)
    hist = olgun_df.sort_values("AY")
    ax.plot(hist["AY"].dt.to_timestamp(), hist[HEDEF] * 100, "o-", color=RENK["GERCEK"], lw=2.2, ms=4,
            label="Gerçekleşen", zorder=5)
    etiket = {"EGITIM": "Eğitim uyumu (in-sample)", "VALIDASYON": "Validasyon (walk-forward)",
              "OOT": "OOT (walk-forward)"}
    for k, s in seg.items():
        ax.plot(s["AY"].dt.to_timestamp(), s["TAHMIN"] * 100, "o-" if k != "EGITIM" else "-",
                color=RENK[k], lw=2, ms=4, label=etiket[k], zorder=4)
    if len(TAHMIN):
        tx = TAHMIN["AY"].dt.to_timestamp()
        ax.plot(tx, TAHMIN[f"{rol}_DUZ"] * 100, "D-", color=RENK["TAHMIN"], lw=2.4, ms=6,
                label="Tahmin (overlay sonrası)", zorder=6)
        ax.plot(tx, TAHMIN[f"{rol}_HAM"] * 100, "D--", color=RENK["TAHMIN"], lw=1.2, ms=4, alpha=0.6,
                label="Tahmin (ham)", zorder=6)
        if "CHAIN_LADDER" in TAHMIN:
            ax.plot(tx, TAHMIN["CHAIN_LADDER"] * 100, "s:", color=RENK["KISMI"], lw=1.4, ms=5,
                    label="Chain-ladder (90_m'den, modelden bağımsız)", zorder=5)
        kp = TAHMIN[f"{rol}_GIRDI"].astype(str).str.startswith("köprü") if f"{rol}_GIRDI" in TAHMIN else None
        if kp is not None and kp.any():
            ax.plot(tx[kp], TAHMIN.loc[kp, f"{rol}_DUZ"] * 100, "D", mfc="white", mec="#d97706", mew=2.2, ms=10,
                    ls="none", label="Köprü tahmini (öncül oranla tahmin edildi)", zorder=7)
            ax.plot(tx[kp], TAHMIN.loc[kp, f"{rol}_HAM"] * 100, "D", mfc="white", mec="#d97706", mew=1.4, ms=7,
                    ls="none", alpha=0.7, zorder=7)
    ax.yaxis.set_major_formatter(PCT_AX)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(alpha=0.25); ax.set_ylabel(f"{HK} (12 ayda 90+ gecikme oranı)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.07), fontsize=8.5, ncol=4, frameon=False)
    ax.set_title(f"{spec['ad']}\n{FORMUL.get(rol, '')}", fontsize=11.5, fontweight="bold", loc="left", pad=62)

    axt = fig.add_subplot(gs[1]); axt.axis("off")
    satirlar, renkler = [], []
    for k in ["EGITIM", "VALIDASYON", "OOT"]:
        s = seg[k]
        mm = metrik(s["GERCEK"], s["TAHMIN"])
        dm = metrik(s["GERCEK"], s["DUMMY"]) if "DUMMY" in s else {}
        satirlar.append([{"EGITIM": "Eğitim (in-sample)", "VALIDASYON": "Validasyon (walk-forward)",
                          "OOT": "OOT (walk-forward)"}[k], f"{alanlar[k][0]} – {alanlar[k][1]}", mm["N"],
                         sayi(mm["WMAPE"]), sayi(mm["MAE_yp"], 3), f"{mm['SAPMA_yp']:+.3f}".replace(".", ","),
                         sayi(mm["R2"], 3), sayi(mm["KORELASYON"], 3), mm["YON"],
                         sayi(1 - mm["WMAPE"] / dm["WMAPE"], 2) if dm else "-"])
        renkler.append(ARKA[k])
    if len(TAHMIN) and TAHMIN[f"{rol}_HAM"].notna().any():
        t_ = TAHMIN.dropna(subset=[f"{rol}_HAM"])
        n_kp = int(t_[f"{rol}_GIRDI"].astype(str).str.startswith("köprü").sum()) if f"{rol}_GIRDI" in t_ else 0
        satirlar.append(["Tahmin (olgunlaşmamış)", f"{t_['AY'].min()} – {t_['AY'].max()}",
                         f"{len(t_)}" + ("*" if n_kp else ""),
                         "-", "-", f"ham ort. {yuzde(t_[f'{rol}_HAM'].mean())}",
                         f"ovl {yuzde(t_[f'{rol}_DUZ'].mean())}",
                         f"CL {yuzde(t_['CHAIN_LADDER'].mean()) if 'CHAIN_LADDER' in t_ else '-'}", "-", "-"])
        renkler.append(ARKA["TAHMIN"])
    tbl = axt.table(cellText=satirlar, colLabels=["Bölge", "Kohortlar", "N", "WMAPE", "MAE (yp)", "Sapma (yp)",
                                                 "R²", "Korelasyon", "Yön isabeti", "Beceri*"],
                    cellLoc="center", loc="center", colWidths=[.17, .15, .05, .07, .08, .12, .07, .09, .09, .08])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1, 1.6)
    for (i, j), c in tbl.get_celld().items():
        c.set_edgecolor("#e5e7eb")
        if i == 0:
            c.set_facecolor("#1f2937"); c.set_text_props(color="white", fontweight="bold")
        else:
            c.set_facecolor(renkler[i - 1])
    axt.text(0, -0.02, "* Beceri = 1 − WMAPE / naif ortalama WMAPE. yp = yüzde puan. Walk-forward: her kohort, kendi "
                       "tahmin anında hedefi belli olan kohortlarla eğitilen modelle tahmin edilir.",
             transform=axt.transAxes, fontsize=8, color="#6b7280")
    plt.show()


def model_karsilastirma():
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.3))
    p = PERF.set_index(["MODEL", "BOLGE"])
    x = np.arange(2)
    renkler = {"CHAMPION": RENK["TAHMIN"], "CHALLENGER": RENK["CHALLENGER"]}
    roller = list(MODELLER)
    gen = 0.8 / (len(roller) + 1)
    for i, rol in enumerate(roller + ["DUMMY"]):
        v = [p.loc[(roller[0], b), "DUMMY_WMAPE"] if rol == "DUMMY" else p.loc[(rol, b), "WMAPE"]
             for b in ["VALIDASYON", "OOT"]]
        bars = axs[0].bar(x + (i - len(roller) / 2) * gen, v, gen, color=renkler.get(rol, RENK["DUMMY"]),
                          label="Naif ortalama" if rol == "DUMMY" else rol.replace("_", " ").title())
        axs[0].bar_label(bars, labels=[sayi(t, 1) for t in v], fontsize=8)
    axs[0].set_xticks(x, ["Validasyon", "OOT"]); axs[0].set_title("WMAPE (düşük = iyi)", fontweight="bold")
    axs[0].legend(fontsize=8)
    for j, (t, baslik, vurgu) in enumerate([(OZET_VARYANT, "Takvim etkisi", "Yok"),
                                             (OZET_BILGI_AY, "Öncül olgunluğu (ay)", CHAMP["bilgi_ay"])], 1):
        t = t.dropna(subset=["EN_IYI_CV_WMAPE"])
        renk = [RENK["TAHMIN"] if str(k) == str(vurgu) else "#cbd5e1" for k in t.index]
        bars = axs[j].bar([str(k) for k in t.index], t["EN_IYI_CV_WMAPE"], color=renk)
        axs[j].bar_label(bars, labels=[sayi(v) for v in t["EN_IYI_CV_WMAPE"]], fontsize=8)
        axs[j].set_title(f"{baslik}: en iyi CV WMAPE", fontweight="bold")
        axs[j].set_ylim(0, t["EN_IYI_CV_WMAPE"].max() * 1.2)
    for a in axs:
        a.grid(axis="y", alpha=0.25)
    plt.tight_layout(); plt.show()


def donusum_grafigi():
    if not oran_onc:
        return
    fig, ax = plt.subplots(figsize=(16, 4))
    ref_ = DONUSUM.loc[DONUSUM.index < str(VAL_BAS.year + 1)]
    for c, renk in zip(oran_onc, [RENK["TAHMIN"], RENK["OOT"], RENK["CHALLENGER"], RENK["VALIDASYON"]]):
        kol_ = f"HEDEF/{kisa_ad(c)}"
        endeks = DONUSUM[kol_] / ref_[kol_].mean() * 100
        ax.plot(DONUSUM.index, endeks, "o-", lw=2.2, color=renk, label=f"{HK} / {kisa_ad(c)}")
        for xx, v in zip(DONUSUM.index, endeks):
            ax.annotate(f"{v:.0f}", (xx, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color=renk)
    ax.axhline(100, color="#9ca3af", ls="--", lw=1)
    ax.set_title(f"Erken gecikmenin batığa dönüşüm oranı (endeks, {DONUSUM.index[0][:4]}–{VAL_BAS.year} ort. = 100)",
                 fontweight="bold", loc="left")
    ax.grid(alpha=0.25); ax.legend(fontsize=9)
    plt.tight_layout(); plt.show()


takvim_grafigi(CHAMP["bilgi_ay"])
for rol in MODELLER:
    ana_grafik(rol)
tahmin_tablosu()
model_karsilastirma()
donusum_grafigi()
