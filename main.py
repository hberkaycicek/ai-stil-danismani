def ts(p):
    return p.to_timestamp() if hasattr(p, "to_timestamp") else pd.Period(p, "M").to_timestamp()


def takvim_grafigi(b, tahmin_araligi=None):
    """Veri bölmesi + bilgi ayı b için walk-forward fold yapısı (taralı alan = olgunlaşma boşluğu)."""
    def span(p0, p1):
        a, c = mdates.date2num(ts(p0)), mdates.date2num(ts(p1 + 1))
        return (a, c - a)
    foldlar = fold_al(b)
    val_bas = AY_CV[foldlar[0][1][0]]
    olgun = d[d["HEDEF_KULLANILABILIR"]]
    ust = [(span(olgun["AY"].min(), val_bas - 1), ARKA["EGITIM"], "Eğitim"),
           (span(val_bas, AY_CV.max()), RENK["VALIDASYON"], "Validasyon"),
           (span(d.loc[oot_idx, "AY"].min(), d.loc[oot_idx, "AY"].max()), RENK["OOT"], "OOT")]
    gelecek = d.loc[d["BOLGE"] == "TAHMIN", "AY"]
    if len(gelecek):
        ust.append((span(gelecek.min(), gelecek.max()), RENK["TAHMIN"], "Tahmin"))
    satir = [("Kohortlar", ust)]
    for i, (tr, val) in enumerate(foldlar, 1):
        parca = [(span(AY_CV[tr[0]], AY_CV[tr[-1]]), "#bfdbfe", None)]
        if val[0] - tr[-1] > 1:
            parca.append((span(AY_CV[tr[-1]] + 1, AY_CV[val[0]] - 1), "#e5e7eb", "hatch"))
        parca.append((span(AY_CV[val[0]], AY_CV[val[-1]]), RENK["VALIDASYON"], None))
        satir.append((f"CV Fold {i}", parca))
    fig, ax = plt.subplots(figsize=(16, 4.2))
    for yy, (ad, parcalar) in enumerate(reversed(satir)):
        for (x, gen), renk, ek in parcalar:
            ax.broken_barh([(x, gen)], (yy - 0.35, 0.7), facecolors=renk, edgecolor="white",
                           hatch="///" if ek == "hatch" else None)
            if isinstance(ek, str) and ek != "hatch":
                ax.text(x + gen / 2, yy, ek, ha="center", va="center", fontsize=9, fontweight="bold",
                        color="white" if renk != ARKA["EGITIM"] else "#334155")
    ax.set_yticks(range(len(satir))); ax.set_yticklabels([s[0] for s in reversed(satir)])
    ax.xaxis_date(); ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m")); ax.grid(axis="x", alpha=0.3)
    ax.legend(handles=[Patch(color="#bfdbfe", label="Fold eğitimi"),
                       Patch(facecolor="#e5e7eb", hatch="///", label="Boşluk: hedef o tarihte henüz belli değil"),
                       Patch(color=RENK["VALIDASYON"], label="Validasyon"), Patch(color=RENK["OOT"], label="OOT")],
              loc="upper left", fontsize=8, ncol=4, bbox_to_anchor=(0, -0.12))
    ax.set_title(f"Walk-forward yapı | bilgi ayı = {b} (öncül {b}. ayda belli, tahmin anı = kohort + {b} ay)",
                 fontweight="bold", loc="left")
    plt.tight_layout(); plt.show()


for b in [2, 3, 6]:
    print(f"=== Fold yapısı | bilgi ayı = {b} ===")
    display(fold_olustur(b)[1])
takvim_grafigi(2)
takvim_grafigi(6)
