olgun_maske = d["HEDEF_KULLANILABILIR"]
KAPSAMA = pd.DataFrame([{
    "DEGISKEN": kisa_ad(c), "GOZLEM_AYI": KOLON_AYI[c],
    "TUR": "Ortalama" if c in ORTALAMA_KOL else "Takvim" if c in TAKVIM_KOL
           else "Öncül" if c in ONCULLER else "Başvuru anı",
    "OLGUN_DOLU": f"{int(d.loc[olgun_maske, c].notna().sum())}/{int(olgun_maske.sum())}",
    "ILK_DOLU_KOHORT": str(d.loc[d[c].notna(), "AY"].min()),
    "TAHMIN_DOLU": f"{int(d.loc[d['BOLGE'] == 'TAHMIN', c].notna().sum())}/{int((d['BOLGE'] == 'TAHMIN').sum())}",
} for c in HAVUZ])
display(KAPSAMA)

ilk_dolu = {"Ham": cv_df["AY"].min()}
for p in CONFIG["ORTALAMA_PENCERELERI"] if CONFIG["ORTALAMA_DENE"] else []:
    ilk_dolu[f"AGORT{p}"] = cv_df.loc[cv_df[f"{ONCULLER[1]}_AGORT{p}"].notna(), "AY"].min()
F1 = pd.DataFrame({tur: {b: int((AY_CV[fold_al(b)[0][0]] >= bas).sum()) for b in range(0, CONFIG["CHAMPION_MAX_AY"] + 1)}
                   for tur, bas in ilk_dolu.items()})
F1.index.name = "BILGI_AY"
print(f"Fold 1 eğitim kohort sayısı (MIN_TRAIN = {MIN_TRAIN}; altı kalan aday elenir):")
display(F1.style.apply(lambda s: ["background-color:#fee2e2" if v < MIN_TRAIN else "" for v in s]))
