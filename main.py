print("DONEM min/max:", m_df["DONEM"].min(), "/", m_df["DONEM"].max())
print("DONEM NaN:", m_df["DONEM"].isna().sum())
print(m_df.groupby("DONEM").size().head(12))
for score, df, ref in MODELLER:
    print(f"{score:20s} ref satır: {len(ref):>8,} | dolu skor: {ref[score].notna().sum():>8,}")
