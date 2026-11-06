print("Boş DONEM:", (lds[DONEM].str.len() != 6).sum())
gercek = lds.loc[lds[DONEM].str.len() == 6, DONEM]
print("Gerçek aralık:", gercek.min(), "→", gercek.max())
display(gercek.value_counts().sort_index().head(30))
display(lds.loc[lds[DONEM].str.len() != 6].head())
