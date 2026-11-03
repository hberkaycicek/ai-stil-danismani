ham_set = lambda k: {c for c in k if c in HAM_HAVUZ}             # ortalama/takvim hariç ham girdiler
farkli = lambda df: df[df["KOLONLAR"].apply(ham_set) != ham_set(CHAMP_SATIR["KOLONLAR"])] \
    if CONFIG["CHALLENGER_FARKLI_GIRDI"] else df
