ID = "BASVURU_NUMARASI"
l = dataiku.Dataset("LDS_MPS_TRD_HARIC").get_dataframe(columns=[ID, "SONUC_DONEM", "ISLEM_SONUCU"])
l["SONUC_DONEM"] = l["SONUC_DONEM"].astype(str).str[:6]
d = l[l[ID].duplicated(keep=False)]
ozet = d.groupby("SONUC_DONEM").agg(tekrar_satir=(ID, "size"),
                                    farkli_statu=(ID, lambda s: d.loc[s.index].groupby(ID)["ISLEM_SONUCU"].nunique().gt(1).sum()))
display(ozet)
