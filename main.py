folder  = dataiku.Folder(FOLDER_ID)
adaylar = [p for p in folder.list_paths_in_partition()
           if os.path.basename(p).lower().startswith(FEATHER_ONEK)]
print("Bulunan:", adaylar)
assert len(adaylar) == 1, "lds_old ile başlayan tek dosya olmalı"

with folder.get_download_stream(adaylar[0]) as s:
    raw = s.read()
print("Boyut:", round(len(raw) / 1e9, 2), "GB | baş:", raw[:6], "| son:", raw[-6:])





f_kolon = pa.ipc.open_file(pa.BufferReader(raw)).schema.names
eksik = [c for c in KOLONLAR if c not in f_kolon]
if eksik:
    raise KeyError(f"Feather'da eksik kolon {eksik} | mevcut: {f_kolon}")

tar = pf.read_feather(pa.BufferReader(raw), columns=KOLONLAR)
del raw
tar = standartla(tar, "FEATHER")
print("FEATHER:", tar.shape, tar[DONEM].min(), "→", tar[DONEM].max())




birlesik = pd.concat([lds, tar], ignore_index=True)
cakisma = birlesik[ID].duplicated(keep="first").sum()
print(f"Çakışan başvuru: {cakisma:,}")
if cakisma:
    birlesik = birlesik.drop_duplicates(ID, keep="first")   # çakışmada güncel LDS kaydı kalır
del lds, tar

print("BİRLEŞİK:", birlesik.shape)
display(birlesik.groupby("KAYNAK")[DONEM].agg(["min", "max", "count"]))



dataiku.Dataset("LDS_MPS_TRD_HARIC_BIRLESIK").write_with_schema(birlesik)
print("Yazıldı:", birlesik.shape)
