import pyarrow.ipc as ipc

# Teşhis: sondaki sıfır bölge ne kadar büyük?
n, mv, i, step = len(raw), memoryview(raw), len(raw), 1 << 20
while i > 0:
    blok = bytes(mv[max(0, i - step):i]).rstrip(b"\x00")
    if blok:
        i = max(0, i - step) + len(blok); break
    i -= step
print(f"Toplam byte: {n:,} | sondaki sıfır: {n - i:,} byte")

# Kurtarma: 'ARROW1\0\0' başlığını atla, record batch'leri sırayla oku
reader = ipc.open_stream(pa.BufferReader(pa.py_buffer(raw).slice(8)))
eksik = [c for c in KOLONLAR if c not in reader.schema.names]
if eksik:
    raise KeyError(f"Feather'da eksik kolon {eksik} | mevcut: {reader.schema.names}")

parcalar, hata = [], None
try:
    for b in reader:
        parcalar.append(pa.Table.from_batches([b]).select(KOLONLAR))
except Exception as e:
    hata = e
print("Okunan batch:", len(parcalar), "| durma sebebi:", repr(hata) if hata else "temiz bitti")

tar = pa.concat_tables(parcalar).to_pandas()
del parcalar, raw, mv
tar = standartla(tar, "FEATHER")
print("FEATHER:", tar.shape, tar[DONEM].min(), "→", tar[DONEM].max())
display(tar[DONEM].value_counts().sort_index())
