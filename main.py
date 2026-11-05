import pyarrow as pa, pyarrow.ipc as ipc

print("Byte:", len(raw))      # lokaldeki dosya boyutuyla karşılaştır (Özellikler → Boyut, byte cinsinden)

reader = ipc.open_stream(pa.BufferReader(pa.py_buffer(raw).slice(8)))   # 'ARROW1\0\0' başlığını atla
batches = []
try:
    for b in reader:
        batches.append(b)
except Exception as e:
    print("Okuma durdu:", type(e).__name__, str(e)[:200])

tar = pa.Table.from_batches(batches, schema=reader.schema).to_pandas()
del batches
print("Kurtarılan:", tar.shape)
print("DONEM:", tar["DONEM"].min(), "→", tar["DONEM"].max())
