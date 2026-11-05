import io

with FEATHER_ARSIV.get_download_stream(feather_path) as s:
    raw = s.read()

print("Boyut:", round(len(raw) / 1e6, 1), "MB")
print("İlk byte'lar:", raw[:12])
print("Son byte'lar:", raw[-12:])
