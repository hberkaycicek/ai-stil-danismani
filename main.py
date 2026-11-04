Kısaca: CV, geçmişte gerçekten yaşayacağın tahmin anını taklit etmeli. O an elinde olmayan bir bilgiyi modele verirsen CV, gerçekte yapacağın işten daha kolay bir işi ölçer.

Gerçek hayatta durum (bugün, 2026-11)

2026-09 kohortunu tahmin etmek istiyorsun. 2. ay öncülü (30_2) bugün belli oldu. Modeli eğitmek için hedefi (12M90) kesinleşmiş kohortlar lazım:

Kohort:   ... 2025-09  2025-10 | 2025-11 ... 2026-08 | 2026-09
12M90:     belli     belli     |  HENÜZ BELLİ DEĞİL  | tahmin edilecek
                               |  (12 ay dolmadı)    |
Eğitim:   ✅ kullanılır         | ❌ 10 kohort boşluk  | 🎯

2025-11 ile 2026-08 arası 10 kohortun 12 ayı daha dolmadı, yani 90+ değerleri yok. Canlıda bu 10 aylık boşluk her zaman olacak. Modelin elindeki en yeni bilgi 10 ay eski.

Yeni çalışmadaki CV ne yapıyor

Fold1'de validasyon 2023-02 kohortuyla başlıyor. Bu kohortu 2. ay modeliyle gerçekte 2023-04'te tahmin ederdin. O tarihte hedefi belli olan son kohort 2022-03. Ama yeni çalışmada eğitim 2023-01'e kadar uzanıyor:

Eğitim (yeni çalışma):   2021-07 ........ 2022-03 | 2022-04 ... 2023-01 | VAL 2023-02..
                                                    ↑ bu 10 kohortun 12M90'ı
                                                      2023-04'te henüz yoktu
                                                      (2023-05 ile 2024-02 arası geldi)

Yani model, o tarihte gelecekte açıklanacak cevaplarla eğitiliyor. Sınavdan önce, henüz açıklanmamış soruların cevap anahtarına bakmak gibi.

Neden bu kadar önemli

	1.	Sızan bilgi en değerli bilgi. Komşu kohortlar birbirine çok benziyor (rho 0,75). Validasyon kohortunu en iyi açıklayan şey, hemen önceki kohortların gerçekleşmesi. Boşluk olmayınca model "son dönemin seviyesini" bedavaya öğreniyor.
	2.	Rejim kırılmasını haksız yere yakalıyor. 2024'te 60+ gecikmelerin batığa dönüşüm oranı düştü. Gerçekte model bunu ancak ~10 ay sonra fark eder. Sızıntılı CV ise hemen biliyor, bu yüzden hata küçük görünüyor.
	3.	Sonuç iyimser rakam. Boşluksuz CV'de hata 6–8, boşluklu (sızıntısız) kurguda 9–12 çıkıyordu. Canlıda göreceğin hata ikinci aralığa yakın olur. Model seçimi de yanlış rakama göre yapılmış olur.

Boşluk ne kadar olmalı

Boşluk, kaçıncı ayda tahmin yaptığına bağlı: 12 − bilgi ayı kadar kohort.

Model	Tahmin anı	Eğitimden çıkan kohort sayısı
2. ay (30_2)	AY + 2	10
3. ay (60_3)	AY + 3	9
6. ay (60_6)	AY + 6	6

Yamadaki fold_al(bilgi_ay) bunu yapıyor: her fold'da sadece AY + 12 < tahmin_ayı olan kohortları eğitimde bırakıyor. CV böylece canlıdaki şartları birebir taklit ediyor.

Aynı mantık OOT için de geçerli (Kritik 2): OOT'de her kohort, kendi tahmin anında hedefi belli olan verilerle tahmin edilmeli.