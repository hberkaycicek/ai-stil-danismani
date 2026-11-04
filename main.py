Walk-forward fold'larında olgunlaşma boşluğu yok. Bölüm 5'teki fold tablosuna göre Fold 1'in eğitimi 2023-01'de bitiyor ve validasyon hemen ardından, 2023-02'de başlıyor. Oysa 2023-02 kohortu 2. ay verisiyle 2023-04'te tahmin ediliyor. O tarihte 2022-04 ile 2023-01 arasındaki 10 kohortun 12M90 değeri henüz belli değil. Yani model, gerçek hayatta elinde olmayacak hedeflerle eğitiliyor ve validasyon sonuçları iyimser çıkacak. Önceki sürümde Fold 1'in eğitimi 2022-03'te bitiyordu (15 kohort); şimdi 19 kohort görünüyor ama bunların bir kısmı tahmin anında bilinemeyecek hedefler.

Bölüm 13'teki OOT walk-forward'da HEDEF_ERISIM < tahmin_ay filtresi hâlâ duruyorsa OOT sonuçları temiz kalır. Bunu da kontrol edeceğim.

Şimdilik not aldığım diğer noktalar:

	•	Trend artık seçilebilir (TREND_SECILEBILIR=True). Daha önce trendin gelecekte doğrusal olarak sınırsız büyüyeceği için seçilmemesine karar vermiştik.
	•	Veri seti sadece olgunlaşmış 57 kohortu içeriyor. Tahmin edilecek 202510-202603 kohortları yok, bu yüzden Bölüm 14'teki üretim tahmini bu veriyle yapılamaz.
	•	Bütün modeller 2021-07'den başlıyor. 6'lı ortalamada tam pencere zorunlu olduğu için ilk 6 kohort, ortalama değişkeni kullanmayan modellerde de eğitimden düşüyor.
	•	İyi taraflar: Geçmiş ortalamalar shift(1) ve sadece geriye bakan rolling ile, sızıntısız hesaplanıyor. Alpha aralığı artık Lasso'nun ölçeğini de kapsıyor. Alpha iç zaman bazlı CV ile seçiliyor.