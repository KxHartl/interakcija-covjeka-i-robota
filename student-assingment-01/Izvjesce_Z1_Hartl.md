# Klasifikacija brzih i sporih kretnji robota

**Zadatak 1**  
**Autor:** Krešimir Hartl

---

## 1. Uvod

U okviru ovog zadatka provedena je analiza klasifikacije pokreta kolaborativnih robota u kategorije "brzo" i "sporo" pomoću objektivnih mjerenja brzine i sile. Cilj istraživanja bio je kvantitativno opisati subjektivnu percepciju brzine pokreta robota te razviti klasifikacijski model koji može automatski razlikovati brze i spore kretnje na temelju izmjerenih podataka. Takva klasifikacija omogućava bolje razumijevanje interakcije čovjeka i robota u suradničkom okruženju te optimizaciju ponašanja robota za sigurniju i ugodniju suradnju.

Pokusi su provedeni na Universal Robots kolaborativnim robotima, pri čemu su prikupljeni podaci o poziciji izvršnog člana i komponentama sile tijekom ručnog vođenja robota. Prikupljeni podaci poslužili su za treniranje i validaciju Linear Discriminant Analysis (LDA) klasifikatora.

---

## 2. Eksperimentalno snimanje

### 2.1 Komunikacija i prikupljanje podataka

Komunikacija između robota i računala uspostavljena je pomoću TCP/IP protokola preko Ethernet LAN kabela. Robot je konfiguriran kao TCP klijent koji šalje podatke u realnom vremenu, dok je računalo postavljeno kao TCP server koji prima i pohranjuje te podatke. Računalo je konfigurirano sa statičkom IP adresom na određenom portu, omogućujući stabilnu mrežnu vezu između robota i računala.

Svako mjerenje sadržavalo je informacije o trenutnoj poziciji vrha alata robota (TCP) u kartezijevim koordinatama (\(x\), \(y\), \(z\)) te komponente sile koje su djelovale na alat (\(f_x\), \(f_y\), \(f_z\)). Podaci su snimani frekvencijom od 100 Hz (svakih 0.01 sekunde) i spremani u CSV format za naknadnu analizu.

### 2.2 Protokol snimanja

Svaki par studenata snimio je 20 kretnji: 10 brzih i 10 sporih. Pokret je izveden ručnim vođenjem robota u kolaborativnom režimu rada, pri čemu je robot bio postavljen u način rada prilagođen vanjskim silama. Kretnje su snimane tako da simuliraju prirodne i prepoznatljive brzine koje čovjek percipira kao brze ili spore.

Dataset korišten za treniranje klasifikatora bio je "norsic", dok su ostali parovi (akrap_gugo, cuk_pongracic, norsic_hartl) korišteni isključivo za validaciju modela na nezavisnim podacima.

---

## 3. Obrada podataka

### 3.1 Pretprocesiranje

Sirovi podaci sadržavali su segmente koji nisu relevantni za klasifikaciju. U fazi pretprocesiranja primijenjeni su sljedeći postupci:

- **Uklanjanje stacionarnih segmenata:** Eliminirani su dijelovi snimki u kojima je brzina gibanja robota bila nula ili vrlo blizu nule. Ovi segmenti predstavljaju početak i kraj snimanja kada robot miruje.

- **Uklanjanje tranzijentnih segmenata:** Filtrirani su početni i završni dijelovi snimanja koji sadržavaju nagle promjene brzine uzrokovane ubrzavanjem ili usporavanjem robota pri startu i zaustavljanju.

### 3.2 Izračun značajki

Za svako snimanje izračunate su dvije ključne značajke:

**Prosječna brzina:**  
Brzina je izračunata kao euklidska udaljenost između uzastopnih pozicija podijeljena s vremenskim korakom:

\[
v_i = \frac{\sqrt{(x_{i+1} - x_i)^2 + (y_{i+1} - y_i)^2 + (z_{i+1} - z_i)^2}}{\Delta t}
\]

gdje je \(\Delta t = 0.01\) s. Prosječna brzina \(\bar{v}\) izračunata je kao aritmetička sredina svih brzina unutar jedne snimke nakon uklanjanja irelevantnih segmenata.

**Prosječna sila:**  
Sila je prikazana kao magnituda (norma) vektora sile:

\[
F_i = \sqrt{f_{x,i}^2 + f_{y,i}^2 + f_{z,i}^2}
\]

Prosječna sila \(\bar{F}\) izračunata je kao aritmetička sredina svih magnitud sile unutar jedne snimke.

### 3.3 Struktura podataka

Obrađeni podaci organizirani su kao set značajki (\(\bar{v}\), \(\bar{F}\)) s pripadajućom oznakom klase (brzo = 1, sporo = 0). Ovakva struktura omogućava direktnu primjenu klasifikacijskih algoritama strojnog učenja.

---

## 4. Klasifikacija - Linear Discriminant Analysis (LDA)

### 4.1 Opis klasifikatora

Linear Discriminant Analysis (LDA) je supervised klasifikacijski algoritam koji pronalazi linearnu granicu između klasa maksimalizirajući razmak između njihovih centara, a minimalizirajući varijabilnost unutar svake klase. LDA pretpostavlja da su klase raspoređene prema normalnoj distribuciji i da imaju slične kovarijance.

Matematički, LDA traži projekciju podataka na nižu dimenzionalnost (u binarnoj klasifikaciji na jednu dimenziju) koja najbolje razdvaja klase. Granica odlučivanja definirana je kao:

\[
w^T x + b = 0
\]

gdje je \(w\) vektor težina, \(x\) vektor značajki, a \(b\) prag. LDA je osobito pogodan za probleme s dvije klase i malim brojem značajki, kao što je slučaj u ovom zadatku.

### 4.2 Trening

Klasifikator je treniran isključivo na skupu podataka "norsic", koji je sadržavao 20 mjerenja (10 brzih i 10 sporih). Zbog male veličine skupa podataka, nije bilo moguće izdvojiti nezavisni test set iz istog izvora. Stoga je cijeli "norsic" dataset korišten za trening, a validacija je provedena na potpuno nezavisnim skupovima drugih studenata.

Ovakav pristup donosi prednosti u smislu bolje iskorištenosti podataka za treniranje, ali postavlja strože uvjete za validaciju, budući da se model testira na podacima iz različitih izvora koji mogu imati drukčije distribucije.

---

## 5. Validacija

### 5.1 Metrike

Performanse klasifikatora evaluirane su pomoću standardnih metrika za binarnu klasifikaciju. Metrike se temelje na confusion matrix, koja opisuje četiri temeljna ishoda:

- **True Positive (TP):** Brza kretnja pravilno klasificirana kao brza
- **True Negative (TN):** Spora kretnja pravilno klasificirana kao spora
- **False Positive (FP):** Spora kretnja pogrešno klasificirana kao brza
- **False Negative (FN):** Brza kretnja pogrešno klasificirana kao spora

#### 5.1.1 Accuracy

Accuracy mjeri globalni postotak točnih predikcija:

\[
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
\]

Ova metrika pruža opći uvid u performanse, ali može biti varljiva kod nebalansiranih skupova podataka.

#### 5.1.2 Precision

Precision mjeri postotak pravilnih pozitivnih predikcija:

\[
\text{Precision} = \frac{TP}{TP + FP}
\]

Visoka vrijednost precision znači da model rijetko označava spore kretnje kao brze (mali broj false positive).

#### 5.1.3 Recall (Sensitivity)

Recall mjeri postotak pravilno detektiranih pozitivnih instanci:

\[
\text{Recall} = \frac{TP}{TP + FN}
\]

Visoka vrijednost recall znači da model rijetko propušta brze kretnje (mali broj false negative).

#### 5.1.4 F1-score

F1-score je harmonijska sredina precision i recall:

\[
F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
\]

F1-score balansira precision i recall, kažnjavajući veliki disbalans između njih.

#### 5.1.5 Matthews Correlation Coefficient (MCC)

MCC je najinformativnija metrika za binarnu klasifikaciju, posebno kod nebalansiranih skupova podataka:

\[
MCC = \frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
\]

MCC se kreće između -1 (potpuno kriva predikcija) i +1 (potpuno točna predikcija), pri čemu 0 označava predikciju koja nije bolja od nasumičnog pogađanja. MCC uzima u obzir sve četiri kategorije confusion matrice i manje je osjetljiv na nebalansirane podatke od accuracy i F1-score.

### 5.2 Rezultati validacije

Validacija je provedena na tri nezavisna skupa podataka:

- **akrap_gugo**
- **cuk_pongracic**
- **norsic_hartl**

Grafički prikazi u direktoriju `plots/` prikazuju:

- **Scatter plot svih podataka** (`scaterPlot_all_data.png`): Prikazuje distribuciju svih mjerenja u prostoru značajki (brzina, sila) s označenim oznakama klasa.
- **Scatter plot klasificiranih podataka** (`scaterPlot_klasificiran.png`): Prikazuje rezultate klasifikacije s graničnom linijom između klasa.
- **Distribucija brzine** (`distribucija_brzine.png`): Prikazuje histograme brzina za brze i spore kretnje.
- **Distribucija sile** (`distribucija_sile.png`): Prikazuje histograme sila za brze i spore kretnje.
- **Sve metrike** (`sveMetrike.png`): Prikazuje sve izračunate metrike za svaki validacijski set.
- **Accuracy** (`accuracy.png`): Prikaz točnosti klasifikacije po skupovima.

Detaljna analiza rezultata pokazuje da model učinkovito klasificira podatke kada su snimke i uvjeti snimanja slični onima u trening skupu. Međutim, kod heterogenih podataka (različiti roboti ili uvjeti), model pokazuje niže performanse, što ukazuje na potrebu za proširenjem trening skupa.

---

## 6. Zaključak

Model temeljen na Linear Discriminant Analysis pokazao se učinkovitim za klasifikaciju brzih i sporih kretnji robota na homogenim podacima. Ipak, generalizacija na različite uvjete snimanja i robotske sustave ograničena je zbog oslanjanja na apsolutne vrijednosti brzine i sile, koje variraju među različitim robotima i uvjetima.

Za razvoj robusnog i univerzalnog klasifikatora potrebno je:

- Proširiti trening skup na podatke iz više robota i uvjeta snimanja
- Razmotriti normalizaciju značajki po tipu robota ili kretnje
- Uvesti relacijske značajke koje su manje ovisne o fizičkim karakteristikama konkretnog robota
- Koristiti naprednije klasifikacijske algoritme (npr. SVM, Random Forest, neuronske mreže)

Rezultati ovog istraživanja pružaju temelje za daljnje proučavanje interakcije čovjeka i robota te razvoj sustava koji mogu autonomno prilagođavati brzinu pokreta robota na temelju konteksta i preferencija korisnika.

---

## 7. Literatura

[1] Chicco, D., & Jurman, G. (2020). The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation. *BMC Genomics*, *21*(1), 6. https://doi.org/10.1186/s12864-019-6413-7

[2] Daydulo, Y. D., Thamineni, B. L., & Dawud, A. A. (2023). Cardiac arrhythmia detection using deep learning approach and time frequency representation of ECG signals. *BMC Medical Informatics and Decision Making*, *23*(1), 232. https://doi.org/10.1186/s12911-023-02326-w

[3] Knežević, T. (2025). *01 Mjerenje i klasifikacija - Interakcija čovjeka i robota*. Fakultet strojarstva i brodogradnje, Sveučilište u Zagrebu.

[4] Knežević, T. (2025). *Ručno vođenje robota i prikupljanje podataka o pokretu - Vježbe UR5*. Fakultet strojarstva i brodogradnje, Sveučilište u Zagrebu.
