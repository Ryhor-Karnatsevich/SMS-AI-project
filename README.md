# Klasyfikacja SMS: sieć neuronowa a regresja logistyczna

## Cel badania

Sprawdzamy, jak ilość danych treningowych wpływa na wykrywanie spamu i czy prosta sieć neuronowa daje lepsze wyniki niż regresja logistyczna. Wiadomości dzielimy na **ham** (zwykłe) i **spam** (niechciane).

## Dane

Prowadzimy dwa osobne eksperymenty:

| Zbiór | Przed czyszczeniem | Po czyszczeniu | Ham | Spam |
|---|---:|---:|---:|---:|
| UCI SMS Spam Collection | 5 574 | 5 156 | 4 514 | 642 |
| Super SMS | 67 010 | 60 858 | 36 309 | 24 549 |

Normalizujemy tekst, usuwamy niepoprawne rekordy, duplikaty i grupy ze sprzecznymi etykietami. W Super SMS dodatkowo odrzucamy krótkie lub uszkodzone wpisy oraz powtarzające się szablony różniące się liczbami, linkami lub interpunkcją. Wykluczamy również pokrycia z oryginalnym UCI, aby nowe badanie nie korzystało ze starych wiadomości według tych reguł.

## Metoda

Tekst zamieniamy na maksymalnie **3 000 cech TF-IDF**, a następnie porównujemy:

- **MLP** — sieć neuronową z jedną warstwą ukrytą zawierającą 32 neurony i aktywacją ReLU;
- **regresję logistyczną** — model liniowy z regularyzacją L2 (`C=1`).

Dla każdego zbioru odkładamy **10% wiadomości na test**, a pozostałe 90% stanowi pulę treningową. Każdy algorytm uczymy na **10%, 20%, …, 100% tej puli**. Mniejsze próbki zawierają się w większych. Oba modele otrzymują te same podwybory danych i ten sam test w ramach danego zbioru. TF-IDF dopasowujemy wyłącznie do aktualnej próbki treningowej. Losowanie jest powtarzalne (`random_state=42`).

## Dwa niezależne badania

Nie porównujemy UCI z Super SMS. W każdym badaniu osobno porównujemy **MLP i regresję logistyczną przy 10%, 20%, …, 100% danych treningowych**. Oś X pokazuje objętość treningu, a dwie linie — dwa modele. Obserwujemy zarówno różnice między modelami przy tym samym rozmiarze próbki, jak i zmianę jakości każdego modelu wraz ze wzrostem próbki.

### Miary jakości

Ocenę opieramy na czterech miarach:

- **Accuracy** — udział poprawnych odpowiedzi w całym teście.
- **Precision** — jaka część wiadomości oznaczonych jako spam rzeczywiście jest spamem.
- **Recall** — jaką część rzeczywistego spamu wykrywa model.
- **F1** — średnia harmoniczna precision i recall.

## Badanie 1 · UCI SMS Spam Collection

**Accuracy — poprawne odpowiedzi**

![Accuracy: MLP vs regresja logistyczna na UCI](results/uci/compare_accuracy.png)

**F1 — jakość wykrywania spamu**

![F1: MLP vs regresja logistyczna na UCI](results/uci/compare_spam_f1.png)

## Badanie 2 · Super SMS

**Accuracy — poprawne odpowiedzi**

![Accuracy: MLP vs regresja logistyczna na Super SMS](results/super_sms/compare_accuracy.png)

**F1 — jakość wykrywania spamu**

![F1: MLP vs regresja logistyczna na Super SMS](results/super_sms/compare_spam_f1.png)

Na wykresach oś pozioma oznacza procent wykorzystanej puli treningowej, a pionowa — wartość miary. Linie przedstawiają oba algorytmy. Wykresy powstają po zapisaniu wyników obu modeli i sprawdzeniu zgodności danych oraz podziału. Do tego czasu nie formułujemy wniosku o przewadze któregoś z nich.

## Ograniczenia

Każdy punkt odpowiada jednemu uczeniu; nie oszacowano zmienności między losowaniami. Czyszczenie szablonów jest heurystyką i nie usuwa wszystkich podobieństw znaczeniowych. Nie przeprowadzono ręcznej kontroli wszystkich etykiet ani automatycznego rozpoznawania języka. Super SMS odczytujemy jako latin-1 zgodnie z podejściem autorów, co nie gwarantuje naprawienia wszystkich błędów kodowania.

Zbiory mają różne testy i proporcje klas, dlatego wyniki porównujemy przede wszystkim **między modelami wewnątrz tego samego badania**. Różnicy między UCI a Super SMS nie można przypisać wyłącznie ilości danych. Parametrów modeli nie dobieramy na zbiorze testowym.

## Uruchomienie

```powershell
python -m pip install -r requirements.txt
python main.py --train --model both
```

Modele są zapisywane do plików i przy kolejnych uruchomieniach wykorzystywane ponownie. Samo odświeżenie wykresów nie wymaga uczenia:

```powershell
python plots.py
```

## Źródła

- [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).
- [Super SMS Dataset — dane i kod autorów](https://github.com/smspamresearch/spstudy): Salman, Ikram, Kaafar (2024), *Investigating Evasive Techniques in SMS Spam Filtering: A Comparative Analysis of Machine Learning Models*, IEEE Access.

Uwaga: starsze wyniki MLP nie mają sumy kontrolnej danych. Ich porównanie jest oznaczone jako historyczne: zapisane parametry podziału są zgodne, ale identyczności danych nie można w pełni potwierdzić. Wykresy odświeżamy bez ponownego uczenia.

W README pokazujemy tylko accuracy i F1 jako osobne duże obrazy. Dodatkowe wykresy precision i recall zapisują się obok nich. Skala pionowa może być zawężona i jej dolna granica jest podana pod wykresem. Aby obrazy były widoczne na GitHubie, pliki PNG muszą zostać dodane do repozytorium razem z README.


