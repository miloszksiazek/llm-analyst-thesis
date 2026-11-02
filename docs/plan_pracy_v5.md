# Plan pracy magisterskiej – wersja 5

**Agent LLM jako analityk inwestycyjny** — wersja odchudzona: jeden model (Claude Opus 5.5) z siecią i bez sieci, 4 spółki po 25%, 28 tygodni testu, uruchamianie w chmurze. Miłosz, Data Science and Business Analytics, WNE UW · 2 listopada 2026

## Co się zmieniło względem wersji 4

Wersja 5 zachowuje rdzeń zaakceptowany 29.10 (Dow Jones 30, test w przód, brak pamięci, IC jako główna miara, XGBoost, Alpaca), ale usuwa drugi model i upraszcza wszystko, co nie służy pytaniom badawczym. Budowa skraca się mniej więcej o połowę.

| Element | Wersja 4 | Wersja 5 (obecna) | Dlaczego |
| --- | --- | --- | --- |
| Modele | Claude Opus 5.5 i Gemini 3.1 Pro | tylko Claude Opus 5.5 | Gemini 3.1 Pro jest w wersji preview i może zostać podmieniony w trakcie 28-tygodniowego testu; jeden dostawca to połowa kodu i testów |
| Warianty | układ 2×2 (4 agentów) | 2 warianty: Claude z siecią i bez sieci | porównanie z siecią vs bez sieci (RQ3) zostaje w całości |
| Pytania badawcze | RQ1–RQ5 | RQ1–RQ4 | RQ5 (Claude vs Gemini) i H5 usunięte |
| Wybór spółek | 3 spółki po 1/3 | 4 spółki po 25% | większa dywersyfikacja portfela; IC liczony na pełnym rankingu, więc bez zmian |
| Długość testu | ok. 30 tygodni | 28 tygodni (6.12.2026 – 21.06.2027) | stała data końca, zapas na analizę latem |
| Alpaca | 5 portfeli na 2 kontach | 3 portfele na 1 koncie | znika problem drugiego konta i regulaminu |
| Środowisko | lokalnie, chmura po pilotażu | GitHub Actions od pilotażu | pominięty tydzień to dziura w danych; chmura nie zależy od włączonego laptopa |
| Analiza jakościowa | forma do ustalenia | kodowanie przez LLM + ręczna weryfikacja próby | najbardziej czasochłonna ręczna część pracy |
| Budżet API | ok. 330–370 zł | ok. 230–260 zł | dwa warianty zamiast czterech |

## 1. Temat w pigułce

Praca sprawdza, czy agent oparty na dużym modelu językowym, który sam wyszukuje informacje w sieci (newsy o spółkach, wyniki finansowe, decyzje Fed, dane makro) i na tej podstawie co tydzień ocenia akcje, podejmuje lepsze decyzje niż ten sam model bez dostępu do sieci oraz model uczenia maszynowego widzący wyłącznie dane liczbowe. Co tydzień agent ocenia wszystkie 30 spółek z indeksu Dow Jones, układa je w ranking i wskazuje cztery spółki do kupienia. Skrypt automatycznie składa zlecenia na rachunkach paper trading w Alpace i niezależnie liczy wyniki wszystkich portfeli.

Kluczowe porównanie to ten sam model z wyszukiwarką i bez niej: ten sam prompt, ten sam pakiet danych liczbowych, różni się wyłącznie narzędzie wyszukiwania. Mierzy to bezpośrednio wartość samego researchu, czego nie da się zrobić, porównując agenta z XGBoost, bo tam zmienia się jednocześnie typ modelu i rodzaj informacji.

Kluczowa decyzja metodologiczna pozostaje bez zmian: test w przód, na okresie po dacie odcięcia wiedzy treningowej modelu. Backtest LLM-a na danych historycznych jest skażony, bo model mógł w treningu zobaczyć przyszłe ceny, a agenta z wyszukiwaniem nie da się cofnąć w czasie, bo internetu nie da się przywrócić do stanu sprzed lat.

### Proponowany tytuł (praca po angielsku)

**EN:** An LLM Agent as an Investment Analyst: Does Web-Based Research Improve Stock Selection? A Forward Test against Machine Learning on the Dow Jones 30

**PL:** Agent LLM jako analityk inwestycyjny – czy research oparty na wyszukiwaniu informacji poprawia wybór akcji? Test w przód na tle uczenia maszynowego na spółkach z indeksu Dow Jones 30

## 2. Cel, pytania badawcze i hipotezy

**Cel główny:** ocena, czy cotygodniowe rankingi agenta LLM prowadzącego samodzielny research w sieci zawierają informację o przyszłych stopach zwrotu, czy research poprawia te rankingi względem tego samego modelu bez dostępu do sieci oraz czy portfele oparte na decyzjach agenta osiągają lepsze wyniki niż indeks, portfel równoważony i model XGBoost.

### Pytania badawcze

- **RQ1 (moc predykcyjna):** Czy cotygodniowe rankingi spółek tworzone przez agenta są dodatnio skorelowane ze stopami zwrotu w kolejnym tygodniu (rank IC > 0)?
- **RQ2 (wyniki portfeli):** Czy portfele agenta osiągają wyższy zwrot skorygowany o ryzyko niż buy & hold DIA, portfel 1/N i strategia XGBoost?
- **RQ3 (wartość researchu):** Czy dostęp do wyszukiwania w sieci poprawia rankingi agenta względem tego samego modelu bez dostępu do sieci?
- **RQ4 (jakość rozumowania):** Jak agent rozumuje o makroekonomii i spółkach, jak reaguje na nieoczekiwane wydarzenia i jak często opiera decyzje na błędnych lub nieaktualnych informacjach?

### Hipotezy

- **H1:** Średni tygodniowy rank IC agenta z wyszukiwaniem jest dodatni, ale słaby (rzędu kilku setnych).
- **H2:** Portfele agenta nie przewyższają istotnie statystycznie DIA, 1/N ani XGBoost w okresie testu, bo 28 tygodni to za mało, by wykazać istotną różnicę w zwrotach.
- **H3:** Wariant z wyszukiwaniem osiąga wyższy średni IC niż wariant bez wyszukiwania, ale różnica jest niewielka.
- **H4:** Raporty agenta zawierają spójne narracje makroekonomiczne, ale część decyzji opiera się na nieaktualnych lub błędnych informacjach, częściej w wariancie bez sieci.

## 3. Struktura pracy (ok. 60 stron)

Budżet stron dotyczy części zasadniczej; bibliografia, spisy i załączniki (prompt, przykładowe raporty agenta, opis repozytorium) są poza limitem. Struktura odpowiada wytycznym promotora: teoria i hipotezy, dane, metodologia z uzasadnieniem wyboru metod, analiza empiryczna. Strony zwolnione po usunięciu porównania Claude vs Gemini przechodzą do rozdziału 5 (stabilność rankingów, SHAP XGBoosta zestawiony z uzasadnieniami agenta).

| Rozdział | Strony | Charakter |
| --- | --- | --- |
| Introduction | 3 | motywacja, luka badawcza, cel, RQ, wkład |
| 1. Markets, macroeconomics and investment strategies | 9 | teoria |
| 2. LLMs and AI agents in finance — literature review | 8 | przegląd krytyczny, hipotezy |
| 3. Data | 5 | uniwersum, pakiet danych, dane do XGBoost, statystyki opisowe |
| 4. Research design and methodology | 10 | agent, benchmarki, miary, testy, kodowanie raportów |
| 5. Quantitative results | 12 | IC, portfele, z siecią vs bez, stabilność rankingów, SHAP |
| 6. Qualitative analysis of agent reports | 7 | makro, wydarzenia, źródła, błędy |
| 7. Discussion and limitations | 4 | interpretacja, ograniczenia, implikacje |
| Conclusions | 2 | weryfikacja hipotez, dalsze badania |
| Razem | 60 | |

### Krótki opis rozdziałów

- **Introduction:** inwestorzy indywidualni coraz częściej pytają czatboty o porady, modele zyskały wyszukiwanie w sieci, a publiczne eksperymenty (Alpha Arena) nie mają kontroli statystycznej. Luka: dominują backtesty obarczone wyciekiem informacji, rzadko testuje się agenta robiącego research na żywo i porównuje go z ML oraz z samym sobą bez sieci.
- **Rozdział 1:** hipoteza rynku efektywnego; wpływ stóp procentowych, inflacji, cyklu koniunkturalnego i wyników spółek na ceny akcji; ML w prognozowaniu stóp zwrotu; miary oceny strategii i pułapki backtestów.
- **Rozdział 2:** od analizy sentymentu do LLM; LLM jako narzędzie prognostyczne (Lopez-Lira i Tang; Pelster i Val); agenci tradingowi (FinMem, TradingAgents); lookahead bias; testy na żywo i ich ograniczenia; sformułowanie hipotez.
- **Rozdział 3:** dlaczego Dow Jones 30; okres badania; skład pakietu danych dla agenta; dane historyczne do XGBoost; definicje zmiennych i statystyki opisowe.
- **Rozdział 4:** architektura agenta i protokół tygodniowy; uzasadnienie wariantu bez sieci i braku pamięci; benchmarki; konstrukcja portfeli; rank IC, Sharpe, maksymalne obsunięcie; testy statystyczne (t-test na IC, test na parach, bootstrap); dobór hiperparametrów XGBoost (walidacja walk-forward); schemat kodowania raportów i jego weryfikacja.
- **Rozdział 5:** statystyki ocen i rankingów; IC wszystkich strategii (RQ1); krzywe kapitału i tabela miar (RQ2); z siecią vs bez sieci (RQ3); stabilność rankingów z tygodnia na tydzień; SHAP XGBoosta zestawiony z uzasadnieniami agenta; zgodność skryptu z Alpaką.
- **Rozdział 6:** narracje makro (Fed, inflacja, rynek pracy); studia przypadków 3–4 tygodni z ważnymi wydarzeniami; z jakich źródeł korzystał agent; błędy i halucynacje w obu wariantach; czy lepsze rozumowanie dawało lepsze decyzje (RQ4).
- **Rozdział 7:** wyniki na tle literatury; ograniczenia (jeden model, krótki okres, jeden reżim rynkowy, symulowane wykonanie, zmiany modelu i wyszukiwarki); wnioski dla inwestora indywidualnego.

## 4. Projekt badania — szczegóły

### 4.1 Rynek i uniwersum spółek

Rynkiem są akcje amerykańskie: Alpaca obsługuje tylko rynek USA, a dla spółek z Dow Jones jest ogromna ilość newsów i analiz po angielsku, więc agent ma z czego robić research. Uniwersum to 30 spółek wchodzących w skład indeksu na dzień startu testu, zamrożone do końca badania: jeśli spółka wypadnie z indeksu, nadal jest oceniana, a nowa nie jest dodawana. Przy 28 tygodniach daje to 840 par „pozycja w rankingu → zwrot w kolejnym tygodniu” na każdy wariant.

### 4.2 Warianty agenta

| Wariant | Model | Co dostaje |
| --- | --- | --- |
| Claude + sieć | Claude Opus 5.5, rozumowanie adaptacyjne | pakiet danych + wyszukiwarka (limit ok. 10 wyszukiwań) |
| Claude bez sieci | Claude Opus 5.5, rozumowanie adaptacyjne | tylko pakiet danych, bez wyszukiwarki |

Wariant bez sieci nie doradza z pamięci. Dostaje dokładnie ten sam pakiet danych liczbowych co wariant z wyszukiwarką, bo jego wiedza treningowa kończy się przed startem testu i bez pakietu nie znałby nawet bieżących cen. Odcięta jest wyłącznie wyszukiwarka, więc różnica w IC między wariantami mierzy wartość samego researchu.

Używany jest jeden model, bo Gemini 3.1 Pro jest dostępny tylko w wersji preview, a w 28-tygodniowym teście z zamrożonymi ustawieniami podmiana modelu przez dostawcę zniszczyłaby spójność danych. Ograniczenie wyników do jednego dostawcy jest opisane w rozdziale 7; porównanie kilku modeli to naturalne rozszerzenie w dalszych badaniach.

### 4.3 Jak działa agent (protokół tygodniowy)

Raz w tygodniu, w niedzielę wieczorem czasu polskiego, zaplanowane zadanie w chmurze uruchamia oba warianty z identycznym promptem po angielsku.

1. **Pakiet danych:** ceny i zmiany z tygodnia, miesiąca i kwartału; podstawowe wskaźniki (P/E, stopa dywidendy, kapitalizacja); kalendarz wyników spółek na najbliższy tydzień; dane makro (stopa Fed, inflacja, rentowności obligacji, kurs dolara). Pakiet jest identyczny dla obu wariantów i nie zawiera informacji o portfelu ani wcześniejszych decyzjach.
2. **Analiza makro (tylko wariant z siecią):** faza cyklu, polityka Fed, sektory zyskujące i tracące.
3. **Research spółek (tylko wariant z siecią):** newsy i wyniki dla najciekawszych kandydatów, sztywny limit wyszukiwań.
4. **Decyzja:** ocena każdej z 30 spółek w skali –5…+5, pełny ranking 1–30 bez remisów, wybór czterech spółek do kupienia, uzasadnienie i lista źródeł.
5. **Zapis:** wynik w formacie JSON (do analizy) i czytelny raport; zapisywany jest też pełny prompt, wersja modelu, lista wyszukiwań i źródeł, streszczenie rozumowania i zużycie tokenów. API udostępnia tylko streszczenie rozumowania, więc w pracy analizujemy deklarowane uzasadnienie agenta, a nie rzeczywiste przyczyny jego decyzji.

**Zasady decyzji.** Agent zawsze wybiera cztery spółki, także w słabym tygodniu; wtedy te, które jego zdaniem wypadną najlepiej względem pozostałych. Nie może zostać w gotówce, bo wtedy mierzylibyśmy market timing, którego XGBoost i 1/N nie mają. Ustawienia (poziom rozumowania, limit wyszukiwań, prompt, reguła awaryjna przy błędnej odpowiedzi) są zamrażane po pilotażu.

### 4.4 Pamięć agenta: brak, każdy tydzień od czystej karty

Agent nie widzi swoich wcześniejszych raportów ani portfela; skrypt zna portfele tylko po to, by złożyć zlecenia. Decyzja i pełne porównanie opcji są takie jak w wersji 4, a przy dwóch wariantach argumenty są te same:

- **Czystość porównania z siecią vs bez sieci:** oba warianty dostają co tydzień bajt w bajt ten sam pakiet. Przy pamięci każdy wariant miałby inną historię portfela i raportów, a różnica narastałaby z tygodnia na tydzień.
- **Poprawność statystyki:** test t na tygodniowych IC zakłada przybliżoną niezależność tygodni; pamięć powoduje autoskorelację rankingów i zawyża istotność.
- **Porównywalność z benchmarkami:** XGBoost i 1/N też co tydzień liczą wszystko od nowa na tych samych danych; rola analityka to ocena spółek w danym momencie.
- **Analiza jakościowa i niezawodność:** błąd z jednego tygodnia nie przechodzi na kolejne, a nieudany tydzień nie psuje następnych.

Koszt: wyższy obrót portfela (Alpaca nie pobiera prowizji od akcji, koszty transakcyjne w analizie wrażliwości) i mniejszy realizm zarządzającego portfelem (opisany w ograniczeniach). Zysk dodatkowy: stabilność rankingów z tygodnia na tydzień pokazuje, jak bardzo agent zmienia zdanie przy podobnych danych.

### 4.5 Portfele i rachunki paper trading (Alpaca)

- **Konstrukcja:** cztery spółki wskazane przez agenta, każda po 25% kapitału. XGBoost działa tak samo: cztery spółki z najwyższą prognozą, po równo. Równe wagi sprawiają, że porównanie dotyczy wyłącznie wyboru spółek.
- **Kapitał:** 100 000 USD na portfel, akcje ułamkowe, w głównej analizie bez kosztów transakcyjnych; koszty doliczane w analizie wrażliwości.
- **Wykonanie:** w poniedziałek zlecenia realizowane na otwarciu sesji (sprzedaż spółek spoza nowego wyboru, zakup nowych). Gdy w poniedziałek giełda w USA jest zamknięta, zlecenia idą na otwarcie najbliższej sesji, a agent nie jest uruchamiany ponownie. Typ zlecenia dla akcji ułamkowych na otwarciu sprawdzamy w pilotażu.
- **Konto:** trzy portfele z decyzjami (Claude z siecią, Claude bez sieci, XGBoost) mieszczą się na jednym koncie Alpaca z trzema rachunkami paper. 1/N, DIA i losowe portfele są liczone tylko w skrypcie.
- **Źródło danych do analizy:** skrypt, który liczy wszystkie portfele po kursach otwarcia w identyczny sposób. Alpaca jest niezależną ewidencją wykonania (forma zaakceptowana przez promotora); rozbieżności między nią a skryptem to mały, osobny wynik. Integracja z Alpaką jest budowana jako ostatni element przed pilotażem; jeśli się nie uda, badanie działa dalej na samym skrypcie.

### 4.6 Benchmarki

| Benchmark | Rola w badaniu (szczegóły w sekcji 9.3) |
| --- | --- |
| Buy & hold DIA | ETF śledzący Dow Jones z dywidendami; minimalny próg: czy aktywne wybieranie spółek ma sens |
| Portfel 1/N | po 1/30 w każdej spółce; naiwna dywersyfikacja, w literaturze trudna do pobicia |
| XGBoost | ML widzący tylko liczby; co tydzień cztery spółki z najwyższą prognozą; wytłumaczalność przez SHAP |
| Losowe 4 spółki | rozkład wyników 10 000 losowych portfeli 4-spółkowych liczony w skrypcie; pokazuje, czy wynik agenta nie jest szczęściem |

**XGBoost w wersji minimalnej.** Model korzysta z dziennych danych z Yahoo Finance (ceny otwarcia, zamknięcia, maksimum, minimum, wolumen, skorygowane o dywidendy i splity), a źródłem zapasowym są dane historyczne Alpaca. Cechy są wyłącznie cenowe i techniczne: momentum z 1, 4, 12 i 26 tygodni, zmienność, wolumen względny, rozpiętość dzienna, odległość od średnich kroczących, zwrot względem indeksu, plus sektor (darmowe dane fundamentalne nie są point-in-time, sekcja 9.2). Próbka treningowa to obserwacje tygodniowe: cechy z piątkowego zamknięcia, cel to zwrot od poniedziałkowego otwarcia do kolejnego poniedziałkowego otwarcia, 30 spółek, ok. 10 lat historii, czyli ok. 15 tys. wierszy. Hiperparametry: walidacja walk-forward na niewielkiej siatce (głębokość drzew, tempo uczenia, liczba drzew), bez rozbudowanej optymalizacji. Model jest trenowany raz przed startem i zamrażany na cały test; survivorship bias opisany jako ograniczenie.

### 4.7 Ewaluacja

- **Moc predykcyjna (RQ1):** rank IC liczony co tydzień na rankingu 1–30; średnia z 28 tygodni, t-statystyka, odsetek tygodni z IC > 0. Ta sama miara dla XGBoost (ranking z prognoz modelu).
- **Wyniki portfeli (RQ2):** zwrot skumulowany, zmienność, wskaźnik Sharpe’a, maksymalne obsunięcie, obrót; pozycja agenta w rozkładzie losowych portfeli. Porównania bootstrapem; przy 28 tygodniach wnioski formułowane ostrożnie.
- **Z siecią vs bez sieci (RQ3):** tygodniowa różnica IC między wariantami (28 par), test t dla par i test Wilcoxona jako kontrola odporności.
- **Stabilność i obrót:** korelacja Spearmana między rankingami z kolejnych tygodni oraz tygodniowy obrót portfela, osobno dla obu wariantów i XGBoost.
- **Analiza jakościowa (RQ4):** sekcja 4.8.

### 4.8 Analiza jakościowa raportów: kodowanie przez LLM z ręczną weryfikacją

Wszystkie 56 raportów (28 tygodni × 2 warianty) koduje osobne wywołanie LLM według stałego schematu, bez dostępu do wyników zwrotów. Autor ręcznie koduje losową próbę 15 raportów, a zgodność z kodowaniem LLM-a mierzona jest współczynnikiem kappa Cohena dla każdej kategorii. Kategorie z kappa poniżej ok. 0,6 są w całości kodowane ręcznie.

Schemat kodowania (do potwierdzenia z promotorem):

- główny temat makro (Fed i stopy, inflacja, rynek pracy, wzrost gospodarczy, geopolityka, brak);
- typ uzasadnienia wyboru (momentum, wycena, wyniki spółki, news, makro/sektor);
- liczba i rodzaj źródeł (media finansowe, komunikaty spółek, dane oficjalne, inne);
- błąd faktograficzny: tak/nie, z opisem.

Część błędów da się wykryć automatycznie: liczby podane przez agenta (ceny, P/E, stopa Fed) skrypt porównuje z pakietem danych z tego samego tygodnia. Pozostałe twierdzenia oznaczone przez LLM jako wątpliwe autor sprawdza ręcznie.

### 4.9 Kosztorys API

Subskrypcja Claude Pro nie pokrywa API: skrypt korzysta z osobnego konta w Claude Console, rozliczanego za każdy token. Ceny (październik 2026): Claude Opus 5.5 — 4 USD za milion tokenów wejściowych i 20 USD za milion wyjściowych (tokeny rozumowania liczone jako wyjściowe), wyszukiwanie 10 USD za 1000 wyszukiwań. Założenia jak w wersji 4: wariant z siecią ok. 150 tys. tokenów wejściowych i 15 tys. wyjściowych, wariant bez sieci ok. 10 tys. i 12 tys.

| Pozycja | Koszt / uruchomienie | 31 uruchomień* |
| --- | --- | --- |
| Claude Opus 5.5 + sieć | ok. 1,00 USD | ok. 31 USD |
| Claude Opus 5.5 bez sieci | ok. 0,30 USD | ok. 9 USD |
| Kodowanie raportów (56 raportów) | — | ok. 3–5 USD |
| Testy i debugowanie skryptu | — | ok. 20–25 USD |
| Razem | | ok. 63–70 USD (ok. 230–260 zł przy kursie ok. 3,7) |

\*28 tygodni testu + 2–3 tygodnie pilotażu. Mieści się z dużym zapasem w budżecie zatwierdzonym przez promotora (400–500 zł). Ograniczenie kosztów: stały poziom rozumowania, limit długości pobieranych stron, cache’owanie powtarzalnej części promptu, twardy limit wydatków w konsoli API.

### 4.10 Środowisko: GitHub Actions od pilotażu

Cotygodniowe uruchomienie działa od początku w chmurze jako zaplanowane zadanie w GitHub Actions, a nie na komputerze autora. Klucze API (Anthropic, Alpaca) są przechowywane jako sekrety repozytorium. Każde uruchomienie zapisuje raporty, pakiet danych i wyniki jako commit do repozytorium, co jednocześnie archiwizuje dane i utrzymuje aktywność repozytorium (GitHub wyłącza zaplanowane zadania w nieaktywnych repozytoriach).

- **Harmonogram:** zadanie główne w niedzielę wieczorem (agent, XGBoost, zapis), zadanie zleceń w poniedziałek przed otwarciem sesji w USA. Harmonogram w GitHub Actions jest w UTC, więc godzina polska przesunie się o godzinę po zmianie czasu 28.03.2027; nie ma to wpływu na badanie.
- **Powiadomienia:** mail przy każdym błędzie lub braku pliku z decyzją; ręczne ponowne uruchomienie możliwe do poniedziałku rana.
- **Reprodukowalność:** repozytorium z kodem, promptem i wszystkimi raportami jest załącznikiem do pracy.

## 5. Ryzyka i jak je ograniczyć

Największe ryzyko wersji 5 to kalendarz: test w przód musi wystartować najpóźniej 6.12.2026, inaczej przesuwa się cała praca. Pozostałe ryzyka mają gotowe obejście.

| Ryzyko | Dlaczego to problem | Jak ograniczyć |
| --- | --- | --- |
| Sztywny kalendarz | Opóźniony start przesuwa koniec testu i analizę. | Budowa do 20.11, pilotaż 22.11 i 29.11, start 6.12; Alpaca budowana na końcu i odpuszczalna. |
| Pominięty tydzień | Błąd API, awaria zadania w chmurze. | GitHub Actions + mail przy błędzie; ręczne ponowne uruchomienie do poniedziałku rana; reguła braku danych opisana w metodologii. |
| Mało decyzji | 28 tygodni to za mało na istotne różnice w zwrotach portfeli. | Główna miara to IC na 30 spółkach (840 obserwacji na wariant); ograniczenie opisane wprost. |
| Słaby test RQ3 | Tylko 28 par tygodniowych różnic IC. | Test na parach + Wilcoxon; hipoteza H3 sformułowana ostrożnie; wynik „brak istotnej różnicy” też jest wynikiem. |
| Jeden model | Wyniki mogą być specyficzne dla Claude’a. | Uzasadnienie wyboru (stabilność wersji), opis w ograniczeniach, kilka modeli jako dalsze badania. |
| Wynik zerowy | Agent może nie pobić indeksu ani XGBoost. | Konserwatywne hipotezy; RQ3 i analiza jakościowa dają wartość niezależnie od zysków. |
| Halucynacje | Decyzja oparta na zmyślonej informacji. | To wynik, nie wada: kodowane w rozdziale 6; wymóg podawania źródeł; automatyczne porównanie liczb z pakietem danych. |
| Zmiana modelu | Dostawca może zmienić model lub wyszukiwarkę. | Przypięta wersja modelu, pełny zapis odpowiedzi; zmiany opisane w ograniczeniach. |
| Wyciek w XGBoost | Bieżące dane fundamentalne w treningu to informacja z przyszłości. | Tylko cechy cenowe i techniczne (sekcja 9.2). |
| Źródło danych | yfinance to nieoficjalna biblioteka i może przestać działać. | Zapasowo dane historyczne Alpaca; dane zapisywane w repozytorium zaraz po pobraniu. |
| Rozrost zakresu | Kusi dodawanie modeli i wariantów. | Plan minimum (sekcja 6) jest zamknięty; rozszerzenia tylko po starcie testu. |

## 6. Plan minimum i rozszerzenia

Plan minimum wystarcza na pełną pracę zgodną z wytycznymi; rozszerzenia dodajemy tylko wtedy, gdy test już działa stabilnie.

| Plan minimum (obowiązkowe) | Rozszerzenia (jeśli starczy czasu) |
| --- | --- |
| Claude Opus 5.5 z siecią i bez sieci | wariant z pamięcią (poprzedni raport) jako test wrażliwości |
| Dow Jones 30, jedno uruchomienie tygodniowo, 28 tygodni | kilka uruchomień w tym samym tygodniu: pomiar losowości odpowiedzi |
| Benchmarki: DIA, 1/N, XGBoost, losowe portfele | SHAP rozszerzony o analizę po sektorach |
| IC + portfele + kodowanie raportów | drugi model (np. GPT lub Gemini po wyjściu z wersji preview) |

## 7. Harmonogram (obrona we wrześniu 2027)

Test trwa 28 tygodni: pierwsze uruchomienie 6.12.2026, ostatnie 13.06.2027, ostatni tydzień rozliczany na otwarciu 21.06.2027.

| Okres | Zadania |
| --- | --- |
| 2–8 listopada 2026 | Mail do promotora o zmianach (wersja 5). Konta: Claude Console z limitem wydatków, Alpaca (3 rachunki paper), repozytorium GitHub. Pobranie danych dziennych. |
| 9–20 listopada | Skrypt: pakiet danych, dwa warianty agenta, zapis raportów, ewidencja portfeli. XGBoost: trening, walidacja walk-forward, zamrożenie. Zadanie w GitHub Actions z powiadomieniami. Na końcu zlecenia w Alpace. |
| 22 i 29 listopada | Pilotaż: dwa pełne uruchomienia w chmurze, kontrola kosztów i formatu odpowiedzi, poprawki promptu. Potem zamrożenie promptu i ustawień. |
| 6.12.2026 – 21.06.2027 | Właściwy test, 28 tygodni. Równolegle rozdziały 1–4 (grudzień–marzec) i cotygodniowa kontrola logów. |
| od kwietnia 2027 | Kodowanie raportów na bieżąco (LLM), ręczna próba po zakończeniu testu. |
| koniec czerwca – sierpień 2027 | Analiza ilościowa i jakościowa, rozdziały 5–7, wstęp i zakończenie, poprawki po uwagach promotora. |
| wrzesień 2027 | Obrona. |

Święta w USA w poniedziałki w okresie testu (Martin Luther King Day 18.01, Presidents' Day 15.02, Memorial Day 31.05.2027): zlecenia przechodzą na otwarcie wtorkowej sesji.

## 8. Decyzje

### 8.1 Podjęte

| Kwestia | Decyzja |
| --- | --- |
| Rynek (promotor) | Dow Jones 30, skład zamrożony na dzień startu |
| Model | tylko Claude Opus 5.5 z rozumowaniem, wersja przypięta, zapisywane streszczenia rozumowania |
| Warianty | z siecią i bez sieci; ten sam prompt i pakiet danych |
| Pamięć agenta | brak; każdy tydzień od czystej karty |
| Portfel | 4 spółki po 25%, zawsze pełne zainwestowanie, bez gotówki |
| Wynik agenta | ocena –5…+5 + ranking 1–30 bez remisów; IC liczony na rankingu |
| Długość testu | 28 tygodni, 6.12.2026 – 21.06.2027 |
| Kapitał | 100 000 USD na portfel, akcje ułamkowe, bez kosztów w głównej analizie |
| Alpaca (promotor) | jedno konto, 3 rachunki paper jako ewidencja wykonania; główne dane ze skryptu |
| XGBoost (promotor) | dane dzienne Yahoo Finance; cechy cenowe i techniczne; trenowany raz i zamrożony |
| Środowisko | GitHub Actions od pilotażu, powiadomienia mailem, repozytorium jako załącznik |
| Analiza jakościowa | kodowanie przez LLM + ręczna próba 15 raportów, kappa Cohena |
| Język | praca, prompt i raporty po angielsku |
| Święta w USA | zlecenia na otwarcie następnej sesji, bez ponownego uruchamiania agenta |

### 8.2 Do potwierdzenia z promotorem

- [ ] Rezygnacja z Gemini i układu 2×2 na rzecz jednego modelu (argument: wersja preview Gemini a stabilność 28-tygodniowego testu).
- [ ] Zmiana z 3 na 4 spółki w portfelu.
- [ ] Brak pamięci agenta (uzasadnienie w sekcji 4.4).
- [ ] Kodowanie raportów przez LLM z ręczną weryfikacją próby: czy to wystarczająca forma analizy jakościowej.
- [ ] Czy survivorship bias w danych treningowych XGBoost wystarczy opisać jako ograniczenie.
- [ ] Potwierdzenie pisania pracy po angielsku i obrony we wrześniu 2027.

## 9. Wyjaśnienia dla autora

*Ta część nie wejdzie do pracy. To notatki do nauki pojęć, które pojawią się w rozmowie z promotorem.*

### 9.1 Information Coefficient (IC)

Agent co tydzień układa 30 spółek od najlepszej do najgorszej. Tydzień później wiemy, jak naprawdę zachowały się ceny, więc te same spółki możemy ułożyć od największego do najmniejszego zwrotu. IC mówi, jak podobne są te dwie kolejności. Formalnie rank IC to współczynnik korelacji rang Spearmana między rankingiem agenta a rankingiem faktycznych zwrotów w kolejnym tygodniu. Przyjmuje wartości od –1 do +1: +1 to idealne przewidzenie kolejności, 0 to brak związku (jak losowanie), –1 to trafienie dokładnie odwrotnie.

Przykład na 5 spółkach: agent ustawia A, B, C, D, E, a zwroty ułożyły się B, A, C, E, D. Zamienione są tylko sąsiednie pary, więc IC wynosi ok. 0,8. Gdyby kolejność była E, D, C, B, A, IC wyniósłby –1. Na rynku akcji nikt nie osiąga takich wartości: IC 0,02–0,05 uważa się za użyteczny sygnał, a 0,1 za bardzo dobry. Dlatego H1 mówi o kilku setnych.

Dlaczego to główna miara? Wynik portfela zależy od czterech spółek tygodniowo, a IC wykorzystuje cały ranking: 30 spółek × 28 tygodni = 840 obserwacji na wariant. IC ocenia samą wiedzę agenta, niezależnie od budowy portfela, i pozwala porównać na jednej skali oba warianty Claude’a i XGBoost. Po 28 tygodniach mamy 28 wartości IC na wariant: podajemy średnią, odsetek tygodni z IC > 0 (hit rate) i t-statystykę = średnia IC / (odchylenie standardowe IC / √28). Wartość powyżej ok. 2 oznacza, że średni IC jest istotnie większy od zera. Dla RQ3 liczymy w każdym tygodniu różnicę IC (z siecią minus bez sieci) i testujemy, czy średnia różnica jest dodatnia. Ranking bez remisów jest potrzebny, bo przy samej skali –5…+5 wiele spółek dostałoby tę samą ocenę, a remisy osłabiają korelację rang.

### 9.2 Dane point-in-time i wyciek informacji w XGBoost

Krytykujemy backtesty LLM-ów za to, że model zna przyszłość z danych treningowych; ten sam błąd łatwo popełnić przy XGBoost. Warunek uczciwości: cecha z tygodnia X musi być dokładnie taką wartością, jaką inwestor mógł znać w tygodniu X. Darmowe dane fundamentalne (np. z yfinance) to zwykle wartości bieżące albo historia bez daty publikacji: wyniki za IV kwartał 2020 spółka publikuje pod koniec stycznia 2021, a baza może je przypisać do początku stycznia. Model nauczony na takich danych wygląda w treningu lepiej, niż byłby w rzeczywistości.

Dlatego XGBoost używa wyłącznie cech liczonych z cen i wolumenu, które zawsze są znane w danym dniu. W samym teście problem nie dotyczy agenta: pakiet danych jest zbierany na żywo w niedzielę, więc jest z definicji point-in-time. Powiązany problem to survivorship bias: trenując model tylko na spółkach obecnych dziś w indeksie, pomijamy te, które wypadły (często przez słabe wyniki). W teście nie występuje, bo skład jest zamrożony od startu, więc opisujemy go jako ograniczenie danych treningowych.

### 9.3 Benchmarki — co jest czym

**Buy & hold DIA.** Kupujesz indeks na starcie i nic nie robisz do końca; to wynik leniwego inwestora. Indeksu nie da się kupić, więc używamy ETF-u DIA, który go odwzorowuje i wypłaca dywidendy; portfele agenta też dostają dywidendy, więc porównanie jest uczciwe. Dow Jones jest ważony ceną akcji, a nie wielkością firmy, więc jest przechylony w stronę kilku drogich akcji.

**Portfel 1/N.** Każda z 30 spółek dostaje 1/30 kapitału. DeMiguel i współautorzy (2009) pokazali, że takiego portfela bardzo trudno pobić wyrafinowanymi metodami. Agent też waży spółki równo, więc jeśli pobije DIA, ale nie 1/N, jego przewaga wynika z równych wag, a nie z trafnego wyboru.

**XGBoost.** Model złożony z wielu drzew decyzyjnych budowanych po kolei tak, że każde poprawia błędy poprzednich (gradient boosting). Na historii uczy się, jakie cechy liczbowe poprzedzały wysokie zwroty, a w teście co tydzień prognozuje zwroty, tworzy ranking i kupuje cztery najlepsze spółki. SHAP pokazuje, które cechy najbardziej wpłynęły na konkretną prognozę, co zestawiamy z uzasadnieniami agenta.

**Losowe portfele.** Skrypt losuje 10 000 portfeli po cztery spółki i liczy ich wyniki, co daje rozkład czystego szczęścia. Jeśli agent wypada lepiej niż np. 90% losowych portfeli, to mocniejsza informacja niż porównanie z jednym benchmarkiem.

**Miary ryzyka.** Wskaźnik Sharpe’a to średni zwrot ponad stopę wolną od ryzyka podzielony przez zmienność, czyli zarobek na jednostkę ryzyka. Maksymalne obsunięcie to największy spadek wartości portfela od szczytu do dołka.

### 9.4 Dlaczego dane dzienne, a nie minutowe

Rozdzielczość danych dopasowujesz do horyzontu decyzji. Decyzja zapada raz w tygodniu i dotyczy tygodnia do przodu, a wahania minutowe w skali tygodnia w większości się znoszą: to szum, do którego model łatwo się przeucza. Do tego skala: dane minutowe dla 30 spółek to kilkadziesiąt milionów wierszy przy 10 latach historii, dane dzienne ok. 76 tys., a po zamianie na obserwacje tygodniowe ok. 15 tys. przykładów. Dwie praktyczne uwagi: cel modelu musi odpowiadać sposobowi handlu (zwrot od poniedziałkowego otwarcia do kolejnego), a ceny muszą być skorygowane o splity i dywidendy, bo split 1:10 wygląda w surowych danych jak krach o 90%.

### 9.5 Kappa Cohena — sprawdzanie kodowania LLM-a

Gdy LLM koduje raporty, trzeba pokazać, że robi to podobnie jak człowiek. Najprostsza miara to odsetek zgodnych ocen, ale zawyża ona zgodność: przy dwóch kategoriach, z których jedna występuje w 90% raportów, dwóch koderów zgadza się często przypadkiem. Kappa Cohena odejmuje zgodność oczekiwaną przypadkowo: kappa = (zgodność obserwowana – zgodność przypadkowa) / (1 – zgodność przypadkowa).

Przykład: na 15 raportach autor i LLM zgadzają się w 13 przypadkach co do „błąd faktograficzny tak/nie”, czyli zgodność obserwowana to 13/15 ≈ 0,867. Jeśli obaj oznaczają błąd w ok. 20% raportów, zgodność przypadkowa to 0,2 × 0,2 + 0,8 × 0,8 = 0,68. Kappa = (0,867 – 0,68) / (1 – 0,68) ≈ 0,58. Wartości powyżej ok. 0,6 uznaje się zwykle za dobrą zgodność, powyżej 0,8 za bardzo dobrą; w tym przykładzie ta kategoria trafiłaby do ręcznego kodowania.

## 10. Literatura przykładowa

*Pozycje z gwiazdką (\*) zostały sprawdzone w sieci przy tworzeniu wcześniejszej wersji planu. Pozostałe sprawdzić w Google Scholar (strony, wydanie) przed wpisaniem do bibliografii w formacie APA.*

### LLM w finansach i agenci tradingowi

- Ding, H., Li, Y., Wang, J., & Chen, H. (2024). Large language model agent in financial trading: A survey. arXiv:2408.06361. \*
- Dong, Y. et al. (2025). Large language model agents in finance: A survey bridging research, practice, and real-world deployment. Findings of EMNLP 2025. \*
- Lopez-Lira, A., & Tang, Y. (2023). Can ChatGPT forecast stock price movements? Return predictability and large language models. arXiv:2304.07619. \*
- Pelster, M., & Val, J. (2024). Can ChatGPT assist in picking stocks? Finance Research Letters. \* (test na żywo z dostępem do internetu — najbliższy tej pracy)
- Kim, A., Muhn, M., & Nikolaev, V. (2024). Financial statement analysis with large language models. Chicago Booth Research Paper.
- Xiao, Y. et al. (2024). TradingAgents: Multi-agents LLM financial trading framework. arXiv:2412.20138. \*
- Yu, Y. et al. (2023). FinMem: A performance-enhanced LLM trading agent with layered memory and character design. arXiv:2311.13743. \*
- Nof1 (2025). Alpha Arena — Season 1 (literatura szara). \*

### Wyciek informacji w modelach językowych

- Glasserman, P., & Lin, C. (2023). Assessing look-ahead bias in stock return predictions generated by GPT sentiment analysis. arXiv:2309.17322. \*
- Sarkar, S., & Vafa, K. (2024). Lookahead bias in pretrained language models. Working paper. \*

### Makroekonomia i ceny akcji

- Chen, N.-F., Roll, R., & Ross, S. A. (1986). Economic forces and the stock market. Journal of Business, 59(3), 383–403.
- Bernanke, B. S., & Kuttner, K. N. (2005). What explains the stock market’s reaction to Federal Reserve policy? Journal of Finance, 60(3), 1221–1257.
- Fama, E. F., & French, K. R. (1989). Business conditions and expected returns on stocks and bonds. Journal of Financial Economics, 25(1), 23–49.

### Uczenie maszynowe w finansach

- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. Review of Financial Studies, 33(5), 2223–2273.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of KDD 2016.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. NeurIPS 2017.
- Michańków, J., Sakowski, P., & Ślepaczuk, R. (2022). LSTM in algorithmic investment strategies on BTC and S&P500 index. Sensors, 22(3), 917. \*
- López de Prado, M. (2018). Advances in financial machine learning. Wiley.

### Teoria rynków i ewaluacja strategii

- Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. Journal of Finance, 25(2), 383–417.
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal versus naive diversification: How inefficient is the 1/N portfolio strategy? Review of Financial Studies, 22(5), 1915–1953.
- Sharpe, W. F. (1966). Mutual fund performance. Journal of Business, 39(1), 119–138.
- Ledoit, O., & Wolf, M. (2008). Robust performance hypothesis testing with the Sharpe ratio. Journal of Empirical Finance, 15(5), 850–859.
- Grinold, R. C., & Kahn, R. N. (2000). Active portfolio management. McGraw-Hill.

### Metody statystyczne i analiza jakościowa (nowe w wersji 5)

- Cohen, J. (1960). A coefficient of agreement for nominal scales. Educational and Psychological Measurement, 20(1), 37–46.
- Wilcoxon, F. (1945). Individual comparisons by ranking methods. Biometrics Bulletin, 1(6), 80–83.
