# CLAUDE.md — praca magisterska: agent LLM jako analityk inwestycyjny

Ten plik czytasz na starcie każdej sesji. Opisuje, co budujemy, które decyzje są zamrożone i jakich zasad pilnować. Pełny plan: `docs/plan_pracy_v5.md`. Bieżące zadania: `docs/harmonogram_do_pilotazu.md`. Wytyczne promotora: `docs/wytyczne.md`.

## O projekcie

Decyzje podjęte po wersji 5 planu są w `docs/zmiany_do_planu.md` – mają pierwszeństwo przed `plan_pracy_v5.md`.

Praca magisterska (Data Science and Business Analytics, WNE UW), pisana po angielsku. Badamy, czy agent LLM (Claude Opus 5.5), który co tydzień ocenia 30 spółek z Dow Jones, wybiera akcje lepiej z dostępem do wyszukiwarki niż bez niego, i jak wypada na tle XGBoost, portfela 1/N i ETF-u DIA. Metoda: test w przód (forward test), 28 tygodni, od 6.12.2026 do 21.06.2027. Główna miara: tygodniowy rank IC (korelacja Spearmana rankingu z faktycznymi zwrotami w kolejnym tygodniu).

Autor: Miłosz. Rozmawiamy po polsku; kod, komentarze, nazwy plików, prompt agenta i raporty agenta są po angielsku.

## Decyzje zamrożone (nie zmieniaj bez wyraźnej zgody autora)

- **Model:** Claude Opus 5.5 przez API (`claude-opus-5-5`), wersja przypięta, rozumowanie adaptacyjne ze stałym poziomem effort, zapisywane streszczenie rozumowania.
- **Dwa warianty agenta:** `claude_web` (pakiet danych + narzędzie web search, limit ok. 10 wyszukiwań) i `claude_noweb` (ten sam pakiet, bez wyszukiwarki). Identyczny prompt, identyczny pakiet danych — jedyna różnica to narzędzie wyszukiwania.
- **Brak pamięci:** agent nie widzi poprzednich raportów, portfela ani decyzji. Każdy tydzień od czystej karty.
- **Uniwersum:** 30 spółek Dow Jones według składu zamrożonego 4.12.2026, zapisanych w `config/universe.yaml`. Lista zweryfikowana 4.12.2026 względem oficjalnych składników DJIA/DIA; zamrożona do końca testu. Jeśli spółka zostanie wycofana z obrotu w trakcie testu, jej ostatnia cena jest ceną wyjścia, a spółka jest pomijana w kolejnych rankingach; IC liczone na pozostałych spółkach.
- **Wynik agenta:** ocena każdej spółki –5…+5, pełny ranking 1–30 bez remisów, cztery spółki do kupienia = pozycje 1–4 rankingu, uzasadnienie i lista źródeł. Format JSON według schematu w `prompts/output_schema.json`.
- **Portfel:** 4 spółki po 25%, zawsze pełne zainwestowanie (bez gotówki), 100 000 USD na portfel, akcje ułamkowe, bez kosztów w głównej analizie.
- **Zwroty:** od otwarcia sesji w poniedziałek do otwarcia w kolejny poniedziałek. Gdy poniedziałek jest dniem wolnym w USA, używamy otwarcia najbliższej sesji; agenta nie uruchamiamy ponownie.
- **XGBoost:** dane dzienne Yahoo Finance (zapasowo dane historyczne Alpaca), wyłącznie cechy cenowe i techniczne plus sektor, cechy z piątkowego zamknięcia, cel = zwrot poniedziałek–poniedziałek (otwarcie), ok. 10 lat historii, walidacja walk-forward na małej siatce. Trenowany raz przed startem i zamrożony.
- **Benchmarki liczone w skrypcie:** DIA (buy & hold), 1/N (po 1/30), 10 000 losowych portfeli 4-spółkowych.
- **Alpaca:** jedno konto, 3 rachunki paper (`claude_web`, `claude_noweb`, `xgb`), wyłącznie ewidencja wykonania. Zlecenia to zlecenia rynkowe składane chwilę po otwarciu sesji w USA, wyzwalane przez zegar rynkowy Alpaca (nie o stałej godzinie – różnica czasu US/EU zmienia się między 14 a 28.03.2027 z powodu różnicy dat zmiany czasu). Do analizy zawsze używamy oficjalnej ceny otwarcia pobieranej ze skryptu; różnica fill–open jest raportowana osobno. Źródłem danych do analizy jest zawsze nasz skrypt, nie Alpaca.
- **Środowisko:** GitHub Actions. Niedziela wieczorem: pakiet danych → oba warianty agenta → XGBoost → zapis i commit. Poniedziałek: zlecenia w Alpace po sygnale z zegara rynkowego Alpaca.

## Zasady, których zawsze pilnujesz

1. **Żadnego wycieku informacji z przyszłości.** Cecha z tygodnia X może zawierać tylko dane znane w tygodniu X. Żadnych bieżących danych fundamentalnych w treningu XGBoost. Przy każdej zmianie w cechach lub celu sprawdź daty.
2. **Ten sam pakiet dla obu wariantów.** Pakiet danych jest generowany raz na tydzień, zapisywany do pliku, a jego hash trafia do logu każdego wariantu. Oba warianty czytają ten sam plik.
3. **Dane tylko dopisujemy.** Folder `data/weekly/<data_niedzieli>/` po commicie jest nienaruszalny. Poprawki zapisujemy jako nowe pliki z adnotacją, nigdy przez nadpisanie.
4. **Walidacja odpowiedzi agenta.** Sprawdź: 30 unikalnych tickerów z uniwersum, ranking jest permutacją 1–30, oceny w zakresie –5…+5, picks = ranking 1–4. Przy błędzie: maksymalnie 2 ponowienia z tym samym wejściem; potem zapis błędu i reguła awaryjna z `config/settings.yaml`. Każde ponowienie jest logowane.
5. **Pełny zapis każdego uruchomienia:** prompt, wersja modelu, surowa odpowiedź, JSON, raport czytelny, lista wyszukiwań i źródeł, streszczenie rozumowania, zużycie tokenów i koszt.
6. **Sekrety tylko w zmiennych środowiskowych** (GitHub Secrets / lokalny `.env` w `.gitignore`). Nigdy nie wpisuj kluczy do kodu, logów ani commitów.
7. **Powtarzalność:** stałe ziarna losowości (losowe portfele, XGBoost), przypięte wersje pakietów w `requirements.txt`.
8. **Po zamrożeniu (po pilotażu, najpóźniej 30.11.2026)** zmiany w `prompts/` i `config/` tylko za wyraźną zgodą autora i z wpisem w `docs/CHANGELOG.md` (data, co, dlaczego).
9. **Koszty:** przed dodaniem czegokolwiek, co zwiększa liczbę tokenów lub wyszukiwań, oszacuj koszt na uruchomienie. Budżet całego badania: ok. 65–70 USD.
10. **Korekta o wydzielenia spółek (spin-off):** wszystkie cechy cenowe XGBoost – w treningu i w cotygodniowej prognozie – liczymy wyłącznie z cen skorygowanych funkcją `src/data/adjust_prices.py` na podstawie `config/corporate_actions.yaml`. Nigdy z surowych cen z Yahoo. Nowe wydzielenie w trakcie testu dopisujemy do `corporate_actions.yaml` z wpisem w `docs/CHANGELOG.md`.

## Struktura repozytorium (docelowa)

```
CLAUDE.md
docs/        plan_pracy_v5.md, wytyczne.md, harmonogram_do_pilotazu.md, CHANGELOG.md
config/      universe.yaml, settings.yaml (model, effort, limit wyszukiwań, reguła awaryjna)
prompts/     agent_prompt.md, output_schema.json
src/data/       pobieranie cen, makro, kalendarz wyników; budowa pakietu tygodniowego
src/agent/      uruchamianie wariantów, walidacja, zapis
src/xgb/        cechy, trening, walidacja walk-forward, prognoza tygodniowa
src/portfolio/  ewidencja portfeli, zwroty, benchmarki, losowe portfele, IC
src/broker/     zlecenia Alpaca, kalendarz sesji
src/analysis/   miary i testy do rozdziału 5, kodowanie raportów do rozdziału 6
data/raw/       surowe dane historyczne
data/weekly/<YYYY-MM-DD>/   pakiet, odpowiedzi agentów, prognoza XGBoost, zlecenia, log
.github/workflows/  weekly_run.yml (niedziela), monday_orders.yml (poniedziałek)
tests/
```

## Pisanie pracy

Tekst pracy (po angielsku) skupia się na metodach, nie na kodzie: bez nazw bibliotek i funkcji, bez fragmentów kodu w treści (kod tylko w załączniku/repozytorium). Bibliografia w stylu APA. Struktura: Introduction, 1. Markets, macroeconomics and investment strategies, 2. Literature review, 3. Data, 4. Research design and methodology, 5. Quantitative results, 6. Qualitative analysis, 7. Discussion and limitations, Conclusions — razem ok. 60 stron.

## Jak ze mną pracować

Gdy prośba autora koliduje z decyzją zamrożoną albo z zasadą powyżej, powiedz to wprost przed wykonaniem i zaproponuj alternatywę. Nie przyjmuj założeń o danych finansowych bez sprawdzenia dat i źródeł.
