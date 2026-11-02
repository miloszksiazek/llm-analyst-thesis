# Harmonogram do pilotażu: 2–29 listopada 2026

Cel: do **piątku 20.11** cały system ma działać w chmurze bez twojego udziału, żeby **w niedzielę 22.11** ruszył pierwszy pilotaż. Drugi pilotaż jest 29.11, potem zamrożenie ustawień i właściwy start **6.12**. Każdy dzień ma zadanie i jedną rzecz do sprawdzenia; zadanie jest skończone dopiero, gdy kontrola przechodzi. Zakładam ok. 2–3 godziny pracy dziennie z Claude Code. Weekendy to rezerwa na to, co się przesunęło.

Kluczowe godziny: sesja w USA otwiera się o **15:30 czasu polskiego** (cały listopad). Niedzielne uruchomienie planujemy na ok. 19:00, zlecenia w poniedziałek ok. 15:00.

## Tydzień 1 (2–8.11): konta, decyzje, dane

| Dzień | Zadanie | Sprawdź, zanim pójdziesz dalej |
| --- | --- | --- |
| Pn 2.11 | Wyślij promotorowi mail z planem v5 i listą z sekcji 8.2. Załóż prywatne repozytorium na GitHubie, wrzuć `CLAUDE.md`, `docs/plan_pracy_v5.md`, `docs/wytyczne.md`, ten plik. | Otwórz repo w VS Code i zapytaj Claude Code „co budujemy i co jest zamrożone?”. Odpowiedź musi zgadzać się z planem. |
| Wt 3.11 | Konto w Claude Console, włączone rozliczanie, **twardy limit wydatków** (np. 30 USD/miesiąc), klucz API w `.env`. Pierwsze testowe wywołanie Claude Opus 5.5 z narzędziem web search i limitem wyszukiwań. | Odpowiedź przyszła, model w odpowiedzi to `claude-opus-5-5`, koszt wywołania widoczny w konsoli. Zapisz, ile kosztowało. |
| Śr 4.11 | Konto Alpaca, trzy rachunki paper (`claude_web`, `claude_noweb`, `xgb`), klucze API każdego z nich. Przeczytaj w dokumentacji: akcje ułamkowe, zlecenia na otwarcie, kalendarz sesji. | Czy na jednym koncie da się mieć 3 rachunki paper z osobnymi kluczami? Jaki typ zlecenia działa dla akcji ułamkowych przed otwarciem? Jeśli coś nie gra — zanotuj; Alpaca i tak jest budowana na końcu. |
| Cz 5.11 | Lista 30 spółek Dow Jones (stan na dziś) w `config/universe.yaml`. Pobranie 10 lat danych dziennych dla 30 spółek + DIA do `data/raw/`. | 31 tickerów, brak dziur w datach, ceny skorygowane o splity i dywidendy (sprawdź jeden znany split, czy nie ma „krachu” o 50–90%). Sprawdź też, czy ogłoszono zmiany w składzie indeksu przed 6.12. |
| Pt 6.11 | Dane makro do pakietu: darmowy klucz FRED, serie: stopa Fed, inflacja CPI, rentowności 2- i 10-letnie, indeks dolara. Źródło kalendarza wyników spółek na najbliższy tydzień. | Dla bieżącego tygodnia masz wszystkie wartości makro i daty wyników. Każda liczba ma datę publikacji. |
| Sb–Nd 7–8.11 | Moduł budowy pakietu tygodniowego: jeden plik JSON z cenami, zmianami (tydzień, miesiąc, kwartał), P/E, dywidendą, kapitalizacją, kalendarzem wyników i makro. Zapis + hash. | Wygeneruj pakiet na ten tydzień, przeczytaj go sam: czy człowiek-analityk mógłby na tej podstawie ocenić spółki? Dwa uruchomienia na tych samych danych dają ten sam hash. |

## Tydzień 2 (9–15.11): agent, XGBoost, ewidencja

| Dzień | Zadanie | Sprawdź, zanim pójdziesz dalej |
| --- | --- | --- |
| Pn 9.11 | Prompt agenta v0 (po angielsku) w `prompts/agent_prompt.md` i schemat odpowiedzi `prompts/output_schema.json`: oceny –5…+5, ranking 1–30 bez remisów, picks = ranking 1–4, uzasadnienie, źródła. | Brak odpowiedzi promotora? Wyślij krótkie przypomnienie. Prompt nie zawiera niczego o portfelu ani poprzednich tygodniach. |
| Wt 10.11 | Uruchamianie dwóch wariantów (z siecią i bez), walidacja odpowiedzi, ponowienia (maks. 2), pełny zapis: surowa odpowiedź, JSON, raport, wyszukiwania, źródła, streszczenie rozumowania, tokeny, koszt. | Celowo zepsuj odpowiedź (np. remis w rankingu) w teście: walidacja musi ją odrzucić i zalogować. |
| Śr 11.11 | (wolne w Polsce) Ręczne uruchomienie obu wariantów na bieżącym tygodniu. Przeczytaj oba raporty. | Koszt na uruchomienie vs plan (ok. 1,00 USD z siecią, 0,30 USD bez). Jeśli ponad 2× — obniż effort lub limit długości stron, zanim pójdziesz dalej. Czy wariant bez sieci nie „udaje”, że zna newsy? |
| Cz 12.11 | XGBoost: panel tygodniowy (cechy z piątkowego zamknięcia, cel poniedziałek–poniedziałek na otwarciach), ok. 15 tys. wierszy. | Wybierz losowo 3 wiersze i policz ręcznie cel i jedną cechę. Żadna cecha nie używa danych późniejszych niż piątkowe zamknięcie. |
| Pt 13.11 | XGBoost: walidacja walk-forward na małej siatce, wybór hiperparametrów, trening końcowy, zapis i zamrożenie modelu. Prognoza i ranking na bieżący tydzień. | IC na okresach walidacyjnych zapisany (spodziewaj się wartości bliskich zera — to normalne). Model zapisany z datą i ziarnem. |
| Sb–Nd 14–15.11 | Ewidencja portfeli: pozycje, zwroty tydzień do tygodnia, DIA, 1/N, 10 000 losowych portfeli, IC, obrót. | Test na sztucznych danych z ręcznie policzonym wynikiem: przykład z 5 spółkami z planu (A–E vs B, A, C, E, D) daje IC = 0,8. Portfel po jednym tygodniu liczony ręcznie zgadza się ze skryptem. |

## Tydzień 3 (16–22.11): automatyzacja i pierwszy pilotaż

| Dzień | Zadanie | Sprawdź, zanim pójdziesz dalej |
| --- | --- | --- |
| Pn 16.11 | GitHub Actions: workflow niedzielny (pakiet → oba warianty → XGBoost → commit wyników), klucze w GitHub Secrets, możliwość ręcznego uruchomienia. | Uruchom ręcznie: w repo pojawia się commit z folderem `data/weekly/<data>/` i wszystkimi plikami. Żaden klucz nie trafił do logów. |
| Wt 17.11 | Workflow poniedziałkowy: zlecenia w Alpace na podstawie ostatnich decyzji, sprawdzenie kalendarza sesji (święta w USA). Test w trakcie sesji (po 15:30). | Zlecenia zrealizowane, pozycje na 3 rachunkach paper zgadzają się z tym, co policzył skrypt. Jeśli Alpaca sprawia problemy — odpuść i zanotuj (badanie działa na samym skrypcie). |
| Śr 18.11 | Powiadomienia: mail przy błędzie workflow i przy braku pliku z decyzją. Instrukcja ręcznego ponownego uruchomienia w `docs/`. | Celowo zepsuj klucz API w kopii workflow: mail dochodzi. Przywróć klucz. |
| Cz 19.11 | Próba generalna: oba workflow uruchomione ręcznie od początku do końca, jak w prawdziwym tygodniu. | Przejdź listę „gotowe do pilotażu” poniżej. |
| Pt 20.11 | **Termin końca budowy.** Poprawki z próby generalnej. Ustaw harmonogram: niedziela ok. 19:00, poniedziałek ok. 15:00 czasu polskiego. | Wszystkie punkty listy „gotowe do pilotażu” zaznaczone. |
| Sb 21.11 | Rezerwa. | — |
| Nd 22.11 | **Pilotaż 1** — uruchomienie automatyczne, bez twojego udziału. Wieczorem przeczytaj oba raporty. | Workflow ruszył sam, commit jest w repo, koszt w konsoli zgodny z planem. |
| Pn 23.11 | Zlecenia pilotażowe w Alpace. | Pozycje zgodne ze skryptem po otwarciu sesji. |

## Tydzień 4 (23–29.11): poprawki i zamrożenie

| Dzień | Zadanie | Sprawdź |
| --- | --- | --- |
| 23–27.11 | Poprawki promptu i ustawień po pilotażu 1 (np. format odpowiedzi, długość raportów, koszt). Każdą zmianę zapisz w `docs/CHANGELOG.md`. | Zmiany wynikają z obserwacji pilotażu, nie z prób „poprawienia wyników”. |
| Nd 29.11 | **Pilotaż 2** — ostatni test na finalnych ustawieniach. | Zero błędów walidacji, koszt w normie. |
| Pn 30.11 | **Zamrożenie:** prompt, ustawienia, model XGBoost, skład uniwersum (sprawdź skład Dow Jones jeszcze raz na 6.12). Tag w repo, np. `freeze-v1`. | Od tego momentu zmiany w `prompts/` i `config/` tylko z wpisem w CHANGELOG. |
| Nd 6.12 | **Start właściwego testu** (tydzień 1 z 28). | — |

## Lista „gotowe do pilotażu” (sprawdź 19–20.11)

- [ ] Pakiet danych generuje się automatycznie, ma hash, oba warianty czytają ten sam plik.
- [ ] Oba warianty zwracają poprawny JSON; walidacja odrzuca błędne odpowiedzi i ponawia maks. 2 razy.
- [ ] Zapisywane są: prompt, wersja modelu, surowa odpowiedź, raport, wyszukiwania, źródła, streszczenie rozumowania, tokeny, koszt.
- [ ] Koszt jednego tygodnia (oba warianty) mieści się w ok. 1,30–2,00 USD; w konsoli ustawiony twardy limit wydatków.
- [ ] XGBoost wytrenowany, zamrożony, prognozuje bieżący tydzień; brak cech z przyszłości.
- [ ] Ewidencja portfeli i IC przechodzi testy na danych z ręcznie policzonym wynikiem.
- [ ] Workflow niedzielny i poniedziałkowy działają z ręcznego uruchomienia; wyniki trafiają do repo jako commit.
- [ ] Mail przy błędzie dochodzi.
- [ ] Alpaca: zlecenia działają albo świadomie odpuszczona (zanotowane).
- [ ] Odpowiedź promotora na zmiany z planu v5 — jest albo wysłane przypomnienie.

## Co robić, gdy coś się przesuwa

Najpierw odpuszczasz Alpakę (badanie działa bez niej), potem przesuwasz pilotaż 2 na 6.12 i start na 13.12 (test kończy się wtedy 28.06.2027 — wciąż w porządku). Nie odpuszczasz walidacji, zapisu i testów IC: błąd w nich wychodzi dopiero przy analizie, kiedy jest już za późno na powtórkę.
