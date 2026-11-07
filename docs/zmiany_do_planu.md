# Zmiany do planu v5 – lista decyzji do scalenia z v5.1

Każda linia = jedna zamknięta decyzja. Mają pierwszeństwo przed `docs/plan_pracy_v5.md`.

- 2026-11-02: picks agenta = pozycje 1–4 rankingu (wymuszone walidacją).
- 2026-11-02: XGBoost trenowany raz i zamrożony; douczanie co tydzień tylko jako test odporności po zakończeniu testu, na zapisanych zrzutach danych.
- 2026-11-02: XGBoost dostaje tylko cechy cenowe/techniczne; migawka wskaźników (P/E, dywidenda, kapitalizacja) i makro trafia tylko do agenta – opisać w metodologii, że porównanie agent vs XGBoost łączy efekt modelu i zakresu informacji; czyste porównanie to tylko RQ3.
- 2026-11-02: każdy niedzielny zrzut danych zapisywany w repo (Yahoo przelicza wstecz ceny skorygowane); końcowe zwroty liczone raz po zakończeniu testu.
- 2026-11-02: środowisko Python 3.12 (lokalnie i w GitHub Actions), anthropic 2.3.0.
- 2026-11-04: skład zamrożony na 2026-12-04; GOOGL zastąpił VZ 29.06.2026, HON został po wydzieleniu Honeywell Aerospace; weryfikacja z pozycjami DIA (SSGA).
- 2026-11-04: spółka wycofana z obrotu – ostatnia cena jako cena sprzedaży, usuwana z kolejnych rankingów, IC na pozostałych spółkach.
- 2026-11-04: DIA = odniesienie rynkowe; właściwy punkt odniesienia dla zamrożonego składu = portfel 1/N.
- 2026-11-04: Alpaca – zlecenia rynkowe po otwarciu sesji wg zegara rynku (ułamki akcji nie obsługują zleceń na otwarcie); analiza zawsze na oficjalnej cenie otwarcia; różnica wykonanie vs otwarcie jako osobny wynik.
- 2026-11-04: korekta cen o wydzielenia spółek (tabela zdarzeń do rozdziału 3); 7 potwierdzonych spin-offów z SEC 8-K (Garrett 2018, Resideo 2018, Organon 2021, Kyndryl 2021, Solventum 2024, Solstice 2025, HON Aerospace 2026); JNJ/Kenvue wykluczone – był to exchange offer, nie pro-rata spin-off.
- 2026-11-05: dni wydzielenia spółek – zwrot spółki zastąpiony zwrotem DIA; cechy XGBoost liczone z oczyszczonego indeksu cen.
- 2026-11-05: kalendarz wyników w pakiecie agenta – tylko daty w oknie 14 dni od uruchomienia, status "unverified (Yahoo)", daty z przeszłości = null.
- 2026-11-06: hash pakietu potwierdza integralność zapisanego pliku (oba warianty czytają ten sam plik); zmiany cen w pakiecie z oczyszczonego indeksu; punkt odniesienia = zamknięcie ostatniej sesji przed niedzielą.
- 2026-11-07: pakiet agenta bez dat po as_of i bez znaczników pobrania (osobny `package_meta.json`); kontrole jednostek i wartości (dividend_yield w [0,15] %, P/E > 0 lub null, implikowana liczba akcji w granicach 20 % tygodnia poprzedniego); market_cap liczymy z close × sharesOutstanding (Yahoo marketCap tylko w meta); earnings – tylko spółki z datą w oknie plus `note` o pozostałych; w prompcie zdanie o preferowaniu źródeł pierwotnych; wariantu z siecią nie testujemy na pakietach z przeszłości.
- 2026-11-07: reguła awaryjna zmieniona – po 2 nieudanych próbach utrzymujemy portfel z poprzedniego tygodnia dla danego wariantu, tydzień zapisujemy jako brak rankingu i wykluczamy go z liczenia IC; zdarzenie logujemy jako błąd pipeline'u.
- 2026-11-07: spółki z wydzieleniem w ostatnich 12 mies. – P/E i kapitalizacja = null z notatką (trailing P/E nieporównywalne).
