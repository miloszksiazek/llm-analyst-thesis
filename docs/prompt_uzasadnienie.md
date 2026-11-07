# Prompt agenta v0 — uzasadnienie decyzji

Dokument roboczy do rozdziału 4 (Research design) i do rozmowy z promotorem. Prompt: `prompts/agent_prompt.md`, schemat odpowiedzi: `prompts/output_schema.json`. Wersja v0 z 2026-11-07; poprawki po pilotażach 22.11 i 29.11, zamrożenie 30.11.

## Decyzje i powody

| # | Decyzja | Dlaczego |
| --- | --- | --- |
| 1 | Jeden, identyczny tekst promptu dla obu wariantów; zdanie „you may or may not have a web search tool”. | RQ3 wymaga, by jedyną różnicą było narzędzie. Dwa różne prompty („masz wyszukiwarkę” / „nie masz”) dokładałyby drugą różnicę. |
| 2 | Pakiet danych jest „authoritative” dla bieżących liczb; wiedza z treningu może być nieaktualna i nie wolno jej podawać jako bieżących newsów. | Ogranicza halucynacje w wariancie bez sieci (model z wiedzą do połowy 2026 r. mógłby „pamiętać” nieaktualne fakty) i kontroluje lookahead bias w drugą stronę: model nie zna przyszłości, ale może mieć stare przekonania. |
| 3 | Ocena –5…+5 jest **względna** (wobec pozostałych 29 spółek), wyśrodkowana wokół zera; horyzont: od najbliższego otwarcia do otwarcia tydzień później. | Główną miarą jest rank IC — korelacja rankingu z przekrojem zwrotów. Ocena bezwzględna („czy spółka wzrośnie”) mieszałaby prognozę rynku z wyborem spółek. Horyzont jest dokładnie taki, na jakim liczymy zwroty. |
| 4 | Ranking 1–30 bez remisów, zgodny z ocenami (wyższa ocena → lepsze miejsce). | Remisy osłabiają korelację rang; zgodność oceny i rankingu pozwala analizować obie wielkości bez sprzeczności. Walidacja to wymusza. |
| 5 | Picks = miejsca 1–4, zawsze cztery, bez gotówki. | Spójność portfela z rankingiem (portfel mierzy to samo co IC) i porównywalność z XGBoost oraz 1/N, które też są zawsze w pełni zainwestowane. |
| 6 | Brak zalecanej strategii („whatever evidence you judge most relevant”). | Nie narzucamy agentowi momentum ani wyceny — badamy, jak rozumuje sam. Typ uzasadnienia mierzymy potem w `key_factors`. |
| 7 | Limit wyszukiwań podany w prompcie i wymuszony w API (`max_uses`). | Model, który zna budżet, planuje wyszukiwania; limit w API chroni koszty niezależnie od posłuszeństwa modelu. |
| 8 | Krótkie uzasadnienia (≤40 słów na spółkę), zamknięte listy `main_themes` i `key_factors`. | Kontrola kosztów (30 uzasadnień to ok. 1,5–2 tys. tokenów) i gotowe zmienne do rozdziału 6. Kategorie odpowiadają schematowi kodowania z planu (sekcja 4.8). |
| 9 | Źródła z identyfikatorami (PKG, W1, W2…), każde twierdzenie o niedawnych wydarzeniach musi mieć źródło. | Umożliwia automatyczne sprawdzenie, czy cytowane źródła istnieją, i ręczną weryfikację halucynacji (H4). |
| 10 | Pole `data_issues`. | Agent może zauważyć błąd w pakiecie (np. dziwny skok ceny). To tani dodatkowy system wczesnego ostrzegania i materiał do rozdziału 6. |
| 11 | Wynik wyłącznie jako jeden obiekt JSON, bez komentarzy i zastrzeżeń. | Prosta, deterministyczna walidacja; czytelny raport generuje skrypt z JSON-a. |

## Ryzyka, które sprawdzamy w pilotażu

- **Kolejność spółek w pakiecie.** Spółki są ułożone alfabetycznie, a modele językowe bywają wrażliwe na pozycję na liście. Kolejność jest identyczna w obu wariantach, więc nie zaburza RQ3, ale może wpływać na RQ1. W pilotażu: sprawdzić, czy rangi nie korelują z pozycją alfabetyczną; jeśli tak — rozważyć stałą, losową kolejność (ta sama co tydzień, ziarno w `config/`).
- **Oceny nie wyśrodkowane.** Model może dać większości spółek +1/+2. Sprawdzić średnią ocen w pilotażu; ranking i tak jest głównym wynikiem.
- **Wariant bez sieci „udaje” newsy** mimo instrukcji. Sprawdzić ręcznie oba raporty z pilotażu.
- **Losowość odpowiedzi.** Przy włączonym rozumowaniu ustawienia losowości mogą być ograniczone przez API — do sprawdzenia przy budowie; powtarzalność mierzymy ewentualnie w rozszerzeniu (kilka uruchomień w tym samym tygodniu).
- **Format JSON.** Jeśli model w pilotażu łamie format, rozważyć wymuszenie schematu po stronie API (structured outputs), o ile działa razem z wyszukiwarką i rozumowaniem — do sprawdzenia w dokumentacji przy budowie.

## Dodatkowe reguły walidacji (do zasady 4 w CLAUDE.md)

1. 30 unikalnych tickerów, identycznych z uniwersum.
2. Rangi to permutacja 1–30.
3. Oceny całkowite w zakresie –5…+5.
4. **Zgodność:** jeśli spółka A ma wyższą ocenę niż B, to ma lepszą (niższą) rangę.
5. `picks` = tickery z rangami 1, 2, 3, 4, w tej kolejności.
6. Każdy `source_id` użyty przy spółkach i w `macro_view` istnieje w `sources`.
7. W wariancie bez sieci nie ma źródeł typu `web`; jeśli są — to zmyślone źródło, zapis do logu jako błąd treści (nie ponawiamy zapytania, to wynik do H4).
8. `run_date` i `as_of` zgodne z uruchomieniem.

Reguły 1–6 i 8 przy naruszeniu → ponowienie (maks. 2). Reguła 7 → odpowiedź zostaje, błąd jest danymi badawczymi.
