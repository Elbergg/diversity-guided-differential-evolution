# Uruchamianie eksperymentów

## Przygotowanie środowiska

```bash
uv sync
```

## Benchmark BBOB

Aby wyniki były takie same jak w dokumentacji nie należy zmieniać ziaren oraz innych parametrów w kodzie. Ważne też aby skrypt `./run_experiment_in_batches.py` uruchomić z `budget_multiplier=5` (pierwszy argument wywołania programu).

```bash
# Zebranie danych
python ./run_experiment_in_batches.py scipy_de 5 4
python ./run_experiment_in_batches.py de_dg 5 4

# Wyniki z poszczególnych batch należy przetworzyć, podając wzór nazw folderów z wynikami oraz nazwę folder gdzie dane zostaną zapisane (poniżej przykładowe nazwy)
python ./merge_and_process_batches.py "exdata/differential_evolution_of_scipy.optimize._differentialevolution_5D_on_bbob_batch*of4" exdata/de_scipy_5D_full_seeded
python ./merge_and_process_batches.py "exdata/de_df_of_de_5D_on_bbob_batch*of4" exdata/de_dg_of_de_5D_on_bbob_batch_full_all_dim_seeded

# Porówananie algorytmów przy pomocy cocopp
python -m cocopp exdata/scipy_full exdata/de_dg_full
```

## Zbieżność

```bash
python ./experiment_convergence.py
```

## Wilcoxon

W skrypcie należy zmienić nazwy folderów z danymi na te wygenerowane w skrypce `./merge_and_process_batches.py`:

```python
# Trzeba zmienić tutaj
ds_scipy = cocopp.load2("exdata/de_scipy_5D_full_seeded")
ds_de_dg = cocopp.load2(
    "exdata/de_dg_of_de_5D_on_bbob_batch_full_all_dim_seeded"
)
```

Uruchomienie:

```bash
python ./calculate_wilcoxon.py
```