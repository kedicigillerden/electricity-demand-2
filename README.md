# Elektrik Talep Tahmini

Bu repo, saatlik elektrik tuketimi, sicaklik ve aylik makroekonomik degiskenlerle elektrik talebi icin ekonometrik modelleme akisini icerir.

Ana dosya:

- `talep_tahmin_tek_dosya.py`

## Veri Yapisi

Ham veriler repoya eklenmez. Asagidaki klasorlere Excel dosyalarini koyun:

- `data/raw/consumption/tuketim_2022.xlsx`
- `data/raw/consumption/tuketim_2023.xlsx`
- `data/raw/consumption/tuketim_2024.xlsx`
- `data/raw/temperature/sicaklik_2022.xlsx`
- `data/raw/temperature/sicaklik_2023.xlsx`
- `data/raw/temperature/sicaklik_2024.xlsx`
- `data/raw/economic/pmi_degerleri.xlsx`
- `data/raw/economic/faiz_oranlari.xlsx`
- `data/raw/economic/kapasite_kullanim_orani.xlsx`

## Modelleme Notlari

- `HDD = max(18 - temperature, 0)`
- `CDD = max(temperature - 24, 0)`
- `HDD x CDD` yapisal olarak anlamsiz oldugu icin interaction adaylarindan cikarilir.
- `month` numeric continuous veya polynomial kaynak olarak kullanilmaz; ay etkisi month dummy ile temsil edilir.
- Ham `hour` 0-23 degeri interactionlara sokulmaz; saat interactionlari `hour_z` ile kurulur.
- Full month dummy iceren modellerde `PMI`, `IR`, `CUR` ana etki olarak kullanilmaz.
- Macro ana etkileri sadece month-dummy'siz robustness modellerinde yer alir.

## Modeller

- Model A: Seasonal baseline, month dummies var, macro ana etki yok.
- Model B: Ana model, month dummies + macro x high-frequency interaction.
- Model C: Robustness, month dummies yok, macro ana etkiler var.
- Model D: Genisletilmis robustness, macro ana etkiler + macro x high-frequency interaction.

## Calistirma

```bash
python -m pip install -r requirements.txt
python talep_tahmin_tek_dosya.py
```

Ciktilar:

- OLS ozetleri: `outputs/reports/*.txt`
- Model metrikleri: `outputs/reports/model_A_B_C_D_ozet_metrikleri.txt`
- Grafikler: `outputs/figures/*.png`

GitHub Actions workflow'u scripti calistirir ve `outputs/reports` ile `outputs/figures` klasorlerini artifact olarak yukler. Veri dosyalari yoksa workflow basarisiz olmaz; eksik veri raporu artifact olarak uretilir.
