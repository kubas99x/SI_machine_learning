# SI_machine_learning

Projekt polega na wykrywaniu znaków ograniczenia prędkości na zdjęciach. Z wykorzystaniem machine learningu. 
## Trenowanie modelu
1. Tworzenie słownika z danymi do nauki modelu z plików xml. Słownik ma strukturę taką jak poniżej:
```
def makingDictionaryForLearning()
```

        imageDictionary = {
            "fileName": nazwa pliku (np. road10.png),
            "width": szerokość zdjęcia,
            "height": wysokość zdjęcia,
            "path": ścieżka do folderu, gdzie znajduja sie foldery "images" oraz "annotations",
            "partDictionaries": wektor słowników elementów
        }
        ,gdzie słownik elementu (wycinka) wygląda tak jak poniżej
        
        partDicionary = {
            "name": co sie znajduje na wycinku (np. speedlimit),
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
            "status": status okreslajacy typ wycinka, "1" oznacza ogr. prędkości, "2" inny znak
        }
2. Poszerzanie słownika o losowe wycinki ze zdjęcia nie zawierające znaków ogr. prędkości oraz wycinki zawierające wycinki znaków ogr. prędkości przy maksymalnym współczynniku iou wynoszącym 0.5. 
```
def addingPartsToTrainData(dataTrain):
```
3. Klasteryzacja, oraz zapis słownika klasteryzacji do pliku typu .npy.
```
def learningBOW(imageDictionary):
```
4. Tworzenie deskryptora dla każdego wycinka i powiększanie słownika każdego wycinka o parametr zawierający dane na temat jego deskryptora.
```
def extract(imageDictionary):
```
5. Trenowanie modelu następuje przy użyciu drzewa losowego o rozmiarze 100.
```
def train(imageDictionary):
```
