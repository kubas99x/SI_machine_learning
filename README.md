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

## Detect
Po przejściu przez cały proces trenowania, program oczekuje na informację wejściową. Jedną z funkjconalności, jaką możemy użyć jest "detect". Polega ona na sprawdzeniu wszystkich zdjęć z folderu "test" i podanie na wyjście informacji o znalezionych znakać ogr. prędkości. Algorytm przebiega następująco:

1. Tworzenie słownika dla zbioru testowego
```
def makingDictionaryForTest():
```
słownik ma następującą strukturę:
```
imageDictionary = {
            "fileName": nazwa pliku,
            "width": szerokosc zdjecia,
            "height": wysokosc zdjecia,
            "path": sciezka do folderu "test",
            "partDictionaries": lista zawierajaca slownik elementow ( w tym momencie jeszcze pusty)
        }
```
2. Wyszukiwanie na zdjęciach miejsc podejrzanych o występowanie znaków.
```
def circleOnImage(dataDict):
```
Algorytm pobiera zdjęcie, zmienia je na format hsv. Następnie nakłada pierwszą maskę. Zdjęcie posiadające tylko dane kolory (po przejściu przez maskę) jest blurowane i wyszukiwane są na nim okręgi, przy użyciu funkcji:
```
cv2.HoughCircles()
```
Jeżeli żadne okręgi nie zostały znalezione, to na obraz zostaje nałożona maska innego koloru i proces powtarza się. Jeżeli przy użyciu wszystkich 3 masek nie zostaną na obrazie zaznaczone żadne okręgi, obraz nie jest dalej używany w procesie.
Jeżeli na obrazie znajdą się okręgi, na podstawie ich współrzędnych odczytujemy współrzędne kwadratów okalających je. Następnie te są zapisywane do słownika wycinka, który jest dodawany do wektora wycinków w naszym słowniku głownym. Słownik wycinka ma postać:
```
tmpDictionary = {    "xmin": xmin,
                     "xmax": xmax,
                     "ymin": ymin,
                     "ymax": ymax}
```
W procesie bierze udział również funkcja
```
def makingMaskForCircles(hsvImage, lowerMaskL, lowerMaskH, higherMaskL, higherMaskH):
```
, która łączy dolne i górne zakresy masek, zmienia format zdjęcia jak i je bluruje i zwraca zblurowane zdjęcie.
3. Ekstrakcja dla każdego znalezionego wycinka na zdjęciu (ta sama funkcja co w przypadku danych treningowych)
```
def extract(imageDictionary):
```
4. Predykcja naszych zdjęć, gdzie ustalany jest status naszego wycinka. Czy znajduje się na nim znak, czy coś innego.
```
def predictImage(rf, data):
```
5. Wyświetlanie informacji na temat zdjęć wejściowych według formatki zadanej w skrypcie.
```
def detectInformation(dataTest):
```

## Classify
Po podaniu na wejście ciągu znaków "classify" następuje proces klasyfikacji zdjęć. Program według formatki podanej w skrypcie przyjmuje daną ilość zdjęć oraz wycinków. Zwracając co znajduje się na każdym z wycinków.
1. Odbieranie danych wejściowych
```
def classifyInput():
```
w funkcji tej następuje również tworzenie słownika zawierające informację na temat każdego wycinka:
```
classifyDict = {        "fileName": filename,
                        "path": mypath,
                        "partDictionaries": partDictionaryArray}
```
, gdzie lista partDictionaryArray zawiera słowniki każdego wycinka jak poniżej:
```
tmpDictionary = {    "xmin": int(xmin),
                     "xmax": int(xmax),
                     "ymin": int(ymin),
                     "ymax": int(ymax)}    
```
2. Ekstrakcja naszych wycinków (tak jak w przypadku danych treningowych oraz przy użyciu funkcjonalności "detect")
```
def extract(imageDictionary):
```
3. Predykcja, co znajduje się na każdym z wycinków.
```
def predictImage(rf, data):
```
4. Wypisanie typów znajdujących się na zadanych wycinkach według formatki.
```
def classifyReturn(dataTest):
```

