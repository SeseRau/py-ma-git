Moin,

die MA-AIS23.py erstellt aus den dänischen AIS Daten (http://web.ais.dk/aisdata/) nach den getroffenen Einstellung eine 'Zwischendatei'. Mit der 'Zwischendatei', welche nur gefilterte AIS-Positionen enthält wird dann das KNN in der MA-KNN14g.py trainiert.

Die Dateien "aisdk-2023-11-08-xs_1_kkn.csv" und "aisdk-2023-11-08-xs_1_kkn_gnu.csv" sind solche 'Zwischendateien'. Die Datei mit der GNU Endung die menschenleserliche in Zeilen ist (braucht GNU-Plot) und trennt die Bewergungsverläufe durch eine Leerzeile. Bei der anderen Datei steht alles in einer Zeile und die verschiedenen Bewegungsverläufte sind durch '$' getrennt werden.

Grüße
Sebastian
