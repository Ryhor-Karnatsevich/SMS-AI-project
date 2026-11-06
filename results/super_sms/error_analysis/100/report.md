# Analiza błędów: Super SMS

Trening: 100% puli. Test: 6086 SMS. Dostępne modele: 3/4.

| Model | Fałszywe alarmy | Przeoczony spam |
|---|---:|---:|
| Regresja logistyczna | 33 | 50 |
| Multinomial Naive Bayes | 101 | 59 |
| Linear SVM | 30 | 42 |

Brak zapisanych modeli: MLP.

Wersje bibliotek różnią się od zapisanych. Dla tych modeli sprawdzono zgodność predykcji z wcześniejszymi miarami zbiorczymi; szczegóły zawiera manifest.json. Zgodność miar nie dowodzi identyczności każdej pojedynczej predykcji ani pełnej zgodności prawdopodobieństw między wersjami.

## Przykłady do ręcznej oceny

Przykłady poniżej są pierwszymi błędami w stałej kolejności testu, nie rankingiem ważności.

### Regresja logistyczna — Fałszywe alarmy

<pre>Amway order &lt;#&gt;</pre>

<pre>Your order from AMAZON booked vide GATI DKT 502024095 is out for delivery, Contact Person: SOURAV/9932909134</pre>

<pre>the last shop you will visit in some structured ways generally gives the best rate</pre>

<pre>can i get a copy of your recent publication please ?</pre>

<pre>Call:ChrisMomNanaDad</pre>


### Regresja logistyczna — Przeoczony spam

<pre>rofl i knew how u get to track my progress i should say its a crappy paper but it has come out really well impact wise though p its beeeeeen long since we saw each othr goodnight da gonna sleep</pre>

<pre>believe mebas thoda sa idhar udhar karna padegai mean ur modified versionthe original has to be much better rite</pre>

<pre>thanks for your precious enquiry and faith shown in agarwal movers group your lifesaving is in your hands dont end up loosing them to fraud cos delivering smiles since</pre>

<pre>dunnom still on painkillers so i dunno how much pain there when they take me off itdepending on that i can sty or leave</pre>

<pre>Replying to USER Lol when I tell some friend this last week they be yabbing me o At the end of XXXX they send the yearly report to my bro and he realise he lose 28k to sms Since then We no even think be again Lol 08/09/2018 10:17</pre>


### Multinomial Naive Bayes — Fałszywe alarmy

<pre>Else send me the complete package</pre>

<pre>Amway order &lt;#&gt;</pre>

<pre>Please confirm once you remove.</pre>

<pre>Can return me 100 this month?</pre>

<pre>wassup wid the status update seemz lyk ur pissed off</pre>


### Multinomial Naive Bayes — Przeoczony spam

<pre>rofl i knew how u get to track my progress i should say its a crappy paper but it has come out really well impact wise though p its beeeeeen long since we saw each othr goodnight da gonna sleep</pre>

<pre>It be Election Day and Dr. Jeff Van Drew be count on your vote Wel 've get a high turnout election and your vote matter Polls be open until 8pm</pre>

<pre>Bond Girl? Which Indian actress you would like to see as Bond Girl? 1) Bipasha 2) Kareena 3) Katrina ^1/ 2/ 3 Reply within 24 hrs to know who's the next Bond Girl.</pre>

<pre>Dating Chatting Please join us for appointment to meet new friend</pre>

<pre>Your package be already on its way http://Your face be see in this video we would like to talk to you http://motochki.ru/b/wc/ fAS.gW 8gU-4mN2 m</pre>


### Linear SVM — Fałszywe alarmy

<pre>Amway order &lt;#&gt;</pre>

<pre>may be, even I think yes. Form &lt;#&gt; is submitted only to save tax. Ucan confirm it from Noida SGC services as well and also ask till whenu can submit it and do u need to save file ITR? Just confirm thesethings.</pre>

<pre>wassup wid the status update seemz lyk ur pissed off</pre>

<pre>Your order from AMAZON booked vide GATI DKT 502024095 is out for delivery, Contact Person: SOURAV/9932909134</pre>

<pre>the last shop you will visit in some structured ways generally gives the best rate</pre>


### Linear SVM — Przeoczony spam

<pre>rofl i knew how u get to track my progress i should say its a crappy paper but it has come out really well impact wise though p its beeeeeen long since we saw each othr goodnight da gonna sleep</pre>

<pre>believe mebas thoda sa idhar udhar karna padegai mean ur modified versionthe original has to be much better rite</pre>

<pre>thanks for your precious enquiry and faith shown in agarwal movers group your lifesaving is in your hands dont end up loosing them to fraud cos delivering smiles since</pre>

<pre>dunnom still on painkillers so i dunno how much pain there when they take me off itdepending on that i can sty or leave</pre>

<pre>*Dial &lt;#&gt; for RingIN Tone -'Football World Cup Song'*</pre>


## Interpretacja

Długość i obecność URL to segmenty opisowe, nie ustalone przyczyny błędów. Proporcje klas i mała liczebność segmentów mogą zmieniać wskaźniki. Jaccard opisuje wspólne błędy; n/a oznacza brak błędów w obu modelach. Margines SVM nie jest prawdopodobieństwem i nie jest porównywalny ze skalą innych modeli. To analiza testu po ocenie. Nie dobieramy na jej podstawie parametrów na tym samym teście; do rozwoju potrzebna będzie walidacja lub nowy test.
