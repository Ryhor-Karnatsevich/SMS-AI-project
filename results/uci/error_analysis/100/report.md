# Analiza błędów: UCI SMS Spam Collection

Trening: 100% puli. Test: 516 SMS. Dostępne modele: 3/4.

| Model | Fałszywe alarmy | Przeoczony spam |
|---|---:|---:|
| Regresja logistyczna | 1 | 16 |
| Multinomial Naive Bayes | 0 | 15 |
| Linear SVM | 1 | 5 |

Brak zapisanych modeli: MLP.

Wersje bibliotek różnią się od zapisanych. Dla tych modeli sprawdzono zgodność predykcji z wcześniejszymi miarami zbiorczymi; szczegóły zawiera manifest.json. Zgodność miar nie dowodzi identyczności każdej pojedynczej predykcji ani pełnej zgodności prawdopodobieństw między wersjami.

## Przykłady do ręcznej oceny

Przykłady poniżej są pierwszymi błędami w stałej kolejności testu, nie rankingiem ważności.

### Regresja logistyczna — Fałszywe alarmy

<pre>Are you free now?can i call now?</pre>


### Regresja logistyczna — Przeoczony spam

<pre>Rock yr chik. Get 100's of filthy films &amp;XXX pics on yr phone now. rply FILTH to 69669. Saristar Ltd, E14 9YT 08701752560. 450p per 5 days. Stop2 cancel</pre>

<pre>cmon babe, make me horny, *turn* me on! Txt me your fantasy now babe -) Im hot, sticky and need you now. All replies cost £1.50. 2 cancel send STOP</pre>

<pre>Camera - You are awarded a SiPix Digital Camera! call 09061221066 fromm landline. Delivery within 28 days</pre>

<pre>Hey I am really horny want to chat or see me naked text hot to 69698 text charged at 150pm to unsubscribe text stop 69698</pre>

<pre>Santa Calling! Would your little ones like a call from Santa Xmas eve? Call 09058094583 to book your time.</pre>


### Multinomial Naive Bayes — Fałszywe alarmy

Brak takich błędów w tym teście.

### Multinomial Naive Bayes — Przeoczony spam

<pre>500 free text msgs. Just text ok to 80488 and we'll credit your account</pre>

<pre>cmon babe, make me horny, *turn* me on! Txt me your fantasy now babe -) Im hot, sticky and need you now. All replies cost £1.50. 2 cancel send STOP</pre>

<pre>U have a secret admirer who is looking 2 make contact with U-find out who they R*reveal who thinks UR so special-call on 09058094599</pre>

<pre>Hey I am really horny want to chat or see me naked text hot to 69698 text charged at 150pm to unsubscribe text stop 69698</pre>

<pre>Santa Calling! Would your little ones like a call from Santa Xmas eve? Call 09058094583 to book your time.</pre>


### Linear SVM — Fałszywe alarmy

<pre>Are you free now?can i call now?</pre>


### Linear SVM — Przeoczony spam

<pre>Back 2 work 2morro half term over! Can U C me 2nite 4 some sexy passion B4 I have 2 go back? Chat NOW 09099726481 Luv DENA Calls £1/minMobsmoreLKPOBOX177HP51FL</pre>

<pre>Guess who am I?This is the first time I created a web page WWW.ASJESUS.COM read all I wrote. I'm waiting for your opinions. I want to be your friend 1/1</pre>

<pre>Hi I'm sue. I am 20 years old and work as a lapdancer. I love sex. Text me live - I'm i my bedroom now. text SUE to 89555. By TextOperator G2 1DA 150ppmsg 18+</pre>

<pre>Babe: U want me dont u baby! Im nasty and have a thing 4 filthyguys. Fancy a rude time with a sexy bitch. How about we go slo n hard! Txt XXX SLO(4msgs)</pre>

<pre>CLAIRE here am havin borin time &amp; am now alone U wanna cum over 2nite? Chat now 09099725823 hope 2 C U Luv CLAIRE xx Calls£1/minmoremobsEMSPOBox45PO139WA</pre>


## Interpretacja

Długość i obecność URL to segmenty opisowe, nie ustalone przyczyny błędów. Proporcje klas i mała liczebność segmentów mogą zmieniać wskaźniki. Jaccard opisuje wspólne błędy; n/a oznacza brak błędów w obu modelach. Margines SVM nie jest prawdopodobieństwem i nie jest porównywalny ze skalą innych modeli. To analiza testu po ocenie. Nie dobieramy na jej podstawie parametrów na tym samym teście; do rozwoju potrzebna będzie walidacja lub nowy test.
