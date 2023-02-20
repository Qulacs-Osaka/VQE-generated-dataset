OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-1.990537084284476) q[0];
rz(-2.0458136706284673) q[0];
ry(-0.00830239251969888) q[1];
rz(-0.369070672331806) q[1];
ry(3.140942282568469) q[2];
rz(2.4008578209825346) q[2];
ry(-0.00039747498224934494) q[3];
rz(-0.8195160898643499) q[3];
ry(2.9940159682420657) q[4];
rz(0.20018944979500652) q[4];
ry(1.5976963503497252) q[5];
rz(-3.139825100479402) q[5];
ry(0.0012335247516244152) q[6];
rz(2.9023519695823756) q[6];
ry(0.0013188336677556478) q[7];
rz(0.20772325341805864) q[7];
ry(-0.006113431905796593) q[8];
rz(1.6537399374335626) q[8];
ry(-0.009862356820759933) q[9];
rz(0.2217641279045326) q[9];
ry(-1.5647307983333398) q[10];
rz(-0.9735144687637387) q[10];
ry(1.574924277877265) q[11];
rz(2.1522008459032502) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(3.047894789521696) q[0];
rz(0.11713641835283538) q[0];
ry(-1.5706725659982181) q[1];
rz(1.742982865543576) q[1];
ry(-0.365556639915547) q[2];
rz(3.0140122340906736) q[2];
ry(-0.0052494229780174345) q[3];
rz(-0.10796708163130388) q[3];
ry(0.037536981677816605) q[4];
rz(2.643899089718504) q[4];
ry(1.5970266077035122) q[5];
rz(1.0992778333179718) q[5];
ry(-0.0031542432478483197) q[6];
rz(-1.9043554437472887) q[6];
ry(3.139913226387704) q[7];
rz(-2.724084708170172) q[7];
ry(1.5710913248568648) q[8];
rz(1.304198009371961) q[8];
ry(1.570156711102701) q[9];
rz(1.3445759613085038) q[9];
ry(-2.067505049131138) q[10];
rz(0.9813296779068228) q[10];
ry(-2.119059063112776) q[11];
rz(2.8047357953972583) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(2.809448132303347) q[0];
rz(0.024828220508942647) q[0];
ry(-1.4926022307623936) q[1];
rz(-2.168912045714252) q[1];
ry(0.036252218092378996) q[2];
rz(-3.121917352685697) q[2];
ry(1.3567315428619575) q[3];
rz(2.335954282873061) q[3];
ry(2.6826560553724645) q[4];
rz(-2.7235697353192045) q[4];
ry(-1.1199814494110403) q[5];
rz(1.6608115089236635) q[5];
ry(-1.599340462721468) q[6];
rz(1.62261103850493) q[6];
ry(3.116905610074833) q[7];
rz(-1.1982112487261336) q[7];
ry(1.348381974562801) q[8];
rz(-2.3291414918734388) q[8];
ry(-1.8325108121876983) q[9];
rz(-2.9356917072403244) q[9];
ry(-0.6261864498752949) q[10];
rz(1.0425523404139372) q[10];
ry(1.427260203427431) q[11];
rz(2.1554069435382788) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.031068127188789596) q[0];
rz(3.128734083487909) q[0];
ry(-3.1342552530644) q[1];
rz(-0.5949903393580206) q[1];
ry(-3.1411320384562456) q[2];
rz(-2.8803691040854127) q[2];
ry(-3.141164819310794) q[3];
rz(0.5397693249975382) q[3];
ry(-3.074021967122153) q[4];
rz(0.07810697123227124) q[4];
ry(0.06765155558809514) q[5];
rz(1.76560906544977) q[5];
ry(-0.3185566521422581) q[6];
rz(-0.08300159465216783) q[6];
ry(-2.380549986499398) q[7];
rz(-2.903972368230707) q[7];
ry(-3.141108639392809) q[8];
rz(-0.28996183904092876) q[8];
ry(0.0034485594602810814) q[9];
rz(2.5252733831693135) q[9];
ry(-0.01350350757704888) q[10];
rz(1.8345958477569635) q[10];
ry(1.47684247282106) q[11];
rz(2.566462942789771) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.3308067657884397) q[0];
rz(0.7790805314147847) q[0];
ry(1.399512116652587) q[1];
rz(2.108142965517543) q[1];
ry(-1.5257208541051779) q[2];
rz(3.1308014872486694) q[2];
ry(-0.35409757819524046) q[3];
rz(2.353958022502679) q[3];
ry(-3.135718705463141) q[4];
rz(1.455276810330177) q[4];
ry(1.9002837417536433) q[5];
rz(2.7044880279180084) q[5];
ry(1.7895667736962482) q[6];
rz(-1.6241477534969313) q[6];
ry(1.5494385296764257) q[7];
rz(3.1305243845483175) q[7];
ry(-3.1336041953383216) q[8];
rz(0.6616614724128737) q[8];
ry(-3.1347787935943425) q[9];
rz(-0.599040433037845) q[9];
ry(0.39454585711805784) q[10];
rz(-2.9027864135507513) q[10];
ry(1.6995083800365007) q[11];
rz(-0.44922612578005156) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-2.796629490317583) q[0];
rz(0.43317421469129147) q[0];
ry(-3.0024698182516896) q[1];
rz(0.42617037528334506) q[1];
ry(1.5616728146247363) q[2];
rz(-0.7397816903860257) q[2];
ry(3.1392004400695637) q[3];
rz(1.9122014160248195) q[3];
ry(0.01188710543714992) q[4];
rz(-2.9478259146490204) q[4];
ry(0.0003640419808058871) q[5];
rz(0.5984765486668948) q[5];
ry(-0.5624510665199995) q[6];
rz(-1.4537661435193554) q[6];
ry(-1.0153264078149835) q[7];
rz(-2.4499707063753884) q[7];
ry(-0.08438700308047586) q[8];
rz(0.5896217259841537) q[8];
ry(0.08332207808198704) q[9];
rz(-2.3292888130314884) q[9];
ry(-2.9990502506957144) q[10];
rz(-1.462773929384515) q[10];
ry(-0.09114488404448551) q[11];
rz(0.061834783780371085) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-2.5163403102641357) q[0];
rz(-1.0540144928259962) q[0];
ry(2.1044559310136566) q[1];
rz(2.860167766737045) q[1];
ry(3.133675328588087) q[2];
rz(-1.0314764675112402) q[2];
ry(3.1407726922703034) q[3];
rz(1.8636540196476046) q[3];
ry(-3.1307882079446476) q[4];
rz(2.8834853919067878) q[4];
ry(-1.5651074729560872) q[5];
rz(1.5653455717527027) q[5];
ry(-1.8530323426887492) q[6];
rz(2.513895701188408) q[6];
ry(-0.03823876980295263) q[7];
rz(1.807317772868208) q[7];
ry(2.7585056455764447) q[8];
rz(1.6410376803800328) q[8];
ry(-1.8280586840353308) q[9];
rz(-3.1141364988083913) q[9];
ry(-1.0903906612725909) q[10];
rz(3.113995816010603) q[10];
ry(2.001406838646613) q[11];
rz(1.8426388902694093) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.7410013806421736) q[0];
rz(-1.5030835296109109) q[0];
ry(1.288806973239864) q[1];
rz(2.4457237036544237) q[1];
ry(-0.015870489389860687) q[2];
rz(-0.4053743317366072) q[2];
ry(-0.5101980261340433) q[3];
rz(0.8748999491352825) q[3];
ry(-1.5917314120593313) q[4];
rz(-0.7011393142818322) q[4];
ry(2.9915779952189134) q[5];
rz(-0.0056777226153279505) q[5];
ry(-0.001187712340684234) q[6];
rz(-0.9686351818385495) q[6];
ry(3.138529805470494) q[7];
rz(-0.5503029906776541) q[7];
ry(0.022410017605881016) q[8];
rz(1.5239859740687398) q[8];
ry(3.096432946503379) q[9];
rz(2.2834918002133056) q[9];
ry(2.2334277297120613) q[10];
rz(-0.30856935364237176) q[10];
ry(-0.7301498419907702) q[11];
rz(2.33528728099268) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(2.5305606400816063) q[0];
rz(2.572116024840899) q[0];
ry(-2.237141971994365) q[1];
rz(0.8053833643938154) q[1];
ry(3.1395171988103026) q[2];
rz(-0.5613460556069754) q[2];
ry(-0.00249561143600463) q[3];
rz(0.9615724095796506) q[3];
ry(-0.000595220439077718) q[4];
rz(-1.5330639635207906) q[4];
ry(1.145767247681246) q[5];
rz(-1.5943016259384546) q[5];
ry(-1.602706794320043) q[6];
rz(2.285685814844707) q[6];
ry(-1.7441356568662565) q[7];
rz(0.6416563359763874) q[7];
ry(1.2932831512655587) q[8];
rz(1.96847875940069) q[8];
ry(-1.636674435243303) q[9];
rz(1.4424754077129185) q[9];
ry(-0.28147674980281145) q[10];
rz(-0.6225300301201858) q[10];
ry(1.4232751208342465) q[11];
rz(-2.8545068422178512) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-0.1786220050146027) q[0];
rz(-2.4949102759097634) q[0];
ry(1.0790147088233741) q[1];
rz(1.0645271636263764) q[1];
ry(-0.3360536732497748) q[2];
rz(-0.3781476123757503) q[2];
ry(-2.4351594429448453) q[3];
rz(0.9276390797908716) q[3];
ry(3.140892649443177) q[4];
rz(1.9670512991032936) q[4];
ry(3.1364672985059565) q[5];
rz(3.11362449354533) q[5];
ry(-1.56089650524192) q[6];
rz(1.5694754304484677) q[6];
ry(-0.04428150804956399) q[7];
rz(-1.3600466202497836) q[7];
ry(2.468171694606031) q[8];
rz(3.1028271324005305) q[8];
ry(2.6118451613948577) q[9];
rz(2.6841188869005124) q[9];
ry(-0.044948531018107286) q[10];
rz(-3.1308481767539496) q[10];
ry(1.089443965423385) q[11];
rz(-1.7200900115757205) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.23231970353603357) q[0];
rz(-2.1220826939425868) q[0];
ry(2.7794860121139617) q[1];
rz(2.0416577616191525) q[1];
ry(-0.001085844775754019) q[2];
rz(-2.5762674026801293) q[2];
ry(0.0019268732595766405) q[3];
rz(-1.6050202718866884) q[3];
ry(0.003080560622842121) q[4];
rz(0.08314001376011416) q[4];
ry(-0.05077925240845893) q[5];
rz(-3.072465039606237) q[5];
ry(-1.5625033977003318) q[6];
rz(-0.9767191998633358) q[6];
ry(1.5716189058915755) q[7];
rz(0.5362746147965307) q[7];
ry(-0.1273718926426956) q[8];
rz(1.293315954323817) q[8];
ry(3.0698996681501516) q[9];
rz(-2.434252344343784) q[9];
ry(-1.5987387583098789) q[10];
rz(1.7271738875916374) q[10];
ry(-0.5241213695969248) q[11];
rz(0.004111689502746429) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(2.0139658068767035) q[0];
rz(0.042634788233254284) q[0];
ry(0.2974380098038374) q[1];
rz(-0.37829611246857464) q[1];
ry(-0.8731905179254387) q[2];
rz(-1.8813769651175378) q[2];
ry(1.5552243515171291) q[3];
rz(1.3674292607319325) q[3];
ry(-2.859719648586705) q[4];
rz(1.6657026702770608) q[4];
ry(-1.5560780695155287) q[5];
rz(2.345659949043625) q[5];
ry(0.05233400324650422) q[6];
rz(0.7474815520965353) q[6];
ry(-3.131084468442484) q[7];
rz(-0.514692628527964) q[7];
ry(-2.3652248256581383) q[8];
rz(-1.5606264300511388) q[8];
ry(-2.343118238093931) q[9];
rz(-1.3994371142219812) q[9];
ry(-0.12297466483369145) q[10];
rz(3.0014019942289103) q[10];
ry(-2.0218155613811297) q[11];
rz(-3.139507020724619) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-0.6025526558449066) q[0];
rz(-2.080527663026696) q[0];
ry(2.7287212288720286) q[1];
rz(0.6840700824396772) q[1];
ry(-0.0002642089994621201) q[2];
rz(1.787416897654897) q[2];
ry(3.140131713900617) q[3];
rz(-1.3916014429491224) q[3];
ry(1.5712456008204043) q[4];
rz(0.005039594601496944) q[4];
ry(-0.023458823097080028) q[5];
rz(-1.4239882627316192) q[5];
ry(-0.00019507424123688113) q[6];
rz(1.4766939197717726) q[6];
ry(-3.1389392999198824) q[7];
rz(-0.3055466152303819) q[7];
ry(1.6098313377556783) q[8];
rz(3.131944953353757) q[8];
ry(1.5906179403617442) q[9];
rz(-0.3414303169221258) q[9];
ry(-1.5719979152430223) q[10];
rz(-1.5771791853935664) q[10];
ry(-1.6408485554993784) q[11];
rz(1.5707027912228997) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-0.10268610745030868) q[0];
rz(-2.9329532971326575) q[0];
ry(-0.2520429981405998) q[1];
rz(0.8157201122396405) q[1];
ry(-1.5703940129273484) q[2];
rz(-1.6579950828921053) q[2];
ry(-1.570831148442188) q[3];
rz(-0.008242966121182811) q[3];
ry(0.6664096814159794) q[4];
rz(1.5681469561768218) q[4];
ry(-0.0049614177754353506) q[5];
rz(-0.8367704763817274) q[5];
ry(0.002292387045569589) q[6];
rz(-2.8182525888549645) q[6];
ry(3.1288805856362645) q[7];
rz(-1.3364024104750114) q[7];
ry(-1.5382584424263406) q[8];
rz(0.7488397145551272) q[8];
ry(1.923139024298145) q[9];
rz(0.5797326517700436) q[9];
ry(1.545524721653436) q[10];
rz(-1.4939914278158712) q[10];
ry(-1.5700955616642465) q[11];
rz(1.5573786584069786) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-0.5444672481228995) q[0];
rz(-0.3356231277730322) q[0];
ry(-0.4238389390461732) q[1];
rz(-1.130051723480056) q[1];
ry(0.0010710647028968623) q[2];
rz(-2.6568932985454357) q[2];
ry(0.00021543524795635156) q[3];
rz(1.5809146626605717) q[3];
ry(1.3460384384068949) q[4];
rz(0.5461447495946058) q[4];
ry(-1.5678806449580727) q[5];
rz(-2.6072816112828305) q[5];
ry(1.570937763144563) q[6];
rz(-0.2804169976282977) q[6];
ry(1.5738916514721952) q[7];
rz(-2.8798987068053594) q[7];
ry(0.8179541392616434) q[8];
rz(1.6106929314733844) q[8];
ry(-0.8422784794690215) q[9];
rz(1.7663173002832897) q[9];
ry(3.1115386915674375) q[10];
rz(1.6148592865627984) q[10];
ry(-1.5414373813725437) q[11];
rz(-1.6105652629072864) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-2.954874775848674) q[0];
rz(2.0100599734117175) q[0];
ry(1.156942543199464) q[1];
rz(0.5057432802390555) q[1];
ry(3.133383379546495) q[2];
rz(-0.5226500566138403) q[2];
ry(1.5638010130091802) q[3];
rz(-3.11423257605157) q[3];
ry(-3.1072116420099696) q[4];
rz(-2.515821796599231) q[4];
ry(-3.1397714940513866) q[5];
rz(-2.29876967310003) q[5];
ry(3.1412752827842394) q[6];
rz(2.879871010403456) q[6];
ry(-3.140567335009457) q[7];
rz(0.09922556070486496) q[7];
ry(1.6569740187815363) q[8];
rz(-1.567002451924403) q[8];
ry(1.6402199182635357) q[9];
rz(0.26499851331177793) q[9];
ry(1.5774316777182467) q[10];
rz(2.3679058544587908) q[10];
ry(1.5829462434518664) q[11];
rz(1.4266425207879927) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-0.17668011369558664) q[0];
rz(2.4965037539463237) q[0];
ry(1.571383385167544) q[1];
rz(0.15872625109001426) q[1];
ry(0.001798567585890254) q[2];
rz(-0.6469124818574841) q[2];
ry(-1.5718474463001535) q[3];
rz(3.14087907621054) q[3];
ry(0.031240334569007537) q[4];
rz(-2.3950317083114467) q[4];
ry(3.140216637241977) q[5];
rz(-2.81525109101219) q[5];
ry(-0.828347993904104) q[6];
rz(0.9517588785997273) q[6];
ry(1.2063433761617945) q[7];
rz(-0.6991845488297209) q[7];
ry(-0.7657431521863982) q[8];
rz(-1.108745097148958) q[8];
ry(-1.8903632063410791) q[9];
rz(-0.5824631870891244) q[9];
ry(0.00747089134447081) q[10];
rz(-0.8128116547164184) q[10];
ry(0.016619888103175596) q[11];
rz(-1.4286200447616606) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.001696834223232635) q[0];
rz(2.200943760512061) q[0];
ry(-0.000836190215734151) q[1];
rz(-1.7153824189369309) q[1];
ry(-1.5707506373542983) q[2];
rz(-1.6202944784846889) q[2];
ry(-1.7063055659821886) q[3];
rz(0.6906242938776908) q[3];
ry(3.141137755540434) q[4];
rz(1.9719162418781375) q[4];
ry(3.0854555347472226) q[5];
rz(-0.753884071301826) q[5];
ry(3.1405476569594017) q[6];
rz(-2.1586208776511073) q[6];
ry(-0.0004462748567109363) q[7];
rz(-2.20696443929282) q[7];
ry(3.140603177340427) q[8];
rz(0.49598703519887344) q[8];
ry(-3.1402686348459565) q[9];
rz(0.26149878262064075) q[9];
ry(3.080665514686364) q[10];
rz(1.5770300362561585) q[10];
ry(1.572595679917686) q[11];
rz(-3.0414829409158606) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.5701254288071556) q[0];
rz(1.5709557780158259) q[0];
ry(3.1415278350098927) q[1];
rz(-3.090450435162135) q[1];
ry(3.1179129809942894) q[2];
rz(1.5652613635576031) q[2];
ry(-3.141472425397696) q[3];
rz(0.6907166046710705) q[3];
ry(0.0017302782897241006) q[4];
rz(1.879446686689322) q[4];
ry(-0.002340553611046246) q[5];
rz(2.356084832565224) q[5];
ry(-2.400131950902666) q[6];
rz(1.8039123551876823) q[6];
ry(1.9629398118245769) q[7];
rz(-2.2906681803456266) q[7];
ry(-1.5773424365800917) q[8];
rz(-1.5685934169455846) q[8];
ry(-1.571863895618745) q[9];
rz(-3.1407386248636637) q[9];
ry(3.1169518686067783) q[10];
rz(-3.122204512459887) q[10];
ry(-1.5773915550846915) q[11];
rz(-2.600938725165985) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-1.570110076039005) q[0];
rz(1.575375480409077) q[0];
ry(1.570891137903702) q[1];
rz(0.002583822431560634) q[1];
ry(-0.010764592921463034) q[2];
rz(-0.6177798176263042) q[2];
ry(-1.5668257405293442) q[3];
rz(1.5432466995066987) q[3];
ry(-1.5693907356791614) q[4];
rz(-0.0007188960649715525) q[4];
ry(1.5691987759132293) q[5];
rz(0.015084577931538471) q[5];
ry(-0.0003087677735669203) q[6];
rz(-2.6892375257503067) q[6];
ry(3.1411151480299386) q[7];
rz(-2.7087047605785504) q[7];
ry(1.5711820113951205) q[8];
rz(-2.7343993265808795) q[8];
ry(-1.5706125904195378) q[9];
rz(-3.1411664794071466) q[9];
ry(1.6688618161428566) q[10];
rz(1.5420772454881826) q[10];
ry(-0.012286884255705533) q[11];
rz(-2.1122697478297607) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.11549322239184533) q[0];
rz(3.1313886632438446) q[0];
ry(-1.5683957558851382) q[1];
rz(1.5089677010660147) q[1];
ry(-3.1407012691526104) q[2];
rz(2.5669737457283603) q[2];
ry(-0.034898991070506646) q[3];
rz(-1.0584191646024728) q[3];
ry(-1.5696147157293678) q[4];
rz(-0.12182439938502032) q[4];
ry(1.5715497127861084) q[5];
rz(1.5662239925984993) q[5];
ry(3.1415781345401976) q[6];
rz(2.235736455871037) q[6];
ry(3.1414601976782803) q[7];
rz(3.101011404698486) q[7];
ry(-3.135707024839819) q[8];
rz(-1.2617326256613992) q[8];
ry(-1.5708201634985102) q[9];
rz(-0.0007476787670155359) q[9];
ry(3.0233522198093117) q[10];
rz(-1.5946828191401459) q[10];
ry(-1.5775299427129275) q[11];
rz(2.993081015563618) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-1.570873988501793) q[0];
rz(2.8228954124231267) q[0];
ry(-3.141480765048946) q[1];
rz(-2.0597152163251593) q[1];
ry(-1.5716250852736957) q[2];
rz(-1.8782118277624562) q[2];
ry(-0.0007669221804429717) q[3];
rz(2.427209013514424) q[3];
ry(-3.139654577290957) q[4];
rz(2.711455335084071) q[4];
ry(-1.4517743954532865) q[5];
rz(0.18207949854526898) q[5];
ry(1.572221181486876) q[6];
rz(2.880095078531902) q[6];
ry(3.1364579696064863) q[7];
rz(-0.6550199480061548) q[7];
ry(3.1402854153501734) q[8];
rz(1.207858206420394) q[8];
ry(1.570773033293128) q[9];
rz(2.0318400932121747) q[9];
ry(-1.5711067291038052) q[10];
rz(2.877410283295165) q[10];
ry(-1.5695582656273768) q[11];
rz(-2.682487685030953) q[11];