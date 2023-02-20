OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(-0.3313097052062721) q[0];
rz(-1.8283313293994603) q[0];
ry(-0.6148568847964881) q[1];
rz(0.06179936198514022) q[1];
ry(-3.1377676786665787) q[2];
rz(2.551113408818978) q[2];
ry(0.03457940938832532) q[3];
rz(-2.9218546457455803) q[3];
ry(-0.004723718068821791) q[4];
rz(1.964478481020953) q[4];
ry(0.10758951701523028) q[5];
rz(-0.2721367364302748) q[5];
ry(0.9295831314550694) q[6];
rz(-2.7179458770907585) q[6];
ry(-0.00020133495213414676) q[7];
rz(0.18021606034936966) q[7];
ry(-0.0005617878398425091) q[8];
rz(-1.152757695784004) q[8];
ry(-1.8528550343319514) q[9];
rz(-2.447099988378483) q[9];
ry(-2.38143968541057) q[10];
rz(-0.9603119319729342) q[10];
ry(0.0008070286282291364) q[11];
rz(-1.4988908767751932) q[11];
ry(1.7736404310110576) q[12];
rz(0.0055160203785777125) q[12];
ry(-3.1399039251397602) q[13];
rz(-0.7687229278122888) q[13];
ry(1.5445698722562762) q[14];
rz(1.9952417452333715) q[14];
ry(-1.8450955402472613) q[15];
rz(-3.123304642907679) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-2.9685118890002595) q[0];
rz(2.235653534672383) q[0];
ry(-0.17879234767681615) q[1];
rz(-1.6927532949630468) q[1];
ry(-2.509850969266666) q[2];
rz(-0.6305544691391197) q[2];
ry(-0.07187039414917251) q[3];
rz(-1.6334271947222172) q[3];
ry(-3.1362857906691524) q[4];
rz(-2.5223682970216195) q[4];
ry(-2.4510808797111068) q[5];
rz(-0.790345845500949) q[5];
ry(-0.715584178515134) q[6];
rz(1.5436647671284756) q[6];
ry(1.5707214550387674) q[7];
rz(1.5568707348793938) q[7];
ry(-0.00028651165555437785) q[8];
rz(1.6073876011109185) q[8];
ry(-2.2511531010180983) q[9];
rz(2.003400244967258) q[9];
ry(2.098322832969111) q[10];
rz(-1.9957895099599243) q[10];
ry(3.1349900916954603) q[11];
rz(1.2042781863757313) q[11];
ry(1.7910167991922208) q[12];
rz(-2.5712650475027217) q[12];
ry(-0.12437683762055801) q[13];
rz(2.0025799188270197) q[13];
ry(-1.4508853492704528) q[14];
rz(0.8846641116889714) q[14];
ry(0.954941037751933) q[15];
rz(2.229664535307907) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(2.7953676728735277) q[0];
rz(-1.524399245717252) q[0];
ry(2.4719734229248758) q[1];
rz(-1.1977555933777715) q[1];
ry(2.2408594561523403) q[2];
rz(1.418155793895225) q[2];
ry(2.9719871927909187) q[3];
rz(-1.9234630224061586) q[3];
ry(3.141251670773454) q[4];
rz(-1.7252127652121223) q[4];
ry(-1.7018671171759137) q[5];
rz(-2.8533151428976606) q[5];
ry(-0.0006093368443238489) q[6];
rz(-0.2543908098852547) q[6];
ry(2.4497511959390534) q[7];
rz(-2.3466437445731456) q[7];
ry(-1.6085287045695107) q[8];
rz(-0.24209564506843143) q[8];
ry(-1.3924176162486126) q[9];
rz(-1.2176120732599598) q[9];
ry(3.027498614259526) q[10];
rz(-0.3786371663401608) q[10];
ry(0.016736743265342465) q[11];
rz(-1.8863468971892043) q[11];
ry(-3.130516815331848) q[12];
rz(0.9546334632719776) q[12];
ry(1.2300792573920944) q[13];
rz(0.9553091639723124) q[13];
ry(1.6359698137045262) q[14];
rz(-0.528002742314423) q[14];
ry(2.582480315676198) q[15];
rz(1.9707867455191121) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(1.2526151746988556) q[0];
rz(0.36183088917858613) q[0];
ry(2.7330179065560727) q[1];
rz(1.0412429547135542) q[1];
ry(3.140018707460763) q[2];
rz(-1.5113468948259738) q[2];
ry(-0.3799601213446689) q[3];
rz(-1.7809695519435123) q[3];
ry(0.0020760446667260624) q[4];
rz(-0.6275859130365884) q[4];
ry(0.3389898572076495) q[5];
rz(-1.4840835781561654) q[5];
ry(-3.0906547637703925) q[6];
rz(1.0152987969991598) q[6];
ry(-0.0002512135742653661) q[7];
rz(-0.2575265977785177) q[7];
ry(0.0013226466485384003) q[8];
rz(2.5906152729459233) q[8];
ry(-1.2688452593631612) q[9];
rz(0.17436077624465315) q[9];
ry(1.7175260976320772) q[10];
rz(-1.932967072461826) q[10];
ry(-3.096696817338188) q[11];
rz(-0.2152988978209415) q[11];
ry(-0.0029294837863702034) q[12];
rz(-0.3238404054417199) q[12];
ry(0.8583632110284096) q[13];
rz(1.6430762180367022) q[13];
ry(3.050942288120992) q[14];
rz(0.9784536084908213) q[14];
ry(0.4221192137290206) q[15];
rz(2.460598605896857) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(1.791062277739977) q[0];
rz(2.235010668332901) q[0];
ry(0.23668706278424967) q[1];
rz(2.588934845995747) q[1];
ry(0.05458162727029236) q[2];
rz(1.0264738491251537) q[2];
ry(0.03557927074406859) q[3];
rz(-1.3673098175730865) q[3];
ry(3.140232374062848) q[4];
rz(-0.38249633779794795) q[4];
ry(0.6147820860195923) q[5];
rz(-2.2969573196876554) q[5];
ry(1.7144231311624019) q[6];
rz(-0.3040626637551749) q[6];
ry(0.030592920530515322) q[7];
rz(1.7769784976326823) q[7];
ry(0.5634597876393919) q[8];
rz(-0.12524186940401513) q[8];
ry(-0.007817918232667316) q[9];
rz(0.8097939108826674) q[9];
ry(0.326597440219781) q[10];
rz(0.15625612610507567) q[10];
ry(-0.08071262645652233) q[11];
rz(-2.182215074816762) q[11];
ry(-0.0048119788632794425) q[12];
rz(2.8211435834183622) q[12];
ry(-1.3045992594878348) q[13];
rz(-0.4193298509490484) q[13];
ry(0.2783925038177895) q[14];
rz(0.4463962861576638) q[14];
ry(-0.3436399676768067) q[15];
rz(-2.61664589327031) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-0.311131866941246) q[0];
rz(-0.3931789962767436) q[0];
ry(-0.5612487837461391) q[1];
rz(-1.337102731820752) q[1];
ry(3.0321913641515925) q[2];
rz(-0.8817796405218635) q[2];
ry(-2.7587430287159855) q[3];
rz(1.088316093902586) q[3];
ry(1.569610251096841) q[4];
rz(-2.9205876527854535) q[4];
ry(2.0430402597773867) q[5];
rz(-1.3781676113173527) q[5];
ry(-2.596726707202136) q[6];
rz(2.9456508638927055) q[6];
ry(-0.0004978725611071155) q[7];
rz(2.563635790192674) q[7];
ry(-3.139701960627746) q[8];
rz(0.5548151720415024) q[8];
ry(2.4875266688416597) q[9];
rz(-2.950889268030783) q[9];
ry(3.1302461817750276) q[10];
rz(-2.0833512632056186) q[10];
ry(3.065559080429626) q[11];
rz(-0.14101543999270216) q[11];
ry(0.005892400057110336) q[12];
rz(-0.48749934851252974) q[12];
ry(0.5751041166336177) q[13];
rz(-1.9750799988415981) q[13];
ry(-2.872012431450413) q[14];
rz(-2.990015566402264) q[14];
ry(-0.8052153680116865) q[15];
rz(-0.5054588868249644) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(0.6463204896321094) q[0];
rz(2.7160051736575883) q[0];
ry(0.4307742673607376) q[1];
rz(2.3064532172435084) q[1];
ry(1.4248742805048322) q[2];
rz(-2.0141171382439094) q[2];
ry(1.539495569614565) q[3];
rz(-2.1499791973345195) q[3];
ry(-1.2634018782420506) q[4];
rz(2.216739741563476) q[4];
ry(-0.378305981177526) q[5];
rz(2.2932792341608876) q[5];
ry(-2.7093668366129933) q[6];
rz(-0.6674401004588307) q[6];
ry(-1.1608317723557606) q[7];
rz(-2.073293723707504) q[7];
ry(0.5095949342976703) q[8];
rz(3.054834259085371) q[8];
ry(0.6480061722366335) q[9];
rz(1.0709488047011861) q[9];
ry(0.9109162240250503) q[10];
rz(-2.723059397448474) q[10];
ry(2.105367078783452) q[11];
rz(3.0910554591394654) q[11];
ry(1.6843621496367085) q[12];
rz(-0.8301718524989806) q[12];
ry(-2.9406121008143256) q[13];
rz(-0.5068052063302473) q[13];
ry(-0.6713434427328396) q[14];
rz(2.1264783995016474) q[14];
ry(2.5000945467342386) q[15];
rz(1.1128547261194832) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(0.4519501194424172) q[0];
rz(-1.768333057481879) q[0];
ry(2.8825616104765173) q[1];
rz(-0.876649896100015) q[1];
ry(-0.0007747923408166919) q[2];
rz(-0.5283106678528483) q[2];
ry(-3.141419963052485) q[3];
rz(2.2833918725446725) q[3];
ry(-3.141424528147207) q[4];
rz(2.9614486222516745) q[4];
ry(-3.130762185472804) q[5];
rz(1.4683246431899337) q[5];
ry(3.123186488360547) q[6];
rz(-0.7525706114913177) q[6];
ry(-0.0006218286647256834) q[7];
rz(2.644788127947942) q[7];
ry(0.002414143880174552) q[8];
rz(-0.7340741644798765) q[8];
ry(0.01993510682893054) q[9];
rz(-2.5876415814883336) q[9];
ry(-3.1408075284884167) q[10];
rz(1.346159023451416) q[10];
ry(3.135846479969403) q[11];
rz(-3.079203328302971) q[11];
ry(-3.1414299270481356) q[12];
rz(-0.8294924645359147) q[12];
ry(3.1400462164202323) q[13];
rz(-2.221878079602825) q[13];
ry(0.9887163292682297) q[14];
rz(0.7729049610515285) q[14];
ry(0.9762651052596159) q[15];
rz(-2.313249048469605) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.7556105123002819) q[0];
rz(-0.7154581950784406) q[0];
ry(0.5318489468467895) q[1];
rz(0.28904490807888017) q[1];
ry(-1.6526741421814635) q[2];
rz(-0.8326419713776735) q[2];
ry(-3.0283128605662424) q[3];
rz(1.7553034917867052) q[3];
ry(2.9161540308888316) q[4];
rz(1.2844993600334078) q[4];
ry(1.7945605403131064) q[5];
rz(-0.5339537008271932) q[5];
ry(2.1310809174017065) q[6];
rz(1.5013072976665693) q[6];
ry(1.6380887528928678) q[7];
rz(-2.3144533635295774) q[7];
ry(2.9124576836568283) q[8];
rz(-0.8210912675472547) q[8];
ry(-0.8459365702213182) q[9];
rz(0.850372162654922) q[9];
ry(-1.8744942837940375) q[10];
rz(1.8929728269103236) q[10];
ry(-2.0533794038041133) q[11];
rz(3.00953347501904) q[11];
ry(1.4797101956760557) q[12];
rz(1.1700386719000908) q[12];
ry(-0.1930885259451374) q[13];
rz(0.3659318192398286) q[13];
ry(0.6910552436895075) q[14];
rz(-0.9546154474140397) q[14];
ry(3.0188993464402323) q[15];
rz(1.3001355846039944) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(0.312198723298956) q[0];
rz(0.3065836627404943) q[0];
ry(0.39913352973131583) q[1];
rz(-1.3784519602819323) q[1];
ry(1.5701949493460026) q[2];
rz(1.3553211428618697) q[2];
ry(0.003457470725341708) q[3];
rz(2.2123370348015907) q[3];
ry(2.4039412676171343) q[4];
rz(-0.23299365995916949) q[4];
ry(0.9223940467801732) q[5];
rz(-2.4444773946058893) q[5];
ry(3.104162535152499) q[6];
rz(-2.348780507130563) q[6];
ry(0.05641805802971737) q[7];
rz(-1.6096737753019354) q[7];
ry(3.1078461739602012) q[8];
rz(2.9478426045057566) q[8];
ry(1.16006300262586) q[9];
rz(-1.9211964103739798) q[9];
ry(-0.2946130510326791) q[10];
rz(-1.299164768227155) q[10];
ry(-1.551624623200615) q[11];
rz(-0.6510165322208643) q[11];
ry(0.012257577650444286) q[12];
rz(1.7866180471127908) q[12];
ry(-0.03077313230505752) q[13];
rz(-1.0303769205480737) q[13];
ry(-1.6339571685796876) q[14];
rz(1.29379736003379) q[14];
ry(-0.24020718113744977) q[15];
rz(-2.9601136819946103) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(2.554827978824394) q[0];
rz(2.894677759074402) q[0];
ry(1.5612811477932267) q[1];
rz(-0.7797217390759837) q[1];
ry(0.39227405621366085) q[2];
rz(-2.2473026304213786) q[2];
ry(-3.136154476249253) q[3];
rz(-2.6922139463806003) q[3];
ry(-3.065155099560189) q[4];
rz(-0.3552733229484577) q[4];
ry(-2.921343133313586) q[5];
rz(-1.356095634873612) q[5];
ry(-3.0905620219023926) q[6];
rz(0.9402972205905815) q[6];
ry(2.857361339473284) q[7];
rz(2.2745106492918734) q[7];
ry(-1.116422795808008) q[8];
rz(1.473630127315512) q[8];
ry(-3.068157520147416) q[9];
rz(1.1202681956507563) q[9];
ry(-1.4125175099413658) q[10];
rz(0.7857426041852661) q[10];
ry(3.028508156878129) q[11];
rz(2.6580118659443843) q[11];
ry(-1.5719590675272712) q[12];
rz(-1.2231134276622273) q[12];
ry(3.12136104127965) q[13];
rz(-1.3233005239841935) q[13];
ry(-0.3282587068381945) q[14];
rz(-2.5616793806973983) q[14];
ry(1.8621744915493572) q[15];
rz(1.2297047063467255) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(1.571626368932817) q[0];
rz(-1.0126791286927457) q[0];
ry(-0.032023129747007946) q[1];
rz(-2.0711207647838874) q[1];
ry(-0.03601598761232451) q[2];
rz(-0.851630473738052) q[2];
ry(-3.1391879102928555) q[3];
rz(-0.08420967217223385) q[3];
ry(0.7413107602029361) q[4];
rz(0.8089326537710435) q[4];
ry(-1.3186943119154355) q[5];
rz(2.8939506977186302) q[5];
ry(2.773940727605093) q[6];
rz(0.7808427034760796) q[6];
ry(3.072199354119414) q[7];
rz(2.1007033981631684) q[7];
ry(3.085505959626572) q[8];
rz(-2.1251113486101607) q[8];
ry(0.02236446672858472) q[9];
rz(-1.7037574324627773) q[9];
ry(-1.9205683942340075) q[10];
rz(-1.1042902307207219) q[10];
ry(-3.1140230681910275) q[11];
rz(0.572504325637781) q[11];
ry(2.6566896337077157) q[12];
rz(1.8000229127104408) q[12];
ry(1.570158879394163) q[13];
rz(2.9579675360469238) q[13];
ry(0.13494172930188064) q[14];
rz(2.3308281499955377) q[14];
ry(1.8056305048153525) q[15];
rz(-2.0535181790199064) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(2.611878693310403) q[0];
rz(3.088681028930226) q[0];
ry(-1.3634432372318959) q[1];
rz(-2.1525632599216165) q[1];
ry(-1.507586637063966) q[2];
rz(2.9826768357983573) q[2];
ry(0.9479217777041924) q[3];
rz(1.7513953424220485) q[3];
ry(-0.9899191361054033) q[4];
rz(-2.19065955497183) q[4];
ry(-0.7089946294149984) q[5];
rz(1.9983964712051838) q[5];
ry(3.1279716789863468) q[6];
rz(0.9282383566350574) q[6];
ry(0.0001541103066154811) q[7];
rz(-2.841672271111499) q[7];
ry(1.7571983369181667) q[8];
rz(1.5651922521343904) q[8];
ry(2.635357113459834) q[9];
rz(2.708734080592649) q[9];
ry(-0.0804679335777633) q[10];
rz(-1.2444034193365532) q[10];
ry(-0.02231097354674283) q[11];
rz(1.2649470255752062) q[11];
ry(-3.1114939098175953) q[12];
rz(1.7851080699617619) q[12];
ry(-0.0016044209616818963) q[13];
rz(-0.8137890788202901) q[13];
ry(-1.6026301998690529) q[14];
rz(2.6396401590930236) q[14];
ry(-2.6342874453982583) q[15];
rz(-2.895381577644128) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(3.129582659272688) q[0];
rz(-0.8528501906218613) q[0];
ry(-3.0811549805028835) q[1];
rz(1.3084452079546436) q[1];
ry(-0.014059016211020818) q[2];
rz(2.827302014132989) q[2];
ry(-3.137790150782692) q[3];
rz(2.454395982699714) q[3];
ry(0.02510578795151739) q[4];
rz(3.0065294715168784) q[4];
ry(-0.00626308155524935) q[5];
rz(1.1660760971631536) q[5];
ry(2.7249684650982964) q[6];
rz(-2.6157198338038232) q[6];
ry(0.03176365626730247) q[7];
rz(1.3887714310330264) q[7];
ry(0.0017611407068152118) q[8];
rz(-2.945069830586453) q[8];
ry(0.006392619176111173) q[9];
rz(-2.7218921103184117) q[9];
ry(-2.1938732331236066) q[10];
rz(2.151624398712623) q[10];
ry(0.04776818090957846) q[11];
rz(-2.742905761687317) q[11];
ry(-2.707478902841529) q[12];
rz(0.09745396180229982) q[12];
ry(3.120372131167154) q[13];
rz(-1.8979793493541237) q[13];
ry(2.393343929188754) q[14];
rz(-0.7750775144964743) q[14];
ry(-0.1440031402580853) q[15];
rz(2.6495628086867753) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(0.8112426487428274) q[0];
rz(2.0398882404910728) q[0];
ry(-1.0885658335077337) q[1];
rz(0.5658300498922227) q[1];
ry(-3.071306540097516) q[2];
rz(0.29946542983310837) q[2];
ry(-3.111282488310369) q[3];
rz(2.3624033249694882) q[3];
ry(0.9992391361103033) q[4];
rz(0.3007091605681924) q[4];
ry(0.3248207964289396) q[5];
rz(-0.23312673115276716) q[5];
ry(-3.0967729322976) q[6];
rz(0.44925361529738234) q[6];
ry(-0.045357908789196585) q[7];
rz(2.3345951744046824) q[7];
ry(1.5625488444978668) q[8];
rz(-0.8303745074024604) q[8];
ry(1.7403577923772096) q[9];
rz(-2.060520851925159) q[9];
ry(3.1050255389157204) q[10];
rz(-0.9303247435756017) q[10];
ry(-2.943018889272987) q[11];
rz(-2.3027591217866283) q[11];
ry(0.8692655373671467) q[12];
rz(-1.392250333271349) q[12];
ry(-1.7884533663547377) q[13];
rz(-1.4303302505139204) q[13];
ry(0.03654617622037648) q[14];
rz(1.5732579029328537) q[14];
ry(2.2810557212135447) q[15];
rz(3.1384211078600983) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-0.007581139932828051) q[0];
rz(-0.7926675576931952) q[0];
ry(-0.19061900989354363) q[1];
rz(1.4111909140444556) q[1];
ry(-0.022209217859718855) q[2];
rz(-1.1823096077645987) q[2];
ry(3.112735894082054) q[3];
rz(2.8360747605033185) q[3];
ry(3.0545442481218887) q[4];
rz(-0.9069422093672612) q[4];
ry(-1.458175256196637) q[5];
rz(2.1357675048534492) q[5];
ry(-0.651731825473753) q[6];
rz(0.6249820646824427) q[6];
ry(0.0027256541694864467) q[7];
rz(-1.283214545046092) q[7];
ry(0.015975940418867043) q[8];
rz(-2.305515222468975) q[8];
ry(0.014969767506483697) q[9];
rz(-2.0445769775267975) q[9];
ry(-0.4500752236818352) q[10];
rz(2.7503485844266553) q[10];
ry(0.05762757013801136) q[11];
rz(1.048272701267801) q[11];
ry(0.05344002594511935) q[12];
rz(2.7405442652661187) q[12];
ry(-3.1373191418967723) q[13];
rz(-0.6656593327056948) q[13];
ry(3.0782853219232185) q[14];
rz(-1.0912212772634229) q[14];
ry(-1.444465786755745) q[15];
rz(0.03602886749011436) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-2.0572686068847412) q[0];
rz(0.972058206205145) q[0];
ry(-0.770763670685194) q[1];
rz(1.4191941894310665) q[1];
ry(-0.8917226038955999) q[2];
rz(-0.5249746981611736) q[2];
ry(-2.481885779704476) q[3];
rz(2.8736305051310085) q[3];
ry(-0.9640344329753496) q[4];
rz(-2.996488826137625) q[4];
ry(-1.9621843776196783) q[5];
rz(0.7611714287151254) q[5];
ry(2.596787594962072) q[6];
rz(-0.24138565165212583) q[6];
ry(-0.05264897947120861) q[7];
rz(-1.5111777798802504) q[7];
ry(0.5098235631211736) q[8];
rz(2.208681575687817) q[8];
ry(2.8785175445728) q[9];
rz(-0.4286600431600709) q[9];
ry(-0.6911210118532332) q[10];
rz(1.756427843728381) q[10];
ry(2.6176418303326923) q[11];
rz(2.127383626877667) q[11];
ry(-2.0871603734142523) q[12];
rz(1.69380829660391) q[12];
ry(-1.4643137203837082) q[13];
rz(-1.6468755326130324) q[13];
ry(-0.12699658875869702) q[14];
rz(2.156368153265766) q[14];
ry(-0.7936120294894513) q[15];
rz(2.9756749088514023) q[15];