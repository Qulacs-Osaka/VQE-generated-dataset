OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(3.1401490629903424) q[0];
rz(-1.9529069131969334) q[0];
ry(-1.5659308671671894) q[1];
rz(-1.5641230985958134) q[1];
ry(1.5743635823119773) q[2];
rz(0.025356684549903363) q[2];
ry(-1.5621734015016466) q[3];
rz(-0.17982138907461476) q[3];
ry(-1.5494629783540035) q[4];
rz(-2.8702427281535012) q[4];
ry(-1.6076332986836712) q[5];
rz(1.363981561464107) q[5];
ry(1.587533197394792) q[6];
rz(1.013383625063955) q[6];
ry(-2.28901937006438) q[7];
rz(-0.011809885686023994) q[7];
ry(0.0004654725433980757) q[8];
rz(-2.458059031836902) q[8];
ry(0.001525124747081108) q[9];
rz(-2.326844867116424) q[9];
ry(-1.509676830452876) q[10];
rz(1.4812450348777997) q[10];
ry(-1.4228745710571555) q[11];
rz(1.3165731538442342) q[11];
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
ry(-0.001079116088041232) q[0];
rz(-1.253153540626557) q[0];
ry(-1.591778166718956) q[1];
rz(-0.5282516445312017) q[1];
ry(-0.007701759927470242) q[2];
rz(-2.933866770551069) q[2];
ry(3.13237925303064) q[3];
rz(-3.0758940311241654) q[3];
ry(3.141425486941514) q[4];
rz(1.974803061014284) q[4];
ry(3.141444497835439) q[5];
rz(2.8949273799576667) q[5];
ry(-3.140866611940665) q[6];
rz(2.5785876406402433) q[6];
ry(1.5582912590393514) q[7];
rz(2.0866089682702915) q[7];
ry(0.002244515034577439) q[8];
rz(-0.37262584381471786) q[8];
ry(3.1410675047455934) q[9];
rz(-1.4465177338320592) q[9];
ry(-1.7412049715426612) q[10];
rz(-1.4972857639352701) q[10];
ry(1.3445100855677918) q[11];
rz(3.050861626665105) q[11];
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
ry(3.02570804726101) q[0];
rz(-0.08049356024209518) q[0];
ry(2.779523300930301) q[1];
rz(1.035871265955282) q[1];
ry(1.805615638198014) q[2];
rz(0.5786000572817773) q[2];
ry(-1.7951134715667407) q[3];
rz(2.701069185029974) q[3];
ry(1.6024622323738287) q[4];
rz(0.000743890630139715) q[4];
ry(1.6532334207998949) q[5];
rz(3.1408172598040145) q[5];
ry(0.765056701918539) q[6];
rz(-2.8702984975815222) q[6];
ry(-1.0559238599656366) q[7];
rz(-2.512479953979058) q[7];
ry(-0.021710378296539545) q[8];
rz(1.4073433281887404) q[8];
ry(-0.06867183332297255) q[9];
rz(-0.9277948344418254) q[9];
ry(-0.4221616930173312) q[10];
rz(-0.6801762016770608) q[10];
ry(0.4225857318939257) q[11];
rz(-0.5110350107757293) q[11];
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
ry(-1.7208808520879515) q[0];
rz(1.7444579666109323) q[0];
ry(-2.76049267189719) q[1];
rz(-1.3620128131042912) q[1];
ry(-3.140654546079633) q[2];
rz(2.2434467341837454) q[2];
ry(-0.002274877340391708) q[3];
rz(2.064673253946781) q[3];
ry(1.5688126660726374) q[4];
rz(2.7233797970320346) q[4];
ry(1.5705887529704075) q[5];
rz(0.5903358595893318) q[5];
ry(-0.011762001636831153) q[6];
rz(2.3830423682848645) q[6];
ry(3.0711059189370795) q[7];
rz(-1.9065935235291698) q[7];
ry(0.0017768540661862176) q[8];
rz(-1.032929089059422) q[8];
ry(-3.139793387111295) q[9];
rz(1.3676371010746715) q[9];
ry(-1.9406246951711967) q[10];
rz(0.25395262456717577) q[10];
ry(-1.4702112585358815) q[11];
rz(1.3039053112293149) q[11];
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
ry(3.030526208586295) q[0];
rz(1.5905563681604347) q[0];
ry(-2.4947885238297554) q[1];
rz(0.5781633800142654) q[1];
ry(-2.770745780459764) q[2];
rz(0.7348782056193717) q[2];
ry(1.6328102786617802) q[3];
rz(-1.9294270515531888) q[3];
ry(-1.7536359869307727) q[4];
rz(2.1680201133753454) q[4];
ry(1.5265548140417176) q[5];
rz(2.8328557489545263) q[5];
ry(-2.8249302965768415) q[6];
rz(0.2333097979840053) q[6];
ry(2.0499620788759056) q[7];
rz(3.0468482430534114) q[7];
ry(0.06281378411564109) q[8];
rz(-1.3383220359112726) q[8];
ry(-3.108784414181176) q[9];
rz(-0.3498541984206219) q[9];
ry(0.9860660787344523) q[10];
rz(0.723119563131851) q[10];
ry(2.7413165843469014) q[11];
rz(-0.22240247216404455) q[11];
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
ry(3.135128949599925) q[0];
rz(-0.5432290097719298) q[0];
ry(-0.004430028703003821) q[1];
rz(-2.6532340252212068) q[1];
ry(-0.005477897889493377) q[2];
rz(1.027148530070443) q[2];
ry(0.0019703154013610558) q[3];
rz(-1.287703679886158) q[3];
ry(-8.497146228079312e-05) q[4];
rz(2.514175958738547) q[4];
ry(3.141461451685261) q[5];
rz(-2.055788068116457) q[5];
ry(-3.117722761014498) q[6];
rz(2.549697977700503) q[6];
ry(-0.005476010361691763) q[7];
rz(-2.897066476638101) q[7];
ry(-3.1397531044321196) q[8];
rz(-1.9927234321091325) q[8];
ry(0.0006512662406824976) q[9];
rz(0.8541811787421192) q[9];
ry(-2.681321496008691) q[10];
rz(1.3022273959182185) q[10];
ry(-0.338784253781804) q[11];
rz(1.7195892182907804) q[11];
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
ry(2.8238074373408826) q[0];
rz(1.5205074371318597) q[0];
ry(2.7455696555721105) q[1];
rz(-2.529986317630545) q[1];
ry(1.4317107633675903) q[2];
rz(-2.5904872112817934) q[2];
ry(-0.14780603137915624) q[3];
rz(-2.2769232460455933) q[3];
ry(-1.5288339576823222) q[4];
rz(-0.2470656528912318) q[4];
ry(-2.6624567702178106) q[5];
rz(1.5095618401658921) q[5];
ry(-2.640105265398715) q[6];
rz(-2.742381542686125) q[6];
ry(-2.521736491046351) q[7];
rz(-1.5843040098438312) q[7];
ry(1.3961106304984403) q[8];
rz(-1.971888841543154) q[8];
ry(-1.4812148398673168) q[9];
rz(2.688602721503311) q[9];
ry(1.9485677414515634) q[10];
rz(-0.09912607816925198) q[10];
ry(-1.011373523516495) q[11];
rz(2.729656335838421) q[11];
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
ry(-0.2249777059044744) q[0];
rz(2.2460054778416962) q[0];
ry(-1.2413165720907904) q[1];
rz(1.006914418743004) q[1];
ry(3.108951292534976) q[2];
rz(1.682864991379952) q[2];
ry(3.1119577800581424) q[3];
rz(1.5584204654773737) q[3];
ry(-3.1108822154144646) q[4];
rz(-2.754607719703095) q[4];
ry(-1.0757457675561475) q[5];
rz(0.6667244046481651) q[5];
ry(-0.024808326284281538) q[6];
rz(-2.9736845508710736) q[6];
ry(0.02340647147098914) q[7];
rz(0.3881628365982933) q[7];
ry(-1.5483898888963123) q[8];
rz(1.6958525138593705) q[8];
ry(0.8266987064787568) q[9];
rz(2.879086418400648) q[9];
ry(1.4779795905436313) q[10];
rz(1.1283994894609162) q[10];
ry(-2.547587115038075) q[11];
rz(-3.1104618895977825) q[11];
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
ry(-0.8883550323530409) q[0];
rz(1.6055349190506654) q[0];
ry(-2.168580759578707) q[1];
rz(0.7057573908372011) q[1];
ry(-0.9025766087182028) q[2];
rz(0.46078352943419565) q[2];
ry(2.2446191622961535) q[3];
rz(-1.795940921433727) q[3];
ry(2.2645385099287862) q[4];
rz(1.4106519335133445) q[4];
ry(2.5210587176972856) q[5];
rz(0.4496153907600837) q[5];
ry(-1.535890060182339) q[6];
rz(-1.4031683237893082) q[6];
ry(1.605076581081576) q[7];
rz(-2.055648461337863) q[7];
ry(-0.31671025076493464) q[8];
rz(-3.0475414409211847) q[8];
ry(-0.2687296241672008) q[9];
rz(-1.0402918014607694) q[9];
ry(-3.0255692182804768) q[10];
rz(2.7156893294238635) q[10];
ry(-3.0986203108439816) q[11];
rz(0.7150781012448791) q[11];
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
ry(0.13648351342969356) q[0];
rz(-2.397729355328133) q[0];
ry(2.4596630191437385) q[1];
rz(-1.2547947560077275) q[1];
ry(3.1361735685543026) q[2];
rz(1.3958399981928005) q[2];
ry(-0.013457502449409908) q[3];
rz(2.9594401110441404) q[3];
ry(-2.1085771419612307) q[4];
rz(2.8673047748557403) q[4];
ry(-0.45148724933606293) q[5];
rz(-1.2669601010766087) q[5];
ry(-0.2665296732862057) q[6];
rz(-1.3096173275406633) q[6];
ry(1.210277473816725) q[7];
rz(1.264940585283232) q[7];
ry(-0.37016672149136914) q[8];
rz(2.669688192554949) q[8];
ry(1.143198271267467) q[9];
rz(-1.3824565958899893) q[9];
ry(-0.45335203670850815) q[10];
rz(-0.7249134692616) q[10];
ry(1.3373949919105195) q[11];
rz(0.5415046747361164) q[11];
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
ry(0.4662791855738577) q[0];
rz(1.995772853527435) q[0];
ry(-1.5374788640026207) q[1];
rz(-2.763602063605997) q[1];
ry(-3.133418406512704) q[2];
rz(-1.305049112602358) q[2];
ry(-0.015460117957888882) q[3];
rz(-2.5696454553017176) q[3];
ry(0.013140079517818528) q[4];
rz(-3.0437974888397417) q[4];
ry(3.1397724944070813) q[5];
rz(1.26003743279254) q[5];
ry(2.7833765710534775) q[6];
rz(0.08852000900912689) q[6];
ry(0.38242610834176016) q[7];
rz(2.723937696508284) q[7];
ry(-0.004183749553469675) q[8];
rz(-1.039567348966882) q[8];
ry(3.1393190191988665) q[9];
rz(2.720083543381802) q[9];
ry(0.2922782141513229) q[10];
rz(-2.9229252710182663) q[10];
ry(0.6284441353981403) q[11];
rz(-2.939173052015424) q[11];
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
ry(-2.5721268580695082) q[0];
rz(-1.0953647375477895) q[0];
ry(0.49487955161043923) q[1];
rz(-0.7992228452834399) q[1];
ry(-0.028624879692905763) q[2];
rz(0.8174000550856224) q[2];
ry(0.04646985999446144) q[3];
rz(-0.3547729486780241) q[3];
ry(-2.814397452815676) q[4];
rz(0.7142255612289816) q[4];
ry(0.34177298267137246) q[5];
rz(2.9518013441874023) q[5];
ry(-0.764470489586406) q[6];
rz(-3.005470824256028) q[6];
ry(-1.3732686585158085) q[7];
rz(-0.651447265105832) q[7];
ry(3.12351212525402) q[8];
rz(1.3567855699777729) q[8];
ry(3.127926262282241) q[9];
rz(2.2933604678107073) q[9];
ry(-0.7153725936383895) q[10];
rz(-0.20243424579480784) q[10];
ry(1.3865080646409504) q[11];
rz(1.6443250777497251) q[11];
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
ry(-2.1858913061009453) q[0];
rz(2.061133641542055) q[0];
ry(-1.5457319853140898) q[1];
rz(-1.2319935830231625) q[1];
ry(-0.7062518895956247) q[2];
rz(2.808790638953869) q[2];
ry(2.4571131130766886) q[3];
rz(-1.3171724482424525) q[3];
ry(-1.9174959230252258) q[4];
rz(-2.643134127320644) q[4];
ry(-2.7240480052664067) q[5];
rz(-0.8150369710163651) q[5];
ry(2.9842305761873105) q[6];
rz(1.1750789643999455) q[6];
ry(2.946710731239267) q[7];
rz(-0.7037765065620035) q[7];
ry(2.8372302863411654) q[8];
rz(-1.8357144089972945) q[8];
ry(-0.36514512428701806) q[9];
rz(-2.075957027292293) q[9];
ry(2.4286247553495652) q[10];
rz(2.9660224021509105) q[10];
ry(-1.5444017919674007) q[11];
rz(1.3046660282935196) q[11];
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
ry(-2.2251010831651534) q[0];
rz(2.197961448068459) q[0];
ry(2.5283786810205044) q[1];
rz(-0.8191263730823826) q[1];
ry(-3.1409208694647384) q[2];
rz(-1.8714865763402884) q[2];
ry(-3.1413075262710746) q[3];
rz(1.8991365229132546) q[3];
ry(-3.129456971664872) q[4];
rz(2.2976289472811358) q[4];
ry(0.8782910510149113) q[5];
rz(-2.832595014567396) q[5];
ry(0.033058778782949716) q[6];
rz(-2.7856066750977586) q[6];
ry(0.00607443716787781) q[7];
rz(1.4287550598488943) q[7];
ry(0.04938600391343151) q[8];
rz(-1.257867082204715) q[8];
ry(-3.094126570029841) q[9];
rz(1.3531795355530756) q[9];
ry(1.2901766463169846) q[10];
rz(-2.6953346991712914) q[10];
ry(-1.7095182583595507) q[11];
rz(0.8370005378277168) q[11];
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
ry(1.0597176927583194) q[0];
rz(2.204400925459571) q[0];
ry(-0.9609614061243369) q[1];
rz(2.0667263830770635) q[1];
ry(0.0022425235936340826) q[2];
rz(-2.070011238131575) q[2];
ry(3.133388770464049) q[3];
rz(0.5081137821000414) q[3];
ry(2.956749995634457) q[4];
rz(2.28286461949777) q[4];
ry(-0.3225889319383804) q[5];
rz(0.3076782730687711) q[5];
ry(-3.1272314256338216) q[6];
rz(-2.1994277763162478) q[6];
ry(3.1270815797136784) q[7];
rz(-0.3605589527066386) q[7];
ry(3.1301926295024747) q[8];
rz(-2.6095646291225614) q[8];
ry(-3.14107992560542) q[9];
rz(2.944167604869548) q[9];
ry(-2.910513873553652) q[10];
rz(-1.1876042821450767) q[10];
ry(2.0503683095404286) q[11];
rz(1.0054017896530143) q[11];
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
ry(-1.8625082244218198) q[0];
rz(-1.3412216247571962) q[0];
ry(-0.02885943618850373) q[1];
rz(2.8237742102589105) q[1];
ry(-0.7576185098887702) q[2];
rz(1.6016291702880685) q[2];
ry(2.364355977944356) q[3];
rz(1.5481444436661158) q[3];
ry(0.7784715637349015) q[4];
rz(2.769988226290123) q[4];
ry(1.0042030248520581) q[5];
rz(0.012463067099261595) q[5];
ry(-0.16582195702860325) q[6];
rz(1.9188307258161217) q[6];
ry(3.0131118884407386) q[7];
rz(1.3917513125601602) q[7];
ry(2.6667119899074527) q[8];
rz(-2.0870380409878018) q[8];
ry(2.650266273205548) q[9];
rz(-0.9774118143537275) q[9];
ry(1.4299316717670922) q[10];
rz(-1.1767983800634791) q[10];
ry(-2.9788109094864366) q[11];
rz(-0.6671340885968196) q[11];
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
ry(0.32967344676997384) q[0];
rz(-2.9217925266415454) q[0];
ry(1.4090233479458503) q[1];
rz(-2.516005071063161) q[1];
ry(-3.1318648317186764) q[2];
rz(-3.0037153762899163) q[2];
ry(3.1394276088217534) q[3];
rz(0.08871069646490781) q[3];
ry(0.7075951544640116) q[4];
rz(1.592599295234229) q[4];
ry(2.532555316929414) q[5];
rz(1.3489749583894106) q[5];
ry(-0.028656134294640243) q[6];
rz(1.685337540106765) q[6];
ry(-0.03007570387129555) q[7];
rz(-1.3329374487597851) q[7];
ry(-2.943240664020652) q[8];
rz(1.296804276317485) q[8];
ry(-2.9795783386516543) q[9];
rz(2.941008790858717) q[9];
ry(-2.0863568188703585) q[10];
rz(3.0119736268449904) q[10];
ry(0.7991298793846084) q[11];
rz(-0.9762897260651684) q[11];
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
ry(-3.0910869236019463) q[0];
rz(-2.324919125040816) q[0];
ry(-0.02857042277495392) q[1];
rz(1.087069424367721) q[1];
ry(2.2798726433984386) q[2];
rz(0.580662170513354) q[2];
ry(0.8706146189588395) q[3];
rz(-0.5130937079487916) q[3];
ry(2.4971927824126023) q[4];
rz(-0.5710408650715503) q[4];
ry(2.0823676699485354) q[5];
rz(-2.9898912325038056) q[5];
ry(-3.1139693985290418) q[6];
rz(1.8655979391336517) q[6];
ry(-0.014545959766421054) q[7];
rz(1.5686186353375255) q[7];
ry(3.1043144803871794) q[8];
rz(-1.4903263657650552) q[8];
ry(3.113192050254886) q[9];
rz(-0.6209306831338807) q[9];
ry(2.4155670991572777) q[10];
rz(-2.938998631846823) q[10];
ry(2.885027781660072) q[11];
rz(0.5086400025060827) q[11];
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
ry(2.6609581029469425) q[0];
rz(2.6954755109634307) q[0];
ry(0.33963804975357625) q[1];
rz(-0.36898510245259614) q[1];
ry(-3.095452718799604) q[2];
rz(2.7833767630714417) q[2];
ry(-3.085171845683793) q[3];
rz(1.0180669234566955) q[3];
ry(-1.9992444487068364) q[4];
rz(-2.0199709099213043) q[4];
ry(0.09131088961144497) q[5];
rz(2.8303336163956754) q[5];
ry(-0.12227798485797159) q[6];
rz(-0.5759316559470147) q[6];
ry(-3.016167443451178) q[7];
rz(2.905722174528222) q[7];
ry(-1.2681270563949765) q[8];
rz(-1.9047952718997987) q[8];
ry(1.834901967131988) q[9];
rz(1.4603288188759391) q[9];
ry(-1.7868915164512689) q[10];
rz(-1.0999657181399523) q[10];
ry(3.07680955982127) q[11];
rz(-2.2440789571937154) q[11];
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
ry(0.9450167413063484) q[0];
rz(-2.6877670480531246) q[0];
ry(1.2885787455935045) q[1];
rz(-0.5109407260751633) q[1];
ry(-3.104309804759025) q[2];
rz(-2.195496767625367) q[2];
ry(0.021195693660215887) q[3];
rz(0.16453563006156696) q[3];
ry(0.7106714172696025) q[4];
rz(-1.8637313953181085) q[4];
ry(-2.9950556092313123) q[5];
rz(-0.040540120362047634) q[5];
ry(0.10455075820109892) q[6];
rz(-2.3786528634433886) q[6];
ry(3.0632174695941234) q[7];
rz(-1.214087166153491) q[7];
ry(-2.4368838174868253) q[8];
rz(-1.7964070810802866) q[8];
ry(-1.9301198768705627) q[9];
rz(1.712794148889941) q[9];
ry(-0.1582766626358521) q[10];
rz(1.0530102443317735) q[10];
ry(-0.07599939673414408) q[11];
rz(-3.083023984510088) q[11];
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
ry(-0.31857599384396806) q[0];
rz(-0.35124673363534836) q[0];
ry(1.6282639003358375) q[1];
rz(-1.4164099155271497) q[1];
ry(-3.1193275956571456) q[2];
rz(-0.09386450427178668) q[2];
ry(3.1210336086233528) q[3];
rz(-3.0192828315356106) q[3];
ry(-2.960525279205582) q[4];
rz(-2.3072988642890104) q[4];
ry(0.30407980912305393) q[5];
rz(-2.7447262076696703) q[5];
ry(-0.0022754873897685712) q[6];
rz(-0.693000472454135) q[6];
ry(-3.1393171549923014) q[7];
rz(-0.5983252283138601) q[7];
ry(1.555678216418941) q[8];
rz(2.3729214101047975) q[8];
ry(-1.5878291520314962) q[9];
rz(-2.2920763521724483) q[9];
ry(-3.0500382241825625) q[10];
rz(-2.9930651081755513) q[10];
ry(0.14894183289071133) q[11];
rz(-0.584858940917468) q[11];
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
ry(1.5039609061344315) q[0];
rz(-1.795203802063539) q[0];
ry(0.21694901709575862) q[1];
rz(-3.092608562056573) q[1];
ry(3.1013377341527413) q[2];
rz(1.9534414599063732) q[2];
ry(0.03980557259613704) q[3];
rz(-0.37291714340842663) q[3];
ry(1.64522612770795) q[4];
rz(1.8229677383939062) q[4];
ry(1.499022469572535) q[5];
rz(-2.625902379092303) q[5];
ry(-0.015349071076540285) q[6];
rz(2.3924413940902265) q[6];
ry(3.0534788363363194) q[7];
rz(1.460330743513374) q[7];
ry(2.565321552074968) q[8];
rz(2.548029489408898) q[8];
ry(-2.894432312163183) q[9];
rz(-2.692703515836191) q[9];
ry(-1.1132870335852458) q[10];
rz(-0.09697496021751295) q[10];
ry(-1.6969045228546742) q[11];
rz(-1.6266926178517391) q[11];
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
ry(0.3040338714242754) q[0];
rz(-0.7453432069337174) q[0];
ry(2.7705648228872186) q[1];
rz(1.5224265977324967) q[1];
ry(0.5174293139796449) q[2];
rz(2.790700929985845) q[2];
ry(1.817103902774858) q[3];
rz(0.6492743401822556) q[3];
ry(-3.07584205657132) q[4];
rz(-2.7272537364967597) q[4];
ry(-0.03322147582172961) q[5];
rz(-0.6784724608863391) q[5];
ry(0.031206388775174232) q[6];
rz(-0.7673824437726102) q[6];
ry(0.08104040611145713) q[7];
rz(-0.040629670967953946) q[7];
ry(-3.100515940003726) q[8];
rz(-1.328956708262942) q[8];
ry(-3.0137879698760512) q[9];
rz(2.1955395157851703) q[9];
ry(2.7509451250541677) q[10];
rz(2.821174834624048) q[10];
ry(2.735490488910952) q[11];
rz(-3.0395489105879108) q[11];
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
ry(-0.508635498091461) q[0];
rz(1.7085056201659368) q[0];
ry(-3.0376041444143174) q[1];
rz(0.7232971864587129) q[1];
ry(1.5062900806073385) q[2];
rz(1.160225221292377) q[2];
ry(-0.09825112573269885) q[3];
rz(0.7503771933758485) q[3];
ry(-0.010008774006891485) q[4];
rz(2.4530303561680133) q[4];
ry(3.1343505766477096) q[5];
rz(1.5773142533571667) q[5];
ry(-2.65417298157629) q[6];
rz(2.7709289092697253) q[6];
ry(2.6118257808921017) q[7];
rz(2.8174074119926016) q[7];
ry(-0.35101996785303324) q[8];
rz(1.7793195368640191) q[8];
ry(0.5199389173911868) q[9];
rz(0.30230743643628516) q[9];
ry(2.406935587534633) q[10];
rz(-1.9371179349315888) q[10];
ry(-2.452815316757326) q[11];
rz(-0.36158991897138554) q[11];
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
ry(0.0965063163034689) q[0];
rz(0.5836653455077219) q[0];
ry(-3.0476377822839775) q[1];
rz(-1.8695332829078666) q[1];
ry(0.01707429296300328) q[2];
rz(0.18487585083232447) q[2];
ry(0.09996799266391071) q[3];
rz(0.8057717672916614) q[3];
ry(0.06737360225624656) q[4];
rz(1.8303390837515265) q[4];
ry(0.14169265523250224) q[5];
rz(0.49089917182370085) q[5];
ry(3.038076582614622) q[6];
rz(-0.40555188161531336) q[6];
ry(0.17594102335119555) q[7];
rz(0.10779847143739829) q[7];
ry(-1.5710822198363494) q[8];
rz(-1.2691587445175074) q[8];
ry(1.6295338364358536) q[9];
rz(-0.21621394463426125) q[9];
ry(0.03228148616526649) q[10];
rz(0.3373885552884708) q[10];
ry(3.121965823420277) q[11];
rz(1.0740750669418742) q[11];
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
ry(-0.6993096210502063) q[0];
rz(-2.871104690724582) q[0];
ry(-0.8707128532988149) q[1];
rz(-2.3144254416221486) q[1];
ry(0.1608088348238681) q[2];
rz(1.3679144241501016) q[2];
ry(-0.016997671626118205) q[3];
rz(-2.951378954628652) q[3];
ry(0.00917903118899588) q[4];
rz(-0.8558154568275838) q[4];
ry(-3.131196499830021) q[5];
rz(2.851089664281154) q[5];
ry(2.9413671688136973) q[6];
rz(-2.43003549698082) q[6];
ry(-0.2414170699895113) q[7];
rz(1.8721046064625062) q[7];
ry(-1.0359754501071627) q[8];
rz(-0.59212025689615) q[8];
ry(-1.7427189042902507) q[9];
rz(1.3145969397665613) q[9];
ry(2.5371230163411806) q[10];
rz(-1.5254079671390457) q[10];
ry(1.6591042424704916) q[11];
rz(-0.8840592299233014) q[11];
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
ry(3.105455735345141) q[0];
rz(-2.250714848455363) q[0];
ry(-3.0976476614725605) q[1];
rz(-1.152761773631329) q[1];
ry(3.1415780955505697) q[2];
rz(1.9050893728952136) q[2];
ry(-0.006713079903574206) q[3];
rz(-1.2663133249198133) q[3];
ry(-0.8548318323644089) q[4];
rz(-1.0523644485577048) q[4];
ry(0.6466953406839977) q[5];
rz(1.3174563661219132) q[5];
ry(0.02162815964669118) q[6];
rz(-1.152406959689685) q[6];
ry(0.03504279749748779) q[7];
rz(0.914832839399276) q[7];
ry(0.014990532570769526) q[8];
rz(-2.6092808203177458) q[8];
ry(-3.1137305533524575) q[9];
rz(0.008927483045104757) q[9];
ry(-3.0953624521992995) q[10];
rz(-1.72051742116963) q[10];
ry(0.00226107374390061) q[11];
rz(0.6003686375226472) q[11];
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
ry(-0.8423812644901227) q[0];
rz(2.318311291713471) q[0];
ry(2.1218597065167613) q[1];
rz(0.8611775064747517) q[1];
ry(-0.04031740388249553) q[2];
rz(2.5350736458064813) q[2];
ry(-1.497601571629942) q[3];
rz(-1.5040069532773854) q[3];
ry(0.0007462604708345921) q[4];
rz(0.037417069921872874) q[4];
ry(0.0017356956169071853) q[5];
rz(0.8263789438546193) q[5];
ry(2.931723112005485) q[6];
rz(0.15489589739967613) q[6];
ry(2.937812005465717) q[7];
rz(-3.0204935248375016) q[7];
ry(1.5656768641117758) q[8];
rz(-3.0410160292910784) q[8];
ry(-1.3784217967205672) q[9];
rz(-2.010070040870322) q[9];
ry(1.022278383353499) q[10];
rz(-2.5823248036630035) q[10];
ry(3.1123936684626843) q[11];
rz(-0.8190235913365838) q[11];
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
ry(-0.7444481638609642) q[0];
rz(-1.0703072320496512) q[0];
ry(0.7448247721878944) q[1];
rz(2.0772917335926637) q[1];
ry(-0.6316514637285087) q[2];
rz(-1.1182952019717423) q[2];
ry(2.213035773409349) q[3];
rz(2.1333750695282925) q[3];
ry(1.12932103331199) q[4];
rz(-2.4992769815598965) q[4];
ry(1.1863696928185208) q[5];
rz(-2.434525470842215) q[5];
ry(0.5976302254522396) q[6];
rz(-1.8032544712748242) q[6];
ry(-0.6389504509513892) q[7];
rz(1.371993067221718) q[7];
ry(-0.958061243687141) q[8];
rz(-1.3713227168251458) q[8];
ry(0.8702284587408925) q[9];
rz(1.823511027512413) q[9];
ry(2.722683194654328) q[10];
rz(1.822019531826743) q[10];
ry(-0.4632067225887644) q[11];
rz(1.9321155301097421) q[11];