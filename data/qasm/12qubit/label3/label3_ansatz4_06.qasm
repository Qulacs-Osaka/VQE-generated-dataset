OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(0.00012245969501112342) q[0];
rz(0.8029259439169892) q[0];
ry(1.6364399502887086) q[1];
rz(2.0615371533520257) q[1];
ry(0.0007986380498492489) q[2];
rz(-1.5736389227382714) q[2];
ry(-1.5011523973586278) q[3];
rz(1.6591466111325863) q[3];
ry(3.1415883354272256) q[4];
rz(0.06971744497509662) q[4];
ry(7.051264769794944e-05) q[5];
rz(1.8667426763442334) q[5];
ry(1.5707946689160712) q[6];
rz(0.6539570375787301) q[6];
ry(1.5707970635720108) q[7];
rz(-1.5705005002453527) q[7];
ry(1.5726941509521057) q[8];
rz(-2.654300992524242) q[8];
ry(3.141581150043014) q[9];
rz(-0.3063530349715719) q[9];
ry(1.5719777195744586) q[10];
rz(-3.0937528060321995) q[10];
ry(-3.141502672230505) q[11];
rz(-2.62739265841336) q[11];
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
ry(-3.1415871348357864) q[0];
rz(-1.4953503410015145) q[0];
ry(3.045886094199367) q[1];
rz(0.40690759986933944) q[1];
ry(-1.5707950891535338) q[2];
rz(1.5028211119464316) q[2];
ry(-1.6524418961491376) q[3];
rz(-0.6817469353017515) q[3];
ry(0.39968382885342946) q[4];
rz(-1.5707826027943161) q[4];
ry(-5.7262987573025725e-05) q[5];
rz(-0.6209994893865787) q[5];
ry(-0.15777280385884396) q[6];
rz(3.0163528742189047) q[6];
ry(1.5708301824875448) q[7];
rz(3.015986515351342) q[7];
ry(-1.5707978082903917) q[8];
rz(-1.5707923386036837) q[8];
ry(1.5707975376015755) q[9];
rz(1.5707953502402379) q[9];
ry(-1.6609807696635661) q[10];
rz(-1.0836565316206652) q[10];
ry(0.0002610428445031232) q[11];
rz(-1.9569224198234232) q[11];
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
ry(-1.5707997878780073) q[0];
rz(-1.5793346313053345) q[0];
ry(1.5706873340521694) q[1];
rz(-1.6973712932206073) q[1];
ry(1.5260342673834162) q[2];
rz(0.003355042589275787) q[2];
ry(-0.0007790111623235061) q[3];
rz(2.256533517068334) q[3];
ry(-1.5708398421836003) q[4];
rz(2.5087637582534854) q[4];
ry(1.570789282885821) q[5];
rz(2.9971029789649233) q[5];
ry(0.0005367945585366131) q[6];
rz(-2.093301411789697) q[6];
ry(-1.5726405289301764) q[7];
rz(-1.5716614249144045) q[7];
ry(-1.5707807290576872) q[8];
rz(-1.5589208968759696) q[8];
ry(-1.5708087183540826) q[9];
rz(1.0769581066938372e-06) q[9];
ry(-0.036172344519706634) q[10];
rz(-1.569465017324844) q[10];
ry(-1.5702316417153892) q[11];
rz(1.682827031217827) q[11];
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
ry(-0.0471546977074393) q[0];
rz(-2.864213840830841) q[0];
ry(-3.1415429438460425) q[1];
rz(1.4288821327911903) q[1];
ry(-0.6959052971052042) q[2];
rz(-0.0032694181384638195) q[2];
ry(-2.167457440100907) q[3];
rz(-3.0736444540022863) q[3];
ry(3.141582980048443) q[4];
rz(-2.203577534243816) q[4];
ry(1.0067231636767815e-05) q[5];
rz(-2.9970497432842524) q[5];
ry(1.5708431026496035) q[6];
rz(1.5719626898330752) q[6];
ry(3.1080282399750776) q[7];
rz(1.5699525042021891) q[7];
ry(-0.005977947183334396) q[8];
rz(-1.5826704560026632) q[8];
ry(2.8272632167915765) q[9];
rz(3.1415877060804562) q[9];
ry(1.5708465719040097) q[10];
rz(0.7555678761210771) q[10];
ry(0.0048729918731602595) q[11];
rz(3.0849769339979987) q[11];
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
ry(3.141572689236654) q[0];
rz(1.8396074378736833) q[0];
ry(-3.141551075237162) q[1];
rz(-1.8916428531876437) q[1];
ry(1.4943668635375646) q[2];
rz(3.0803766755990893) q[2];
ry(1.5677269976307358) q[3];
rz(-1.6530646715398323) q[3];
ry(1.5708731597604142) q[4];
rz(1.3451840547154275) q[4];
ry(1.5710498069849617) q[5];
rz(-1.5585959626581314) q[5];
ry(0.24152753001994043) q[6];
rz(3.1404644610336856) q[6];
ry(1.570902017419968) q[7];
rz(-1.5707953317273935) q[7];
ry(1.5648218638282652) q[8];
rz(-1.5707927400123074) q[8];
ry(1.5707886201295302) q[9];
rz(1.5706533469387747) q[9];
ry(-1.083835784195315e-05) q[10];
rz(-0.7555813177336727) q[10];
ry(8.238980326635514e-06) q[11];
rz(3.086327832897559) q[11];
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
ry(-1.5707645324211352) q[0];
rz(1.7929381550894314) q[0];
ry(0.0003960370770981214) q[1];
rz(-3.0039192763710845) q[1];
ry(-3.1415672335624407) q[2];
rz(3.080714103100722) q[2];
ry(3.141591955490176) q[3];
rz(-0.014516740038561517) q[3];
ry(3.1415681223815937) q[4];
rz(2.909633114707644) q[4];
ry(3.141533917747824) q[5];
rz(-1.5586529575817507) q[5];
ry(1.3047590434688456) q[6];
rz(-3.1415657722215906) q[6];
ry(-1.5707811016872701) q[7];
rz(0.0009064084736611625) q[7];
ry(-1.57079656007717) q[8];
rz(-3.1328542220545397) q[8];
ry(-1.5705586211146265) q[9];
rz(0.12892156171175892) q[9];
ry(1.570875630670093) q[10];
rz(1.6676470536766481) q[10];
ry(1.4578934821232685) q[11];
rz(-7.715246464989889e-05) q[11];
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
ry(-3.527105208298999e-06) q[0];
rz(1.797929941288042) q[0];
ry(3.141521632932154) q[1];
rz(1.4707906223164535) q[1];
ry(1.570793822865448) q[2];
rz(0.11363325903581423) q[2];
ry(3.141578637200275) q[3];
rz(0.06805993086698069) q[3];
ry(-3.062675324203421) q[4];
rz(0.935655797013113) q[4];
ry(1.6555177555605258) q[5];
rz(-0.047195799412366554) q[5];
ry(-1.5707986780619314) q[6];
rz(0.769213971183934) q[6];
ry(1.5708082334823734) q[7];
rz(-0.2062457291940261) q[7];
ry(1.5695867819729816) q[8];
rz(0.07600988891796663) q[8];
ry(-3.1414472855429185) q[9];
rz(-1.5212821267817773) q[9];
ry(-0.08693654535296365) q[10];
rz(-0.449017836174761) q[10];
ry(1.570684142349549) q[11];
rz(-1.5704943293212041) q[11];
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
ry(0.020213866304549025) q[0];
rz(-1.7766276384450208) q[0];
ry(-1.5788285572943002) q[1];
rz(-1.016758624388089) q[1];
ry(3.1401750368375545) q[2];
rz(-1.4571407064930915) q[2];
ry(-1.5707953072120953) q[3];
rz(7.631960408538419e-06) q[3];
ry(1.9879153375440766e-05) q[4];
rz(1.2035668127490249) q[4];
ry(0.010815772731210593) q[5];
rz(1.0390479060092421) q[5];
ry(-1.2877649041550853e-05) q[6];
rz(2.134294497351252) q[6];
ry(7.164530968495342e-06) q[7];
rz(-0.669483553654703) q[7];
ry(1.5702740273385514) q[8];
rz(0.0055816387870287355) q[8];
ry(1.5707989238771107) q[9];
rz(-3.141591834260684) q[9];
ry(-3.17951931905227e-05) q[10];
rz(-2.7877011434659056) q[10];
ry(1.5708142080551335) q[11];
rz(-0.0015031493794964424) q[11];
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
ry(2.3311583208972693e-05) q[0];
rz(2.934391653816875) q[0];
ry(-3.141591916189695) q[1];
rz(-2.5909863020330772) q[1];
ry(-1.570806396161459) q[2];
rz(1.567728876610091) q[2];
ry(1.570793654575608) q[3];
rz(1.614829791836243) q[3];
ry(1.2895067014184747e-06) q[4];
rz(0.13098639098824627) q[4];
ry(3.141570030178126) q[5];
rz(2.5603073244595502) q[5];
ry(3.1415845801707527) q[6];
rz(-1.81133273392319) q[6];
ry(-6.442908545021551e-06) q[7];
rz(-1.431664112602327) q[7];
ry(1.570776597597419) q[8];
rz(1.5735676162098549) q[8];
ry(1.5708702269217918) q[9];
rz(-0.7596931933880534) q[9];
ry(-1.5706525304640753) q[10];
rz(3.138823092415678) q[10];
ry(-1.5707945411605437) q[11];
rz(-0.5466469233534027) q[11];
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
ry(-1.573839884225266) q[0];
rz(-2.604257385062087) q[0];
ry(-1.573044067463446) q[1];
rz(-2.6042281473945597) q[1];
ry(1.6082728208914956) q[2];
rz(-1.0335773610686305) q[2];
ry(3.1407743336568155) q[3];
rz(-2.560336120119327) q[3];
ry(0.003757087041337566) q[4];
rz(1.4028523204370948) q[4];
ry(1.5676114718731031) q[5];
rz(-1.0337705722881625) q[5];
ry(-1.5679796543097448) q[6];
rz(2.1086004455578746) q[6];
ry(-3.1381399056519337) q[7];
rz(1.3717788930235688) q[7];
ry(1.5675200344962388) q[8];
rz(-2.603862068614508) q[8];
ry(3.138514551877048) q[9];
rz(-1.7933919426971885) q[9];
ry(-1.574130973142947) q[10];
rz(-2.603848059910071) q[10];
ry(0.003747384936288789) q[11];
rz(1.0837477648322498) q[11];