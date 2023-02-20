OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(2.8468926122790905) q[0];
rz(-3.0551498663412664) q[0];
ry(-1.8185107595307866) q[1];
rz(1.0368646587864994) q[1];
ry(0.0020704348369913425) q[2];
rz(1.7454778706975924) q[2];
ry(3.1409075664357995) q[3];
rz(-0.30260609305663927) q[3];
ry(1.5678168829846184) q[4];
rz(2.888723188393741) q[4];
ry(1.5706583214451901) q[5];
rz(2.8012868439149843) q[5];
ry(3.1410759390224996) q[6];
rz(3.0577657157668803) q[6];
ry(3.1412773408495953) q[7];
rz(-1.2948823620072605) q[7];
ry(-0.6159819616661003) q[8];
rz(2.6023222680064833) q[8];
ry(-1.236567181396957) q[9];
rz(0.5525432784590693) q[9];
ry(3.000277441376024) q[10];
rz(-2.4150650406910334) q[10];
ry(2.6747244486881288) q[11];
rz(1.378185570148938) q[11];
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
ry(0.6417296544716454) q[0];
rz(-1.8986107988142686) q[0];
ry(1.7038389827171316) q[1];
rz(-2.753004380354849) q[1];
ry(-0.27443478959164125) q[2];
rz(-1.7732005618052917) q[2];
ry(-3.0866450260001557) q[3];
rz(-0.3199682407382545) q[3];
ry(-0.9533300043696287) q[4];
rz(2.9713895499720375) q[4];
ry(2.482534170781993) q[5];
rz(2.5521691051373647) q[5];
ry(-2.758821212821917) q[6];
rz(-2.391235748301056) q[6];
ry(0.5559831885295212) q[7];
rz(0.22007720636174713) q[7];
ry(-0.5397681917270098) q[8];
rz(-1.116766870415205) q[8];
ry(1.4225582672290573) q[9];
rz(0.9151634241408705) q[9];
ry(2.1064235657765176) q[10];
rz(2.2494220183928673) q[10];
ry(2.1719682094623662) q[11];
rz(0.22231986107695792) q[11];
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
ry(1.9060298039707086) q[0];
rz(-0.7167379115032819) q[0];
ry(-1.9951455254794286) q[1];
rz(3.119887596097633) q[1];
ry(-1.9700079342689047) q[2];
rz(0.2728975119332535) q[2];
ry(-1.6479839944905152) q[3];
rz(2.6136503324303857) q[3];
ry(1.6614933947283221) q[4];
rz(-0.2240975347968952) q[4];
ry(-1.66310232607493) q[5];
rz(-0.23928219916244764) q[5];
ry(0.2187476701870095) q[6];
rz(-2.864533012857778) q[6];
ry(0.6083418118730908) q[7];
rz(3.0381431144680344) q[7];
ry(-2.0191262096272453) q[8];
rz(2.55378924649315) q[8];
ry(-0.9043014069407799) q[9];
rz(1.7619012311186912) q[9];
ry(2.365668398761247) q[10];
rz(-3.118312816475897) q[10];
ry(-1.1507279721057966) q[11];
rz(0.7111355098369022) q[11];
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
ry(1.3542223063134946) q[0];
rz(0.5740208023042603) q[0];
ry(0.6328075825745714) q[1];
rz(-0.6727112736267944) q[1];
ry(-0.6231858838002236) q[2];
rz(-2.1712316434958385) q[2];
ry(-0.604861084182849) q[3];
rz(-0.2088594603504407) q[3];
ry(3.0307346653439873) q[4];
rz(0.31799457822638294) q[4];
ry(0.11056619987670581) q[5];
rz(-0.297805376627367) q[5];
ry(0.37809447217990355) q[6];
rz(-0.5732634958609109) q[6];
ry(1.1547921512188601) q[7];
rz(-1.5184210245889993) q[7];
ry(0.17594241675222808) q[8];
rz(1.8783811856357893) q[8];
ry(-0.4848475389055276) q[9];
rz(-1.077219062124665) q[9];
ry(-1.3819623529411267) q[10];
rz(2.6044700852259393) q[10];
ry(2.6513754539305987) q[11];
rz(0.812633714965898) q[11];
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
ry(-1.513057668948332) q[0];
rz(0.22309131555452708) q[0];
ry(-1.7623634490750264) q[1];
rz(-1.1692161563654908) q[1];
ry(-1.113430170577277) q[2];
rz(1.9462185865463528) q[2];
ry(-2.616649869920694) q[3];
rz(-0.6510084329389556) q[3];
ry(-0.25605994283208666) q[4];
rz(-0.322592223121375) q[4];
ry(2.886011584173613) q[5];
rz(2.825471880072339) q[5];
ry(2.469070808412556) q[6];
rz(1.6394206137021594) q[6];
ry(-0.7185313690369903) q[7];
rz(-0.9493347730862861) q[7];
ry(2.0188148873813816) q[8];
rz(-2.683241839146646) q[8];
ry(-0.24423738380772786) q[9];
rz(-3.0155230972133564) q[9];
ry(-1.5386583666181926) q[10];
rz(1.3759612789789264) q[10];
ry(0.7929827477172483) q[11];
rz(-0.6235488236583233) q[11];
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
ry(1.6088076200859245) q[0];
rz(-0.7207445592501641) q[0];
ry(0.7482515055322588) q[1];
rz(0.7096684966232556) q[1];
ry(0.6601139172918842) q[2];
rz(-2.5554680393837494) q[2];
ry(1.8836961861272847) q[3];
rz(-1.7403744388286275) q[3];
ry(-0.887171985597069) q[4];
rz(-0.12353620869907263) q[4];
ry(-2.252013457581034) q[5];
rz(0.12376623762870632) q[5];
ry(2.2937348821522603) q[6];
rz(1.1160572266466833) q[6];
ry(2.1933000149483544) q[7];
rz(-1.92549129611583) q[7];
ry(-1.9245952040461818) q[8];
rz(-2.887701036750598) q[8];
ry(3.120217987238048) q[9];
rz(-2.609160886299081) q[9];
ry(-1.0485073960371223) q[10];
rz(2.5765474671615807) q[10];
ry(-1.2571301906333714) q[11];
rz(-1.9300768083934718) q[11];
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
ry(3.1202519418535855) q[0];
rz(-1.766941709622251) q[0];
ry(0.15346302775798648) q[1];
rz(2.237175423209841) q[1];
ry(-1.9308824502845754) q[2];
rz(-1.7263425204724485) q[2];
ry(-0.658319023122328) q[3];
rz(-1.809243927866719) q[3];
ry(2.1188635371421363) q[4];
rz(-3.0930662246044602) q[4];
ry(2.1166872501252305) q[5];
rz(-1.536097134123458) q[5];
ry(0.3976384608222502) q[6];
rz(-0.40924787624383263) q[6];
ry(1.5157766875446237) q[7];
rz(-2.487952041283749) q[7];
ry(-0.6046757098188715) q[8];
rz(0.9632320001972314) q[8];
ry(1.6006491125118805) q[9];
rz(-3.021503016952151) q[9];
ry(-0.6372799654127572) q[10];
rz(1.9972172550709484) q[10];
ry(-2.514222010045922) q[11];
rz(0.8575141640427892) q[11];
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
ry(0.9935127909810735) q[0];
rz(0.638286104518018) q[0];
ry(-2.844697549498292) q[1];
rz(1.938901407311476) q[1];
ry(-1.219827074284531) q[2];
rz(-2.3879423140971445) q[2];
ry(0.5415997136121389) q[3];
rz(1.6841693891709586) q[3];
ry(-3.1359267808277913) q[4];
rz(1.7077357013037355) q[4];
ry(0.003435740529457495) q[5];
rz(-3.0817230672743126) q[5];
ry(0.0790223708798643) q[6];
rz(-2.139333586440295) q[6];
ry(-2.017041704975889) q[7];
rz(-2.35170002338586) q[7];
ry(-1.0363616453177489) q[8];
rz(-2.626712547321387) q[8];
ry(-1.2275065846202375) q[9];
rz(0.002144957840702233) q[9];
ry(-3.131535421534234) q[10];
rz(-3.02996963099798) q[10];
ry(1.5190339753361686) q[11];
rz(3.0923263124060147) q[11];
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
ry(-0.41237665119314926) q[0];
rz(0.8726335274343543) q[0];
ry(-1.1550612796247552) q[1];
rz(-2.9158072706504883) q[1];
ry(2.80167282714998) q[2];
rz(-1.9650682360692135) q[2];
ry(2.182051330223215) q[3];
rz(1.778975860039111) q[3];
ry(-2.5230754396417696) q[4];
rz(0.8862074205427705) q[4];
ry(0.6177252588304732) q[5];
rz(-0.8535819189822433) q[5];
ry(0.7470587817575867) q[6];
rz(-2.8183228010282204) q[6];
ry(-2.590798272094543) q[7];
rz(0.4024656707963929) q[7];
ry(1.5212818948660407) q[8];
rz(2.7018160460448026) q[8];
ry(-1.1161616256175177) q[9];
rz(-2.4680220523219036) q[9];
ry(-1.2167851001710748) q[10];
rz(2.539280908846929) q[10];
ry(-1.0417450492283749) q[11];
rz(1.9246215839698029) q[11];
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
ry(0.7128440667772749) q[0];
rz(0.6032374659078569) q[0];
ry(1.7535314071081691) q[1];
rz(-2.884724138798246) q[1];
ry(-0.39982099643748903) q[2];
rz(-1.8760822963709112) q[2];
ry(2.940443945561203) q[3];
rz(1.6358346821322316) q[3];
ry(-1.6631929174389803) q[4];
rz(1.3949761481760596) q[4];
ry(1.4883555941741835) q[5];
rz(1.5619971866964302) q[5];
ry(-2.190715314401474) q[6];
rz(0.9916860446665171) q[6];
ry(-2.3698938470904483) q[7];
rz(1.6259010171334216) q[7];
ry(0.3173514355627916) q[8];
rz(2.510111209503945) q[8];
ry(-1.3730197495376382) q[9];
rz(2.43900749885857) q[9];
ry(1.4667491016289418) q[10];
rz(-1.138483899660824) q[10];
ry(-1.939891902232047) q[11];
rz(-3.0353811612347976) q[11];
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
ry(-1.5172426855039296) q[0];
rz(-0.37550402877887956) q[0];
ry(-2.294366784524655) q[1];
rz(0.568106808210378) q[1];
ry(3.0889560786815546) q[2];
rz(1.4433349760603162) q[2];
ry(-1.6279271703931135) q[3];
rz(2.135097680799442) q[3];
ry(-3.1105003853237485) q[4];
rz(2.3744704171913944) q[4];
ry(3.1102708095197227) q[5];
rz(-3.012124228517717) q[5];
ry(-1.545805477543004) q[6];
rz(-1.5379186064909547) q[6];
ry(1.5827898206774282) q[7];
rz(-1.215888827239831) q[7];
ry(-1.259898684656541) q[8];
rz(-2.4105162691808584) q[8];
ry(-0.1870573773222687) q[9];
rz(1.2094072423107625) q[9];
ry(-2.754159131939626) q[10];
rz(1.392552435676948) q[10];
ry(2.1181640165923397) q[11];
rz(-2.438277389790583) q[11];
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
ry(-1.3237644896943608) q[0];
rz(1.0311717066823913) q[0];
ry(0.05220848775463782) q[1];
rz(-0.8389025806949858) q[1];
ry(1.9269215420217956) q[2];
rz(2.9432903030495265) q[2];
ry(-0.08860987023503863) q[3];
rz(-2.132202568729807) q[3];
ry(-1.6659159091637619) q[4];
rz(2.132711160240401) q[4];
ry(-1.6010049915475877) q[5];
rz(-0.12482473750723477) q[5];
ry(-2.2741449699926646) q[6];
rz(-3.1135889055746375) q[6];
ry(3.0888703654874203) q[7];
rz(-2.7929572874516944) q[7];
ry(1.4855511020762675) q[8];
rz(-0.6923380756933295) q[8];
ry(1.962726007305213) q[9];
rz(2.300165206619588) q[9];
ry(-2.247890016532674) q[10];
rz(-2.624275827475456) q[10];
ry(2.575661334671784) q[11];
rz(1.8571620090569692) q[11];
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
ry(0.27719038484732006) q[0];
rz(-3.0118912788153547) q[0];
ry(2.3988615051832842) q[1];
rz(0.7015483490637389) q[1];
ry(-1.5299048459403002) q[2];
rz(2.431410605288671) q[2];
ry(-0.1402885514282417) q[3];
rz(2.8671963347791953) q[3];
ry(3.1376675042698206) q[4];
rz(-2.1923346660576666) q[4];
ry(-0.09890829545597234) q[5];
rz(-0.8634430362023534) q[5];
ry(-1.5761558451993734) q[6];
rz(3.124546178418815) q[6];
ry(-1.570631563884283) q[7];
rz(3.133566020215317) q[7];
ry(1.7219841146922823) q[8];
rz(-2.3514043583973367) q[8];
ry(1.5156965173591206) q[9];
rz(-2.0656520349032323) q[9];
ry(2.053392098677837) q[10];
rz(0.2949476522987691) q[10];
ry(-1.389152618189578) q[11];
rz(0.6610199157958173) q[11];
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
ry(-2.8533341100470846) q[0];
rz(-0.04463194009237075) q[0];
ry(1.5841632651499342) q[1];
rz(0.4701221912611801) q[1];
ry(0.09826865121244843) q[2];
rz(0.5945937773356383) q[2];
ry(-1.4777108761651778) q[3];
rz(1.1127888611167984) q[3];
ry(3.139971007238491) q[4];
rz(2.091162014729499) q[4];
ry(3.1373544774527646) q[5];
rz(0.6190060706235272) q[5];
ry(1.5768891688846622) q[6];
rz(-1.4855045951073693) q[6];
ry(-1.5803570947465406) q[7];
rz(0.22872859183665972) q[7];
ry(2.3764685913461583) q[8];
rz(0.47562097118733915) q[8];
ry(-2.9553840557589592) q[9];
rz(-1.903145984536927) q[9];
ry(-0.2991298524865595) q[10];
rz(2.5923764503121767) q[10];
ry(2.0907105824799936) q[11];
rz(-0.293295276275027) q[11];
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
ry(1.4781184637948206) q[0];
rz(0.09896945211405139) q[0];
ry(3.016902604563017) q[1];
rz(-1.096088259411384) q[1];
ry(1.5862054289779781) q[2];
rz(0.10282191736944277) q[2];
ry(0.23249774869994155) q[3];
rz(2.159805843962727) q[3];
ry(3.055201055954589) q[4];
rz(-3.063556774229646) q[4];
ry(0.0009564493316771562) q[5];
rz(1.5734590193242206) q[5];
ry(-0.037845172897174706) q[6];
rz(2.5820751580781622) q[6];
ry(-0.11366979276009208) q[7];
rz(0.9502654118712063) q[7];
ry(1.781019676934631) q[8];
rz(-0.11973924310944639) q[8];
ry(0.5321881198059994) q[9];
rz(-1.9488037124635964) q[9];
ry(1.0536509791848838) q[10];
rz(2.87717398238003) q[10];
ry(1.9096137002972973) q[11];
rz(-2.969950291191872) q[11];
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
ry(-1.6950697929480196) q[0];
rz(3.0595485834126652) q[0];
ry(-0.5080640319132862) q[1];
rz(1.6019830661250958) q[1];
ry(2.029542304598252) q[2];
rz(-3.0330223570378836) q[2];
ry(3.0126528734029434) q[3];
rz(1.6286420076961148) q[3];
ry(0.022236298199873137) q[4];
rz(-2.317145504906443) q[4];
ry(0.04226083507878653) q[5];
rz(-0.6582247986394885) q[5];
ry(-1.6238644651141643) q[6];
rz(3.1302765186741155) q[6];
ry(-1.5741077199625553) q[7];
rz(1.7933331554467145) q[7];
ry(-2.960428661521913) q[8];
rz(1.763586227778065) q[8];
ry(-0.918380785637158) q[9];
rz(2.8477158843344688) q[9];
ry(2.9643109196072053) q[10];
rz(0.7083210706535051) q[10];
ry(0.8060028524300558) q[11];
rz(1.7678828654233287) q[11];
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
ry(1.5386357947554787) q[0];
rz(-2.9569041626884935) q[0];
ry(1.8986253499779686) q[1];
rz(3.0164594888391165) q[1];
ry(1.599862673921407) q[2];
rz(0.8089975492112474) q[2];
ry(-1.4640207305659052) q[3];
rz(-1.6588353349919633) q[3];
ry(-3.132409626591966) q[4];
rz(-0.3179121377542939) q[4];
ry(-3.1347379145071415) q[5];
rz(-2.343401466120513) q[5];
ry(0.011559464553081404) q[6];
rz(-1.4443983570577679) q[6];
ry(3.115790943979974) q[7];
rz(0.2061464379867116) q[7];
ry(3.1196516274858976) q[8];
rz(0.03368567664236455) q[8];
ry(-0.027219084824479548) q[9];
rz(-0.04695838282170926) q[9];
ry(1.6476683877123433) q[10];
rz(-1.4822943072181172) q[10];
ry(2.3657464328771654) q[11];
rz(-1.6450341542497924) q[11];
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
ry(-0.19969565357785615) q[0];
rz(2.934945110989882) q[0];
ry(-0.07367733696140799) q[1];
rz(1.9981269049706745) q[1];
ry(-0.08383135240526955) q[2];
rz(1.4192579393124185) q[2];
ry(-1.7147069454579178) q[3];
rz(-1.5432367269328093) q[3];
ry(-0.003057230254427168) q[4];
rz(-1.9303788183178652) q[4];
ry(-1.5700408576554632) q[5];
rz(-1.5705826210294642) q[5];
ry(2.678170612783125) q[6];
rz(-3.0159823424574763) q[6];
ry(0.3832264780437256) q[7];
rz(-3.1164287895887854) q[7];
ry(-2.2467597870411806) q[8];
rz(-1.9410395438094548) q[8];
ry(1.2450220303948842) q[9];
rz(-0.7729545871153958) q[9];
ry(2.205067569417875) q[10];
rz(0.8956281784110686) q[10];
ry(3.029678878966839) q[11];
rz(-1.1816098979224243) q[11];
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
ry(1.568747318361173) q[0];
rz(3.0709671409776274) q[0];
ry(-0.7719667687117067) q[1];
rz(-0.41671494998741465) q[1];
ry(0.002891708280578743) q[2];
rz(-0.8552978720766146) q[2];
ry(-1.5624796093116649) q[3];
rz(-1.5705324487745351) q[3];
ry(0.094739098669407) q[4];
rz(2.808457504669187) q[4];
ry(2.7776104529758077) q[5];
rz(1.131952231656534) q[5];
ry(0.9251064803640913) q[6];
rz(1.5575357045475295) q[6];
ry(-1.571047783031041) q[7];
rz(-0.5376543192503747) q[7];
ry(-0.14501159543051845) q[8];
rz(2.8332271690007014) q[8];
ry(-1.4604472273371714) q[9];
rz(2.195318161395491) q[9];
ry(1.4069898974494655) q[10];
rz(-0.2976947842059779) q[10];
ry(-1.3754511427967133) q[11];
rz(2.740470393480733) q[11];
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
ry(-1.582514295308421) q[0];
rz(-0.7353642598750358) q[0];
ry(1.5643218973983233) q[1];
rz(-3.1303051183992845) q[1];
ry(-6.98319157584514e-05) q[2];
rz(-2.930157014818873) q[2];
ry(1.5620217084528079) q[3];
rz(-3.071624651180558) q[3];
ry(3.1165647026766283) q[4];
rz(-1.9060556689365908) q[4];
ry(-0.005195479560049254) q[5];
rz(-1.1279365443334122) q[5];
ry(0.018717695552760816) q[6];
rz(0.013894312714032075) q[6];
ry(3.141579322476497) q[7];
rz(-2.1126440225849867) q[7];
ry(-1.5759954891196475) q[8];
rz(-2.874814877448542) q[8];
ry(0.8463765409739135) q[9];
rz(-1.5691141351563527) q[9];
ry(0.022854366572473417) q[10];
rz(-2.115900217976325) q[10];
ry(-1.2147822419600558) q[11];
rz(-1.1353354301752923) q[11];
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
ry(3.1411003746903234) q[0];
rz(0.8336944230553289) q[0];
ry(2.91438431036089) q[1];
rz(-3.130993401456834) q[1];
ry(-2.64902456916926) q[2];
rz(1.4523127974222856) q[2];
ry(-0.1613059756554529) q[3];
rz(-0.058193230515784136) q[3];
ry(-1.5731514219909732) q[4];
rz(-0.8317442651960305) q[4];
ry(1.565627283295806) q[5];
rz(1.5703119116154767) q[5];
ry(1.5719548749519991) q[6];
rz(-3.1348616438621546) q[6];
ry(-1.5727706317958114) q[7];
rz(-2.6118888716622606) q[7];
ry(0.06935141322139834) q[8];
rz(-0.2625558620828141) q[8];
ry(1.5723157369501626) q[9];
rz(0.8743858336961883) q[9];
ry(-1.569494537305948) q[10];
rz(1.5748044975687359) q[10];
ry(1.5741912682664465) q[11];
rz(-1.5663245243984845) q[11];
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
ry(1.632538886837454) q[0];
rz(-3.141204682307067) q[0];
ry(1.5781314169174872) q[1];
rz(-0.0987438531590303) q[1];
ry(-3.1380977335890297) q[2];
rz(-1.6936391555713444) q[2];
ry(-1.307123167122037) q[3];
rz(-0.0013533824035922848) q[3];
ry(-0.004969088115055698) q[4];
rz(-2.013933665724683) q[4];
ry(1.5740306506168642) q[5];
rz(0.04564006880868978) q[5];
ry(3.140765466468425) q[6];
rz(0.010316494436723113) q[6];
ry(-3.1414777542200025) q[7];
rz(0.5293058411053665) q[7];
ry(-1.5608794034131126) q[8];
rz(0.2445410738139753) q[8];
ry(0.0026348668400313358) q[9];
rz(-2.4336544142714382) q[9];
ry(-1.567898957614519) q[10];
rz(1.9216501033319542) q[10];
ry(1.5813800004325744) q[11];
rz(-1.26953588297184) q[11];
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
ry(1.5619615104455882) q[0];
rz(-3.140724098100108) q[0];
ry(-0.0034645298299773586) q[1];
rz(-1.1310848869030474) q[1];
ry(-1.5706561008834736) q[2];
rz(0.005717374936640597) q[2];
ry(-1.5718160988749432) q[3];
rz(0.0007095683218868824) q[3];
ry(0.00739203030968922) q[4];
rz(-2.54002441746496) q[4];
ry(-3.1255306210478633) q[5];
rz(1.134912692243403) q[5];
ry(-1.5703216992435065) q[6];
rz(-1.1675009456125784) q[6];
ry(-2.0386148592415645) q[7];
rz(0.1636280967003424) q[7];
ry(0.21302473073578196) q[8];
rz(-1.819010309892561) q[8];
ry(-1.5782766583168224) q[9];
rz(2.8922055027319895) q[9];
ry(-0.4229752012484713) q[10];
rz(1.4490963041432918) q[10];
ry(-0.11231227354364452) q[11];
rz(-1.824430655834459) q[11];
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
ry(-1.5751423500791026) q[0];
rz(3.1366119237469867) q[0];
ry(3.1406838906829107) q[1];
rz(0.3311755148937979) q[1];
ry(1.5712350951554523) q[2];
rz(-3.1371227400820536) q[2];
ry(1.5701422550170294) q[3];
rz(1.5642929923565674) q[3];
ry(-3.1408706253494536) q[4];
rz(-0.6631242536994844) q[4];
ry(0.0001803297346676037) q[5];
rz(2.0521413312403674) q[5];
ry(-3.1415119714894266) q[6];
rz(0.35667093015084295) q[6];
ry(0.0022676496383813227) q[7];
rz(-0.0752778415673463) q[7];
ry(-1.5719989654023871) q[8];
rz(-1.3564237722350443) q[8];
ry(-1.570677654959514) q[9];
rz(-0.004700188595951893) q[9];
ry(-3.1395445916256484) q[10];
rz(1.1203157312979921) q[10];
ry(-3.135405391217842) q[11];
rz(-1.5245210413200532) q[11];
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
ry(-1.5926532108944855) q[0];
rz(1.5702133496550135) q[0];
ry(-1.560146654073202) q[1];
rz(-2.2094972666464026) q[1];
ry(1.249584461929266) q[2];
rz(-0.02146762711677397) q[2];
ry(-1.5649210038631471) q[3];
rz(-1.573276508102846) q[3];
ry(1.571694629499028) q[4];
rz(-2.2032831247930647) q[4];
ry(1.574672629029604) q[5];
rz(-2.9837180961377827) q[5];
ry(0.0046688149531224354) q[6];
rz(-0.43077488768238115) q[6];
ry(-3.141234173909635) q[7];
rz(0.07402818467554792) q[7];
ry(-3.125060755811496) q[8];
rz(-1.3599485061047631) q[8];
ry(1.5703461330084298) q[9];
rz(-2.050527408433666) q[9];
ry(0.005983631271083922) q[10];
rz(-1.1039883855322685) q[10];
ry(-1.5661592718623254) q[11];
rz(3.116326011691382) q[11];
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
ry(-1.5692186183765353) q[0];
rz(2.0088992355603956) q[0];
ry(-3.141257893739673) q[1];
rz(1.2897108331941336) q[1];
ry(1.6046263702012213) q[2];
rz(-2.6981807473123793) q[2];
ry(-1.8184626386029648) q[3];
rz(2.77348814928412) q[3];
ry(0.0005137562602754997) q[4];
rz(-1.1959509512282247) q[4];
ry(-8.432744431347514e-05) q[5];
rz(-1.732827406262911) q[5];
ry(3.1414632804608975) q[6];
rz(1.09422385433792) q[6];
ry(1.4879143811263873e-05) q[7];
rz(0.34270627419380567) q[7];
ry(1.565451634488018) q[8];
rz(0.002658643869538599) q[8];
ry(-3.141255641024273) q[9];
rz(1.191822729224498) q[9];
ry(-1.412056813342736) q[10];
rz(0.00010846677095352991) q[10];
ry(-1.571246563112858) q[11];
rz(-1.9085435537281303) q[11];
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
ry(-3.141454944770399) q[0];
rz(2.268535744487681) q[0];
ry(3.14113715406325) q[1];
rz(2.5008372576414324) q[1];
ry(-0.0005635493802511604) q[2];
rz(-0.21230334835167142) q[2];
ry(0.0003166259099058877) q[3];
rz(-2.213378807760232) q[3];
ry(0.0025168152867739665) q[4];
rz(2.0839664852751274) q[4];
ry(-1.5669728179426885) q[5];
rz(-2.5760382131059796) q[5];
ry(1.5715333675794287) q[6];
rz(0.25382334039016374) q[6];
ry(-0.0018097378220263128) q[7];
rz(1.809710664706547) q[7];
ry(1.5692325559924098) q[8];
rz(0.11928652525240814) q[8];
ry(0.0038368488804065848) q[9];
rz(1.835492802619248) q[9];
ry(-1.5736826761428668) q[10];
rz(-2.8812100205517504) q[10];
ry(-3.1409363747103387) q[11];
rz(0.2486197990256581) q[11];