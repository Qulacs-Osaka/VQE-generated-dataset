OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(3.1413225742971296) q[0];
rz(1.9802607836525095) q[0];
ry(1.5421196559484651) q[1];
rz(-1.6754148000061013) q[1];
ry(-3.122401742171442) q[2];
rz(-0.8186946020324735) q[2];
ry(-1.5401839739585117) q[3];
rz(-1.6283184036169587) q[3];
ry(-0.0002573211138612308) q[4];
rz(1.867212662542337) q[4];
ry(-1.6590055293848867) q[5];
rz(1.5323643344061602) q[5];
ry(-0.0006536713251197313) q[6];
rz(-1.6612580981955936) q[6];
ry(-1.5712307007409017) q[7];
rz(-1.5709239645576716) q[7];
ry(1.675451565488504) q[8];
rz(1.8899504886058631) q[8];
ry(1.5706607519178366) q[9];
rz(2.7613218959806805) q[9];
ry(-3.1413339159754767) q[10];
rz(2.1538619559759127) q[10];
ry(-0.3903115716601624) q[11];
rz(-0.2169846453168254) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.3431260133344254) q[0];
rz(0.5904997596577547) q[0];
ry(2.8437357236327543) q[1];
rz(-0.10809323504356302) q[1];
ry(-1.568425857441479) q[2];
rz(2.958349235213892) q[2];
ry(2.651188271805433) q[3];
rz(3.0762887152157816) q[3];
ry(1.8525097136545157) q[4];
rz(0.00018468290783556023) q[4];
ry(-1.9807238784168497) q[5];
rz(-1.6538920479307926) q[5];
ry(1.8779118056610002) q[6];
rz(-0.00032444696640835595) q[6];
ry(-1.48839094864351) q[7];
rz(1.5708358191502168) q[7];
ry(-1.5708238044894909) q[8];
rz(-9.689472832707936e-05) q[8];
ry(-0.11650849602056024) q[9];
rz(1.9497042255459949) q[9];
ry(1.5708417785613493) q[10];
rz(-1.1073065859336433) q[10];
ry(-2.2218422474646022) q[11];
rz(-0.08484001467202695) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.010851480444130217) q[0];
rz(2.457857210934351) q[0];
ry(-1.5819669208907925) q[1];
rz(0.8150820793263165) q[1];
ry(3.1409282611071303) q[2];
rz(-0.1884179207490351) q[2];
ry(0.07617395518375503) q[3];
rz(0.00015432700219886186) q[3];
ry(-1.489322026831053) q[4];
rz(-1.5711424275434294) q[4];
ry(-4.942186959233652e-06) q[5];
rz(-1.032550849891006) q[5];
ry(-1.57028898117857) q[6];
rz(-0.0014014841869034458) q[6];
ry(2.6607596889556873) q[7];
rz(1.570854883745635) q[7];
ry(-1.560658064074092) q[8];
rz(-0.613797506929645) q[8];
ry(-2.6823471556931) q[9];
rz(1.6431546026575323) q[9];
ry(-0.0021465117260230926) q[10];
rz(-1.2396174391039887) q[10];
ry(1.57083583207445) q[11];
rz(0.7944135308940982) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.9301074586376887) q[0];
rz(-0.6431587736352605) q[0];
ry(-3.139972626297686) q[1];
rz(-1.1406678471131473) q[1];
ry(-1.5699457122586369) q[2];
rz(1.5716734282935567) q[2];
ry(-1.5699688228113564) q[3];
rz(-0.00023310122555209031) q[3];
ry(-1.571382675489371) q[4];
rz(1.2881783599461682) q[4];
ry(-8.181454409950106e-06) q[5];
rz(-0.5526315095988298) q[5];
ry(-1.2132023277803876) q[6];
rz(-3.140767366213164) q[6];
ry(1.5713404299213638) q[7];
rz(1.0271965590092746) q[7];
ry(3.1415874213813706) q[8];
rz(-2.1842116486692422) q[8];
ry(1.5521569636152373) q[9];
rz(0.2571589000046294) q[9];
ry(1.3594835991842014) q[10];
rz(7.59799220812648e-06) q[10];
ry(-1.3802745199754842) q[11];
rz(-0.19584230740583142) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.014156605577291259) q[0];
rz(2.1218267225477447) q[0];
ry(0.004870430985633334) q[1];
rz(-2.7567554527826212) q[1];
ry(1.5707612434824407) q[2];
rz(0.9965009065964417) q[2];
ry(2.842266061077876) q[3];
rz(-0.00024674372918287446) q[3];
ry(1.9577181514024335) q[4];
rz(2.184137461928612) q[4];
ry(-3.1006369491212897) q[5];
rz(-0.001257848871051167) q[5];
ry(1.571954710553672) q[6];
rz(-1.5713240301259077) q[6];
ry(-3.140836588919014) q[7];
rz(-0.3897587289440088) q[7];
ry(1.5698017322432793) q[8];
rz(-2.831267164268424) q[8];
ry(0.8422039988195998) q[9];
rz(0.00031151560583880247) q[9];
ry(-1.5704322435694185) q[10];
rz(-1.5709500204298994) q[10];
ry(1.570726386248902) q[11];
rz(-1.6323855282228434) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.5912477288776579) q[0];
rz(-2.9187377408169453) q[0];
ry(1.5723755142393463) q[1];
rz(0.6475990950489452) q[1];
ry(0.7972878562988464) q[2];
rz(-0.1574030714553942) q[2];
ry(1.5729203503378768) q[3];
rz(2.1185675395395966) q[3];
ry(3.1412581753296975) q[4];
rz(-0.9572675129370322) q[4];
ry(-1.57265527642124) q[5];
rz(0.00013259484990512234) q[5];
ry(1.673591868959868) q[6];
rz(-1.9973931569769308) q[6];
ry(-0.7354394380005933) q[7];
rz(3.1160144898010484) q[7];
ry(-1.6409506297238607) q[8];
rz(-2.872579711132964) q[8];
ry(-1.57083654801382) q[9];
rz(-1.2278905225899472) q[9];
ry(1.5710536071910408) q[10];
rz(1.5710310096405733) q[10];
ry(0.00014421272672127117) q[11];
rz(-1.5092587799577677) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.31550251781034544) q[0];
rz(0.7492181589840515) q[0];
ry(-3.1405011798771825) q[1];
rz(1.475209123202683) q[1];
ry(-0.00044989832015249914) q[2];
rz(0.5821147008134604) q[2];
ry(-0.00016431304746715653) q[3];
rz(-0.5479305690008118) q[3];
ry(0.28815259304624036) q[4];
rz(-1.9693806340483242) q[4];
ry(-0.37680591447163697) q[5];
rz(1.570796483057535) q[5];
ry(-0.0004975463542835617) q[6];
rz(-2.8462774599035887) q[6];
ry(-0.007685993880392595) q[7];
rz(-1.659011317704948) q[7];
ry(3.1401004481243167) q[8];
rz(2.0568705171445334) q[8];
ry(3.14148973368134) q[9];
rz(-1.2278977667750735) q[9];
ry(1.5709288124657184) q[10];
rz(0.3491434168991221) q[10];
ry(1.5706456997982636) q[11];
rz(0.0007058779789902918) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-3.1414834385496864) q[0];
rz(-0.2884420595725681) q[0];
ry(-0.0019513279587272175) q[1];
rz(-1.1866771996665868) q[1];
ry(2.3213446405406213) q[2];
rz(-2.309938527601795) q[2];
ry(1.5708052263030974) q[3];
rz(-0.3612688109490003) q[3];
ry(3.141194408099018) q[4];
rz(-3.0080320252966923) q[4];
ry(1.570782403108157) q[5];
rz(2.7823864996729286) q[5];
ry(-1.5853315936915642) q[6];
rz(-2.4995553670479334) q[6];
ry(-1.673168441922246) q[7];
rz(0.36848658665961587) q[7];
ry(-0.3182070570550337) q[8];
rz(1.8756000479435972) q[8];
ry(1.570887754787141) q[9];
rz(1.2095734746815816) q[9];
ry(-3.141568564598248) q[10];
rz(-0.6894802179865716) q[10];
ry(-1.5711124591754868) q[11];
rz(-1.9319235215910429) q[11];