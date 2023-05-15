OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-2.676640188806865) q[0];
rz(-2.7427762021171875) q[0];
ry(1.587836799758219) q[1];
rz(0.015060415094181835) q[1];
ry(3.13958660039231) q[2];
rz(0.937602301983383) q[2];
ry(0.00020649706778019805) q[3];
rz(1.5281453623005277) q[3];
ry(-3.007977193594234) q[4];
rz(2.254117202404479) q[4];
ry(1.5928511239075824) q[5];
rz(-1.2946759604601026) q[5];
ry(-3.119357341543652) q[6];
rz(-2.641340282177833) q[6];
ry(-1.5838899009276501) q[7];
rz(-0.8414421473220277) q[7];
ry(-0.8125064482437576) q[8];
rz(1.1517863745442565) q[8];
ry(-1.7073472779533636) q[9];
rz(-3.132257001168574) q[9];
ry(1.5439635139095502) q[10];
rz(-2.742151266082759) q[10];
ry(-3.086729201190622) q[11];
rz(-2.0746379055867963) q[11];
ry(0.0005000016471345958) q[12];
rz(0.6675508508878589) q[12];
ry(-1.5268206260422958) q[13];
rz(-1.6487885953677126) q[13];
ry(-1.5722010718782167) q[14];
rz(-2.24890422280874) q[14];
ry(1.5217276078665325) q[15];
rz(-2.0251761098513263) q[15];
ry(-3.041557503824282) q[16];
rz(0.8225670833223783) q[16];
ry(-3.141075839972922) q[17];
rz(0.7609606598332181) q[17];
ry(1.182452483674537) q[18];
rz(2.7796700307583815) q[18];
ry(1.7922783512999425) q[19];
rz(2.7956120482157063) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(-1.5843202477530545) q[0];
rz(2.3308265450811176) q[0];
ry(1.542149888773609) q[1];
rz(-2.7371892403845925) q[1];
ry(-0.6699361020243932) q[2];
rz(0.25713987750279976) q[2];
ry(-1.570295175647371) q[3];
rz(0.41426212727236505) q[3];
ry(0.2538075242555712) q[4];
rz(0.11970462227940447) q[4];
ry(0.0011796399193563525) q[5];
rz(2.860279459087721) q[5];
ry(3.128451965546288) q[6];
rz(2.641454789733298) q[6];
ry(3.140815800784943) q[7];
rz(2.303005375429255) q[7];
ry(-2.7735179034138784) q[8];
rz(1.1005061455733383) q[8];
ry(0.07166378586858825) q[9];
rz(1.661187355469834) q[9];
ry(0.06304960709928613) q[10];
rz(0.3595490355770581) q[10];
ry(-6.939813729633926e-05) q[11];
rz(-2.96413083438452) q[11];
ry(-0.268392400568768) q[12];
rz(-2.3534304928807206) q[12];
ry(2.895022325866347) q[13];
rz(-0.6202051447964836) q[13];
ry(0.003219753679137405) q[14];
rz(-2.502023373188522) q[14];
ry(-2.2626846469031827) q[15];
rz(1.0381965088294547) q[15];
ry(-3.1389816611861026) q[16];
rz(1.1337579895832288) q[16];
ry(-0.0016114374951768018) q[17];
rz(-1.6971455464849403) q[17];
ry(2.6285949813015494) q[18];
rz(-0.3542170541708094) q[18];
ry(-1.7872669533898908) q[19];
rz(0.9568098201760609) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(-0.012362900394588648) q[0];
rz(1.2204935843042104) q[0];
ry(-1.594912129775147) q[1];
rz(2.2214352506180606) q[1];
ry(-3.116051442527395) q[2];
rz(0.2960086610315015) q[2];
ry(0.00455151712342694) q[3];
rz(-0.20857318198630193) q[3];
ry(2.8306891472011486) q[4];
rz(2.2802903644488417) q[4];
ry(1.6523699620761632) q[5];
rz(-0.006307418040618629) q[5];
ry(-0.5435172936031742) q[6];
rz(-0.44527254144864076) q[6];
ry(1.584581160383589) q[7];
rz(0.47995828323920176) q[7];
ry(2.348361229326454) q[8];
rz(-1.6282937904440342) q[8];
ry(1.7773524829762382) q[9];
rz(1.2064190760934244) q[9];
ry(3.133541921597835) q[10];
rz(2.348498375582075) q[10];
ry(-0.3859496304971704) q[11];
rz(-2.28589489250027) q[11];
ry(0.000692630777893551) q[12];
rz(2.5862343985001557) q[12];
ry(0.09740086110725125) q[13];
rz(-2.573645649538646) q[13];
ry(-0.8515670533926379) q[14];
rz(0.01805883679779363) q[14];
ry(-3.0677483166583883) q[15];
rz(-0.08096859141018697) q[15];
ry(-3.0116428981471155) q[16];
rz(-2.763186018345799) q[16];
ry(0.0005492868267845444) q[17];
rz(0.7726091077214575) q[17];
ry(-0.18243411471386306) q[18];
rz(-0.601081095037241) q[18];
ry(0.5900527528779094) q[19];
rz(2.9954514573488944) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(0.31254331636731436) q[0];
rz(1.990686195025594) q[0];
ry(-0.01593313583102693) q[1];
rz(0.7703264943966506) q[1];
ry(-1.6260723430786355) q[2];
rz(-2.317275796609505) q[2];
ry(1.1523854959390416) q[3];
rz(2.5560179450758436) q[3];
ry(3.1388012684345186) q[4];
rz(2.1212115699617584) q[4];
ry(1.4881134763349868) q[5];
rz(-2.3240893292700537) q[5];
ry(-1.5537254660908504) q[6];
rz(-3.103904924184515) q[6];
ry(-0.00047684522842921466) q[7];
rz(-0.7265622911988974) q[7];
ry(-3.01741609269665) q[8];
rz(-3.0697124484188674) q[8];
ry(0.00038019951813400045) q[9];
rz(0.23913978631367422) q[9];
ry(3.1032785581905764) q[10];
rz(1.4448260107945097) q[10];
ry(0.00019222282090405646) q[11];
rz(2.2358484329768142) q[11];
ry(1.0138051318876908) q[12];
rz(1.531807861924653) q[12];
ry(0.1811688327065628) q[13];
rz(2.3708434815887127) q[13];
ry(1.7890846968688379) q[14];
rz(1.2432149007871303) q[14];
ry(2.410603150239366) q[15];
rz(1.7660007353576952) q[15];
ry(0.016030280720580183) q[16];
rz(-2.7539712683338253) q[16];
ry(-3.138498373502973) q[17];
rz(2.3916709792275905) q[17];
ry(1.5105781189619956) q[18];
rz(3.024598400258002) q[18];
ry(-0.7772016552584489) q[19];
rz(1.9744470257709148) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(-0.044359109644667605) q[0];
rz(0.7161294765124467) q[0];
ry(3.0879183251626454) q[1];
rz(-1.7861900666466004) q[1];
ry(-0.005417336318525159) q[2];
rz(2.281641802857341) q[2];
ry(-0.015880923900660093) q[3];
rz(-2.57736768963981) q[3];
ry(-3.1046774159707984) q[4];
rz(-0.9944736111320421) q[4];
ry(-3.1387918402886754) q[5];
rz(-2.466347337895794) q[5];
ry(2.642585652855693) q[6];
rz(0.19868507936740235) q[6];
ry(-1.5945697891136996) q[7];
rz(-1.5529299446414173) q[7];
ry(3.132619831408793) q[8];
rz(-0.004258922309646529) q[8];
ry(-1.4464366909998365) q[9];
rz(0.9805948238657463) q[9];
ry(-0.0016520715652355022) q[10];
rz(-3.0649565861143055) q[10];
ry(2.539710876915283) q[11];
rz(-0.7480408950507372) q[11];
ry(3.141514505330096) q[12];
rz(0.2969775961622883) q[12];
ry(-3.136198714707653) q[13];
rz(-0.7860868651208018) q[13];
ry(-0.007439296283357599) q[14];
rz(-1.1995437285498407) q[14];
ry(2.1241264664347685) q[15];
rz(-2.1801434766055925) q[15];
ry(-3.081105969811484) q[16];
rz(3.075574765561692) q[16];
ry(-0.0010407981637881036) q[17];
rz(1.3480125134085794) q[17];
ry(-3.10116112823741) q[18];
rz(0.7565523400311265) q[18];
ry(1.1538425880496552) q[19];
rz(1.1709007310966488) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(2.8570334928063756) q[0];
rz(3.0725339276928216) q[0];
ry(-1.205330945193861) q[1];
rz(1.4438457215655172) q[1];
ry(0.2765886471527444) q[2];
rz(2.692549271160383) q[2];
ry(1.9692093437043074) q[3];
rz(1.4698630813389995) q[3];
ry(-0.013195496893670996) q[4];
rz(0.8783852399616956) q[4];
ry(1.566483390282985) q[5];
rz(2.7684000760779) q[5];
ry(1.568616444039625) q[6];
rz(1.5826129138315428) q[6];
ry(1.5734652754316951) q[7];
rz(1.633287655046658) q[7];
ry(-1.5845520791667806) q[8];
rz(-0.0417575504712131) q[8];
ry(-3.138788206200481) q[9];
rz(2.0599833893215482) q[9];
ry(1.4390569763321985) q[10];
rz(2.7699506346966727) q[10];
ry(-1.4463751369568) q[11];
rz(-0.5602178293123253) q[11];
ry(1.8576305419798045) q[12];
rz(2.181373143507119) q[12];
ry(0.5721131881244688) q[13];
rz(0.034708213708160685) q[13];
ry(2.8322263749713987) q[14];
rz(3.0385972006150035) q[14];
ry(0.847526829994976) q[15];
rz(-0.7466507162438196) q[15];
ry(-0.7781171558219685) q[16];
rz(2.816699668846562) q[16];
ry(-1.5949228413148595) q[17];
rz(0.04843008471613341) q[17];
ry(1.5995840642297143) q[18];
rz(-2.8706350623362717) q[18];
ry(2.246470658720516) q[19];
rz(-1.85317555096665) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(2.877448644775752) q[0];
rz(-0.46935137068273214) q[0];
ry(3.0791559102740553) q[1];
rz(-2.2391902219850435) q[1];
ry(-2.16718349913398) q[2];
rz(-0.013245908537435014) q[2];
ry(-8.240914037216385e-05) q[3];
rz(0.7215337566495228) q[3];
ry(0.1995146539951603) q[4];
rz(-2.09827918266388) q[4];
ry(0.00018627187058981338) q[5];
rz(-2.770410732326307) q[5];
ry(-1.5868428238578218) q[6];
rz(1.4391495203258042) q[6];
ry(-0.22472755402059236) q[7];
rz(-0.09351952835199207) q[7];
ry(0.05437355626294327) q[8];
rz(1.6112343737216) q[8];
ry(3.1415846169134207) q[9];
rz(1.4575935942977085) q[9];
ry(3.1413170611142616) q[10];
rz(-0.014030531651485548) q[10];
ry(3.140698087446496) q[11];
rz(2.5883001012964355) q[11];
ry(0.029513718473711318) q[12];
rz(-3.1335525120536456) q[12];
ry(-2.893732848814104) q[13];
rz(1.4961699398962498) q[13];
ry(0.0007375251076009447) q[14];
rz(2.5424515573198647) q[14];
ry(-1.6012011675480649) q[15];
rz(3.1406697821706553) q[15];
ry(-3.086735918704744) q[16];
rz(0.4108577535738407) q[16];
ry(-0.0014215329332339457) q[17];
rz(-0.04736410814612728) q[17];
ry(1.3294612595933701) q[18];
rz(1.9890972169591372) q[18];
ry(-0.00043499015923753603) q[19];
rz(-0.7723361188345983) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(-0.0019262572827720703) q[0];
rz(1.8508990996621282) q[0];
ry(-1.96356076565967) q[1];
rz(1.2382525906802668) q[1];
ry(-1.2054640437403359) q[2];
rz(0.005821797249451421) q[2];
ry(-3.1358883251981964) q[3];
rz(-2.2213910748283157) q[3];
ry(2.759318552048029e-05) q[4];
rz(-0.5158006251548952) q[4];
ry(1.5450284408450692) q[5];
rz(-1.9999192549101705) q[5];
ry(-1.602838655430136) q[6];
rz(-2.9408373319993073) q[6];
ry(1.6999898841136034) q[7];
rz(1.3394834799340476) q[7];
ry(0.00046285097302322325) q[8];
rz(0.3106849136515554) q[8];
ry(0.0003875802118837868) q[9];
rz(-0.08044474419314757) q[9];
ry(1.643319620676993) q[10];
rz(2.4355767720777344) q[10];
ry(1.3868369343404445) q[11];
rz(-1.0062594897571413) q[11];
ry(-1.5092747243005336) q[12];
rz(0.042025607257396125) q[12];
ry(3.1403008026033055) q[13];
rz(-0.05860943581138365) q[13];
ry(-3.1370842471766918) q[14];
rz(0.1453752574758746) q[14];
ry(-1.8362741216150382) q[15];
rz(-1.5048480599627045) q[15];
ry(-2.518793170030894) q[16];
rz(0.2718316102287357) q[16];
ry(0.6155817210137811) q[17];
rz(-1.832534157741832) q[17];
ry(-0.001502436853230371) q[18];
rz(1.8683968751954088) q[18];
ry(-1.3091090762141881) q[19];
rz(-1.2556423179168972) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(-1.4394836580417905) q[0];
rz(2.588946091014886) q[0];
ry(0.2247657561542823) q[1];
rz(-2.829530643325865) q[1];
ry(1.001598154210477) q[2];
rz(3.009649194637989) q[2];
ry(0.0037096472344624533) q[3];
rz(-2.3655578639473998) q[3];
ry(4.513611682810392e-05) q[4];
rz(-1.4627265317559792) q[4];
ry(3.138365972356272) q[5];
rz(-0.24844106977170277) q[5];
ry(1.580531281357219) q[6];
rz(-0.012656606160873043) q[6];
ry(1.5427839160015786) q[7];
rz(-1.8383212632674162) q[7];
ry(-0.0003210823894604681) q[8];
rz(1.2152976302166945) q[8];
ry(0.03054166476112978) q[9];
rz(-1.8242412242019073) q[9];
ry(0.0015335245187424464) q[10];
rz(0.34805535160301665) q[10];
ry(0.3590340136774422) q[11];
rz(-2.233513134075144) q[11];
ry(3.1394716934009184) q[12];
rz(-2.6504856825509866) q[12];
ry(1.5871920048473946) q[13];
rz(-0.4776305964415411) q[13];
ry(-0.006942996345064356) q[14];
rz(-2.4236417077590153) q[14];
ry(-2.423480216896222) q[15];
rz(1.5949783507972795) q[15];
ry(-3.0931956793960524) q[16];
rz(-0.7072901641298612) q[16];
ry(3.1414028705062558) q[17];
rz(-2.2724596476995664) q[17];
ry(-0.0014434579002557744) q[18];
rz(2.2781591145590885) q[18];
ry(-0.3646525453569716) q[19];
rz(1.2779642873747434) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(0.05467310770250897) q[0];
rz(1.1121040635015746) q[0];
ry(-2.2762504941870336) q[1];
rz(-0.0073112226083571485) q[1];
ry(0.6104573794710432) q[2];
rz(-3.030032489795706) q[2];
ry(3.1102933133998967) q[3];
rz(1.8302975515135271) q[3];
ry(1.1785242736789987) q[4];
rz(3.1035851945453734) q[4];
ry(-1.522379552321137) q[5];
rz(0.2685569755406334) q[5];
ry(3.089972365978887) q[6];
rz(-0.9824222189992674) q[6];
ry(2.7983809880409876) q[7];
rz(-0.3600048406012312) q[7];
ry(2.869111332329712) q[8];
rz(-0.8460379227091447) q[8];
ry(2.636401008467415) q[9];
rz(-2.075621351220359) q[9];
ry(-1.5799340920690783) q[10];
rz(-1.4295078188261794) q[10];
ry(-0.0009398137442451814) q[11];
rz(-1.1158355085354303) q[11];
ry(3.140327248434396) q[12];
rz(0.6323515738254253) q[12];
ry(-3.1371046813270955) q[13];
rz(1.9986314962541427) q[13];
ry(0.06274497237521223) q[14];
rz(1.4948396619976698) q[14];
ry(-1.5614578213294747) q[15];
rz(-0.01056114508047489) q[15];
ry(2.2199084727125324) q[16];
rz(0.6095132987062666) q[16];
ry(0.004732970328741595) q[17];
rz(-1.1320282076523098) q[17];
ry(0.515920782969868) q[18];
rz(-3.134477224970698) q[18];
ry(1.81921744454905) q[19];
rz(-2.483082394294536) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(-0.63236304555836) q[0];
rz(2.6493271777335146) q[0];
ry(1.3729655231028532) q[1];
rz(1.3630754069069742) q[1];
ry(0.00029860553066391214) q[2];
rz(-3.080755459385465) q[2];
ry(3.14075868110765) q[3];
rz(1.5060418836764284) q[3];
ry(3.1065442616722754) q[4];
rz(1.0732203135370006) q[4];
ry(-3.132522994624645) q[5];
rz(0.6980918031004331) q[5];
ry(-3.141191109933919) q[6];
rz(-2.4852532734917387) q[6];
ry(-3.126519667600259) q[7];
rz(-0.002313454318998431) q[7];
ry(-3.117862703010249) q[8];
rz(1.1768404803108155) q[8];
ry(-0.03441740829354525) q[9];
rz(-1.2487611743926352) q[9];
ry(0.009206558649682783) q[10];
rz(3.0265810447852686) q[10];
ry(0.005255534970548972) q[11];
rz(1.1867512944434333) q[11];
ry(-3.132384034554085) q[12];
rz(-1.3800360907880873) q[12];
ry(1.2696335834427273) q[13];
rz(-2.7714181517412046) q[13];
ry(3.1108297138914516) q[14];
rz(0.063760092935064) q[14];
ry(1.4970160110544593) q[15];
rz(-1.565724893301788) q[15];
ry(-0.007119660238696901) q[16];
rz(-2.676561611899588) q[16];
ry(-0.17620791396821112) q[17];
rz(-3.1392543712601118) q[17];
ry(-0.698967556839517) q[18];
rz(-0.07001137533309354) q[18];
ry(2.9771204521516155) q[19];
rz(1.5777885326433303) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(-1.9463474758883281) q[0];
rz(1.5072323973300756) q[0];
ry(-1.3374614801128333) q[1];
rz(2.347048348806785) q[1];
ry(1.525542605354433) q[2];
rz(0.04181704822911669) q[2];
ry(-3.1336136182588774) q[3];
rz(0.9660937223361507) q[3];
ry(2.41080063505267) q[4];
rz(0.9578723496599656) q[4];
ry(-1.4941997293931717) q[5];
rz(0.023325479623184542) q[5];
ry(-1.0763924958601137) q[6];
rz(-1.6295135998318417) q[6];
ry(2.416882352829277) q[7];
rz(1.701585033147387) q[7];
ry(0.5886997215569574) q[8];
rz(-2.0174670260190997) q[8];
ry(-2.065529700254545) q[9];
rz(3.050495021995059) q[9];
ry(1.064603783177989) q[10];
rz(0.01587634110888837) q[10];
ry(-1.5702241257694236) q[11];
rz(-3.141296308745578) q[11];
ry(1.5657983464192125) q[12];
rz(-3.1327842320719816) q[12];
ry(1.5541843708537353) q[13];
rz(0.009027662322195814) q[13];
ry(-0.012431846894319646) q[14];
rz(1.465082000368402) q[14];
ry(1.69193210834866) q[15];
rz(-1.5325669439954241) q[15];
ry(3.117681920431133) q[16];
rz(-1.4415563835124647) q[16];
ry(1.6037719903012784) q[17];
rz(-1.5697772429515837) q[17];
ry(-0.511441524760039) q[18];
rz(-3.083896818657169) q[18];
ry(-1.5785530901812255) q[19];
rz(-0.011780962647056192) q[19];