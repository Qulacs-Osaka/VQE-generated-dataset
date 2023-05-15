OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(0.9616423491259625) q[0];
rz(-1.6514889118075695) q[0];
ry(0.004965373197517343) q[1];
rz(-0.4629575369243028) q[1];
ry(1.56740273115374) q[2];
rz(0.7375526083265747) q[2];
ry(1.5692925242635765) q[3];
rz(3.141347342018659) q[3];
ry(-2.714885286125795) q[4];
rz(1.5856612491110011) q[4];
ry(-0.005293396126327904) q[5];
rz(-1.5621792319146826) q[5];
ry(3.128456769389086) q[6];
rz(2.926533075763746) q[6];
ry(2.3501753550634614) q[7];
rz(-1.1041533615516486) q[7];
ry(-1.518502099776148) q[8];
rz(-0.6291126605634306) q[8];
ry(1.5781286889215844) q[9];
rz(-1.7096168546882797) q[9];
ry(3.1405587856099206) q[10];
rz(-1.7860834052659529) q[10];
ry(3.1412877362603937) q[11];
rz(-0.023086305322790834) q[11];
ry(0.003395439770425571) q[12];
rz(-2.536948625795801) q[12];
ry(-0.8523913147405822) q[13];
rz(2.4068983559603887) q[13];
ry(0.001248574725988938) q[14];
rz(0.2788195974134114) q[14];
ry(-0.0005879327324480599) q[15];
rz(-2.4723542237284577) q[15];
ry(1.859091685045108) q[16];
rz(-2.0946567765422452) q[16];
ry(-1.5815457737306193) q[17];
rz(-1.4659543320189892) q[17];
ry(-1.6508710245578953) q[18];
rz(-1.124091135168336) q[18];
ry(-1.3862582849993828) q[19];
rz(1.7256625582486875) q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.568671999027444) q[0];
rz(1.826778719461495) q[0];
ry(-3.080968552928866) q[1];
rz(0.3045851348601033) q[1];
ry(-0.0117607780380391) q[2];
rz(0.7938715196145658) q[2];
ry(1.5681591400424422) q[3];
rz(1.8368421072368726) q[3];
ry(3.140419745792342) q[4];
rz(-0.7079570418672999) q[4];
ry(-3.1333886907833826) q[5];
rz(-1.0516119989872763) q[5];
ry(1.5679806466816348) q[6];
rz(-1.5809419085801917) q[6];
ry(1.5720305893500017) q[7];
rz(0.0011027977663431445) q[7];
ry(1.6662892857707199) q[8];
rz(-1.6377888808490244) q[8];
ry(-0.7258483127483596) q[9];
rz(-2.965334571302469) q[9];
ry(-3.1415277998054156) q[10];
rz(3.0778734972004083) q[10];
ry(-0.0002428746653629679) q[11];
rz(0.2471642825550748) q[11];
ry(-1.5683506450423934) q[12];
rz(-0.5720918169843302) q[12];
ry(-2.2748401608100854) q[13];
rz(-2.524572137195809) q[13];
ry(-0.00017957267228307927) q[14];
rz(-0.39741617665865187) q[14];
ry(0.0014903376142469944) q[15];
rz(-1.1424250680540482) q[15];
ry(-0.3883943219997965) q[16];
rz(2.3458877910340044) q[16];
ry(-1.7319260779026313) q[17];
rz(2.179266781706853) q[17];
ry(2.417822307133176) q[18];
rz(0.29145898836711925) q[18];
ry(0.7193086661883614) q[19];
rz(2.0757141766909815) q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(0.6486375815279528) q[0];
rz(-1.5398174564123608) q[0];
ry(-3.1091167234054375) q[1];
rz(2.3799207455534366) q[1];
ry(-1.0384940997082397) q[2];
rz(-1.1040745758383883) q[2];
ry(2.731361348911239) q[3];
rz(1.4434730330495162) q[3];
ry(1.568478371723479) q[4];
rz(1.5875091258753813) q[4];
ry(-1.558193127378485) q[5];
rz(-0.031998416573788845) q[5];
ry(1.5741222066013911) q[6];
rz(-0.22079210578011035) q[6];
ry(1.6939239023696153) q[7];
rz(-0.0030219180671996787) q[7];
ry(1.5735900358920962) q[8];
rz(0.9414288118522632) q[8];
ry(-1.5692483709857994) q[9];
rz(0.4001991354442353) q[9];
ry(0.7257417748315024) q[10];
rz(2.41430866950762) q[10];
ry(0.3896810828853923) q[11];
rz(-0.6801568012596456) q[11];
ry(1.334080884703269) q[12];
rz(-1.3916394443593523) q[12];
ry(2.526960143125225) q[13];
rz(2.6078071336251876) q[13];
ry(-1.0397869440668321) q[14];
rz(-1.9397273331486735) q[14];
ry(-1.1033177351462364) q[15];
rz(1.7993911936641527) q[15];
ry(-2.1139909547303852) q[16];
rz(-0.3722058318857843) q[16];
ry(0.30679738912699595) q[17];
rz(-0.03309197653968267) q[17];
ry(2.5106435288972824) q[18];
rz(-2.522484490350681) q[18];
ry(2.1503935410053456) q[19];
rz(1.2883759731134434) q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.5843820383026657) q[0];
rz(-0.12175019070709059) q[0];
ry(0.10916690072454038) q[1];
rz(1.7450153074460104) q[1];
ry(-0.3399634761836648) q[2];
rz(0.003062496097920539) q[2];
ry(-2.78966910328562) q[3];
rz(0.18430805704131117) q[3];
ry(-1.5996006662444051) q[4];
rz(1.546566563496081) q[4];
ry(-1.5736067374428817) q[5];
rz(-1.5694624828790786) q[5];
ry(-0.000580988812941996) q[6];
rz(1.9872088498111706) q[6];
ry(1.5844533261890792) q[7];
rz(-2.6127288490362033) q[7];
ry(3.059583678206126) q[8];
rz(2.80367576903323) q[8];
ry(3.05119222779128) q[9];
rz(1.9267395514211054) q[9];
ry(-2.1267849673605426) q[10];
rz(-1.6873319998417147) q[10];
ry(-1.9035758941651553) q[11];
rz(-2.5492767611658085) q[11];
ry(-0.8642361429121669) q[12];
rz(1.6554146218249484) q[12];
ry(0.8633200497269424) q[13];
rz(1.1206746939102832) q[13];
ry(1.3251498319584734) q[14];
rz(1.8204708154663374) q[14];
ry(1.5653015831915416) q[15];
rz(-3.0102478340225653) q[15];
ry(-2.339335575457475) q[16];
rz(-2.474509844787841) q[16];
ry(0.9697620417024069) q[17];
rz(2.265099393484634) q[17];
ry(1.16617958975851) q[18];
rz(-2.701932972519049) q[18];
ry(-2.1694583979476323) q[19];
rz(-0.45592043293184176) q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.417172047394614) q[0];
rz(1.2210746420202163) q[0];
ry(-1.555093483949788) q[1];
rz(-2.301099989918938) q[1];
ry(-0.002968416411907886) q[2];
rz(-2.377728323446035) q[2];
ry(-0.0041850008484214385) q[3];
rz(-2.4176106378796254) q[3];
ry(1.5732105158053724) q[4];
rz(-2.3310805645293318) q[4];
ry(-1.622175442014905) q[5];
rz(-1.0675258227791835) q[5];
ry(-0.0073804135596571285) q[6];
rz(1.409161076786964) q[6];
ry(3.134063674844851) q[7];
rz(2.0980224012440036) q[7];
ry(-3.141470278289894) q[8];
rz(-1.943908108797113) q[8];
ry(-3.1410441626509975) q[9];
rz(-1.7012031227882394) q[9];
ry(-1.9323661890114847) q[10];
rz(0.11376615776853692) q[10];
ry(-1.4013402827661918) q[11];
rz(1.2562325065802915) q[11];
ry(3.1217872479914126) q[12];
rz(-2.899394202844135) q[12];
ry(-3.1169062169522173) q[13];
rz(-0.6077107863131888) q[13];
ry(0.23819582738633294) q[14];
rz(-0.7076117315966634) q[14];
ry(2.242680791257981) q[15];
rz(-2.2934591597744234) q[15];
ry(2.1201275783019295) q[16];
rz(0.7286575768934148) q[16];
ry(0.9272183371310352) q[17];
rz(-2.8579132022287967) q[17];
ry(2.8140911149947208) q[18];
rz(3.0534304537846717) q[18];
ry(-2.4905678586377364) q[19];
rz(-2.56771025583817) q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.6019002815636334) q[0];
rz(-0.10772920218118412) q[0];
ry(3.1395105614893573) q[1];
rz(-2.4088652515109983) q[1];
ry(1.3690669366361545) q[2];
rz(0.45756043793693807) q[2];
ry(1.2448396327408373) q[3];
rz(-1.1581123965496058) q[3];
ry(0.004419223100966541) q[4];
rz(-2.3825217614165775) q[4];
ry(3.1116506613656947) q[5];
rz(-2.5344573775456842) q[5];
ry(1.5691936717534372) q[6];
rz(0.15531691744403897) q[6];
ry(-1.572068559320834) q[7];
rz(-1.8420241647814706) q[7];
ry(-0.06994354058978032) q[8];
rz(-0.9169298553469484) q[8];
ry(3.068549054977833) q[9];
rz(-1.6882726100923806) q[9];
ry(-1.8226910377407552) q[10];
rz(2.582727489405656) q[10];
ry(-0.9236328611671987) q[11];
rz(-2.92792125728828) q[11];
ry(1.7578985471168282) q[12];
rz(-1.165216110943601) q[12];
ry(1.3835960375575276) q[13];
rz(-1.1264260890076543) q[13];
ry(-2.0766929207140294) q[14];
rz(0.36918122015667587) q[14];
ry(-2.8418361419931197) q[15];
rz(1.8591384938428657) q[15];
ry(1.2937744421369555) q[16];
rz(-2.1955356090153195) q[16];
ry(-1.3310036623032342) q[17];
rz(1.5465538558105996) q[17];
ry(1.4248301023969896) q[18];
rz(1.681395101073119) q[18];
ry(0.1909296878096498) q[19];
rz(-1.2201874171768128) q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.5141782932163055) q[0];
rz(-1.015307321378197) q[0];
ry(-2.200491928744607) q[1];
rz(1.3947595743302394) q[1];
ry(-0.0034178157042406886) q[2];
rz(2.1458891892920846) q[2];
ry(-0.014261845725014623) q[3];
rz(0.44214160353837517) q[3];
ry(-1.4369982148563922) q[4];
rz(2.436669043959398) q[4];
ry(0.014711388890932932) q[5];
rz(-1.7151835119890275) q[5];
ry(-1.5689835652170823) q[6];
rz(3.133504941203791) q[6];
ry(0.0370947457174573) q[7];
rz(1.8595656029671188) q[7];
ry(0.49617226341306075) q[8];
rz(-0.017453815631387926) q[8];
ry(0.81115511942724) q[9];
rz(-0.8540912796637865) q[9];
ry(1.4412810824409756) q[10];
rz(-1.8232928094047893) q[10];
ry(2.6972258909353184) q[11];
rz(2.2924162529422576) q[11];
ry(0.013003082659357065) q[12];
rz(-2.070311767116598) q[12];
ry(-0.012844035090916086) q[13];
rz(-2.4322892492846027) q[13];
ry(-2.399395970988115) q[14];
rz(0.976318113103571) q[14];
ry(-2.6712479743717257) q[15];
rz(1.251343608877436) q[15];
ry(-2.2527694653046546) q[16];
rz(3.1284526373144543) q[16];
ry(1.3367898289994933) q[17];
rz(-0.5210588594572292) q[17];
ry(2.671877102048359) q[18];
rz(-1.2360999228312755) q[18];
ry(-3.1069167057136724) q[19];
rz(1.780166713424542) q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(2.0749406145385363) q[0];
rz(1.6061994303097125) q[0];
ry(-0.496037784883331) q[1];
rz(-2.0081839798764065) q[1];
ry(0.10941250824021766) q[2];
rz(-0.29855231566015106) q[2];
ry(-0.11732845911201045) q[3];
rz(-0.6859839518488001) q[3];
ry(-0.0028916179036517775) q[4];
rz(2.247207817431544) q[4];
ry(0.57474961968326) q[5];
rz(-2.757600451900132) q[5];
ry(1.635475935237543) q[6];
rz(0.319536954761554) q[6];
ry(-1.571268858108817) q[7];
rz(-0.11204798339238686) q[7];
ry(0.002350133429336504) q[8];
rz(-0.6051164178399896) q[8];
ry(0.004635052710943303) q[9];
rz(2.435447951680314) q[9];
ry(-0.9932831200356561) q[10];
rz(-0.9172482158927853) q[10];
ry(0.42714564348265877) q[11];
rz(-2.0580117743333375) q[11];
ry(-1.6064859781261467) q[12];
rz(1.720964091314552) q[12];
ry(-0.029941942051119774) q[13];
rz(2.544747242572181) q[13];
ry(-1.176863853760337) q[14];
rz(-0.1324408223309048) q[14];
ry(-0.45025470111920135) q[15];
rz(-2.267522087739085) q[15];
ry(0.9989943955913608) q[16];
rz(-1.929717736325928) q[16];
ry(-1.5239821034810839) q[17];
rz(-1.2153490821953443) q[17];
ry(-0.7023984545942916) q[18];
rz(-1.81619780620872) q[18];
ry(-1.2811080787329896) q[19];
rz(0.3081739715043188) q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-0.5608736870558957) q[0];
rz(0.3629336480635184) q[0];
ry(-2.5396517882680727) q[1];
rz(-1.983011558837834) q[1];
ry(-3.129424787602417) q[2];
rz(3.072268494772859) q[2];
ry(3.1294676850380987) q[3];
rz(2.575591000434614) q[3];
ry(0.17043400783600152) q[4];
rz(0.8040720999979649) q[4];
ry(0.00932290744959019) q[5];
rz(2.7612394191005305) q[5];
ry(3.139790769533094) q[6];
rz(1.7538383929989874) q[6];
ry(0.0027445162692281144) q[7];
rz(1.421731186136131) q[7];
ry(-0.00038531718788567765) q[8];
rz(-1.3474045316631693) q[8];
ry(-1.579568221433953) q[9];
rz(-0.06728863351437726) q[9];
ry(3.14060924389918) q[10];
rz(1.110075389060546) q[10];
ry(-3.1334695079695685) q[11];
rz(-1.5725709752368138) q[11];
ry(3.139623291748717) q[12];
rz(-1.399313578436891) q[12];
ry(0.0026321932404202997) q[13];
rz(-2.19586454494603) q[13];
ry(-4.133636890557568e-06) q[14];
rz(-0.03138839581667696) q[14];
ry(-0.007980878204998376) q[15];
rz(-1.5385635695023516) q[15];
ry(-2.0417357245027654) q[16];
rz(-2.844548836276773) q[16];
ry(-2.830044071991019) q[17];
rz(-2.4116930002898327) q[17];
ry(2.560766336595538) q[18];
rz(-3.0832629176302677) q[18];
ry(-1.4109699459424574) q[19];
rz(-0.23653691717000316) q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-2.547045307591006) q[0];
rz(-2.2044137606428498) q[0];
ry(-1.9836212762721526) q[1];
rz(-1.9420155446838816) q[1];
ry(0.05922930752200361) q[2];
rz(-2.2526715933486074) q[2];
ry(-0.05864324465531257) q[3];
rz(1.7364307405166453) q[3];
ry(-3.130579100312588) q[4];
rz(-2.312831513273954) q[4];
ry(0.6392503598922126) q[5];
rz(1.5832895877324629) q[5];
ry(3.1412255591132014) q[6];
rz(-1.7052128770164456) q[6];
ry(0.0006928163120552) q[7];
rz(-2.912203890626289) q[7];
ry(-3.1411474620170687) q[8];
rz(1.1716265561814705) q[8];
ry(-0.0034873551118764423) q[9];
rz(1.6394563030325457) q[9];
ry(-1.5700950020448685) q[10];
rz(-1.5673570211802752) q[10];
ry(-1.5700493619773255) q[11];
rz(1.5987502045771387) q[11];
ry(1.9910028652529528) q[12];
rz(-2.074539426227827) q[12];
ry(-0.47679856792689945) q[13];
rz(-2.10079290469533) q[13];
ry(-1.4851764472927833) q[14];
rz(-0.963398204217814) q[14];
ry(1.5995019539476751) q[15];
rz(2.04761703550192) q[15];
ry(2.134180783506982) q[16];
rz(-0.2198588531675161) q[16];
ry(-0.9404093959344708) q[17];
rz(1.7739173496447291) q[17];
ry(2.763128564139596) q[18];
rz(-3.1040784129565435) q[18];
ry(-1.6412201772322887) q[19];
rz(-1.4194633577137852) q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-0.21631485063504652) q[0];
rz(-0.7571338918540341) q[0];
ry(-2.7090934792082684) q[1];
rz(-3.0615462924177628) q[1];
ry(-3.1277718965528853) q[2];
rz(-0.7905559384875414) q[2];
ry(0.0038539774733727943) q[3];
rz(2.776817280929902) q[3];
ry(3.0174141082650427) q[4];
rz(-0.8036830582644939) q[4];
ry(1.5729807442996566) q[5];
rz(-2.470049336867107) q[5];
ry(-1.5712068109096513) q[6];
rz(1.5407227044119052) q[6];
ry(-0.16429452743883344) q[7];
rz(0.030416053495714954) q[7];
ry(1.5714809229484983) q[8];
rz(1.5550225283501964) q[8];
ry(-1.5670877425591383) q[9];
rz(0.007397715714683173) q[9];
ry(-1.1182028649526616) q[10];
rz(0.6478031482232333) q[10];
ry(2.8951627171819387) q[11];
rz(-1.4095887903861997) q[11];
ry(2.578003116482672) q[12];
rz(-0.5333457513304047) q[12];
ry(-2.577862846381295) q[13];
rz(1.1586371981792152) q[13];
ry(1.2725805670012145) q[14];
rz(-1.6124779199280983) q[14];
ry(2.6244772220893484) q[15];
rz(-1.6514064068078733) q[15];
ry(-1.1415077749678648) q[16];
rz(1.6236761514651512) q[16];
ry(3.016492339067546) q[17];
rz(2.708841049200616) q[17];
ry(-2.4112334499201196) q[18];
rz(-2.7672615261851856) q[18];
ry(-1.2372776552120852) q[19];
rz(2.5427248507349067) q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.5366530612675122) q[0];
rz(1.0856863726123676) q[0];
ry(2.9940985016593413) q[1];
rz(0.7345679946227692) q[1];
ry(0.0005564636969843662) q[2];
rz(1.4513488228690954) q[2];
ry(-3.141276111180433) q[3];
rz(-0.10843844240696043) q[3];
ry(0.004211894625854401) q[4];
rz(0.058900772362166454) q[4];
ry(-3.1386608055989393) q[5];
rz(-0.9384585703755084) q[5];
ry(-1.6158431508948565) q[6];
rz(-2.871634781586037) q[6];
ry(1.5712350646286037) q[7];
rz(-3.1221685285352447) q[7];
ry(1.570343430950061) q[8];
rz(2.5595261933712306) q[8];
ry(1.5709928113013458) q[9];
rz(-2.896434821287711) q[9];
ry(1.0349273042796092) q[10];
rz(1.6914502373651485) q[10];
ry(-1.963984928980759) q[11];
rz(1.48798229039937) q[11];
ry(-0.028301982235561945) q[12];
rz(-3.083375348694946) q[12];
ry(2.7090805526024826) q[13];
rz(-1.5029633960171132) q[13];
ry(2.7257861991847236) q[14];
rz(-1.2032101918750557) q[14];
ry(-1.7410723877736345) q[15];
rz(-0.5432788275858457) q[15];
ry(1.3641213657924034) q[16];
rz(0.8351359256767834) q[16];
ry(2.2081083863087505) q[17];
rz(2.3734530691707256) q[17];
ry(3.0838519497773063) q[18];
rz(-1.1672044277721687) q[18];
ry(-2.844154752204848) q[19];
rz(-2.174540585733684) q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.2327499119182201) q[0];
rz(-1.5475603404329534) q[0];
ry(1.1294857251669423) q[1];
rz(-0.27968330285746656) q[1];
ry(0.08633493720862973) q[2];
rz(-1.0904780719354115) q[2];
ry(0.276069068961994) q[3];
rz(-1.5990503578773145) q[3];
ry(-3.1401723427175816) q[4];
rz(-2.3933880626226607) q[4];
ry(-1.5687309036682802) q[5];
rz(-1.4818038435301526) q[5];
ry(3.1368095816131563) q[6];
rz(-2.1049460812829146) q[6];
ry(-1.4544693337726269) q[7];
rz(1.2546395992150714) q[7];
ry(3.141072386448163) q[8];
rz(-2.254080028732983) q[8];
ry(0.0004903466377825794) q[9];
rz(-2.009937278159745) q[9];
ry(3.1215854524220337) q[10];
rz(-0.6701982802464332) q[10];
ry(-3.1351063669472072) q[11];
rz(2.244738936581806) q[11];
ry(1.6193824199557785) q[12];
rz(-3.1408583837529367) q[12];
ry(-1.5222817424262838) q[13];
rz(0.0013711835540144435) q[13];
ry(-2.173200328004992) q[14];
rz(-2.875400889541074) q[14];
ry(-1.5508607006597797) q[15];
rz(3.108125056251502) q[15];
ry(0.9880238417259654) q[16];
rz(1.0393057545973212) q[16];
ry(1.8663101785464897) q[17];
rz(-0.10402717598261418) q[17];
ry(-2.0855366418747225) q[18];
rz(-0.16263419300020882) q[18];
ry(0.4924916674890157) q[19];
rz(-1.7711560961584025) q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-0.01122651067124103) q[0];
rz(2.796190512311243) q[0];
ry(-3.039435464747465) q[1];
rz(0.8815963384710948) q[1];
ry(1.5634273104813676) q[2];
rz(1.4175906384787889) q[2];
ry(1.4909289531662042) q[3];
rz(-0.12049540472351249) q[3];
ry(-0.0021246985038247956) q[4];
rz(-3.1158514828726633) q[4];
ry(3.1409730938587983) q[5];
rz(2.527065823728145) q[5];
ry(-0.00024319495330877127) q[6];
rz(-2.1372087810543814) q[6];
ry(0.01005566247485934) q[7];
rz(-0.6113919636388875) q[7];
ry(1.5721619717268966) q[8];
rz(-0.0018498389889905927) q[8];
ry(3.141522485744492) q[9];
rz(1.3334361401717127) q[9];
ry(2.3759087325726247) q[10];
rz(2.9041161854470112) q[10];
ry(-2.966569753059111) q[11];
rz(0.8152478977681142) q[11];
ry(-1.5698246974821402) q[12];
rz(-0.19537624906365372) q[12];
ry(-1.5718511541723394) q[13];
rz(2.45795636799789) q[13];
ry(0.15638983954707264) q[14];
rz(-1.9210786266287703) q[14];
ry(1.560015398862011) q[15];
rz(-1.497815161980913) q[15];
ry(-2.943214690197498) q[16];
rz(-0.3052008922435813) q[16];
ry(1.6120382774559185) q[17];
rz(-3.0975270088343887) q[17];
ry(1.406266749676213) q[18];
rz(1.9015898017302908) q[18];
ry(-1.9908073583443595) q[19];
rz(0.7304983587110356) q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(0.039253664128135135) q[0];
rz(-2.521415935770153) q[0];
ry(3.135466158079862) q[1];
rz(1.3110336314039595) q[1];
ry(1.8307432776311063) q[2];
rz(1.482864589105989) q[2];
ry(-3.08877604655301) q[3];
rz(-0.08767699130816729) q[3];
ry(-1.5702812000305724) q[4];
rz(0.9782561037425795) q[4];
ry(-3.139851154527306) q[5];
rz(-1.83383915933889) q[5];
ry(3.1366510392800375) q[6];
rz(0.03127926592578412) q[6];
ry(3.1410359532678465) q[7];
rz(-0.1128989978056598) q[7];
ry(1.5710157762576928) q[8];
rz(0.0005694718986413024) q[8];
ry(-1.570037524867959) q[9];
rz(-3.1413554381749327) q[9];
ry(3.1387503253157645) q[10];
rz(-1.666077055868159) q[10];
ry(-1.6231729332876288) q[11];
rz(-1.5067209715849947) q[11];
ry(3.1414213554329584) q[12];
rz(0.702354862487879) q[12];
ry(-3.1414187090994976) q[13];
rz(-0.832153493192325) q[13];
ry(-0.017264179835147282) q[14];
rz(-1.3880377429234982) q[14];
ry(0.6338201771593992) q[15];
rz(0.703503911243545) q[15];
ry(-0.01677345808441988) q[16];
rz(-2.827421172577809) q[16];
ry(-3.0327080538763487) q[17];
rz(1.5642741649230612) q[17];
ry(0.006783781131401873) q[18];
rz(-0.7851146392703097) q[18];
ry(-1.9456774202687246) q[19];
rz(-1.1990205581434532) q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.7512696828574459) q[0];
rz(2.5092405063768175) q[0];
ry(2.3681851904389446) q[1];
rz(0.37533999121041006) q[1];
ry(1.5708440167575883) q[2];
rz(1.6955139099525829) q[2];
ry(1.5709516941394306) q[3];
rz(-1.5866326006613924) q[3];
ry(-0.0017615509679445244) q[4];
rz(1.6981184235152436) q[4];
ry(-0.00030985267186158254) q[5];
rz(-1.7467977895689808) q[5];
ry(-3.141194468307948) q[6];
rz(-2.720733479448908) q[6];
ry(3.141462817068158) q[7];
rz(2.385677713259833) q[7];
ry(-1.5711901881698493) q[8];
rz(0.06991405782150553) q[8];
ry(-1.5704759190252915) q[9];
rz(-3.0833117471844482) q[9];
ry(-1.5716115750090123) q[10];
rz(-0.045498881311776034) q[10];
ry(0.0029159657820942617) q[11];
rz(-1.6655374469923814) q[11];
ry(-0.5711947058309085) q[12];
rz(-2.885619974154867) q[12];
ry(-0.3101592047454851) q[13];
rz(2.279084131526682) q[13];
ry(-1.7183751552817246) q[14];
rz(1.5492152872198635) q[14];
ry(3.113848602928158) q[15];
rz(-0.6613917568730628) q[15];
ry(2.0141850134139974) q[16];
rz(0.252970463685517) q[16];
ry(1.5966680819485932) q[17];
rz(1.5867646062482625) q[17];
ry(-2.974034186099514) q[18];
rz(-1.3841049829874372) q[18];
ry(-0.021039335573715512) q[19];
rz(-0.5864952367368801) q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.5065939690216146) q[0];
rz(-1.5451958049858583) q[0];
ry(-0.006815337316992576) q[1];
rz(-2.614217036418245) q[1];
ry(-0.030262390118202143) q[2];
rz(-2.61244702792502) q[2];
ry(3.074690487268621) q[3];
rz(1.5826862951216967) q[3];
ry(1.5676445710231877) q[4];
rz(-2.788994212144617) q[4];
ry(1.8132755813560495) q[5];
rz(-1.8352672035833129) q[5];
ry(-0.007075102739087013) q[6];
rz(0.9332838000288043) q[6];
ry(1.4849358141635864) q[7];
rz(0.6679594187914273) q[7];
ry(-0.16586527172122434) q[8];
rz(-2.9183237430824263) q[8];
ry(1.5867789966376078) q[9];
rz(-1.619395296353508) q[9];
ry(3.139287500748856) q[10];
rz(1.531716115065577) q[10];
ry(0.012765057996430684) q[11];
rz(-1.5401646831362859) q[11];
ry(0.0005865345983225128) q[12];
rz(3.1027684755570495) q[12];
ry(3.141002933032794) q[13];
rz(-0.12190724399557594) q[13];
ry(-0.6837505443008366) q[14];
rz(-0.001581827461775376) q[14];
ry(0.0014596681589722491) q[15];
rz(-2.1100020956528516) q[15];
ry(0.029631481056910727) q[16];
rz(-0.9348690614386225) q[16];
ry(-0.10473079383899937) q[17];
rz(-0.4922119280531554) q[17];
ry(1.182194725470751) q[18];
rz(-2.445053771209942) q[18];
ry(3.0358669516159646) q[19];
rz(1.3583922870347225) q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.5439539202104127) q[0];
rz(1.8257659604305057) q[0];
ry(-0.033356460259111397) q[1];
rz(-0.8277460887358375) q[1];
ry(0.0037263840069151053) q[2];
rz(0.30255654821120004) q[2];
ry(0.0151468874143319) q[3];
rz(-0.026898148230254826) q[3];
ry(-3.1410435766909752) q[4];
rz(1.992260578838358) q[4];
ry(3.14147151972202) q[5];
rz(-3.060348965506752) q[5];
ry(3.140346544019299) q[6];
rz(3.068410939924998) q[6];
ry(3.1405933099488883) q[7];
rz(3.0358035455566093) q[7];
ry(0.0002475617396892048) q[8];
rz(-0.21062591712362763) q[8];
ry(-1.5677470416858998) q[9];
rz(-2.8833663704019825) q[9];
ry(-1.5716508774785838) q[10];
rz(1.9301308302310973) q[10];
ry(0.591573128730404) q[11];
rz(0.4095872523842286) q[11];
ry(0.48806828747071496) q[12];
rz(-1.75987816432328) q[12];
ry(-1.3540843648585577) q[13];
rz(-0.17408049531289915) q[13];
ry(-1.601204347182226) q[14];
rz(-3.1388843292593926) q[14];
ry(-0.005200311654378601) q[15];
rz(0.4072419670066354) q[15];
ry(3.13678877953098) q[16];
rz(-1.8175658517182745) q[16];
ry(3.137171205326077) q[17];
rz(1.1429929564680175) q[17];
ry(3.1008027505733287) q[18];
rz(-1.3605725880743587) q[18];
ry(-1.563459978982557) q[19];
rz(0.28082163665306054) q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.539125303987019) q[0];
rz(0.9514612918513958) q[0];
ry(-1.8899964870068775) q[1];
rz(0.8090874919911143) q[1];
ry(0.0003185164649801564) q[2];
rz(2.1412518221347376) q[2];
ry(-1.5711799580782317) q[3];
rz(0.7487622894290578) q[3];
ry(1.159856391245854) q[4];
rz(-0.583196572337064) q[4];
ry(-2.639183801321953) q[5];
rz(-2.0436014249180694) q[5];
ry(1.5382172195155934) q[6];
rz(-2.7833543812798496) q[6];
ry(0.14241168086612715) q[7];
rz(2.507079879806671) q[7];
ry(-1.5323109734071538) q[8];
rz(2.9169814234863956) q[8];
ry(3.1303883596881534) q[9];
rz(-1.5201086097237972) q[9];
ry(0.02342321845634398) q[10];
rz(-1.3883941877760486) q[10];
ry(3.118038385920638) q[11];
rz(1.3804855087351704) q[11];
ry(1.5814416661463548) q[12];
rz(-0.00031162149602614875) q[12];
ry(1.5697919819641923) q[13];
rz(3.132784335221182) q[13];
ry(1.633677444333471) q[14];
rz(1.5706297108313638) q[14];
ry(-0.02158081668955564) q[15];
rz(1.5253566881476373) q[15];
ry(-3.1339359672050207) q[16];
rz(0.9567705615310854) q[16];
ry(-1.748156974257725) q[17];
rz(2.6165884126447496) q[17];
ry(2.2921864475657583) q[18];
rz(2.123606005516375) q[18];
ry(-0.9769341300723653) q[19];
rz(-2.7323061916834677) q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.570867134500797) q[0];
rz(-1.628230626110736) q[0];
ry(1.5704996216500349) q[1];
rz(-1.4319400069154626) q[1];
ry(-3.1384548717005916) q[2];
rz(-1.4567904120854134) q[2];
ry(-0.003418067892374385) q[3];
rz(-0.8302548319879861) q[3];
ry(-3.140431280153095) q[4];
rz(2.36229818156159) q[4];
ry(0.0022034650861950666) q[5];
rz(0.8252400055268596) q[5];
ry(-3.1410075307586856) q[6];
rz(-2.1447466896723952) q[6];
ry(0.0309303637124847) q[7];
rz(0.8462446475820421) q[7];
ry(-3.1410064713133807) q[8];
rz(-1.7994769462692974) q[8];
ry(3.140451886626689) q[9];
rz(-1.7578715415526922) q[9];
ry(-0.0001322159500963928) q[10];
rz(-2.0774004500042036) q[10];
ry(3.1413688669681683) q[11];
rz(1.0214256653789873) q[11];
ry(1.57057619442256) q[12];
rz(-1.044821951115654) q[12];
ry(-1.5802510800406515) q[13];
rz(-2.583153882199525) q[13];
ry(1.5724380441030776) q[14];
rz(-0.6682550218745602) q[14];
ry(-1.5682222709031137) q[15];
rz(2.9918203487502457) q[15];
ry(-3.141292857191576) q[16];
rz(2.6438977832390873) q[16];
ry(-0.0002777284461155105) q[17];
rz(1.141641802466646) q[17];
ry(-3.1287491989259304) q[18];
rz(-1.5054806141466281) q[18];
ry(-3.124854322945856) q[19];
rz(-0.809761180977671) q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.3684450819990959) q[0];
rz(0.011214055897275088) q[0];
ry(-0.4107544889678021) q[1];
rz(-0.12286883184570076) q[1];
ry(-3.084282450671909) q[2];
rz(1.7292995086342122) q[2];
ry(1.6248455800774044) q[3];
rz(1.5744558241248987) q[3];
ry(0.04765563677646424) q[4];
rz(-2.916376240496207) q[4];
ry(-1.5876403850074772) q[5];
rz(-2.932442170750758) q[5];
ry(-0.050354422899508755) q[6];
rz(-0.6374046025568648) q[6];
ry(-3.047903474222066) q[7];
rz(-0.5794113649811464) q[7];
ry(1.6884679808422076) q[8];
rz(1.6091371269723993) q[8];
ry(-1.5332541911008493) q[9];
rz(1.551490929829133) q[9];
ry(-1.578680117631551) q[10];
rz(0.0005878289163774695) q[10];
ry(-1.5912950168743094) q[11];
rz(-0.015757828291066645) q[11];
ry(-1.7681422896561205) q[12];
rz(1.474982193865693) q[12];
ry(-1.9362979859269567) q[13];
rz(1.5539955908114802) q[13];
ry(-1.6044402056780545) q[14];
rz(-1.3719987973532561) q[14];
ry(-2.7420358517823575) q[15];
rz(-1.7728087309599312) q[15];
ry(0.9928770692378471) q[16];
rz(-2.7566277332302422) q[16];
ry(-1.9656948187770027) q[17];
rz(1.5044177697204104) q[17];
ry(0.09236982107482207) q[18];
rz(-2.7576399166936323) q[18];
ry(-0.7485452440103995) q[19];
rz(-0.3192144990737294) q[19];