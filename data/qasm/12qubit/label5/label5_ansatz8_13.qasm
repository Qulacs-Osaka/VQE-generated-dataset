OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(2.0918562592631407) q[0];
ry(-2.187755650398091) q[1];
cx q[0],q[1];
ry(2.707636712466816) q[0];
ry(-0.635216870142686) q[1];
cx q[0],q[1];
ry(1.9001995943210075) q[2];
ry(-1.7608807468107708) q[3];
cx q[2],q[3];
ry(0.7601679999626388) q[2];
ry(-1.8518822240949966) q[3];
cx q[2],q[3];
ry(-1.2924668482511146) q[4];
ry(-1.386912179080884) q[5];
cx q[4],q[5];
ry(0.16393897585838424) q[4];
ry(-0.8331959777097868) q[5];
cx q[4],q[5];
ry(0.4769612618850824) q[6];
ry(-0.9618017910078632) q[7];
cx q[6],q[7];
ry(-1.1191418550003254) q[6];
ry(1.2474514604188514) q[7];
cx q[6],q[7];
ry(1.792433626501972) q[8];
ry(1.75867930936887) q[9];
cx q[8],q[9];
ry(0.03076794301252841) q[8];
ry(2.4509386073506336) q[9];
cx q[8],q[9];
ry(1.7397559875047506) q[10];
ry(1.9365900046153728) q[11];
cx q[10],q[11];
ry(0.7577295779035953) q[10];
ry(0.6677980339447798) q[11];
cx q[10],q[11];
ry(2.3721897107592698) q[0];
ry(0.3512737323533436) q[2];
cx q[0],q[2];
ry(2.200587079538645) q[0];
ry(-0.8821555682376362) q[2];
cx q[0],q[2];
ry(0.29140220166895325) q[2];
ry(-0.17916490034018118) q[4];
cx q[2],q[4];
ry(0.9101271769658529) q[2];
ry(-1.2701295896222147) q[4];
cx q[2],q[4];
ry(-2.9241857242876015) q[4];
ry(-0.772938956032827) q[6];
cx q[4],q[6];
ry(0.5204927313628043) q[4];
ry(-0.3732074611055607) q[6];
cx q[4],q[6];
ry(-1.1166260350731416) q[6];
ry(-0.12958642306474585) q[8];
cx q[6],q[8];
ry(-2.011942914852896) q[6];
ry(0.6855331001826275) q[8];
cx q[6],q[8];
ry(0.3354269877759393) q[8];
ry(-2.55884046170174) q[10];
cx q[8],q[10];
ry(2.3838864299677938) q[8];
ry(1.6687272028345814) q[10];
cx q[8],q[10];
ry(-0.11176055493867934) q[1];
ry(0.7749388780446163) q[3];
cx q[1],q[3];
ry(1.8335700856200705) q[1];
ry(0.639192642986953) q[3];
cx q[1],q[3];
ry(3.108535003539074) q[3];
ry(-2.17292523690515) q[5];
cx q[3],q[5];
ry(2.127088010597353) q[3];
ry(1.1356050493847123) q[5];
cx q[3],q[5];
ry(3.0920640120745824) q[5];
ry(-2.2374189574006627) q[7];
cx q[5],q[7];
ry(-1.867972482305325) q[5];
ry(-2.1392140785731906) q[7];
cx q[5],q[7];
ry(0.921344162546112) q[7];
ry(-0.9673315547538692) q[9];
cx q[7],q[9];
ry(-2.1193850444254636) q[7];
ry(-0.7867280013992017) q[9];
cx q[7],q[9];
ry(-0.837325649497435) q[9];
ry(1.8936227458934252) q[11];
cx q[9],q[11];
ry(-2.0062434370350695) q[9];
ry(0.8817703558726144) q[11];
cx q[9],q[11];
ry(0.5017486863732881) q[0];
ry(-1.1873136255132104) q[1];
cx q[0],q[1];
ry(1.50103468824472) q[0];
ry(-3.1398018184928698) q[1];
cx q[0],q[1];
ry(1.0137371077739956) q[2];
ry(2.7833459730015395) q[3];
cx q[2],q[3];
ry(-1.6470903232701053) q[2];
ry(-2.160217840235779) q[3];
cx q[2],q[3];
ry(0.3231334128080876) q[4];
ry(-1.6409243745524362) q[5];
cx q[4],q[5];
ry(-2.8695759644254806) q[4];
ry(-1.739790592525596) q[5];
cx q[4],q[5];
ry(0.682130156775929) q[6];
ry(1.5658526418660306) q[7];
cx q[6],q[7];
ry(-1.437892334280359) q[6];
ry(-0.8152218660598081) q[7];
cx q[6],q[7];
ry(-0.9360141345072077) q[8];
ry(-1.3504053971161054) q[9];
cx q[8],q[9];
ry(-2.1847841859318216) q[8];
ry(-3.0570033295560806) q[9];
cx q[8],q[9];
ry(0.5027401480804218) q[10];
ry(-3.1353964372972856) q[11];
cx q[10],q[11];
ry(-1.3639378319595261) q[10];
ry(1.459881626723532) q[11];
cx q[10],q[11];
ry(2.368358487725712) q[0];
ry(-1.1864449638744672) q[2];
cx q[0],q[2];
ry(1.9561027415289212) q[0];
ry(-1.5580902647973727) q[2];
cx q[0],q[2];
ry(-1.6780285849872694) q[2];
ry(0.9631055566568323) q[4];
cx q[2],q[4];
ry(-0.9000320655312666) q[2];
ry(0.9244678107751104) q[4];
cx q[2],q[4];
ry(-0.13300686063639786) q[4];
ry(-1.8045605841298071) q[6];
cx q[4],q[6];
ry(-1.9725679249359618) q[4];
ry(0.5937887511403098) q[6];
cx q[4],q[6];
ry(1.7920653899383494) q[6];
ry(-0.8379989477358665) q[8];
cx q[6],q[8];
ry(-2.0675069236920445) q[6];
ry(2.5183231208208827) q[8];
cx q[6],q[8];
ry(-2.810418607147619) q[8];
ry(-2.87784648206248) q[10];
cx q[8],q[10];
ry(1.8052400172691336) q[8];
ry(1.3109523943927028) q[10];
cx q[8],q[10];
ry(-0.6892150250546045) q[1];
ry(-0.21301519076060624) q[3];
cx q[1],q[3];
ry(1.729121107539112) q[1];
ry(0.5048883823485415) q[3];
cx q[1],q[3];
ry(-0.3151486000721228) q[3];
ry(0.7332704109947729) q[5];
cx q[3],q[5];
ry(-0.16054003320186538) q[3];
ry(-1.3129720233472435) q[5];
cx q[3],q[5];
ry(3.1342712550257774) q[5];
ry(2.5994315247317523) q[7];
cx q[5],q[7];
ry(-1.100364006909448) q[5];
ry(-1.181421715984892) q[7];
cx q[5],q[7];
ry(-0.5335057329352981) q[7];
ry(0.9752230590889699) q[9];
cx q[7],q[9];
ry(2.0487135139664336) q[7];
ry(2.3030444115612023) q[9];
cx q[7],q[9];
ry(-0.3828473079516028) q[9];
ry(-1.6332887354769996) q[11];
cx q[9],q[11];
ry(2.3889702462378097) q[9];
ry(3.0062658922754597) q[11];
cx q[9],q[11];
ry(-2.3664262303725643) q[0];
ry(1.4094950687570822) q[1];
cx q[0],q[1];
ry(1.1214061631848955) q[0];
ry(1.2315798400617062) q[1];
cx q[0],q[1];
ry(2.218386389119461) q[2];
ry(-1.9403351511551445) q[3];
cx q[2],q[3];
ry(1.0604207034092235) q[2];
ry(2.5440168212477157) q[3];
cx q[2],q[3];
ry(-2.920270264286351) q[4];
ry(2.7647067228240054) q[5];
cx q[4],q[5];
ry(-0.4875264721682364) q[4];
ry(2.8174640953698495) q[5];
cx q[4],q[5];
ry(2.61021220656269) q[6];
ry(-2.6647564462153923) q[7];
cx q[6],q[7];
ry(1.5185025110487846) q[6];
ry(1.3760408203163124) q[7];
cx q[6],q[7];
ry(-0.3596317865992553) q[8];
ry(-0.6958169040265477) q[9];
cx q[8],q[9];
ry(0.13918422118122198) q[8];
ry(1.5502017706218125) q[9];
cx q[8],q[9];
ry(-2.735478690125574) q[10];
ry(1.555231075644875) q[11];
cx q[10],q[11];
ry(1.8245314609691774) q[10];
ry(0.8725354297609793) q[11];
cx q[10],q[11];
ry(-0.5333274508936847) q[0];
ry(1.3957187779629878) q[2];
cx q[0],q[2];
ry(2.352391096034795) q[0];
ry(-0.7045352368095301) q[2];
cx q[0],q[2];
ry(-1.0653713529077917) q[2];
ry(-2.166290721609997) q[4];
cx q[2],q[4];
ry(-1.465293795607857) q[2];
ry(1.9764685776693627) q[4];
cx q[2],q[4];
ry(2.7471810234379053) q[4];
ry(-2.2483208600196165) q[6];
cx q[4],q[6];
ry(-1.72366732002331) q[4];
ry(1.6824209322904826) q[6];
cx q[4],q[6];
ry(-1.1272603648358943) q[6];
ry(-1.7677880747510806) q[8];
cx q[6],q[8];
ry(-0.5141080988353579) q[6];
ry(2.2742231102183093) q[8];
cx q[6],q[8];
ry(2.800351207191746) q[8];
ry(1.553511507362039) q[10];
cx q[8],q[10];
ry(-0.5382024270952597) q[8];
ry(-2.019079143280453) q[10];
cx q[8],q[10];
ry(2.095539308958975) q[1];
ry(1.4247944596240252) q[3];
cx q[1],q[3];
ry(1.9162995907807958) q[1];
ry(0.45324353667130374) q[3];
cx q[1],q[3];
ry(1.2615430605044065) q[3];
ry(1.0596142839580907) q[5];
cx q[3],q[5];
ry(2.308894112886129) q[3];
ry(2.5536410926555213) q[5];
cx q[3],q[5];
ry(-0.4939048989426906) q[5];
ry(-3.000907853551525) q[7];
cx q[5],q[7];
ry(2.257859741041531) q[5];
ry(-2.0501220426022955) q[7];
cx q[5],q[7];
ry(-0.14130205375284652) q[7];
ry(0.5410336414354157) q[9];
cx q[7],q[9];
ry(-1.9590468986442249) q[7];
ry(-2.8116597995090027) q[9];
cx q[7],q[9];
ry(-0.012258419661247019) q[9];
ry(-2.648613587880949) q[11];
cx q[9],q[11];
ry(0.6108236207481728) q[9];
ry(-1.885755281257583) q[11];
cx q[9],q[11];
ry(1.6087909228377084) q[0];
ry(2.4106038943411634) q[1];
cx q[0],q[1];
ry(-0.6907919636538207) q[0];
ry(1.5629879053270792) q[1];
cx q[0],q[1];
ry(-1.8806043375775505) q[2];
ry(-2.339356415829558) q[3];
cx q[2],q[3];
ry(-2.125424004095146) q[2];
ry(0.9139126159548886) q[3];
cx q[2],q[3];
ry(1.1374147950712434) q[4];
ry(-2.573125900200755) q[5];
cx q[4],q[5];
ry(-2.4379726132235895) q[4];
ry(1.2050568664714727) q[5];
cx q[4],q[5];
ry(-1.9329656513428386) q[6];
ry(-2.885730560931231) q[7];
cx q[6],q[7];
ry(2.720536464981604) q[6];
ry(2.0759034047287894) q[7];
cx q[6],q[7];
ry(0.0636002061268055) q[8];
ry(3.0386401714330655) q[9];
cx q[8],q[9];
ry(-1.4831848129243248) q[8];
ry(0.9682725037246298) q[9];
cx q[8],q[9];
ry(-0.8420682955250216) q[10];
ry(1.4622817241256865) q[11];
cx q[10],q[11];
ry(0.5939983278828018) q[10];
ry(2.937037027485083) q[11];
cx q[10],q[11];
ry(1.6986449618934891) q[0];
ry(-3.0579834964287054) q[2];
cx q[0],q[2];
ry(1.2144489985559028) q[0];
ry(-0.4123196095343813) q[2];
cx q[0],q[2];
ry(-2.0177546976007923) q[2];
ry(1.2812221756975384) q[4];
cx q[2],q[4];
ry(1.4900006652047555) q[2];
ry(1.512601233949617) q[4];
cx q[2],q[4];
ry(-1.5294962400902752) q[4];
ry(1.9512856295191252) q[6];
cx q[4],q[6];
ry(2.8095710018444726) q[4];
ry(2.09225196106075) q[6];
cx q[4],q[6];
ry(2.065674940550316) q[6];
ry(0.11826867509154661) q[8];
cx q[6],q[8];
ry(-2.568908755271478) q[6];
ry(2.599019102883261) q[8];
cx q[6],q[8];
ry(1.020111338627831) q[8];
ry(-0.74779473777578) q[10];
cx q[8],q[10];
ry(1.851672167254823) q[8];
ry(2.6718349661699) q[10];
cx q[8],q[10];
ry(-0.3825210955246119) q[1];
ry(2.197943691316862) q[3];
cx q[1],q[3];
ry(2.013244742871937) q[1];
ry(2.093297359641383) q[3];
cx q[1],q[3];
ry(-0.8318420709914847) q[3];
ry(3.1078190177737346) q[5];
cx q[3],q[5];
ry(2.9061923115931174) q[3];
ry(1.715735671399641) q[5];
cx q[3],q[5];
ry(1.2593657781850274) q[5];
ry(-1.2345332604965584) q[7];
cx q[5],q[7];
ry(1.7059369482848297) q[5];
ry(-2.0247081582123063) q[7];
cx q[5],q[7];
ry(2.0809543297305737) q[7];
ry(-1.3174643208937662) q[9];
cx q[7],q[9];
ry(-0.6536548760579252) q[7];
ry(1.7579729769007162) q[9];
cx q[7],q[9];
ry(2.596461169676797) q[9];
ry(2.0987377841305284) q[11];
cx q[9],q[11];
ry(-1.829874462057103) q[9];
ry(-0.882464301360732) q[11];
cx q[9],q[11];
ry(-0.10045885887631201) q[0];
ry(2.002281290692244) q[1];
cx q[0],q[1];
ry(-1.6165208213720155) q[0];
ry(-2.405257607760315) q[1];
cx q[0],q[1];
ry(-0.3661077556980844) q[2];
ry(-0.6876729522672589) q[3];
cx q[2],q[3];
ry(1.4771844708802508) q[2];
ry(-1.350549378778052) q[3];
cx q[2],q[3];
ry(-2.1627721052069835) q[4];
ry(-2.6850995020199933) q[5];
cx q[4],q[5];
ry(-2.5061360600896445) q[4];
ry(2.024732589261097) q[5];
cx q[4],q[5];
ry(-1.3177017547117558) q[6];
ry(2.4059088602478105) q[7];
cx q[6],q[7];
ry(-0.6881674931608472) q[6];
ry(2.1014427119062757) q[7];
cx q[6],q[7];
ry(-2.3938929947637804) q[8];
ry(1.3757467657703044) q[9];
cx q[8],q[9];
ry(-1.9687554279799568) q[8];
ry(-1.037442075853635) q[9];
cx q[8],q[9];
ry(0.22046112227953876) q[10];
ry(-1.4807784463271316) q[11];
cx q[10],q[11];
ry(-0.29491368166563625) q[10];
ry(-0.5724400279928983) q[11];
cx q[10],q[11];
ry(0.9411356617473162) q[0];
ry(-2.0667330987284247) q[2];
cx q[0],q[2];
ry(2.594562355495327) q[0];
ry(-1.1550021889686175) q[2];
cx q[0],q[2];
ry(-0.6744738261444709) q[2];
ry(2.5766076651588192) q[4];
cx q[2],q[4];
ry(-0.49151450267975094) q[2];
ry(-2.1833852855442757) q[4];
cx q[2],q[4];
ry(1.0663938843799343) q[4];
ry(0.5118610449693977) q[6];
cx q[4],q[6];
ry(1.2114258774117679) q[4];
ry(2.6682639153465675) q[6];
cx q[4],q[6];
ry(1.1147038000605398) q[6];
ry(0.5268660078010512) q[8];
cx q[6],q[8];
ry(2.052223974720441) q[6];
ry(-0.7145353783210204) q[8];
cx q[6],q[8];
ry(2.288911651053195) q[8];
ry(1.4798593993090714) q[10];
cx q[8],q[10];
ry(0.6337905934657778) q[8];
ry(1.8521245496981136) q[10];
cx q[8],q[10];
ry(-2.578235026735517) q[1];
ry(-1.8546564725661092) q[3];
cx q[1],q[3];
ry(2.3318087455363954) q[1];
ry(1.9328862573199883) q[3];
cx q[1],q[3];
ry(-0.2959189736091308) q[3];
ry(-1.1329803814667012) q[5];
cx q[3],q[5];
ry(1.859385165324156) q[3];
ry(2.786711012316696) q[5];
cx q[3],q[5];
ry(0.22282124405299855) q[5];
ry(-0.02598131063502329) q[7];
cx q[5],q[7];
ry(-3.052607572947428) q[5];
ry(0.19319062439866097) q[7];
cx q[5],q[7];
ry(0.7579747320521387) q[7];
ry(1.9175050060327505) q[9];
cx q[7],q[9];
ry(-1.1237216713741667) q[7];
ry(0.7335908896845494) q[9];
cx q[7],q[9];
ry(-1.1980621925859216) q[9];
ry(2.9895535101734687) q[11];
cx q[9],q[11];
ry(-2.0262648568089188) q[9];
ry(-2.129812157081778) q[11];
cx q[9],q[11];
ry(2.322056170569638) q[0];
ry(-2.8721266103529324) q[1];
cx q[0],q[1];
ry(0.5235303473879398) q[0];
ry(1.3479516110898775) q[1];
cx q[0],q[1];
ry(1.9277559415686278) q[2];
ry(2.9582197324092343) q[3];
cx q[2],q[3];
ry(1.1934886751130946) q[2];
ry(-1.9873489217304432) q[3];
cx q[2],q[3];
ry(1.3903195672471096) q[4];
ry(0.035746112577859535) q[5];
cx q[4],q[5];
ry(1.1955846554909577) q[4];
ry(1.3047331468834722) q[5];
cx q[4],q[5];
ry(2.486675101735351) q[6];
ry(2.0280366862869026) q[7];
cx q[6],q[7];
ry(0.9640604093835369) q[6];
ry(-2.4815513647147576) q[7];
cx q[6],q[7];
ry(2.495950944806729) q[8];
ry(0.7711692423667209) q[9];
cx q[8],q[9];
ry(1.8160039130083963) q[8];
ry(-1.9767121788881292) q[9];
cx q[8],q[9];
ry(0.3252449636028975) q[10];
ry(2.1652432169484284) q[11];
cx q[10],q[11];
ry(-1.5485938312568717) q[10];
ry(2.071922326835502) q[11];
cx q[10],q[11];
ry(-1.6149167907002298) q[0];
ry(-2.6070658896117145) q[2];
cx q[0],q[2];
ry(0.846474642306186) q[0];
ry(-2.2702417564575175) q[2];
cx q[0],q[2];
ry(-3.135865300599963) q[2];
ry(-0.7140852456277882) q[4];
cx q[2],q[4];
ry(-2.870225018311487) q[2];
ry(-1.7864401146671696) q[4];
cx q[2],q[4];
ry(2.549829555214657) q[4];
ry(1.2597225635070004) q[6];
cx q[4],q[6];
ry(0.6571695820311901) q[4];
ry(-2.715467509332925) q[6];
cx q[4],q[6];
ry(-2.070441551337539) q[6];
ry(-1.998777849247058) q[8];
cx q[6],q[8];
ry(-2.115097538268212) q[6];
ry(-1.167417223050101) q[8];
cx q[6],q[8];
ry(-2.3917917207179324) q[8];
ry(-2.7364434606435633) q[10];
cx q[8],q[10];
ry(0.27995396219970115) q[8];
ry(-0.17263817619570818) q[10];
cx q[8],q[10];
ry(-0.37724035762335184) q[1];
ry(1.270033273780846) q[3];
cx q[1],q[3];
ry(-0.4131894294463717) q[1];
ry(2.3469650467300105) q[3];
cx q[1],q[3];
ry(2.303539906046809) q[3];
ry(-2.9450867650413106) q[5];
cx q[3],q[5];
ry(-0.9309709711905763) q[3];
ry(0.8192923542420675) q[5];
cx q[3],q[5];
ry(2.0175922422808563) q[5];
ry(0.10190145179001053) q[7];
cx q[5],q[7];
ry(0.9432286445309135) q[5];
ry(0.6576203443688939) q[7];
cx q[5],q[7];
ry(2.5305419928223296) q[7];
ry(2.8009361550339316) q[9];
cx q[7],q[9];
ry(2.3946253329361173) q[7];
ry(1.0308309885202114) q[9];
cx q[7],q[9];
ry(1.5292060743506302) q[9];
ry(0.2637150086122605) q[11];
cx q[9],q[11];
ry(-1.1370074577719123) q[9];
ry(-0.36111793049063273) q[11];
cx q[9],q[11];
ry(2.7998638234951874) q[0];
ry(-1.706963604518557) q[1];
cx q[0],q[1];
ry(2.2837500721995267) q[0];
ry(1.5308671242380987) q[1];
cx q[0],q[1];
ry(-1.7870749802912336) q[2];
ry(-0.6255150238073544) q[3];
cx q[2],q[3];
ry(-2.5623810249864056) q[2];
ry(-1.5658026921632242) q[3];
cx q[2],q[3];
ry(0.5243343569555166) q[4];
ry(3.054130174976488) q[5];
cx q[4],q[5];
ry(0.25979589772267975) q[4];
ry(-0.4315635383288602) q[5];
cx q[4],q[5];
ry(1.6228140472898769) q[6];
ry(1.1077609508012314) q[7];
cx q[6],q[7];
ry(1.4662930039530788) q[6];
ry(-0.367185021144637) q[7];
cx q[6],q[7];
ry(0.19543327577952938) q[8];
ry(2.5113962426976078) q[9];
cx q[8],q[9];
ry(-1.2836051229961951) q[8];
ry(-1.2891043068580874) q[9];
cx q[8],q[9];
ry(0.06983868070733744) q[10];
ry(1.3527116002448647) q[11];
cx q[10],q[11];
ry(-0.7812726646295268) q[10];
ry(-2.6961122494266956) q[11];
cx q[10],q[11];
ry(1.177788992363202) q[0];
ry(-1.4237040645620862) q[2];
cx q[0],q[2];
ry(-1.408707608005613) q[0];
ry(-2.558608719807788) q[2];
cx q[0],q[2];
ry(0.9075690231983886) q[2];
ry(-1.605918959935862) q[4];
cx q[2],q[4];
ry(0.9769809652745645) q[2];
ry(1.2561027101541258) q[4];
cx q[2],q[4];
ry(2.647407253855527) q[4];
ry(-0.6449714616666757) q[6];
cx q[4],q[6];
ry(2.7781541056376367) q[4];
ry(-0.8680292145868529) q[6];
cx q[4],q[6];
ry(-1.5488013800874683) q[6];
ry(-1.4942039200757082) q[8];
cx q[6],q[8];
ry(-0.9657206937601609) q[6];
ry(-0.7146143852771144) q[8];
cx q[6],q[8];
ry(-2.2967118813636227) q[8];
ry(0.1727594865621969) q[10];
cx q[8],q[10];
ry(-0.7151359295634349) q[8];
ry(0.558160477914124) q[10];
cx q[8],q[10];
ry(2.893499401881545) q[1];
ry(0.26324457537517115) q[3];
cx q[1],q[3];
ry(-0.9592705255688756) q[1];
ry(0.6058034782431188) q[3];
cx q[1],q[3];
ry(0.3539200441761565) q[3];
ry(1.966957791191505) q[5];
cx q[3],q[5];
ry(-2.411974002625977) q[3];
ry(-1.9249142402257917) q[5];
cx q[3],q[5];
ry(-0.02242887764766888) q[5];
ry(0.6474078773927037) q[7];
cx q[5],q[7];
ry(-0.29911470473121854) q[5];
ry(-0.28650559833085687) q[7];
cx q[5],q[7];
ry(1.4248896317644828) q[7];
ry(0.3897058916567833) q[9];
cx q[7],q[9];
ry(2.1640756837631487) q[7];
ry(2.285519872448628) q[9];
cx q[7],q[9];
ry(-0.08410978656209345) q[9];
ry(-3.1157966335335057) q[11];
cx q[9],q[11];
ry(0.9881314671000121) q[9];
ry(0.4734019115624424) q[11];
cx q[9],q[11];
ry(2.3719966832416115) q[0];
ry(-1.0204646417223748) q[1];
cx q[0],q[1];
ry(-1.7136490486742622) q[0];
ry(-2.3602275069107774) q[1];
cx q[0],q[1];
ry(0.17395591775608693) q[2];
ry(1.3937830126530546) q[3];
cx q[2],q[3];
ry(-0.5153432733915069) q[2];
ry(1.5949920326184694) q[3];
cx q[2],q[3];
ry(-2.9916882036630925) q[4];
ry(-0.7732737549918083) q[5];
cx q[4],q[5];
ry(-2.8869048603159584) q[4];
ry(3.0298751198983735) q[5];
cx q[4],q[5];
ry(0.20483215752246048) q[6];
ry(-0.9497619055182334) q[7];
cx q[6],q[7];
ry(-2.191455926043006) q[6];
ry(1.5848866099572003) q[7];
cx q[6],q[7];
ry(-1.8320235563367693) q[8];
ry(3.0205799568181684) q[9];
cx q[8],q[9];
ry(-0.6867567366686259) q[8];
ry(-1.406872583920337) q[9];
cx q[8],q[9];
ry(1.940154618317977) q[10];
ry(-0.6650650310325948) q[11];
cx q[10],q[11];
ry(-1.378287917689744) q[10];
ry(-1.68125515845498) q[11];
cx q[10],q[11];
ry(0.039097006776495746) q[0];
ry(2.9584913689427403) q[2];
cx q[0],q[2];
ry(1.8238338375734093) q[0];
ry(-0.6148418241938245) q[2];
cx q[0],q[2];
ry(1.9321585956732972) q[2];
ry(-0.7754430378945699) q[4];
cx q[2],q[4];
ry(-0.7307835417669661) q[2];
ry(-0.1864489331116629) q[4];
cx q[2],q[4];
ry(-3.0044867844835235) q[4];
ry(0.4959301240236211) q[6];
cx q[4],q[6];
ry(0.7489941510803263) q[4];
ry(2.331883856984027) q[6];
cx q[4],q[6];
ry(-2.3790342576828003) q[6];
ry(-1.5969060206408838) q[8];
cx q[6],q[8];
ry(0.36323990019415325) q[6];
ry(1.4170865588107313) q[8];
cx q[6],q[8];
ry(-1.2257180571589432) q[8];
ry(3.0310860019039767) q[10];
cx q[8],q[10];
ry(0.4493985307300727) q[8];
ry(2.4343471277059883) q[10];
cx q[8],q[10];
ry(2.3896691361926803) q[1];
ry(-2.2028619022522147) q[3];
cx q[1],q[3];
ry(0.5701540975084214) q[1];
ry(2.452739157037458) q[3];
cx q[1],q[3];
ry(2.345972028162736) q[3];
ry(-0.95632362444953) q[5];
cx q[3],q[5];
ry(-0.1415921140155239) q[3];
ry(0.3595512629741186) q[5];
cx q[3],q[5];
ry(-0.23180837087462985) q[5];
ry(0.23750200579106961) q[7];
cx q[5],q[7];
ry(-2.7426724080628944) q[5];
ry(0.9489554395008619) q[7];
cx q[5],q[7];
ry(3.054843840372382) q[7];
ry(-2.190528551473717) q[9];
cx q[7],q[9];
ry(3.0071488130068427) q[7];
ry(-2.683016902690024) q[9];
cx q[7],q[9];
ry(1.2831985566812305) q[9];
ry(1.4815776929604358) q[11];
cx q[9],q[11];
ry(1.8310619130550132) q[9];
ry(-1.3010160052602968) q[11];
cx q[9],q[11];
ry(-2.480596586037882) q[0];
ry(2.0639688797628053) q[1];
cx q[0],q[1];
ry(2.748622955144762) q[0];
ry(-2.75239264053169) q[1];
cx q[0],q[1];
ry(-0.9831881235584107) q[2];
ry(1.140999210715626) q[3];
cx q[2],q[3];
ry(1.579451857675888) q[2];
ry(1.1898112126708025) q[3];
cx q[2],q[3];
ry(0.1552796188663743) q[4];
ry(-2.0020229389676567) q[5];
cx q[4],q[5];
ry(0.6236129938078666) q[4];
ry(2.6915385744755405) q[5];
cx q[4],q[5];
ry(-1.5803042812423067) q[6];
ry(1.642372494683741) q[7];
cx q[6],q[7];
ry(-2.821695287488867) q[6];
ry(1.0857528029567538) q[7];
cx q[6],q[7];
ry(-0.2478594386992334) q[8];
ry(0.6755389435419613) q[9];
cx q[8],q[9];
ry(0.8667200954346086) q[8];
ry(-0.7973125779541307) q[9];
cx q[8],q[9];
ry(-1.7985972017485574) q[10];
ry(-1.7772946625290835) q[11];
cx q[10],q[11];
ry(2.6434797599881317) q[10];
ry(1.9684753346257908) q[11];
cx q[10],q[11];
ry(-0.8657486322870012) q[0];
ry(0.5925934379768895) q[2];
cx q[0],q[2];
ry(2.550209629758172) q[0];
ry(2.996529225032831) q[2];
cx q[0],q[2];
ry(1.5839805599155892) q[2];
ry(-1.4761061339665147) q[4];
cx q[2],q[4];
ry(-1.3249635636680932) q[2];
ry(-1.9709002366404986) q[4];
cx q[2],q[4];
ry(2.9185460068032545) q[4];
ry(0.01580261957658801) q[6];
cx q[4],q[6];
ry(0.6761371641391029) q[4];
ry(0.10130171641990893) q[6];
cx q[4],q[6];
ry(-2.0056108302575373) q[6];
ry(2.823106509907384) q[8];
cx q[6],q[8];
ry(-0.22316846145127742) q[6];
ry(-0.4371073601241493) q[8];
cx q[6],q[8];
ry(-2.5247734041438727) q[8];
ry(2.627322859480171) q[10];
cx q[8],q[10];
ry(0.38203042859185476) q[8];
ry(-2.7384155357518263) q[10];
cx q[8],q[10];
ry(1.5319690581812084) q[1];
ry(1.6396099401762312) q[3];
cx q[1],q[3];
ry(0.39971521144488387) q[1];
ry(-0.25447413410243797) q[3];
cx q[1],q[3];
ry(3.0833787206699976) q[3];
ry(-0.7103740774496979) q[5];
cx q[3],q[5];
ry(2.6610791368228455) q[3];
ry(2.113320342588426) q[5];
cx q[3],q[5];
ry(2.7567356732561956) q[5];
ry(-1.9739434690560662) q[7];
cx q[5],q[7];
ry(-1.8409001828198583) q[5];
ry(2.5207660530943405) q[7];
cx q[5],q[7];
ry(2.084767307367043) q[7];
ry(2.8652338286641097) q[9];
cx q[7],q[9];
ry(0.7312154338161827) q[7];
ry(0.9370185633998216) q[9];
cx q[7],q[9];
ry(0.4981901390031149) q[9];
ry(1.593084876212898) q[11];
cx q[9],q[11];
ry(-2.2644325046024143) q[9];
ry(-1.2645741652597877) q[11];
cx q[9],q[11];
ry(-2.5865759243647135) q[0];
ry(-1.5989971943456638) q[1];
cx q[0],q[1];
ry(-0.24843595225597515) q[0];
ry(2.723113588024426) q[1];
cx q[0],q[1];
ry(-1.232393105991524) q[2];
ry(1.820701927125997) q[3];
cx q[2],q[3];
ry(1.5438018877115922) q[2];
ry(1.0761299264144961) q[3];
cx q[2],q[3];
ry(-2.3491484023392077) q[4];
ry(1.2978395566256031) q[5];
cx q[4],q[5];
ry(-0.25975791934259984) q[4];
ry(0.5253719598399541) q[5];
cx q[4],q[5];
ry(1.5182931362579646) q[6];
ry(0.31343534392592254) q[7];
cx q[6],q[7];
ry(-1.544711025137703) q[6];
ry(2.0048320521337297) q[7];
cx q[6],q[7];
ry(-1.5367284139030384) q[8];
ry(1.88489368151465) q[9];
cx q[8],q[9];
ry(0.4636634503316568) q[8];
ry(-1.129974223368457) q[9];
cx q[8],q[9];
ry(-1.0542688637886564) q[10];
ry(-0.6955650445342778) q[11];
cx q[10],q[11];
ry(-1.9676641333424332) q[10];
ry(2.609910040073908) q[11];
cx q[10],q[11];
ry(-0.1271297264875176) q[0];
ry(3.134650768236195) q[2];
cx q[0],q[2];
ry(0.3212030473497166) q[0];
ry(0.5984714945791447) q[2];
cx q[0],q[2];
ry(1.0506405418811875) q[2];
ry(-2.4532305986763467) q[4];
cx q[2],q[4];
ry(-2.131259477095667) q[2];
ry(1.350002587806408) q[4];
cx q[2],q[4];
ry(2.645604659374807) q[4];
ry(-2.1533212326283824) q[6];
cx q[4],q[6];
ry(-2.5331388733962727) q[4];
ry(-1.187051636373683) q[6];
cx q[4],q[6];
ry(-2.618990705153203) q[6];
ry(1.602104905866384) q[8];
cx q[6],q[8];
ry(0.6225312404878592) q[6];
ry(2.70495912347729) q[8];
cx q[6],q[8];
ry(-1.491529490887684) q[8];
ry(1.3072963974126195) q[10];
cx q[8],q[10];
ry(0.44864911476879366) q[8];
ry(2.6641518683982075) q[10];
cx q[8],q[10];
ry(-0.0518075906870621) q[1];
ry(1.9253226036874507) q[3];
cx q[1],q[3];
ry(-2.738501612422983) q[1];
ry(-1.944303206112917) q[3];
cx q[1],q[3];
ry(-1.4627594183839197) q[3];
ry(-2.861721582648726) q[5];
cx q[3],q[5];
ry(0.8972962465608099) q[3];
ry(-1.9267363592666487) q[5];
cx q[3],q[5];
ry(-2.271027902051814) q[5];
ry(1.6559598142255145) q[7];
cx q[5],q[7];
ry(-0.414289914057485) q[5];
ry(2.500563940441192) q[7];
cx q[5],q[7];
ry(1.0876026728782708) q[7];
ry(-0.5401658781676071) q[9];
cx q[7],q[9];
ry(-1.1586066812713574) q[7];
ry(-0.2935411943995368) q[9];
cx q[7],q[9];
ry(-0.09246688108197704) q[9];
ry(-0.3787695801277227) q[11];
cx q[9],q[11];
ry(0.9161487487377579) q[9];
ry(-0.90973688025458) q[11];
cx q[9],q[11];
ry(-0.8502566253001831) q[0];
ry(-2.265949688881509) q[1];
cx q[0],q[1];
ry(3.0452126544321567) q[0];
ry(1.4537848983352637) q[1];
cx q[0],q[1];
ry(0.8934711015433016) q[2];
ry(-1.4869519712201498) q[3];
cx q[2],q[3];
ry(2.8013223029425682) q[2];
ry(-0.15154750593432542) q[3];
cx q[2],q[3];
ry(-1.2623056040238505) q[4];
ry(-1.5472698907636788) q[5];
cx q[4],q[5];
ry(2.164720673686322) q[4];
ry(1.0932363511132985) q[5];
cx q[4],q[5];
ry(0.02694420758359871) q[6];
ry(-1.0295971939597481) q[7];
cx q[6],q[7];
ry(2.2633881012837715) q[6];
ry(1.7818154101791746) q[7];
cx q[6],q[7];
ry(0.05308189261949895) q[8];
ry(-1.2173224815133246) q[9];
cx q[8],q[9];
ry(2.5347017794778766) q[8];
ry(2.0005210330639827) q[9];
cx q[8],q[9];
ry(2.29589976430489) q[10];
ry(-0.7470365477315792) q[11];
cx q[10],q[11];
ry(1.259742321253093) q[10];
ry(2.4176414859566373) q[11];
cx q[10],q[11];
ry(1.4084462730351106) q[0];
ry(2.8493360409457456) q[2];
cx q[0],q[2];
ry(-0.29949433966177996) q[0];
ry(2.413777700698525) q[2];
cx q[0],q[2];
ry(0.9179459339577863) q[2];
ry(-2.6465623198330097) q[4];
cx q[2],q[4];
ry(-1.5946292814954326) q[2];
ry(-1.9756647526083888) q[4];
cx q[2],q[4];
ry(1.3479692006820827) q[4];
ry(-1.850785045596223) q[6];
cx q[4],q[6];
ry(-2.5012574271900254) q[4];
ry(-2.0565355748245944) q[6];
cx q[4],q[6];
ry(2.1798263416216592) q[6];
ry(-2.2266388283420806) q[8];
cx q[6],q[8];
ry(1.0569315601351243) q[6];
ry(1.9638530783894153) q[8];
cx q[6],q[8];
ry(-2.476646137835849) q[8];
ry(-2.41198866401522) q[10];
cx q[8],q[10];
ry(-2.418645079624819) q[8];
ry(0.33876667576437125) q[10];
cx q[8],q[10];
ry(2.240234559641933) q[1];
ry(-2.673217549415806) q[3];
cx q[1],q[3];
ry(1.0700552362440918) q[1];
ry(1.0400353288730497) q[3];
cx q[1],q[3];
ry(0.5560391218320005) q[3];
ry(0.31002210004000386) q[5];
cx q[3],q[5];
ry(1.2508620288936594) q[3];
ry(1.8250815895590833) q[5];
cx q[3],q[5];
ry(2.617452746584513) q[5];
ry(-1.5791226424208213) q[7];
cx q[5],q[7];
ry(-0.8736564650164267) q[5];
ry(-2.8327129163094127) q[7];
cx q[5],q[7];
ry(-3.0435089091439003) q[7];
ry(-0.783897031149297) q[9];
cx q[7],q[9];
ry(-0.8589688088329777) q[7];
ry(-0.9829368208261361) q[9];
cx q[7],q[9];
ry(-1.1575039780635974) q[9];
ry(0.32828494821432574) q[11];
cx q[9],q[11];
ry(1.8555517979623612) q[9];
ry(-2.837817927566562) q[11];
cx q[9],q[11];
ry(0.48933286738333776) q[0];
ry(-1.1126225763582862) q[1];
cx q[0],q[1];
ry(-2.957006891250363) q[0];
ry(2.879496598188202) q[1];
cx q[0],q[1];
ry(3.085241878880946) q[2];
ry(-1.314207946698005) q[3];
cx q[2],q[3];
ry(-2.0921780263574368) q[2];
ry(-1.8186727448677047) q[3];
cx q[2],q[3];
ry(2.7082510158210824) q[4];
ry(-2.030913489333037) q[5];
cx q[4],q[5];
ry(-1.0712937973766783) q[4];
ry(-2.44516577807421) q[5];
cx q[4],q[5];
ry(0.6135084360980765) q[6];
ry(-2.309141477196811) q[7];
cx q[6],q[7];
ry(2.435979587800147) q[6];
ry(-2.1930512763570853) q[7];
cx q[6],q[7];
ry(-2.685671388574515) q[8];
ry(0.4928975454314956) q[9];
cx q[8],q[9];
ry(-1.99411953583057) q[8];
ry(-2.9819931232197407) q[9];
cx q[8],q[9];
ry(0.017520342675864775) q[10];
ry(3.1079704583074923) q[11];
cx q[10],q[11];
ry(-2.9707684627750637) q[10];
ry(-2.217856457890531) q[11];
cx q[10],q[11];
ry(-1.267491992306591) q[0];
ry(2.6123061677709902) q[2];
cx q[0],q[2];
ry(-2.094156909596223) q[0];
ry(1.8465569493269547) q[2];
cx q[0],q[2];
ry(0.6639242271386253) q[2];
ry(-2.5539686962153456) q[4];
cx q[2],q[4];
ry(0.8484932955201208) q[2];
ry(-1.1436144543335776) q[4];
cx q[2],q[4];
ry(-0.012532925312307874) q[4];
ry(1.7294174830462998) q[6];
cx q[4],q[6];
ry(-0.7031504069630993) q[4];
ry(1.0584340833747552) q[6];
cx q[4],q[6];
ry(-0.7651540205842403) q[6];
ry(1.0137464927441513) q[8];
cx q[6],q[8];
ry(0.3873102738255785) q[6];
ry(-1.9139672682226783) q[8];
cx q[6],q[8];
ry(-0.2303829165864325) q[8];
ry(0.23176305917885104) q[10];
cx q[8],q[10];
ry(2.6387511899977754) q[8];
ry(0.9835487796674851) q[10];
cx q[8],q[10];
ry(-0.8077140130373484) q[1];
ry(2.043608679549223) q[3];
cx q[1],q[3];
ry(2.843286919468218) q[1];
ry(1.7101146608934725) q[3];
cx q[1],q[3];
ry(2.9563975375216387) q[3];
ry(-2.1905492434296168) q[5];
cx q[3],q[5];
ry(3.0004323192648346) q[3];
ry(-2.817452760893844) q[5];
cx q[3],q[5];
ry(-0.25969351203503077) q[5];
ry(0.4653159478292093) q[7];
cx q[5],q[7];
ry(0.750584512719187) q[5];
ry(-1.8102139404200204) q[7];
cx q[5],q[7];
ry(-0.2668261129849793) q[7];
ry(2.188829735682913) q[9];
cx q[7],q[9];
ry(-1.6578997723993005) q[7];
ry(1.3947032052179882) q[9];
cx q[7],q[9];
ry(-2.8093866543555213) q[9];
ry(2.7880630253335994) q[11];
cx q[9],q[11];
ry(1.29721906673649) q[9];
ry(-1.1958988438484157) q[11];
cx q[9],q[11];
ry(2.272863340434049) q[0];
ry(-1.333122453629767) q[1];
cx q[0],q[1];
ry(1.2690092307655738) q[0];
ry(-1.5617615634031434) q[1];
cx q[0],q[1];
ry(1.6767338363589073) q[2];
ry(1.0557209331063162) q[3];
cx q[2],q[3];
ry(-0.31014765066695577) q[2];
ry(-2.1186356611611297) q[3];
cx q[2],q[3];
ry(2.34353532836715) q[4];
ry(-0.7339546069406513) q[5];
cx q[4],q[5];
ry(2.494061954757931) q[4];
ry(-0.36367153441115563) q[5];
cx q[4],q[5];
ry(-0.7591352311493154) q[6];
ry(-0.09802302984636047) q[7];
cx q[6],q[7];
ry(0.6492004149996462) q[6];
ry(-2.833651503146604) q[7];
cx q[6],q[7];
ry(2.417595854709334) q[8];
ry(2.7908435472610145) q[9];
cx q[8],q[9];
ry(-1.1927531460361407) q[8];
ry(-1.6485371743192692) q[9];
cx q[8],q[9];
ry(-0.5155788983978447) q[10];
ry(-0.9611604916342971) q[11];
cx q[10],q[11];
ry(-1.8815817702533628) q[10];
ry(1.846696448617271) q[11];
cx q[10],q[11];
ry(-2.458633971993116) q[0];
ry(-2.8727340724714496) q[2];
cx q[0],q[2];
ry(0.5069324424117505) q[0];
ry(1.3617082372464457) q[2];
cx q[0],q[2];
ry(-1.8249992271804079) q[2];
ry(-1.7392803849917808) q[4];
cx q[2],q[4];
ry(-2.8766072247534216) q[2];
ry(-0.37734394997052423) q[4];
cx q[2],q[4];
ry(-2.384836345534909) q[4];
ry(-0.3886325016780239) q[6];
cx q[4],q[6];
ry(1.3436445342180827) q[4];
ry(3.0552336846439467) q[6];
cx q[4],q[6];
ry(-0.10700275144191984) q[6];
ry(2.419348765850914) q[8];
cx q[6],q[8];
ry(0.14929305654236336) q[6];
ry(1.6631989837618733) q[8];
cx q[6],q[8];
ry(0.23810507994970376) q[8];
ry(-2.7000756767810836) q[10];
cx q[8],q[10];
ry(1.8908522182606626) q[8];
ry(0.3610883362964741) q[10];
cx q[8],q[10];
ry(1.953805795661408) q[1];
ry(0.8215470403900671) q[3];
cx q[1],q[3];
ry(1.7447185958210856) q[1];
ry(-2.6148434045845548) q[3];
cx q[1],q[3];
ry(3.0107477586128337) q[3];
ry(-1.0027224466552624) q[5];
cx q[3],q[5];
ry(1.135278549256376) q[3];
ry(-2.1344678307748213) q[5];
cx q[3],q[5];
ry(-0.6888678176415199) q[5];
ry(0.8883880527935824) q[7];
cx q[5],q[7];
ry(-0.9798951400106413) q[5];
ry(2.553025238881886) q[7];
cx q[5],q[7];
ry(-0.375459979369289) q[7];
ry(-2.4382188296260967) q[9];
cx q[7],q[9];
ry(-2.4291487364959488) q[7];
ry(-2.0194727172909324) q[9];
cx q[7],q[9];
ry(2.873592996600181) q[9];
ry(1.1995090754979572) q[11];
cx q[9],q[11];
ry(2.0340942102785347) q[9];
ry(-0.08263646326898844) q[11];
cx q[9],q[11];
ry(-1.2681836143645144) q[0];
ry(-2.0723119688523983) q[1];
cx q[0],q[1];
ry(-2.616225558933053) q[0];
ry(2.1933381473624878) q[1];
cx q[0],q[1];
ry(-2.2536105880209005) q[2];
ry(-2.397660864876222) q[3];
cx q[2],q[3];
ry(-0.558520737030558) q[2];
ry(1.5380044556514696) q[3];
cx q[2],q[3];
ry(2.4554443674830733) q[4];
ry(3.049229445636303) q[5];
cx q[4],q[5];
ry(1.119477710666958) q[4];
ry(1.776476776121969) q[5];
cx q[4],q[5];
ry(2.390218447501256) q[6];
ry(1.9966000830565047) q[7];
cx q[6],q[7];
ry(2.6081877597790175) q[6];
ry(0.23957842176462416) q[7];
cx q[6],q[7];
ry(0.5412520535460841) q[8];
ry(-0.675741954733151) q[9];
cx q[8],q[9];
ry(1.2559310220801805) q[8];
ry(0.13033337154989244) q[9];
cx q[8],q[9];
ry(-0.42199781109300916) q[10];
ry(-2.927042919159272) q[11];
cx q[10],q[11];
ry(1.9500702502685283) q[10];
ry(-2.699585290286921) q[11];
cx q[10],q[11];
ry(-0.890185243860822) q[0];
ry(0.8278625158454053) q[2];
cx q[0],q[2];
ry(0.15479097031934863) q[0];
ry(1.860241459738384) q[2];
cx q[0],q[2];
ry(-0.8806860093325101) q[2];
ry(1.6179461907381272) q[4];
cx q[2],q[4];
ry(-1.3566668575291356) q[2];
ry(0.9378303294775567) q[4];
cx q[2],q[4];
ry(-3.0307945284476308) q[4];
ry(0.7601958993182077) q[6];
cx q[4],q[6];
ry(2.407992783824779) q[4];
ry(-2.7182273888994755) q[6];
cx q[4],q[6];
ry(2.30933175763947) q[6];
ry(-2.294905516011629) q[8];
cx q[6],q[8];
ry(1.29834223715863) q[6];
ry(2.0072609374421075) q[8];
cx q[6],q[8];
ry(-1.5439983615976507) q[8];
ry(0.362945348662529) q[10];
cx q[8],q[10];
ry(0.9486558467918403) q[8];
ry(-2.422384353772882) q[10];
cx q[8],q[10];
ry(2.4087066645180646) q[1];
ry(2.747801716006836) q[3];
cx q[1],q[3];
ry(-1.190799905135278) q[1];
ry(0.7898480409719895) q[3];
cx q[1],q[3];
ry(-1.0907246508956954) q[3];
ry(-3.032699625799358) q[5];
cx q[3],q[5];
ry(2.890663225566767) q[3];
ry(-1.7158011331218546) q[5];
cx q[3],q[5];
ry(0.26475187836552894) q[5];
ry(-0.12425843308166139) q[7];
cx q[5],q[7];
ry(-1.1228653531217114) q[5];
ry(2.6502950108090055) q[7];
cx q[5],q[7];
ry(-2.0798372242959493) q[7];
ry(1.0660387772002582) q[9];
cx q[7],q[9];
ry(2.726706386567293) q[7];
ry(-2.5464269147071343) q[9];
cx q[7],q[9];
ry(-0.0716519066827157) q[9];
ry(-1.8424105026304616) q[11];
cx q[9],q[11];
ry(-0.9585584059344312) q[9];
ry(2.8048023058318994) q[11];
cx q[9],q[11];
ry(-1.933421524098442) q[0];
ry(1.5741098878077242) q[1];
cx q[0],q[1];
ry(0.7931180156222224) q[0];
ry(1.026708976440869) q[1];
cx q[0],q[1];
ry(3.066749317670092) q[2];
ry(2.7388975696384645) q[3];
cx q[2],q[3];
ry(-1.5698936181066108) q[2];
ry(-0.9055976243434456) q[3];
cx q[2],q[3];
ry(-2.0759861667319863) q[4];
ry(-0.07599682721772982) q[5];
cx q[4],q[5];
ry(-1.8151579931735136) q[4];
ry(2.7588417562164924) q[5];
cx q[4],q[5];
ry(-0.011035039465375486) q[6];
ry(2.43147230542958) q[7];
cx q[6],q[7];
ry(3.0293677280527724) q[6];
ry(-2.5894743647312346) q[7];
cx q[6],q[7];
ry(1.514493978662402) q[8];
ry(0.10440676935889748) q[9];
cx q[8],q[9];
ry(-2.064899416402568) q[8];
ry(-2.92317856434788) q[9];
cx q[8],q[9];
ry(2.813120298003661) q[10];
ry(-0.43231789785378677) q[11];
cx q[10],q[11];
ry(-2.600688270760345) q[10];
ry(1.675482110689467) q[11];
cx q[10],q[11];
ry(-2.186529342245565) q[0];
ry(-2.8359666194545166) q[2];
cx q[0],q[2];
ry(0.776090809917693) q[0];
ry(0.9886285426199776) q[2];
cx q[0],q[2];
ry(-1.3153060305983555) q[2];
ry(-1.9600312297198574) q[4];
cx q[2],q[4];
ry(1.1803260476508806) q[2];
ry(2.2037328436787424) q[4];
cx q[2],q[4];
ry(-0.2514904870261592) q[4];
ry(-1.5919707668794603) q[6];
cx q[4],q[6];
ry(-2.286610050054041) q[4];
ry(2.9486503261719643) q[6];
cx q[4],q[6];
ry(1.2128211324627605) q[6];
ry(-2.7038696216676326) q[8];
cx q[6],q[8];
ry(-1.3660306325946054) q[6];
ry(1.3728838285886216) q[8];
cx q[6],q[8];
ry(-0.20083422674240858) q[8];
ry(2.9321125809508) q[10];
cx q[8],q[10];
ry(1.0461740400086856) q[8];
ry(1.2646138136783538) q[10];
cx q[8],q[10];
ry(1.5665262021374031) q[1];
ry(-2.320459793244199) q[3];
cx q[1],q[3];
ry(-2.470762761143575) q[1];
ry(-0.38496720724088457) q[3];
cx q[1],q[3];
ry(0.240241156363445) q[3];
ry(0.01341272095333608) q[5];
cx q[3],q[5];
ry(-2.579433122821106) q[3];
ry(-2.262464796224415) q[5];
cx q[3],q[5];
ry(0.3688627600335358) q[5];
ry(-2.3966695380795984) q[7];
cx q[5],q[7];
ry(1.6766011884229666) q[5];
ry(-1.7970268024675957) q[7];
cx q[5],q[7];
ry(-0.9783595224849302) q[7];
ry(-1.051909026482606) q[9];
cx q[7],q[9];
ry(-1.1730595594051245) q[7];
ry(2.661684351370576) q[9];
cx q[7],q[9];
ry(0.40200603087532133) q[9];
ry(2.825301873004991) q[11];
cx q[9],q[11];
ry(-1.561737337416478) q[9];
ry(0.567309113993373) q[11];
cx q[9],q[11];
ry(2.131646543236556) q[0];
ry(0.36810311374496457) q[1];
cx q[0],q[1];
ry(2.7480065040258723) q[0];
ry(2.7560307522451355) q[1];
cx q[0],q[1];
ry(-0.06622580456949034) q[2];
ry(-2.8025710604829324) q[3];
cx q[2],q[3];
ry(-2.8432754362564094) q[2];
ry(-0.6253711726669889) q[3];
cx q[2],q[3];
ry(1.0088305570414342) q[4];
ry(2.8296918872365864) q[5];
cx q[4],q[5];
ry(-1.6020725913108118) q[4];
ry(0.7785635214570652) q[5];
cx q[4],q[5];
ry(0.39043234232120305) q[6];
ry(-2.0292935710947875) q[7];
cx q[6],q[7];
ry(1.0909239711277916) q[6];
ry(0.7726575220927773) q[7];
cx q[6],q[7];
ry(-0.6238906403140935) q[8];
ry(-2.759761250988865) q[9];
cx q[8],q[9];
ry(2.3767784910885554) q[8];
ry(2.0027398692807683) q[9];
cx q[8],q[9];
ry(2.415182104570919) q[10];
ry(-1.4558465893616281) q[11];
cx q[10],q[11];
ry(3.0065024092939634) q[10];
ry(-0.6971697245439308) q[11];
cx q[10],q[11];
ry(-1.8128830011242196) q[0];
ry(1.6922566957911638) q[2];
cx q[0],q[2];
ry(2.6997432678791045) q[0];
ry(-2.546791054868347) q[2];
cx q[0],q[2];
ry(0.9154188471290683) q[2];
ry(1.5068315716904364) q[4];
cx q[2],q[4];
ry(-0.6287971233128979) q[2];
ry(-0.7711060753748645) q[4];
cx q[2],q[4];
ry(-0.9895856823701985) q[4];
ry(1.2822232067549582) q[6];
cx q[4],q[6];
ry(2.4924998215913052) q[4];
ry(-1.0839682194755529) q[6];
cx q[4],q[6];
ry(0.1933422767087869) q[6];
ry(-1.9507074686290788) q[8];
cx q[6],q[8];
ry(0.8786711826955669) q[6];
ry(1.7359334903090478) q[8];
cx q[6],q[8];
ry(-2.2384867809574547) q[8];
ry(-1.4701271075844833) q[10];
cx q[8],q[10];
ry(-2.7642042233723125) q[8];
ry(-0.7971425709343601) q[10];
cx q[8],q[10];
ry(0.4600356705257935) q[1];
ry(-2.341819669850864) q[3];
cx q[1],q[3];
ry(-1.4359575941856892) q[1];
ry(1.2888498341214254) q[3];
cx q[1],q[3];
ry(-0.15138929919541688) q[3];
ry(2.5703109894057214) q[5];
cx q[3],q[5];
ry(-0.8631644538205778) q[3];
ry(-0.017602563414293945) q[5];
cx q[3],q[5];
ry(-2.526877928548332) q[5];
ry(-1.6211656657915292) q[7];
cx q[5],q[7];
ry(-1.0185512231129576) q[5];
ry(0.5598910524685827) q[7];
cx q[5],q[7];
ry(2.4638783545598852) q[7];
ry(-0.5346708897330118) q[9];
cx q[7],q[9];
ry(2.225680688188223) q[7];
ry(0.8147122781952145) q[9];
cx q[7],q[9];
ry(0.5426624193496973) q[9];
ry(-2.9928394232122066) q[11];
cx q[9],q[11];
ry(-1.0264175732479544) q[9];
ry(0.9243181852676524) q[11];
cx q[9],q[11];
ry(-1.7971364884154508) q[0];
ry(-2.972280045171196) q[1];
ry(-0.2723676309883144) q[2];
ry(1.463323032634582) q[3];
ry(0.1691081349823819) q[4];
ry(1.344757302381586) q[5];
ry(1.4642484851697937) q[6];
ry(2.3012219881066343) q[7];
ry(-0.9597644450055878) q[8];
ry(-3.056663983932278) q[9];
ry(-0.0009983297098075743) q[10];
ry(-1.068849268260501) q[11];