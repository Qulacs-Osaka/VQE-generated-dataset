OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(1.7578355880112007) q[0];
rz(2.9444242983621627) q[0];
ry(1.4434525571932193) q[1];
rz(-0.2102243986813832) q[1];
ry(0.005940941323856919) q[2];
rz(-2.299316534137935) q[2];
ry(-3.1415695757370408) q[3];
rz(0.326533460468025) q[3];
ry(-1.579645457245224) q[4];
rz(-1.24876995043571) q[4];
ry(3.1411238752628186) q[5];
rz(1.471832751048044) q[5];
ry(3.1361063892555596) q[6];
rz(-3.1082379685030515) q[6];
ry(8.363711700987152e-05) q[7];
rz(1.5504260365407465) q[7];
ry(1.5707933225610062) q[8];
rz(-0.17012973263830225) q[8];
ry(1.5709511430739793) q[9];
rz(-2.4987974066916734) q[9];
ry(-1.571152356337886) q[10];
rz(-2.314193335825267) q[10];
ry(3.141403602825842) q[11];
rz(0.29386682179297624) q[11];
ry(1.5709423881536742) q[12];
rz(2.812942729496987) q[12];
ry(-1.532066587589733) q[13];
rz(3.1415624683595755) q[13];
ry(3.1413641483230434) q[14];
rz(2.476237059016077) q[14];
ry(1.5708199328972219) q[15];
rz(-1.5709480936489477) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(2.032465537734608) q[0];
rz(-3.138853396668232) q[0];
ry(-2.0861574030000085) q[1];
rz(-0.13063680941407974) q[1];
ry(-3.1415625525228092) q[2];
rz(0.45047943236628907) q[2];
ry(-0.24871596155788064) q[3];
rz(0.39468372771071475) q[3];
ry(3.1135949958695415) q[4];
rz(1.6137959225011151) q[4];
ry(-1.5707894840368202) q[5];
rz(-3.141591501298758) q[5];
ry(1.5707912654017508) q[6];
rz(0.017894821902365684) q[6];
ry(1.915646441572274) q[7];
rz(-3.0936280909805767) q[7];
ry(2.5935269767926942e-05) q[8];
rz(-2.9120148041035003) q[8];
ry(4.4080725541266584e-05) q[9];
rz(-2.2137967095501843) q[9];
ry(2.0726775242784433e-06) q[10];
rz(-1.3932260251353688) q[10];
ry(9.936532238466147e-07) q[11];
rz(2.408154734172362) q[11];
ry(0.8934643116863813) q[12];
rz(-2.69150950740594) q[12];
ry(0.6198589308972475) q[13];
rz(3.986510323628067e-05) q[13];
ry(1.5706889651967595) q[14];
rz(-2.43869371386774) q[14];
ry(1.7682637776089345) q[15];
rz(-0.004612555628029471) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-0.38429920565378684) q[0];
rz(-0.079027261283557) q[0];
ry(1.3676165236481808) q[1];
rz(-2.4938251115460446) q[1];
ry(1.1334583810044406e-05) q[2];
rz(-0.08216589230372405) q[2];
ry(-3.141579660635472) q[3];
rz(1.9897280462084368) q[3];
ry(-1.229018831212488e-05) q[4];
rz(0.4303135108959463) q[4];
ry(-1.5707802242512257) q[5];
rz(1.5707998569291517) q[5];
ry(-1.5686354749391587) q[6];
rz(-0.09313292621902744) q[6];
ry(-1.7904849030969672e-05) q[7];
rz(3.0935806848460614) q[7];
ry(-1.6114131147109987) q[8];
rz(1.610993327614306) q[8];
ry(1.6245541728981396) q[9];
rz(-1.5709348024064518) q[9];
ry(-0.0005351281177689552) q[10];
rz(-2.5757526874657386) q[10];
ry(-1.5709599008834425) q[11];
rz(2.0697134808901128) q[11];
ry(-1.5265190921240315) q[12];
rz(-2.855987586318216) q[12];
ry(1.5708046772057216) q[13];
rz(-1.435677499089835) q[13];
ry(-0.0008926900532806314) q[14];
rz(0.8278208074121025) q[14];
ry(3.0435818887484207) q[15];
rz(-1.5404927633964338) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(0.0735461705981919) q[0];
rz(1.5211308155480672) q[0];
ry(-0.4982053093004053) q[1];
rz(2.5269877051303795) q[1];
ry(1.7136005054333793e-05) q[2];
rz(-2.818273931410247) q[2];
ry(-1.570549912533687) q[3];
rz(1.6300446991554354) q[3];
ry(-4.66604037541174e-06) q[4];
rz(-1.1876551980288585) q[4];
ry(-1.570804047039597) q[5];
rz(1.6493837992323646) q[5];
ry(-0.0016006517687285893) q[6];
rz(-1.812078995571019) q[6];
ry(1.5707992540225284) q[7];
rz(-1.5708161269978955) q[7];
ry(1.6127116673141861) q[8];
rz(1.431148095722301) q[8];
ry(1.56701972847796) q[9];
rz(1.3154735632376893) q[9];
ry(1.5709640073441697) q[10];
rz(2.8599531774217013) q[10];
ry(-3.14158433967472) q[11];
rz(0.9253257394993809) q[11];
ry(-1.2766251105134074e-06) q[12];
rz(2.1541817841521285) q[12];
ry(-1.5707822751015756) q[13];
rz(-1.2902569134475188) q[13];
ry(-1.7600150925055489) q[14];
rz(-2.472955211745397) q[14];
ry(1.5669726909253363) q[15];
rz(-1.5707134804463367) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(1.5708082335487017) q[0];
rz(-3.1366997232333813) q[0];
ry(0.00015742317560644124) q[1];
rz(-1.554618183465518) q[1];
ry(0.017693146519287262) q[2];
rz(-2.8585623468502814) q[2];
ry(-4.6233191253286905e-05) q[3];
rz(-3.078887107071376) q[3];
ry(3.1413623986239783) q[4];
rz(2.1051004412414582) q[4];
ry(-0.25409773306850825) q[5];
rz(3.0764731433514134) q[5];
ry(3.141586537318554) q[6];
rz(2.8072309295489717) q[6];
ry(-1.5686557925387872) q[7];
rz(3.1415065796234174) q[7];
ry(-1.5707726186653475) q[8];
rz(2.976374296025843) q[8];
ry(3.1414862890743094) q[9];
rz(2.8862266344487684) q[9];
ry(3.1415880055731207) q[10];
rz(-1.557208771512573) q[10];
ry(3.1415908604096243) q[11];
rz(-2.704555018036622) q[11];
ry(2.397871630156156e-05) q[12];
rz(-3.127419394012431) q[12];
ry(-2.214460740379565e-05) q[13];
rz(-1.39605892163307) q[13];
ry(-3.1415225338673802) q[14];
rz(-1.5424705108218388) q[14];
ry(1.569770849723612) q[15];
rz(-0.2378658189512297) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-1.8585023405800867) q[0];
rz(1.5684792264557388) q[0];
ry(1.5708240372276903) q[1];
rz(3.1411612557445228) q[1];
ry(-1.5707847606288687) q[2];
rz(3.063482111528832) q[2];
ry(0.0005153033145015585) q[3];
rz(-2.390824854698587) q[3];
ry(1.570812493011683) q[4];
rz(1.5709398834669657) q[4];
ry(-1.570797821199375) q[5];
rz(0.030713632032019152) q[5];
ry(-1.5707905456739386) q[6];
rz(1.6220276836360963) q[6];
ry(-1.5635705487309755) q[7];
rz(-1.5592818524594474) q[7];
ry(3.141582821782013) q[8];
rz(2.9617373197933294) q[8];
ry(-1.7893657474211597) q[9];
rz(3.1404976981676227) q[9];
ry(1.5022384005840382e-05) q[10];
rz(-2.76382421656129) q[10];
ry(-3.1166719694904486) q[11];
rz(1.0680068744533244) q[11];
ry(1.5709472044385666) q[12];
rz(0.24456963019509545) q[12];
ry(3.1403039677831948) q[13];
rz(2.040711662458616) q[13];
ry(1.5698872745815942) q[14];
rz(0.21364178944786083) q[14];
ry(-0.028635313717997983) q[15];
rz(1.8085864391540518) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-1.5713939006245807) q[0];
rz(2.699945839779611) q[0];
ry(-1.570964456948468) q[1];
rz(-2.236669913230112) q[1];
ry(1.571735953533505) q[2];
rz(1.5744713207228873) q[2];
ry(0.000775114984825187) q[3];
rz(-2.438714985229071) q[3];
ry(-1.579133194608323) q[4];
rz(-2.5137477239499484) q[4];
ry(0.04136413452254394) q[5];
rz(0.23110338877885364) q[5];
ry(1.5707800448654787) q[6];
rz(0.045147203221250834) q[6];
ry(-1.570471770303209) q[7];
rz(-0.9499505040019337) q[7];
ry(0.03412782607115972) q[8];
rz(-1.5560821158665092) q[8];
ry(-1.9575939798463162) q[9];
rz(-1.7089672112081224) q[9];
ry(3.1415868560337046) q[10];
rz(-1.533708509238748) q[10];
ry(8.040962123700979e-06) q[11];
rz(2.0996871098973235) q[11];
ry(-3.141555119067235) q[12];
rz(1.940493556223668) q[12];
ry(-0.016800865190079506) q[13];
rz(-0.01449928440906856) q[13];
ry(-3.883987426778697e-05) q[14];
rz(-0.2783618182642334) q[14];
ry(-1.5707159819487275) q[15];
rz(-2.062472674117178) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-3.1415372688314145) q[0];
rz(-2.244621829240921) q[0];
ry(3.1412619883505535) q[1];
rz(0.2977600315132234) q[1];
ry(-2.841005543786407) q[2];
rz(1.5738535452254396) q[2];
ry(3.122943938613046) q[3];
rz(-1.5653218071047452) q[3];
ry(0.0023912847818499117) q[4];
rz(-2.1987577513244076) q[4];
ry(-3.1368391638199387) q[5];
rz(0.2428148610312695) q[5];
ry(3.1415549902888467) q[6];
rz(1.6158875197867117) q[6];
ry(-1.7220172131817553e-06) q[7];
rz(2.6776339426369824) q[7];
ry(-1.5709721876575564) q[8];
rz(0.9135772740565983) q[8];
ry(5.255029884310147e-05) q[9];
rz(2.6019453736536673) q[9];
ry(-0.00025090468366961716) q[10];
rz(2.2902005208861063) q[10];
ry(-1.4967436684233273) q[11];
rz(3.140541403448047) q[11];
ry(3.3407049882186815e-05) q[12];
rz(1.442446655564669) q[12];
ry(-1.5700407006896382) q[13];
rz(3.1415249597062176) q[13];
ry(-0.0002136752633432765) q[14];
rz(0.05510594443923697) q[14];
ry(3.1311860715630093) q[15];
rz(-1.9122682938687916) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(3.1385101183313187) q[0];
rz(1.2663532407319824) q[0];
ry(7.118931697466818e-05) q[1];
rz(-2.3462074805253987) q[1];
ry(1.5697947858686567) q[2];
rz(-3.1283260405341244) q[2];
ry(1.5704143263120889) q[3];
rz(-1.0992594289072606) q[3];
ry(-1.570650570025158) q[4];
rz(0.3499496134520639) q[4];
ry(0.11444579205723837) q[5];
rz(-2.5708853996542707) q[5];
ry(-1.5706059655072213) q[6];
rz(-1.5824434240597565) q[6];
ry(-5.606348177216257e-05) q[7];
rz(1.4154122308480135) q[7];
ry(7.370645910498297e-06) q[8];
rz(-2.0058926744159686) q[8];
ry(3.1415777110084653) q[9];
rz(0.8929589891709789) q[9];
ry(1.5707671802711303) q[10];
rz(-1.5707071760873212) q[10];
ry(-1.5708061514438023) q[11];
rz(-1.570821701688386) q[11];
ry(-3.128886261872375) q[12];
rz(0.755666241790629) q[12];
ry(3.1252204254772025) q[13];
rz(2.9854252758009046) q[13];
ry(2.7143336981787347) q[14];
rz(-2.1611618211418646) q[14];
ry(-2.3376663113795138) q[15];
rz(1.6938528003827997) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(1.6845570943500183) q[0];
rz(1.504256153877909) q[0];
ry(1.4544634648831305) q[1];
rz(2.2945991458957025) q[1];
ry(-3.141577130010647) q[2];
rz(-2.995532511013835) q[2];
ry(-5.147465137746549e-05) q[3];
rz(-1.849364048030261) q[3];
ry(1.5707959842042936) q[4];
rz(3.1415814457193267) q[4];
ry(1.5707355936487584) q[5];
rz(-2.586389519443344e-06) q[5];
ry(-3.071936772322392) q[6];
rz(0.1727830549007265) q[6];
ry(-1.570731222002645) q[7];
rz(1.364727893647892) q[7];
ry(3.1409472366169475) q[8];
rz(1.3895180550403499) q[8];
ry(1.5704566025925704) q[9];
rz(1.6881692209515977) q[9];
ry(1.0933102716096794) q[10];
rz(-1.5707880854714964) q[10];
ry(1.5595231777770637) q[11];
rz(-1.5707875863739182) q[11];
ry(1.624459482485463e-06) q[12];
rz(0.7667941287907993) q[12];
ry(-3.141591040431333) q[13];
rz(0.8030877405861911) q[13];
ry(3.141355829699532) q[14];
rz(0.2872058266540971) q[14];
ry(1.5777215758972956) q[15];
rz(1.5683641126459875) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-3.141398157955437) q[0];
rz(2.0527481341496348) q[0];
ry(2.4572536396760825e-06) q[1];
rz(-0.3669367489468352) q[1];
ry(0.000107778677886472) q[2];
rz(2.9519413760288113) q[2];
ry(3.140609105494929) q[3];
rz(0.2145421841955787) q[3];
ry(1.5707933022317506) q[4];
rz(1.5468685656426497) q[4];
ry(-1.57082251499543) q[5];
rz(-3.114851518447434) q[5];
ry(3.141590627644025) q[6];
rz(1.1261362283361755) q[6];
ry(1.839757625354821e-06) q[7];
rz(1.7977678132782564) q[7];
ry(3.1415915380371424) q[8];
rz(-2.211192945601108) q[8];
ry(3.1415908045600003) q[9];
rz(2.935248778648042) q[9];
ry(1.5708056144009164) q[10];
rz(-1.4432103639620806) q[10];
ry(-1.5707923971797955) q[11];
rz(-1.5786420435135842) q[11];
ry(5.328816405292258e-05) q[12];
rz(3.045149374119194) q[12];
ry(2.458936462043846e-05) q[13];
rz(0.6188816679084103) q[13];
ry(-0.0002807267974470391) q[14];
rz(2.2778018755570137) q[14];
ry(-1.5777133629474935) q[15];
rz(1.413034513817557) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(0.09790166295993963) q[0];
rz(1.7222205332290212) q[0];
ry(-0.22325134675182268) q[1];
rz(0.32251380124722123) q[1];
ry(1.6055182120554043) q[2];
rz(-0.8748381862505469) q[2];
ry(1.6072927187988348) q[3];
rz(-0.8727431796050559) q[3];
ry(-2.0955071515665304) q[4];
rz(-0.8869993422669379) q[4];
ry(1.5384679729752522) q[5];
rz(1.05403916283989) q[5];
ry(3.1079806789064075) q[6];
rz(1.6377707389572604) q[6];
ry(1.6070973060618685) q[7];
rz(-0.8754338975657615) q[7];
ry(1.5709004421520523) q[8];
rz(-0.8236734080751046) q[8];
ry(0.022325236287481662) q[9];
rz(-2.1233397676388943) q[9];
ry(-3.133525066850189) q[10];
rz(-2.318101052739186) q[10];
ry(-2.0541611214881996) q[11];
rz(0.6908859161594524) q[11];
ry(3.1199192331415437) q[12];
rz(0.5531839345184156) q[12];
ry(1.5494712414524894) q[13];
rz(0.6876137868111222) q[13];
ry(-1.1650203651094273) q[14];
rz(0.6893838905469032) q[14];
ry(-0.7676828371824573) q[15];
rz(-0.8276443273255939) q[15];