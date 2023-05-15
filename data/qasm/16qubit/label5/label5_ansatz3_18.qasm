OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(-0.8845816109759077) q[0];
rz(-2.4473103599960035) q[0];
ry(-1.6787660343511028) q[1];
rz(1.322142660225912) q[1];
ry(-0.6344115429981134) q[2];
rz(-0.5106537907946778) q[2];
ry(1.0664086043634158) q[3];
rz(2.7573529791310003) q[3];
ry(3.1364293567402304) q[4];
rz(-2.6923121650658928) q[4];
ry(-0.9993184675002124) q[5];
rz(-0.3869039133079857) q[5];
ry(2.511033125818686) q[6];
rz(-0.42730202083640856) q[6];
ry(-0.00014275095284865813) q[7];
rz(1.5736183555604357) q[7];
ry(-1.5716739971276512) q[8];
rz(-0.2762470298925912) q[8];
ry(6.538143340328872e-05) q[9];
rz(2.235351827641434) q[9];
ry(1.1999181902493157) q[10];
rz(-2.361364553017338) q[10];
ry(-1.5713275392884152) q[11];
rz(-0.47668942654008845) q[11];
ry(0.001480698844034448) q[12];
rz(-1.4430857976077853) q[12];
ry(1.5708760088428781) q[13];
rz(5.01577607163739e-05) q[13];
ry(-0.6887936860583057) q[14];
rz(-3.1409116081358377) q[14];
ry(-3.1390393027423777) q[15];
rz(3.108962855913656) q[15];
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
ry(2.4477654913057014) q[0];
rz(1.2709699412983224) q[0];
ry(1.9456435453347876) q[1];
rz(0.08698907862373988) q[1];
ry(-1.2812176869868555) q[2];
rz(-0.3407130048290474) q[2];
ry(0.5501836766833019) q[3];
rz(-1.472010415846058) q[3];
ry(3.059459429367866) q[4];
rz(-0.7437189224973405) q[4];
ry(1.9923238012659068) q[5];
rz(-2.24721478232022) q[5];
ry(1.572389104310771) q[6];
rz(-0.7703620376282486) q[6];
ry(0.0003962765240430276) q[7];
rz(-1.828858546800841) q[7];
ry(0.001926631448047722) q[8];
rz(-1.8294734091444345) q[8];
ry(3.139752985667826) q[9];
rz(3.023925635199289) q[9];
ry(1.635824084705514) q[10];
rz(-0.2575506501262277) q[10];
ry(1.5721739515343278) q[11];
rz(3.0951345827856325) q[11];
ry(-3.1406388182329046) q[12];
rz(0.7288633579763587) q[12];
ry(-1.5748106284875325) q[13];
rz(-1.8934712343836555) q[13];
ry(0.6889687258709936) q[14];
rz(0.37256216936855635) q[14];
ry(-1.8809164143777073) q[15];
rz(-1.923240290710848) q[15];
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
ry(1.2578473705594293) q[0];
rz(-2.9550402775540365) q[0];
ry(-0.6974056777158362) q[1];
rz(0.4487072217905769) q[1];
ry(1.9408619949814714) q[2];
rz(0.653583002162313) q[2];
ry(2.057638977115635) q[3];
rz(-2.3886835006615352) q[3];
ry(3.141005620498541) q[4];
rz(1.125084885973795) q[4];
ry(-1.7091237581074834) q[5];
rz(2.2447729784148005) q[5];
ry(0.0507472981267674) q[6];
rz(2.3583846513155278) q[6];
ry(3.1404981439688715) q[7];
rz(-2.08131011243319) q[7];
ry(0.3417137250795859) q[8];
rz(1.6293985704146436) q[8];
ry(-0.1717794374619972) q[9];
rz(-2.483084140066526) q[9];
ry(2.7664562460759843) q[10];
rz(-2.116779129187589) q[10];
ry(-2.5449718841693043) q[11];
rz(-1.9793863763699782) q[11];
ry(0.0014791307229060635) q[12];
rz(1.2063304763769251) q[12];
ry(-0.08280692697813925) q[13];
rz(1.3199824361139498) q[13];
ry(3.1401098706250568) q[14];
rz(2.5576616794881613) q[14];
ry(-0.7783360465833251) q[15];
rz(-1.1405478370695075) q[15];
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
ry(-1.9028021958750456) q[0];
rz(2.24203900858564) q[0];
ry(-1.2818524871672639) q[1];
rz(-1.2552055988013349) q[1];
ry(2.6827330781523404) q[2];
rz(0.02476976303580969) q[2];
ry(-2.1375008867997787) q[3];
rz(0.46443204491704615) q[3];
ry(-1.101790767216071) q[4];
rz(-2.4422192202977167) q[4];
ry(1.858518848784981) q[5];
rz(-2.4014622037880975) q[5];
ry(-2.0252642242455527) q[6];
rz(1.2945716358436974) q[6];
ry(0.00011509078686877672) q[7];
rz(2.480080008788282) q[7];
ry(-0.0026591707566667997) q[8];
rz(-1.1103796877844059) q[8];
ry(-3.138808308761173) q[9];
rz(-2.48834049425181) q[9];
ry(3.1411778315668903) q[10];
rz(-2.617907071840172) q[10];
ry(3.139632844873513) q[11];
rz(-1.9230089696873405) q[11];
ry(-0.0024793202024664396) q[12];
rz(0.667225283718943) q[12];
ry(3.1365004057106645) q[13];
rz(-0.41671596655841725) q[13];
ry(0.0009159818844040046) q[14];
rz(-2.645665041530246) q[14];
ry(-2.554791716503232) q[15];
rz(0.5243137204173705) q[15];
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
ry(1.167611848086305) q[0];
rz(1.9527939126558342) q[0];
ry(1.6384290816238507) q[1];
rz(1.7240871492347372) q[1];
ry(-2.9772477403851316) q[2];
rz(-1.2666396983579138) q[2];
ry(-2.7235153203387137) q[3];
rz(-2.9447266103803824) q[3];
ry(3.1407078905695514) q[4];
rz(-2.397148875727918) q[4];
ry(-0.8341921762149251) q[5];
rz(-0.10982062515088398) q[5];
ry(3.1398520311758182) q[6];
rz(1.3093286055567912) q[6];
ry(3.141339801611738) q[7];
rz(2.956550653031271) q[7];
ry(1.704632705744729) q[8];
rz(2.3000014756002423) q[8];
ry(2.9702965026219688) q[9];
rz(1.9324703715643898) q[9];
ry(1.308225758191842) q[10];
rz(3.002981938823317) q[10];
ry(2.166184578774704) q[11];
rz(2.9671346541712857) q[11];
ry(3.140836416404635) q[12];
rz(3.120980017010034) q[12];
ry(-1.090012583932423) q[13];
rz(1.8883142501691421) q[13];
ry(1.9175146261384992) q[14];
rz(2.2536471427898164) q[14];
ry(1.1135116303067063) q[15];
rz(0.30169281658651936) q[15];
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
ry(-2.5625836647942095) q[0];
rz(-3.0686839599190487) q[0];
ry(-1.9191472199075372) q[1];
rz(3.130239198993053) q[1];
ry(1.5178648415564933) q[2];
rz(0.7166981082871083) q[2];
ry(2.866533060453116) q[3];
rz(-0.9855567550393268) q[3];
ry(2.5151642410881943) q[4];
rz(1.6587527319572566) q[4];
ry(-1.9497568815023225) q[5];
rz(2.984299379441031) q[5];
ry(-1.3027847189915205) q[6];
rz(2.98967678969747) q[6];
ry(-0.00010210008635247055) q[7];
rz(2.396374661263021) q[7];
ry(-0.000444660709077798) q[8];
rz(3.075958542336723) q[8];
ry(3.141383065184387) q[9];
rz(-0.5477182578591631) q[9];
ry(-0.0008566776644896024) q[10];
rz(-2.8256397190011655) q[10];
ry(-2.844078006565949) q[11];
rz(-2.2871442473812142) q[11];
ry(-3.14120012447805) q[12];
rz(2.919016077819835) q[12];
ry(2.794738089931561) q[13];
rz(3.140121753872071) q[13];
ry(-0.30322711184745804) q[14];
rz(-2.2304061075284114) q[14];
ry(1.9778735231283848) q[15];
rz(-0.9929725239986098) q[15];
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
ry(-2.254626686144988) q[0];
rz(1.9247137788554947) q[0];
ry(1.1258328052459337) q[1];
rz(-0.9270511864128977) q[1];
ry(0.3360093367643781) q[2];
rz(-1.9314095760739012) q[2];
ry(-1.6532803688188595) q[3];
rz(2.3826690542600493) q[3];
ry(3.140418957442664) q[4];
rz(-0.055066193920791484) q[4];
ry(-0.6610254236257314) q[5];
rz(-1.7166265945562218) q[5];
ry(-2.2672510420809946) q[6];
rz(-1.605983831511491) q[6];
ry(0.0014760654497987249) q[7];
rz(2.986991072944326) q[7];
ry(-2.0999618181442115) q[8];
rz(1.0985828660355743) q[8];
ry(3.1409599238885946) q[9];
rz(-1.9455793814981996) q[9];
ry(-1.6388498610543922) q[10];
rz(-0.5140331567011369) q[10];
ry(3.1411656119444586) q[11];
rz(0.6801688664085231) q[11];
ry(0.00041403323730509296) q[12];
rz(1.753237594063651) q[12];
ry(-0.0018330003711920625) q[13];
rz(0.15689019389578274) q[13];
ry(-0.7853421430023806) q[14];
rz(1.513724714252172) q[14];
ry(0.0019694564064650223) q[15];
rz(-2.1508053884886555) q[15];
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
ry(1.965225895195524) q[0];
rz(-2.4347117041673365) q[0];
ry(-0.5740314948819041) q[1];
rz(2.375089918468486) q[1];
ry(2.204235672830521) q[2];
rz(1.9718187141897383) q[2];
ry(-3.040070897616039) q[3];
rz(-1.7681442134585037) q[3];
ry(3.132634807095688) q[4];
rz(-2.8827222597735815) q[4];
ry(-0.1527889445142117) q[5];
rz(3.0168689458174334) q[5];
ry(-3.1401650502971257) q[6];
rz(2.5204420266607412) q[6];
ry(-3.141570133061939) q[7];
rz(-2.7777064575706456) q[7];
ry(1.161632996804668) q[8];
rz(0.3943497194172947) q[8];
ry(-1.9739586206248259) q[9];
rz(1.7694574324660346) q[9];
ry(3.1382177186741393) q[10];
rz(2.801834447587521) q[10];
ry(-3.1248963705739774) q[11];
rz(1.3547823880308156) q[11];
ry(-3.1413019638262143) q[12];
rz(2.7400276740689424) q[12];
ry(1.9268412808401871) q[13];
rz(1.64337318583794) q[13];
ry(-1.2601799427096285) q[14];
rz(-2.830929517268758) q[14];
ry(-1.047803674186552) q[15];
rz(2.9170050095216333) q[15];
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
ry(2.029352533924415) q[0];
rz(-2.367301105042388) q[0];
ry(-0.9863402167589237) q[1];
rz(-1.2165399273549609) q[1];
ry(2.698786719691545) q[2];
rz(0.87948509034337) q[2];
ry(-2.0412837871354146) q[3];
rz(1.6621117870384337) q[3];
ry(0.5224397632585367) q[4];
rz(0.953470004157807) q[4];
ry(1.6095204056486958) q[5];
rz(1.2165528607905245) q[5];
ry(1.5940693586514048) q[6];
rz(1.3129344375551393) q[6];
ry(-3.132346108948583) q[7];
rz(0.8030405691264846) q[7];
ry(3.1412057398028077) q[8];
rz(-2.631538519193937) q[8];
ry(0.00036559596104979164) q[9];
rz(-1.6142985888531003) q[9];
ry(-0.0004362905395831618) q[10];
rz(0.856890226528461) q[10];
ry(3.141485124857295) q[11];
rz(0.47809001952943125) q[11];
ry(-3.0591907206898843) q[12];
rz(2.5258734973369226) q[12];
ry(-1.018760441890426) q[13];
rz(-0.04184824788199659) q[13];
ry(2.4907832701886434) q[14];
rz(-0.16465408972859932) q[14];
ry(-0.1903163883885277) q[15];
rz(3.129094441766925) q[15];
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
ry(1.943729595382309) q[0];
rz(-2.5983750022621783) q[0];
ry(3.028405905223984) q[1];
rz(1.7762894114792669) q[1];
ry(1.7724595245642565) q[2];
rz(0.004112508378538494) q[2];
ry(1.611267524196836) q[3];
rz(1.5973809662229141) q[3];
ry(-3.1355703954523473) q[4];
rz(-2.2338444980147183) q[4];
ry(3.135352291670654) q[5];
rz(3.0164027801644133) q[5];
ry(0.010035930474697531) q[6];
rz(-2.7916801282994848) q[6];
ry(-0.00041381203949669904) q[7];
rz(-0.8445468714522513) q[7];
ry(-0.9764002068978161) q[8];
rz(-1.7144361504655392) q[8];
ry(1.5830610886773175) q[9];
rz(-1.2614854778056974) q[9];
ry(0.002177141683915656) q[10];
rz(2.234841301340409) q[10];
ry(-1.5668627766404137) q[11];
rz(-1.1156076803644936) q[11];
ry(0.0004118568141733579) q[12];
rz(-1.6483745602349609) q[12];
ry(-1.4924041899309544) q[13];
rz(-0.04335225706883793) q[13];
ry(-3.1213705936268012) q[14];
rz(-1.0660056116250816) q[14];
ry(1.5700046699647976) q[15];
rz(-0.3118560656015838) q[15];
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
ry(0.00018721086045346937) q[0];
rz(-0.3124382229798897) q[0];
ry(-1.8738599023305074) q[1];
rz(2.132351281429475) q[1];
ry(-0.7226997158844384) q[2];
rz(-1.5494859810884947) q[2];
ry(-1.563091567001238) q[3];
rz(3.1352734502913884) q[3];
ry(-3.01834848299095) q[4];
rz(2.2679191850648532) q[4];
ry(-2.147748268820033) q[5];
rz(-1.7714515188538114) q[5];
ry(-2.0854156661639918) q[6];
rz(-1.5708714859104527) q[6];
ry(-3.1307185446566614) q[7];
rz(0.06273022146256224) q[7];
ry(3.1414676040556557) q[8];
rz(3.0603348364548983) q[8];
ry(-0.0003371801503626517) q[9];
rz(0.08898573767414313) q[9];
ry(0.0003493015027213886) q[10];
rz(0.5360309105613243) q[10];
ry(-3.138941816671074) q[11];
rz(2.8604233209517287) q[11];
ry(-1.5623833552716961) q[12];
rz(-1.2820346370719902) q[12];
ry(2.8362829828137097) q[13];
rz(1.459371604134783) q[13];
ry(0.20068834667903027) q[14];
rz(-0.9787535717018602) q[14];
ry(0.012902684468420682) q[15];
rz(1.8884131667363668) q[15];
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
ry(1.321344275494621) q[0];
rz(1.5694661768616074) q[0];
ry(3.129499161621439) q[1];
rz(2.077859664745767) q[1];
ry(-1.5708927818798442) q[2];
rz(-1.6950787961135327) q[2];
ry(0.5573559927888265) q[3];
rz(1.5813004393214587) q[3];
ry(8.348874012342323e-05) q[4];
rz(2.9335416977013167) q[4];
ry(-2.85147480292293) q[5];
rz(-2.2561547263850863) q[5];
ry(-0.0010699070266095845) q[6];
rz(1.3040361177979654) q[6];
ry(-0.0011997798112659554) q[7];
rz(-2.871215132979341) q[7];
ry(1.5242775502137569) q[8];
rz(-1.0530803435213398) q[8];
ry(-0.5320650260262266) q[9];
rz(2.062427201027382) q[9];
ry(3.141135574586599) q[10];
rz(-1.4508679952375463) q[10];
ry(-0.005964972873219843) q[11];
rz(-2.5146517947770355) q[11];
ry(-3.141454032234371) q[12];
rz(-2.848472426295049) q[12];
ry(-0.9148009447758836) q[13];
rz(2.5849865523118765) q[13];
ry(2.5528109980577085e-05) q[14];
rz(1.7488129617729244) q[14];
ry(0.0020570534914687855) q[15];
rz(0.6655576926004975) q[15];
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
ry(1.8976873699861905) q[0];
rz(1.4064753099720502) q[0];
ry(-1.8288330237159176) q[1];
rz(-3.1076170457692456) q[1];
ry(-1.569410401625671) q[2];
rz(1.0172578091935058) q[2];
ry(-1.5622024304075857) q[3];
rz(-0.6020680386607774) q[3];
ry(0.0014051548682258964) q[4];
rz(-0.31596525189218655) q[4];
ry(0.020041839507315144) q[5];
rz(-1.5027188875082913) q[5];
ry(0.5828083702643649) q[6];
rz(0.017650082796845008) q[6];
ry(3.1349408091812414) q[7];
rz(0.3132007152299287) q[7];
ry(8.88908336538208e-06) q[8];
rz(-1.595149724936354) q[8];
ry(-0.000253677970290504) q[9];
rz(0.9446703246161527) q[9];
ry(1.571319782395384) q[10];
rz(0.9237139524126388) q[10];
ry(0.0027109364906641176) q[11];
rz(-3.0332015195586957) q[11];
ry(-1.697336084206599) q[12];
rz(0.006140434593594256) q[12];
ry(0.02594628788498543) q[13];
rz(2.1953444186713638) q[13];
ry(-1.4260472164709261) q[14];
rz(-2.1545718592584935) q[14];
ry(-3.139790887180753) q[15];
rz(-2.4678071225154734) q[15];
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
ry(-1.5700932178011147) q[0];
rz(3.1323453474329837) q[0];
ry(1.666307410857701) q[1];
rz(1.056697988146491) q[1];
ry(3.1286880348108204) q[2];
rz(2.583596936610308) q[2];
ry(-0.45514561375307316) q[3];
rz(1.7626121795634657) q[3];
ry(0.0002407868687868131) q[4];
rz(1.6631330280268788) q[4];
ry(3.139184787524435) q[5];
rz(2.5353759218513896) q[5];
ry(1.5790780564271145) q[6];
rz(-0.005886234183618846) q[6];
ry(-0.0003453095481171787) q[7];
rz(1.7482119332250572) q[7];
ry(-2.412794005776123) q[8];
rz(-1.288137645353516) q[8];
ry(-1.0700051764553906) q[9];
rz(0.06849164803836297) q[9];
ry(3.141442020665998) q[10];
rz(0.14323591978453276) q[10];
ry(1.5695727852512147) q[11];
rz(3.141455085771539) q[11];
ry(1.5705863818986794) q[12];
rz(-2.530007071519552) q[12];
ry(1.63350580372968) q[13];
rz(1.73413920626326) q[13];
ry(-1.5649389438410806) q[14];
rz(-2.1740841904157167) q[14];
ry(1.5741462490349791) q[15];
rz(-0.8296702177038998) q[15];
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
ry(-0.3906628438839954) q[0];
rz(3.121945080184516) q[0];
ry(1.561420632534557) q[1];
rz(-0.001476885618942752) q[1];
ry(-1.5727856032549896) q[2];
rz(1.7229487127514425) q[2];
ry(-1.574170111480429) q[3];
rz(3.13473399054816) q[3];
ry(0.002096975616550978) q[4];
rz(0.05648821627262672) q[4];
ry(-1.9558095688489177) q[5];
rz(2.1536052423061633) q[5];
ry(2.116384159216472) q[6];
rz(1.7953935268684595) q[6];
ry(-1.5688893830327284) q[7];
rz(-0.018647728166276245) q[7];
ry(0.00011999709235333) q[8];
rz(-0.5462359119888286) q[8];
ry(3.1412881641263795) q[9];
rz(-0.543469514580817) q[9];
ry(3.137235108865559) q[10];
rz(1.8797041138823483) q[10];
ry(-1.6599406172535938) q[11];
rz(-0.853125807368758) q[11];
ry(-0.028801945308371657) q[12];
rz(0.32813069054798744) q[12];
ry(-0.00024030132400363949) q[13];
rz(-2.6543411734966096) q[13];
ry(2.8834518594062475) q[14];
rz(2.796444249247637) q[14];
ry(-3.141104216497334) q[15];
rz(1.7954419495532052) q[15];
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
ry(1.5808202355513323) q[0];
rz(2.0526499325645364) q[0];
ry(1.5469188611180194) q[1];
rz(1.8008672419263636) q[1];
ry(-1.54260001477495) q[2];
rz(0.39924061037195635) q[2];
ry(-2.5447062787525283) q[3];
rz(-3.12480214257467) q[3];
ry(0.0008045945172492353) q[4];
rz(-1.9872394058654705) q[4];
ry(0.23381759166684835) q[5];
rz(3.098416912077813) q[5];
ry(-3.137770624089332) q[6];
rz(-1.3428724831607695) q[6];
ry(-3.141244166332895) q[7];
rz(-0.5425263624849704) q[7];
ry(-1.560212413393483) q[8];
rz(-2.441018752179985) q[8];
ry(-1.8066847215164545) q[9];
rz(0.5016937884697992) q[9];
ry(0.0005550952118973385) q[10];
rz(0.4738687315513682) q[10];
ry(3.1412865249658397) q[11];
rz(2.836971495486041) q[11];
ry(-9.296497876043475e-05) q[12];
rz(1.078299804590447) q[12];
ry(0.0007871048572525297) q[13];
rz(-2.4212359715866585) q[13];
ry(-3.1078061615125536) q[14];
rz(-2.8684844999664483) q[14];
ry(-0.003472008000598368) q[15];
rz(-0.117833380287154) q[15];
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
ry(0.0059401048831845505) q[0];
rz(-0.35702010563144393) q[0];
ry(2.826550045086529) q[1];
rz(1.5508768317738904) q[1];
ry(0.6057698597770944) q[2];
rz(0.1503015219137982) q[2];
ry(2.0947658749143088) q[3];
rz(3.1412346466906143) q[3];
ry(-0.0023710848820315752) q[4];
rz(0.5781929508109075) q[4];
ry(0.300930139667023) q[5];
rz(0.025360614190014452) q[5];
ry(-1.7248001626131013) q[6];
rz(0.3238427113849349) q[6];
ry(-3.1386876523120653) q[7];
rz(-2.5740524673108656) q[7];
ry(3.141403768398906) q[8];
rz(-1.2770650967294732) q[8];
ry(-3.1414400294133) q[9];
rz(0.5214592868093906) q[9];
ry(1.5701023140602919) q[10];
rz(0.17245662252641483) q[10];
ry(-9.441971734247545e-05) q[11];
rz(2.5933216736243585) q[11];
ry(3.1228009442485933) q[12];
rz(-1.1255013315328106) q[12];
ry(0.000838177402108344) q[13];
rz(-1.4417053136605875) q[13];
ry(-1.359445721341575) q[14];
rz(-0.20525161307768694) q[14];
ry(-3.1415879128088657) q[15];
rz(2.306331422139673) q[15];
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
ry(1.5623747019561494) q[0];
rz(-3.137547225673065) q[0];
ry(-1.5716186159810643) q[1];
rz(1.5647006519219868) q[1];
ry(0.017539541220498205) q[2];
rz(-1.8661987394694646) q[2];
ry(-1.5480299408223321) q[3];
rz(0.012144129704948272) q[3];
ry(3.051158138360959) q[4];
rz(-1.6024178180442026) q[4];
ry(0.1021931751148669) q[5];
rz(1.900613976458163) q[5];
ry(-3.1339471014370752) q[6];
rz(2.010453089837328) q[6];
ry(-1.5707150093084086) q[7];
rz(1.570504908692696) q[7];
ry(1.5699700324524661) q[8];
rz(-1.6505140408061045) q[8];
ry(0.363794905767441) q[9];
rz(0.8694384910698183) q[9];
ry(-0.0002893644921266869) q[10];
rz(-0.7330619925978796) q[10];
ry(1.5709264342704419) q[11];
rz(1.570736656062949) q[11];
ry(-1.571346808474177) q[12];
rz(-3.0785879420682254) q[12];
ry(-0.0003496676086758299) q[13];
rz(3.0472056532143728) q[13];
ry(1.5775392220108742) q[14];
rz(0.07033864117412587) q[14];
ry(3.1409592797980026) q[15];
rz(-1.0130561781792713) q[15];
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
ry(-1.6102941627908522) q[0];
rz(1.5838532418507996) q[0];
ry(1.3090903217775096) q[1];
rz(1.569385474750666) q[1];
ry(3.1400999068791884) q[2];
rz(-0.32560725167440463) q[2];
ry(1.5829871650298974) q[3];
rz(3.097630220947933) q[3];
ry(3.1411429358727423) q[4];
rz(0.9885710117939934) q[4];
ry(-0.0001779427322447686) q[5];
rz(-1.2117163866960148) q[5];
ry(-3.1414635138008786) q[6];
rz(-2.7373579224161855) q[6];
ry(-1.5771514822297705) q[7];
rz(0.001186130639195217) q[7];
ry(3.141089662694086) q[8];
rz(0.5208483531252631) q[8];
ry(-1.5706502822015977) q[9];
rz(1.5728887557597275) q[9];
ry(0.00028033245919440174) q[10];
rz(3.1300247667649423) q[10];
ry(-1.570896907738506) q[11];
rz(0.9684901941677095) q[11];
ry(0.07059712215587534) q[12];
rz(0.28334398945529227) q[12];
ry(1.570781829258222) q[13];
rz(3.1414373853260407) q[13];
ry(1.5685263197524941) q[14];
rz(-1.5722911127536543) q[14];
ry(-0.0011171081524050663) q[15];
rz(-0.042929198636425525) q[15];
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
ry(0.3612706492313205) q[0];
rz(1.5672775179900524) q[0];
ry(1.5596247548853865) q[1];
rz(-3.137081664012387) q[1];
ry(3.137474058299454) q[2];
rz(0.08365752732924925) q[2];
ry(1.588349922992439) q[3];
rz(0.44261512273490927) q[3];
ry(1.4945565450210363) q[4];
rz(0.06471866125951119) q[4];
ry(-0.00011045476910354068) q[5];
rz(2.294507206481618) q[5];
ry(-3.1415672347557697) q[6];
rz(-2.887620240011023) q[6];
ry(1.5713746254926275) q[7];
rz(-0.15406095879204165) q[7];
ry(3.1415376674338944) q[8];
rz(-3.1262897717139992) q[8];
ry(0.13430509038330118) q[9];
rz(3.139933060663207) q[9];
ry(3.1409298130848256) q[10];
rz(-1.5119319657221748) q[10];
ry(0.5935740418042594) q[11];
rz(-0.12516861764910026) q[11];
ry(3.141522703254681) q[12];
rz(-0.32273486628632053) q[12];
ry(1.5707375497213132) q[13];
rz(-3.1414995266486616) q[13];
ry(-1.5706518838845158) q[14];
rz(0.6090460565765656) q[14];
ry(3.4291504734440537e-05) q[15];
rz(0.8533163333485576) q[15];
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
ry(1.2503837985203363) q[0];
rz(-3.1248274647432988) q[0];
ry(-1.5823230866530338) q[1];
rz(1.5599438088020567) q[1];
ry(-1.582314532248965) q[2];
rz(-3.13665430250177) q[2];
ry(1.5507596264549506) q[3];
rz(-0.7096799572719635) q[3];
ry(0.0025452797330949295) q[4];
rz(-3.1170040997776574) q[4];
ry(-3.1415191730058005) q[5];
rz(-0.21623480598819975) q[5];
ry(-3.1410319505557425) q[6];
rz(0.9899432926358901) q[6];
ry(-1.577086868849734) q[7];
rz(-1.5718623056231928) q[7];
ry(-0.0004264400512568045) q[8];
rz(2.302160548768537) q[8];
ry(-1.57070096120318) q[9];
rz(0.18370818554986362) q[9];
ry(-2.643888061903965e-05) q[10];
rz(2.4186676483589844) q[10];
ry(-3.141425784449453) q[11];
rz(-0.3010619994670307) q[11];
ry(-0.00012906898040920822) q[12];
rz(-2.8476634576408406) q[12];
ry(-1.5707489846244975) q[13];
rz(7.073893101670625e-06) q[13];
ry(-0.09679185985020224) q[14];
rz(-0.9994189408422107) q[14];
ry(-1.5695730750499237) q[15];
rz(-2.7962136007637994) q[15];
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
ry(-1.5736098376349341) q[0];
rz(2.5810052075167147) q[0];
ry(-1.5771931580706329) q[1];
rz(-1.9827268219428724) q[1];
ry(-0.004144087084266168) q[2];
rz(1.0029799476550743) q[2];
ry(0.00013253947355629236) q[3];
rz(1.9101693919934668) q[3];
ry(3.128503775260157) q[4];
rz(-0.5117174886676085) q[4];
ry(1.553669197055938) q[5];
rz(2.4996225749540324) q[5];
ry(-3.1379384427847645) q[6];
rz(-1.1090870901120935) q[6];
ry(1.571077745500938) q[7];
rz(0.5512871983516803) q[7];
ry(-0.00014657003530809025) q[8];
rz(-2.235330009038128) q[8];
ry(0.0008119682669525118) q[9];
rz(-1.9764832651521909) q[9];
ry(-0.0004187037043825726) q[10];
rz(-2.044032597709343) q[10];
ry(-0.00011604525437292068) q[11];
rz(0.6944366796226138) q[11];
ry(3.141039173554787) q[12];
rz(-0.9403228867756965) q[12];
ry(1.5705946817790881) q[13];
rz(-1.8595863114653497) q[13];
ry(-0.0005789183450589227) q[14];
rz(2.5479938660869226) q[14];
ry(3.141030392829148) q[15];
rz(-1.5797292807329786) q[15];