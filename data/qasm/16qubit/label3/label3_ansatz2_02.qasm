OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(-0.8055150913110986) q[0];
rz(-3.1393232131782267) q[0];
ry(-0.04412781845314483) q[1];
rz(3.138473614959466) q[1];
ry(-1.5771634167844566) q[2];
rz(0.000136213765888904) q[2];
ry(-8.238250360534526e-05) q[3];
rz(-1.803148816067266) q[3];
ry(0.6113521992352852) q[4];
rz(-3.140762321686572) q[4];
ry(3.0837274021436087) q[5];
rz(0.0006638316754955509) q[5];
ry(-3.013544416336963) q[6];
rz(0.000612381116516862) q[6];
ry(-0.019984310903880698) q[7];
rz(0.0010618921294689445) q[7];
ry(3.141497793275886) q[8];
rz(2.7385451345696046) q[8];
ry(-3.1331876841610127) q[9];
rz(-0.0071202031017829626) q[9];
ry(-3.1141459769626443) q[10];
rz(-3.1410328404941574) q[10];
ry(-0.003294502674076802) q[11];
rz(-0.03739291104846236) q[11];
ry(-2.7543384089631392e-08) q[12];
rz(-1.1720721848125786) q[12];
ry(-3.139082221796187) q[13];
rz(-0.005934944319953672) q[13];
ry(0.003064076165396834) q[14];
rz(3.117649583673922) q[14];
ry(0.002205426024297855) q[15];
rz(0.08364338151621778) q[15];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(3.0910718779072246) q[0];
rz(-1.5682472962135634) q[0];
ry(-0.004126053936645846) q[1];
rz(1.5737957580701405) q[1];
ry(1.462601564331399) q[2];
rz(-1.570278272292576) q[2];
ry(-0.006060867155911499) q[3];
rz(3.134687630190742) q[3];
ry(0.180443928581359) q[4];
rz(-1.573319882295308) q[4];
ry(3.1172836086544873) q[5];
rz(-1.5701603921184883) q[5];
ry(2.450364710947614) q[6];
rz(-1.569701332855187) q[6];
ry(-3.1123756464974877) q[7];
rz(-1.5701560601394378) q[7];
ry(-1.5708150950626711) q[8];
rz(-0.03294683249394392) q[8];
ry(-0.10155329614201936) q[9];
rz(-1.5706169237143481) q[9];
ry(0.6547992662326108) q[10];
rz(1.569315231719167) q[10];
ry(0.04876323517546677) q[11];
rz(-1.5562834846203213) q[11];
ry(-5.740170698457772e-08) q[12];
rz(3.0249756776681704) q[12];
ry(3.1345036913142716) q[13];
rz(1.793982329081457) q[13];
ry(-0.18601609725239343) q[14];
rz(1.5733862655540447) q[14];
ry(3.1265387401069953) q[15];
rz(-2.7852887633972863) q[15];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(-3.030491373892679) q[0];
rz(-1.602469197116525) q[0];
ry(-2.2620340781431194) q[1];
rz(1.58931752614787) q[1];
ry(-3.038051141006631) q[2];
rz(1.5625349805367277) q[2];
ry(1.5706927247183613) q[3];
rz(-3.1311419516300356) q[3];
ry(-0.018803739025119448) q[4];
rz(-1.5708057319004578) q[4];
ry(-0.5806138630474201) q[5];
rz(1.5547283950130153) q[5];
ry(-0.03793353433318845) q[6];
rz(-1.5716199546790337) q[6];
ry(3.0096204978648124) q[7];
rz(1.5720273188196519) q[7];
ry(1.5707335735605796) q[8];
rz(3.1412906604665243) q[8];
ry(0.03048682869668351) q[9];
rz(1.5709226310025008) q[9];
ry(-0.0195746310417384) q[10];
rz(1.5724015276329188) q[10];
ry(3.1361290296565634) q[11];
rz(1.5847447223359343) q[11];
ry(2.4038371725811113e-08) q[12];
rz(-1.1340135741924753) q[12];
ry(-3.1412717975940456) q[13];
rz(1.7940320920184023) q[13];
ry(-3.1350179344660094) q[14];
rz(-1.5680869096322505) q[14];
ry(-3.141547837328825) q[15];
rz(-2.784773582514026) q[15];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(-0.0073253835057593975) q[0];
rz(-3.1089071555649963) q[0];
ry(0.007827771368597247) q[1];
rz(3.1072799402389557) q[1];
ry(-3.1306727365689775) q[2];
rz(-0.008963925958280819) q[2];
ry(1.5710442431308822) q[3];
rz(1.5789519543779933) q[3];
ry(-3.12598030975339) q[4];
rz(-0.0021535170860201196) q[4];
ry(3.132288426575562) q[5];
rz(-0.019846277339188485) q[5];
ry(3.099479884673898) q[6];
rz(-3.141378015393883) q[6];
ry(-0.03256913784439116) q[7];
rz(3.140192219886118) q[7];
ry(-3.076044947472311) q[8];
rz(3.1413167579267642) q[8];
ry(0.13443314031479142) q[9];
rz(0.0002986171778891816) q[9];
ry(0.07219764005243585) q[10];
rz(3.1414566354573776) q[10];
ry(2.565475881770658) q[11];
rz(-0.051753635378333614) q[11];
ry(-3.14159256479959) q[12];
rz(-0.2188308340607771) q[12];
ry(-1.5893907702021608) q[13];
rz(-3.141531762762777) q[13];
ry(0.04814010176299721) q[14];
rz(3.1414646807373483) q[14];
ry(-0.8317222955158359) q[15];
rz(3.141089686764194) q[15];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(-3.114753758044148) q[0];
rz(-2.351446715374313) q[0];
ry(-3.136772322523965) q[1];
rz(0.7938033990970964) q[1];
ry(3.1032023821484804) q[2];
rz(-0.2223936948625719) q[2];
ry(0.01021975775284469) q[3];
rz(-0.3305156071392439) q[3];
ry(-0.041927489430860235) q[4];
rz(-1.2892073173670973) q[4];
ry(0.012309317008607934) q[5];
rz(1.4575870684966974) q[5];
ry(0.15943217561488737) q[6];
rz(2.132887503966783) q[6];
ry(3.1272306171274247) q[7];
rz(1.1495700517353875) q[7];
ry(0.361619933753885) q[8];
rz(-1.1463451451666273) q[8];
ry(-0.01730562174769959) q[9];
rz(-1.1791835751214181) q[9];
ry(-0.6505993345574386) q[10];
rz(-2.1538080032847935) q[10];
ry(-0.0011131965858677262) q[11];
rz(2.6635915615376238) q[11];
ry(1.570796461828273) q[12];
rz(-5.928850654868256e-08) q[12];
ry(-1.4940345269554314) q[13];
rz(0.10257835459791666) q[13];
ry(2.1081775427846496) q[14];
rz(-1.8787003699749703) q[14];
ry(0.1044252411008027) q[15];
rz(-0.09506954940541235) q[15];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(5.828570515120646e-08) q[0];
rz(-1.567255272181073) q[0];
ry(1.0932134807817087e-08) q[1];
rz(-1.5873591213979772) q[1];
ry(3.1415925616557585) q[2];
rz(2.14153750113939) q[2];
ry(3.390839786732158e-08) q[3];
rz(-0.4554002497875079) q[3];
ry(2.9171626092505676e-07) q[4];
rz(0.5118275700116328) q[4];
ry(3.141592612732072) q[5];
rz(-2.465550320253297) q[5];
ry(-3.141592407863582) q[6];
rz(1.3556640715369053) q[6];
ry(2.9777183005264354e-07) q[7];
rz(1.2141387237568377) q[7];
ry(-5.2510500836433494e-08) q[8];
rz(-2.7724499356454433) q[8];
ry(3.1415925041389734) q[9];
rz(-1.955770598866927) q[9];
ry(-3.141592643176624) q[10];
rz(0.21059861175430158) q[10];
ry(-3.1415925403147114) q[11];
rz(-1.3066619690446513) q[11];
ry(1.5707964217693045) q[12];
rz(2.3644229182033154) q[12];
ry(-3.1415923402187906) q[13];
rz(-0.6743272789774304) q[13];
ry(-3.7627858282657984e-08) q[14];
rz(1.1015283037856491) q[14];
ry(-3.141592395183789) q[15];
rz(-0.872480242405021) q[15];