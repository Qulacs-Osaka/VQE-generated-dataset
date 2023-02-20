OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(-2.6453226594956503) q[0];
rz(-0.702897325084397) q[0];
ry(2.875418105178778) q[1];
rz(-0.5212752773627054) q[1];
ry(-0.0017494905398897401) q[2];
rz(-1.3468559294347235) q[2];
ry(0.2714324597120532) q[3];
rz(1.5695819510444016) q[3];
ry(-1.6421590563192787) q[4];
rz(1.2854492284685481) q[4];
ry(0.03209989439675898) q[5];
rz(2.3731567340823263) q[5];
ry(-0.31469397736338584) q[6];
rz(-1.8209342681334726) q[6];
ry(-2.9683032298611085) q[7];
rz(2.510304404792905) q[7];
ry(-0.01118700262050451) q[8];
rz(-2.957147192752942) q[8];
ry(1.1115124926820457) q[9];
rz(-3.138375527124249) q[9];
ry(3.14132653707683) q[10];
rz(0.6347724690958039) q[10];
ry(-1.7783042702949963) q[11];
rz(0.028530051182010574) q[11];
ry(-0.8974449550274214) q[12];
rz(-0.4551502489264653) q[12];
ry(0.006452819718502943) q[13];
rz(0.7875556118479848) q[13];
ry(-0.0038551571414138135) q[14];
rz(-2.941991160266359) q[14];
ry(-0.1277038437464202) q[15];
rz(-0.6214058471333567) q[15];
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
ry(-1.0735812949605021) q[0];
rz(-1.4183053835983044) q[0];
ry(-2.976454190458715) q[1];
rz(-2.9579024411225703) q[1];
ry(-1.2104030608322185) q[2];
rz(-1.436743790101815) q[2];
ry(1.9031886350580862) q[3];
rz(0.2270167356467621) q[3];
ry(-2.503768519417471) q[4];
rz(2.2038947340765374) q[4];
ry(3.1330944895948734) q[5];
rz(-2.5196211726867093) q[5];
ry(0.00011224792818514376) q[6];
rz(-1.527970684599373) q[6];
ry(3.1368186012813806) q[7];
rz(2.61558986452125) q[7];
ry(3.109192341324076) q[8];
rz(2.601296360716665) q[8];
ry(-2.8403868236636653) q[9];
rz(1.6318696881785666) q[9];
ry(-0.0166364153301615) q[10];
rz(1.6598870799267436) q[10];
ry(2.8249574586167987) q[11];
rz(-3.0984905964095515) q[11];
ry(0.7438431626055865) q[12];
rz(-0.03580405643653606) q[12];
ry(3.140083026909874) q[13];
rz(1.7096437220424425) q[13];
ry(-1.7076295809136288) q[14];
rz(2.1354743680928507) q[14];
ry(-3.050782733273281) q[15];
rz(1.8478982485595725) q[15];
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
ry(2.717291015354153) q[0];
rz(0.284460751346785) q[0];
ry(0.7620145865752413) q[1];
rz(1.6131265685170408) q[1];
ry(-0.0009124870971977117) q[2];
rz(-1.962082758386171) q[2];
ry(1.020397344903479) q[3];
rz(-1.8880279129785729) q[3];
ry(-3.107320785896402) q[4];
rz(2.71166292361387) q[4];
ry(-3.1346998760303513) q[5];
rz(-2.978706173443832) q[5];
ry(0.7417828833054453) q[6];
rz(2.6775484146373003) q[6];
ry(2.945878061884898) q[7];
rz(2.0775570455252383) q[7];
ry(2.8455660638290214) q[8];
rz(-0.3317834492254773) q[8];
ry(3.0828577310958587) q[9];
rz(1.7531958932312133) q[9];
ry(2.4153081792845277) q[10];
rz(-1.6698422854456645) q[10];
ry(1.4758180134613366) q[11];
rz(3.1237033301122445) q[11];
ry(-3.13534975385369) q[12];
rz(-1.9705691567094585) q[12];
ry(-3.020018208613605) q[13];
rz(1.635145842254941) q[13];
ry(0.010356806202044783) q[14];
rz(0.1820145992419855) q[14];
ry(2.764507792385699) q[15];
rz(2.395594035504575) q[15];
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
ry(0.7858913560935725) q[0];
rz(-1.6104888282771739) q[0];
ry(2.734562810838592) q[1];
rz(2.7513560219750013) q[1];
ry(-2.141616978069486) q[2];
rz(-0.8051276638809772) q[2];
ry(-1.325827671550519) q[3];
rz(1.5644675953560765) q[3];
ry(2.7388177261142377) q[4];
rz(-2.704573937045617) q[4];
ry(0.006431374219615371) q[5];
rz(3.001243497151917) q[5];
ry(-3.1398365924707234) q[6];
rz(-0.6672910524545507) q[6];
ry(3.1316890503325956) q[7];
rz(-0.936119886035665) q[7];
ry(-3.1398990792105805) q[8];
rz(2.8728138146400823) q[8];
ry(-2.0027677598837434) q[9];
rz(-1.1472892171593478) q[9];
ry(-3.138313762125146) q[10];
rz(-0.6165625670982794) q[10];
ry(0.009006763623764838) q[11];
rz(2.6866530501862376) q[11];
ry(0.26309140558061894) q[12];
rz(-1.0832017705964363) q[12];
ry(0.0009625334013667515) q[13];
rz(-1.7674221271026154) q[13];
ry(-0.2187807668217548) q[14];
rz(1.9151689417299076) q[14];
ry(-3.1358418238597165) q[15];
rz(-2.359159401989022) q[15];
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
ry(-1.8159092934094812) q[0];
rz(-0.25461873022784653) q[0];
ry(0.016351054559915888) q[1];
rz(-0.5741279594377398) q[1];
ry(3.1402001288530874) q[2];
rz(-2.167110051410222) q[2];
ry(0.8110579555592698) q[3];
rz(-0.2740234796061199) q[3];
ry(-0.052583388495510364) q[4];
rz(0.42561143750200725) q[4];
ry(-1.6481843421668128) q[5];
rz(2.26163365938923) q[5];
ry(-0.5518762483991958) q[6];
rz(-0.9226320370383941) q[6];
ry(2.209272409063626) q[7];
rz(-0.7024994248397094) q[7];
ry(0.12160961958862605) q[8];
rz(-1.3224879404127188) q[8];
ry(0.07224309808517582) q[9];
rz(1.0372514153816383) q[9];
ry(-2.4403279215374476) q[10];
rz(-2.8999805502848846) q[10];
ry(-1.537558600349474) q[11];
rz(0.8624720842248755) q[11];
ry(3.1335760862098296) q[12];
rz(2.324988598358337) q[12];
ry(-3.0187294717959947) q[13];
rz(0.8308761561035363) q[13];
ry(0.019163089058395148) q[14];
rz(-2.6893417764922254) q[14];
ry(1.4868182531436185) q[15];
rz(1.1390603907209074) q[15];
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
ry(-3.1172311803532495) q[0];
rz(-2.36717852362774) q[0];
ry(-1.8601765266703651) q[1];
rz(-1.8631948375595115) q[1];
ry(-0.01933476455397148) q[2];
rz(-0.5200350680040212) q[2];
ry(-0.002098325611761709) q[3];
rz(-1.3285936718366287) q[3];
ry(2.8440109017566835) q[4];
rz(2.433757803248761) q[4];
ry(-0.0007133331565159831) q[5];
rz(-2.5104301130518665) q[5];
ry(-0.0005317071440780192) q[6];
rz(-0.6085833420113307) q[6];
ry(-3.1300614300360183) q[7];
rz(-1.2768488222038232) q[7];
ry(-0.0006004272750402895) q[8];
rz(-0.6469835656693403) q[8];
ry(0.0648549858971359) q[9];
rz(-2.6046910939966095) q[9];
ry(3.133541395694823) q[10];
rz(-2.7776521646855246) q[10];
ry(0.009656972603741354) q[11];
rz(2.3016953644365175) q[11];
ry(2.0273972538319223) q[12];
rz(1.8461591565048607) q[12];
ry(-1.5898162494046575) q[13];
rz(-0.165556991477259) q[13];
ry(-3.0987397283767475) q[14];
rz(-1.4729877619794935) q[14];
ry(-1.4887615433796577) q[15];
rz(-1.492003182872283) q[15];
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
ry(-2.941533521392193) q[0];
rz(2.2478705077190244) q[0];
ry(-0.8581064036208821) q[1];
rz(0.24390982267169772) q[1];
ry(3.1391009278850652) q[2];
rz(0.516353161437209) q[2];
ry(1.5791287416208126) q[3];
rz(-0.4145287309127679) q[3];
ry(3.1116542440986388) q[4];
rz(-1.5090386986842808) q[4];
ry(-0.003945649849402762) q[5];
rz(1.1774429159803184) q[5];
ry(0.6055189710851805) q[6];
rz(-2.3537741221576765) q[6];
ry(-0.7932852489373811) q[7];
rz(3.03564468690442) q[7];
ry(3.091508463605149) q[8];
rz(-2.5891776501851664) q[8];
ry(1.5929835335375975) q[9];
rz(0.8723301255030197) q[9];
ry(0.022469995327613367) q[10];
rz(0.7627815178195433) q[10];
ry(1.5700024955593266) q[11];
rz(1.8289198977218497) q[11];
ry(3.1415581361607203) q[12];
rz(-1.3472542589361038) q[12];
ry(0.001043899846246998) q[13];
rz(2.5631482490790423) q[13];
ry(3.141348115052213) q[14];
rz(1.5881223412663648) q[14];
ry(1.5502199376189403) q[15];
rz(-1.5706077357056207) q[15];
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
ry(2.02412630654659) q[0];
rz(-0.06418561556810834) q[0];
ry(0.3901211054258802) q[1];
rz(-1.9842887798601092) q[1];
ry(0.7468862148613136) q[2];
rz(0.39222690083527834) q[2];
ry(-3.135926123381599) q[3];
rz(2.9998731128758522) q[3];
ry(0.4838143388913852) q[4];
rz(-2.961336874498836) q[4];
ry(0.0003159090038649811) q[5];
rz(0.18723596866787262) q[5];
ry(3.140335477423012) q[6];
rz(-0.7670614957342742) q[6];
ry(-0.00011358357944857289) q[7];
rz(-1.1063365755019152) q[7];
ry(-3.1402742649679674) q[8];
rz(-1.191298425302728) q[8];
ry(0.00030277300323311814) q[9];
rz(2.2892798539704473) q[9];
ry(3.084630762855363) q[10];
rz(1.562513471660539) q[10];
ry(0.2462292866125983) q[11];
rz(3.1391358531509024) q[11];
ry(-1.8779527908870266) q[12];
rz(-2.8558761915622983) q[12];
ry(3.137650702771958) q[13];
rz(1.1452729963035297) q[13];
ry(-1.5994113987434198) q[14];
rz(-1.5215095192543435) q[14];
ry(0.28493544494789536) q[15];
rz(1.520727518274799) q[15];
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
ry(0.5561519772346529) q[0];
rz(1.8958623720715224) q[0];
ry(0.09240323287014186) q[1];
rz(-1.9763667465448307) q[1];
ry(1.6043781667077022) q[2];
rz(-0.9314940173436304) q[2];
ry(2.7400858527094063) q[3];
rz(0.08727451448788334) q[3];
ry(-0.3349663853857985) q[4];
rz(1.6619182057968835) q[4];
ry(2.113855901179324) q[5];
rz(-3.008509508214058) q[5];
ry(-1.5475677471269425) q[6];
rz(-2.4012658116646226) q[6];
ry(2.630795269284014) q[7];
rz(-0.9705628915798895) q[7];
ry(1.5812530881073519) q[8];
rz(3.138697581897888) q[8];
ry(-1.5754996734684286) q[9];
rz(1.9216152994281985) q[9];
ry(-1.6242987970389526) q[10];
rz(-2.494289862269211) q[10];
ry(1.640529794408572) q[11];
rz(0.3057175649538877) q[11];
ry(1.1728819596423865) q[12];
rz(-2.3917385663547206) q[12];
ry(-3.1414994533090668) q[13];
rz(0.6195065899894523) q[13];
ry(-3.1310641223607463) q[14];
rz(-0.3223889646695816) q[14];
ry(1.9815623236063216) q[15];
rz(-0.7218690992197688) q[15];
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
ry(-2.050965160193993) q[0];
rz(0.7398534011705094) q[0];
ry(-0.37569250063885645) q[1];
rz(-1.2817342298258976) q[1];
ry(0.31201885683273917) q[2];
rz(-2.1015917898474656) q[2];
ry(0.0117093490105864) q[3];
rz(-1.9260305192937137) q[3];
ry(3.1407147142871374) q[4];
rz(-1.8535204059917696) q[4];
ry(3.1414653267750667) q[5];
rz(-1.8693650316228965) q[5];
ry(-3.1403890737146347) q[6];
rz(1.5715608646335986) q[6];
ry(-3.141331872001964) q[7];
rz(-1.2348800010268608) q[7];
ry(-0.8251683685216591) q[8];
rz(-1.4393496303130597) q[8];
ry(-1.5332979637303517) q[9];
rz(-1.584702665226196) q[9];
ry(0.0003391479065388836) q[10];
rz(-0.6786906560311774) q[10];
ry(0.009183113481312663) q[11];
rz(1.6937779100472852) q[11];
ry(3.119941796677516) q[12];
rz(-2.379241420353043) q[12];
ry(-1.5975097285924855) q[13];
rz(-1.9581491426469857) q[13];
ry(-1.5272353711978652) q[14];
rz(1.5311786265143832) q[14];
ry(-2.8267590979975936) q[15];
rz(-2.3730228008777785) q[15];
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
ry(0.05740122879767107) q[0];
rz(-0.9387900110562781) q[0];
ry(0.22441571874775063) q[1];
rz(-0.99860518799093) q[1];
ry(-1.816755691371669) q[2];
rz(2.6359500499615964) q[2];
ry(0.0369087943438416) q[3];
rz(0.2594471815632957) q[3];
ry(-0.49648675750503224) q[4];
rz(0.03948074371771396) q[4];
ry(-3.0174862111543) q[5];
rz(-1.3315558216195384) q[5];
ry(-1.6089780814272854) q[6];
rz(-1.6018056854565819) q[6];
ry(0.026136680005586735) q[7];
rz(-2.4579770559270386) q[7];
ry(2.741816940381146) q[8];
rz(-2.2874938034943053) q[8];
ry(1.582627656741174) q[9];
rz(0.4933181337002436) q[9];
ry(2.9492492771443612) q[10];
rz(-2.983069946822496) q[10];
ry(2.160382055707681e-06) q[11];
rz(1.1380725411212431) q[11];
ry(1.330868664790719) q[12];
rz(2.73473397523738) q[12];
ry(-0.00010279422162291717) q[13];
rz(-0.919467743773235) q[13];
ry(2.1706092789301024) q[14];
rz(3.1340258206418117) q[14];
ry(-1.3925290673613204) q[15];
rz(2.1871184821251797) q[15];
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
ry(1.627065556823747) q[0];
rz(1.2104286830488797) q[0];
ry(-0.15087334359291837) q[1];
rz(-1.5073041259237732) q[1];
ry(-2.6723296622080004) q[2];
rz(2.6664150963143176) q[2];
ry(-0.001234951919466131) q[3];
rz(-2.7317849228004847) q[3];
ry(3.140321811404632) q[4];
rz(0.621725883995678) q[4];
ry(-3.141210083030543) q[5];
rz(1.6625588930740822) q[5];
ry(-3.140359749048683) q[6];
rz(-2.897587903530617) q[6];
ry(0.00016630810826755887) q[7];
rz(-2.5771245937288247) q[7];
ry(0.005840544317685203) q[8];
rz(2.4331791401311276) q[8];
ry(0.015879892540522092) q[9];
rz(-0.062353450896269536) q[9];
ry(3.1415177929422664) q[10];
rz(1.8216585352030503) q[10];
ry(0.3493699017054969) q[11];
rz(2.749018949941846) q[11];
ry(0.0010346393067205552) q[12];
rz(2.0598598465118005) q[12];
ry(-3.1250458619792405) q[13];
rz(-1.434597333261639) q[13];
ry(0.5197590018395898) q[14];
rz(-0.11773907604516853) q[14];
ry(-3.0942645635937693) q[15];
rz(-0.0032579762466937723) q[15];
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
ry(-1.6742087592737098) q[0];
rz(2.4534226358863824) q[0];
ry(-1.4772168838896533) q[1];
rz(1.2170668803209472) q[1];
ry(-0.9700214360833291) q[2];
rz(-1.2522041317134112) q[2];
ry(2.707721936018076) q[3];
rz(2.2473551737349515) q[3];
ry(-0.22562599178139475) q[4];
rz(-0.35352830090485926) q[4];
ry(2.9240116630904196) q[5];
rz(-0.16680789217895514) q[5];
ry(-0.07266618500217524) q[6];
rz(-0.2927674742876416) q[6];
ry(3.1295543822386707) q[7];
rz(-0.41071355958626476) q[7];
ry(0.44387153282904684) q[8];
rz(-1.77504071929227) q[8];
ry(-1.7937052032031682) q[9];
rz(-2.8203621374857892) q[9];
ry(1.0646158111486717) q[10];
rz(-1.7660678745110892) q[10];
ry(-0.0005440704013949682) q[11];
rz(-0.9853348224064424) q[11];
ry(0.09369850321579365) q[12];
rz(-0.5539013381004025) q[12];
ry(-1.6574548287486466) q[13];
rz(-3.1408320321895618) q[13];
ry(2.6836891100884) q[14];
rz(0.5984222237685338) q[14];
ry(2.1838128446185574) q[15];
rz(1.176650477723034) q[15];
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
ry(-2.8820440749535985) q[0];
rz(-1.4269075933028528) q[0];
ry(2.7082754794337442) q[1];
rz(-2.538855355725033) q[1];
ry(-1.7874020804703112) q[2];
rz(0.651715166416021) q[2];
ry(3.1323272299544214) q[3];
rz(-2.1624563729413326) q[3];
ry(-3.139589424353813) q[4];
rz(-1.9621399968724464) q[4];
ry(0.00016253970979867547) q[5];
rz(-2.4448870421200786) q[5];
ry(-3.127029192155163) q[6];
rz(-1.59153841492811) q[6];
ry(0.00023635864365711967) q[7];
rz(-1.0541202495210613) q[7];
ry(0.0013299245752120825) q[8];
rz(-0.5185255804043231) q[8];
ry(-0.01857865130355396) q[9];
rz(0.8284619381038308) q[9];
ry(3.1414620840450707) q[10];
rz(1.6192945171960698) q[10];
ry(-3.141479720376666) q[11];
rz(-0.9571290959780354) q[11];
ry(3.139064614968774) q[12];
rz(-2.371802782816415) q[12];
ry(-0.3333029623895864) q[13];
rz(-0.9112203414443467) q[13];
ry(3.136923196333881) q[14];
rz(0.9115725626229477) q[14];
ry(-3.117222359191275) q[15];
rz(-1.3241777870045262) q[15];
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
ry(1.6526302568780826) q[0];
rz(-2.8008588424113237) q[0];
ry(-2.8171927273424164) q[1];
rz(1.2700614367138119) q[1];
ry(0.15847102332999619) q[2];
rz(1.5483460486275475) q[2];
ry(-1.6881539139457962) q[3];
rz(-2.871582451955353) q[3];
ry(-2.939323696183388) q[4];
rz(-1.9836231913037299) q[4];
ry(-2.910691595332924) q[5];
rz(1.6414924166412235) q[5];
ry(1.5767714395508934) q[6];
rz(3.1275581509884445) q[6];
ry(-0.026893486672245004) q[7];
rz(2.3518450871425616) q[7];
ry(0.0757473791012826) q[8];
rz(0.7227062997926713) q[8];
ry(-2.66485493553393) q[9];
rz(1.946482586272035) q[9];
ry(-2.0820223830998215) q[10];
rz(2.719255216403237) q[10];
ry(3.141019561207325) q[11];
rz(1.916001583032675) q[11];
ry(0.28534742636530686) q[12];
rz(-2.819496072049293) q[12];
ry(-3.001297824408609) q[13];
rz(2.235827178074378) q[13];
ry(-0.1277967314930386) q[14];
rz(-2.411222567193384) q[14];
ry(1.3534929282233725) q[15];
rz(1.6730423106621037) q[15];
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
ry(-1.8063195480857859) q[0];
rz(0.1485496248173516) q[0];
ry(1.2525569135918366) q[1];
rz(2.294077051812595) q[1];
ry(0.18520490123166233) q[2];
rz(-1.1465398715322157) q[2];
ry(0.016969530820294842) q[3];
rz(1.6382814900351372) q[3];
ry(-0.001358629043234372) q[4];
rz(-2.979658531631387) q[4];
ry(-1.6610398641270283) q[5];
rz(-0.1128164636184117) q[5];
ry(-3.1357599116880928) q[6];
rz(0.3429561505489547) q[6];
ry(0.00041780940620839147) q[7];
rz(-0.5046830162950708) q[7];
ry(2.3864574828983294) q[8];
rz(1.5666069126131912) q[8];
ry(1.3574409315247546) q[9];
rz(1.2338324395695532) q[9];
ry(3.1413152110052405) q[10];
rz(-0.4393423317769036) q[10];
ry(3.113899818066143) q[11];
rz(-2.827494644753193) q[11];
ry(-1.5672943406392363) q[12];
rz(1.5420449337490814) q[12];
ry(-2.311896773141357) q[13];
rz(-1.5622962161114131) q[13];
ry(0.012331548993558905) q[14];
rz(-2.5997696298857402) q[14];
ry(1.575010894569261) q[15];
rz(1.5922048482476112) q[15];
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
ry(-1.1882956152689168) q[0];
rz(1.9715094636075632) q[0];
ry(-1.446232783798046) q[1];
rz(1.840567225142231) q[1];
ry(-0.25909051223158563) q[2];
rz(1.7298511285262705) q[2];
ry(-0.00017808721743412185) q[3];
rz(-1.6460820428165308) q[3];
ry(-3.1393374767197377) q[4];
rz(0.40324617466330537) q[4];
ry(-0.016830191653039783) q[5];
rz(-3.028516203120023) q[5];
ry(-1.5300530783448125) q[6];
rz(2.0174390672589846) q[6];
ry(3.1412675262963172) q[7];
rz(1.1118286846994119) q[7];
ry(-0.0838203076865982) q[8];
rz(-1.6793654445278183) q[8];
ry(-0.01082259588055745) q[9];
rz(-0.008598711066463594) q[9];
ry(-1.5020589927258996) q[10];
rz(0.12622326649161497) q[10];
ry(0.00011330391579118655) q[11];
rz(-2.6239826652802685) q[11];
ry(1.8119244045915845) q[12];
rz(-3.0738523929328876) q[12];
ry(-3.0176897470614557) q[13];
rz(-3.1328974744890337) q[13];
ry(1.9371349290381787) q[14];
rz(2.9029622296945576) q[14];
ry(1.632130405774915) q[15];
rz(-2.796544298328572) q[15];
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
ry(-0.632718355121414) q[0];
rz(-3.037080533550648) q[0];
ry(-1.8854484481945262) q[1];
rz(0.045066917900032484) q[1];
ry(0.2686747061535351) q[2];
rz(-1.6415331803246698) q[2];
ry(0.004937864989382204) q[3];
rz(-2.0413596131345493) q[3];
ry(-0.002027776281976207) q[4];
rz(0.8038500626398344) q[4];
ry(-1.661438014110082) q[5];
rz(1.1952890537012504) q[5];
ry(-3.129425443285018) q[6];
rz(-1.0732504589312282) q[6];
ry(3.124205294696353) q[7];
rz(1.6095582918525884) q[7];
ry(0.0235743167286431) q[8];
rz(-1.9094621186215006) q[8];
ry(2.3911936334189456) q[9];
rz(3.0925821778549984) q[9];
ry(3.140660648482464) q[10];
rz(-2.7592208255879567) q[10];
ry(0.0008835107086076277) q[11];
rz(2.812951512850357) q[11];
ry(-3.1399871251281546) q[12];
rz(-2.956043511195682) q[12];
ry(-1.5696563526961542) q[13];
rz(2.024700850438167) q[13];
ry(-1.57144235759948) q[14];
rz(1.759618726560759) q[14];
ry(-1.5692138844633634) q[15];
rz(-0.16245691083113256) q[15];
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
ry(-2.610997659710823) q[0];
rz(-1.9747185096385995) q[0];
ry(1.4587854681143169) q[1];
rz(1.6473056590629485) q[1];
ry(-1.3107777352983285) q[2];
rz(-3.021343549738045) q[2];
ry(-2.2309695272035963) q[3];
rz(-1.0983279470715719) q[3];
ry(-0.2212510610365666) q[4];
rz(0.7527262002450463) q[4];
ry(-1.8001125252516843) q[5];
rz(1.4983843790789209) q[5];
ry(-1.5937865862152014) q[6];
rz(3.1284268114991245) q[6];
ry(-1.61595538089996) q[7];
rz(-1.5664067439476765) q[7];
ry(0.003099952541902354) q[8];
rz(2.022320249317593) q[8];
ry(-1.5592867017609155) q[9];
rz(0.0015776170376291534) q[9];
ry(3.071921476710863) q[10];
rz(0.25273280010125454) q[10];
ry(-0.0006753457595953805) q[11];
rz(2.6078364592158576) q[11];
ry(-1.5765003045994925) q[12];
rz(0.001277522342137026) q[12];
ry(-3.1291123195001482) q[13];
rz(0.4673120719920636) q[13];
ry(-0.0017289037925962703) q[14];
rz(-0.19515907934148924) q[14];
ry(-3.139059931858747) q[15];
rz(1.4068515093675993) q[15];