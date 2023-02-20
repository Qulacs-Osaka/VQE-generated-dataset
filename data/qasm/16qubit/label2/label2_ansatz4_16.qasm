OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(-1.1403848019844984) q[0];
rz(1.568428796028727) q[0];
ry(0.5110170892814185) q[1];
rz(-1.1327094063481884) q[1];
ry(0.0023384410096722874) q[2];
rz(1.664547836467202) q[2];
ry(-0.0001127543445362435) q[3];
rz(1.2622825687967072) q[3];
ry(1.5622969341052135) q[4];
rz(-3.0749978232543453) q[4];
ry(-1.5937688057052082) q[5];
rz(2.762175892827082) q[5];
ry(-3.141424957336682) q[6];
rz(-1.6741025300748966) q[6];
ry(3.140167125988963) q[7];
rz(-1.5930217139502012) q[7];
ry(-1.5841043012343416) q[8];
rz(0.4169162300907481) q[8];
ry(1.5839265086908219) q[9];
rz(2.761696289082976) q[9];
ry(-0.0001734222681522013) q[10];
rz(-0.9002388526852695) q[10];
ry(-5.545901748626913e-05) q[11];
rz(-0.7561861982943308) q[11];
ry(-1.5732129086237432) q[12];
rz(-2.876094546904025) q[12];
ry(-1.5711839506320435) q[13];
rz(-0.24496199600879984) q[13];
ry(3.1397536256034844) q[14];
rz(2.979429907346641) q[14];
ry(3.1116384826582495) q[15];
rz(1.7521544840034844) q[15];
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
ry(0.5362989349329738) q[0];
rz(2.7326556657934735) q[0];
ry(0.8531022963567207) q[1];
rz(-0.7393089960261352) q[1];
ry(-0.07180296770927914) q[2];
rz(-0.2605310130668519) q[2];
ry(-0.036006574349330314) q[3];
rz(3.0425590222738506) q[3];
ry(1.738525678299564) q[4];
rz(-1.5037297362015911) q[4];
ry(0.1655601626978056) q[5];
rz(1.9021926111633727) q[5];
ry(-1.57984343653236) q[6];
rz(3.1288439642016916) q[6];
ry(-1.5854298328576395) q[7];
rz(3.0217790062064176) q[7];
ry(-1.572463698317737) q[8];
rz(2.1166475203627435) q[8];
ry(-0.7317906611290592) q[9];
rz(1.051250954645962) q[9];
ry(0.20133499143428335) q[10];
rz(-1.9961594672112821) q[10];
ry(-3.0623554895848084) q[11];
rz(2.6494848519644174) q[11];
ry(0.7387587099956955) q[12];
rz(2.587341551365853) q[12];
ry(-0.846780458702221) q[13];
rz(-0.13538487185643344) q[13];
ry(-0.9731073847090013) q[14];
rz(-1.4381629720088136) q[14];
ry(1.3034471350353014) q[15];
rz(1.2637119139004411) q[15];
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
ry(-2.459788729677153) q[0];
rz(-0.6770905460825576) q[0];
ry(-1.1743868595309348) q[1];
rz(-0.9630018401718784) q[1];
ry(2.9815714442825034) q[2];
rz(2.1647736729911697) q[2];
ry(2.8395936218969218) q[3];
rz(-2.745281061853528) q[3];
ry(2.0818126273759745) q[4];
rz(-1.4694819197990299) q[4];
ry(1.053199568880279) q[5];
rz(2.0771636980939263) q[5];
ry(-2.6095321366065205) q[6];
rz(-0.20960170261325697) q[6];
ry(-0.5039179815032208) q[7];
rz(1.8758930263234017) q[7];
ry(-3.1337739785171745) q[8];
rz(1.3243196465303253) q[8];
ry(-0.05182407817606475) q[9];
rz(-0.7522256443127998) q[9];
ry(3.115725691000781) q[10];
rz(2.257589204971848) q[10];
ry(0.0202307677390933) q[11];
rz(1.8164032075627687) q[11];
ry(2.3195161095615857) q[12];
rz(1.6312740601936984) q[12];
ry(0.8195728876109946) q[13];
rz(2.2280265038517504) q[13];
ry(0.021890044541421464) q[14];
rz(3.005675756083361) q[14];
ry(2.2376491986666416) q[15];
rz(-2.042408954343295) q[15];
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
ry(-2.9417671973316115) q[0];
rz(-1.0499842827437327) q[0];
ry(1.3621007875223192) q[1];
rz(-0.8343001833300381) q[1];
ry(-0.004456105479905936) q[2];
rz(-2.6305605468313287) q[2];
ry(2.843982269778374) q[3];
rz(-2.6688490488149412) q[3];
ry(-0.003834440143257289) q[4];
rz(-2.2546108899912514) q[4];
ry(0.0018414397469833708) q[5];
rz(2.721492232089353) q[5];
ry(-2.419731848009407) q[6];
rz(-0.9618136210533333) q[6];
ry(2.436314297889385) q[7];
rz(0.8175171180369981) q[7];
ry(1.7716370526301277) q[8];
rz(2.284408320085325) q[8];
ry(-0.28871439533358423) q[9];
rz(-2.645574637777907) q[9];
ry(1.8804668023675108) q[10];
rz(-1.045855040958144) q[10];
ry(-0.5847726844088409) q[11];
rz(1.9813494142549342) q[11];
ry(0.023910652472624836) q[12];
rz(2.321929490631721) q[12];
ry(-0.009958311183342337) q[13];
rz(2.930632888600781) q[13];
ry(-1.7267769068428664) q[14];
rz(1.6488935377464362) q[14];
ry(0.3154917666253949) q[15];
rz(-1.8280365361022235) q[15];
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
ry(1.2710281705571636) q[0];
rz(-2.6258883365834222) q[0];
ry(-2.5714923694063407) q[1];
rz(-2.5419483679739687) q[1];
ry(-3.035134180926639) q[2];
rz(-1.0972037665858636) q[2];
ry(-0.2552135661783863) q[3];
rz(-0.5989541092092753) q[3];
ry(0.03692814907485431) q[4];
rz(1.6902361393272174) q[4];
ry(-3.111737140870615) q[5];
rz(1.945215677060217) q[5];
ry(-3.0967617791810818) q[6];
rz(3.0739383385667645) q[6];
ry(3.1316855066736817) q[7];
rz(0.6677805072404867) q[7];
ry(0.006306436086888746) q[8];
rz(-1.0099393601236057) q[8];
ry(0.010406326023880181) q[9];
rz(-0.9533725271543904) q[9];
ry(0.028011782321065014) q[10];
rz(-1.5369642356877566) q[10];
ry(3.1343501916125183) q[11];
rz(2.2924572018496185) q[11];
ry(0.01565099584272932) q[12];
rz(1.8244937643440204) q[12];
ry(0.010036312761049437) q[13];
rz(1.5768025874041856) q[13];
ry(-0.3231665915761992) q[14];
rz(0.849846155906607) q[14];
ry(0.9770477612814819) q[15];
rz(0.03600079982316194) q[15];
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
ry(1.8325431792675353) q[0];
rz(1.0300669390709825) q[0];
ry(-1.823419636915388) q[1];
rz(0.5584147045725726) q[1];
ry(3.120533343757257) q[2];
rz(1.5939173533737858) q[2];
ry(3.1337769464501286) q[3];
rz(-3.0024341384585806) q[3];
ry(3.1159235596245947) q[4];
rz(2.8799570249374917) q[4];
ry(3.1246957312388717) q[5];
rz(-1.5117509386654753) q[5];
ry(0.0947983213485557) q[6];
rz(-2.832932621756204) q[6];
ry(0.5897826525439074) q[7];
rz(0.5345712144860134) q[7];
ry(-1.4607326788174522) q[8];
rz(2.5629995135134256) q[8];
ry(-1.505399587842652) q[9];
rz(-3.0832297264367274) q[9];
ry(-1.5951127371152043) q[10];
rz(-1.2454430900419924) q[10];
ry(0.8593667390782089) q[11];
rz(-1.1933694395606695) q[11];
ry(1.4313719045798852) q[12];
rz(1.5194430896106883) q[12];
ry(-1.7149031293433676) q[13];
rz(-2.292926634472872) q[13];
ry(1.6528761845684852) q[14];
rz(-0.6042427652922147) q[14];
ry(1.151780381029397) q[15];
rz(-2.8955893338947813) q[15];
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
ry(-1.9250972992041904) q[0];
rz(-2.716742471669738) q[0];
ry(0.5307960611827355) q[1];
rz(1.5171056229335136) q[1];
ry(-1.6855471549719647) q[2];
rz(0.2751116425235636) q[2];
ry(2.0914430011197647) q[3];
rz(0.4367930999336286) q[3];
ry(-2.1902216779736494) q[4];
rz(2.283411674016411) q[4];
ry(-0.9663252918189906) q[5];
rz(-2.132339054865969) q[5];
ry(-1.201221443482134) q[6];
rz(3.1268136728304037) q[6];
ry(1.010176004567783) q[7];
rz(0.005266170251495339) q[7];
ry(-3.10551081985342) q[8];
rz(2.0115505745580284) q[8];
ry(3.0886104784017023) q[9];
rz(1.1043403039371182) q[9];
ry(2.2905244472149766) q[10];
rz(-1.508167914611762) q[10];
ry(-2.873056922205048) q[11];
rz(1.353071299220754) q[11];
ry(3.1356945230526216) q[12];
rz(-1.4865498103767498) q[12];
ry(0.0022657097349911837) q[13];
rz(-0.9794817674320307) q[13];
ry(2.553924772332359) q[14];
rz(-1.5984367230613854) q[14];
ry(-3.003898392737074) q[15];
rz(-0.5492457930557294) q[15];
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
ry(0.9421286098565858) q[0];
rz(-2.410681012728674) q[0];
ry(-2.6596502806997315) q[1];
rz(1.3695503956541524) q[1];
ry(-1.2307173270323934) q[2];
rz(-1.8874400862290575) q[2];
ry(-1.606236400644307) q[3];
rz(-2.5565083341716615) q[3];
ry(2.486027921289311) q[4];
rz(1.5221809230675674) q[4];
ry(1.6064743346494768) q[5];
rz(-2.1257203823754534) q[5];
ry(-0.6111760178171591) q[6];
rz(2.4902719191658176) q[6];
ry(-2.557550050875133) q[7];
rz(0.40545846906285377) q[7];
ry(-2.928769730110658) q[8];
rz(-1.8698918933864477) q[8];
ry(2.96268966908488) q[9];
rz(1.0856978629814023) q[9];
ry(-1.1974297455884164) q[10];
rz(1.8495747774846805) q[10];
ry(-0.680877465891925) q[11];
rz(3.0253346123663527) q[11];
ry(1.8232866088838344) q[12];
rz(0.6434039868187708) q[12];
ry(1.320861506525481) q[13];
rz(0.6459532355669416) q[13];
ry(2.535381188550914) q[14];
rz(1.6857133019524413) q[14];
ry(-0.6261888724750633) q[15];
rz(-2.619131664926905) q[15];
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
ry(0.1730831159942691) q[0];
rz(-0.7230805021020253) q[0];
ry(0.018802842562840684) q[1];
rz(-0.4299460129714782) q[1];
ry(-0.00035187152282478706) q[2];
rz(1.6556989994331666) q[2];
ry(-3.138410755193627) q[3];
rz(-2.5095710312597097) q[3];
ry(-3.1399942259096054) q[4];
rz(-1.849390153944812) q[4];
ry(3.134045398042369) q[5];
rz(-0.66117891143917) q[5];
ry(-0.005367835395814576) q[6];
rz(1.9903258598157896) q[6];
ry(-3.1373434203564483) q[7];
rz(2.467260868303572) q[7];
ry(-2.536682132037628) q[8];
rz(-1.1549000932063676) q[8];
ry(0.6021174792323608) q[9];
rz(2.261071957252218) q[9];
ry(2.888026597998188) q[10];
rz(0.946886624827556) q[10];
ry(0.5965373146361942) q[11];
rz(2.23408035613933) q[11];
ry(2.8180529393673748) q[12];
rz(-1.4187559779854721) q[12];
ry(0.3135380344967986) q[13];
rz(-0.6037967714846404) q[13];
ry(2.876410605099894) q[14];
rz(2.1442864384703317) q[14];
ry(-0.658192032406888) q[15];
rz(0.12044571561359343) q[15];
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
ry(1.9665312422125316) q[0];
rz(-2.8352771258779583) q[0];
ry(-1.8734552147791046) q[1];
rz(1.4382172767115842) q[1];
ry(1.6506231743264994) q[2];
rz(1.7539978103668845) q[2];
ry(1.3271435135031941) q[3];
rz(1.9848426164689839) q[3];
ry(2.140082312748187) q[4];
rz(-1.5690047466210872) q[4];
ry(1.1358105435253707) q[5];
rz(-0.13745524834466225) q[5];
ry(-1.1043205206719913) q[6];
rz(-0.08182228867421149) q[6];
ry(0.5609874366512981) q[7];
rz(-2.1321598801380555) q[7];
ry(3.136841919944597) q[8];
rz(0.0012868545094280037) q[8];
ry(-0.030754104787921437) q[9];
rz(-0.061320272367719114) q[9];
ry(-1.9330084842701947) q[10];
rz(-1.0407459801754788) q[10];
ry(0.543432367341186) q[11];
rz(0.08287836033589435) q[11];
ry(3.1396838991178218) q[12];
rz(3.058401557883022) q[12];
ry(-3.1385293985151703) q[13];
rz(-0.401694193105957) q[13];
ry(2.1138130315586583) q[14];
rz(2.4398558664536525) q[14];
ry(0.42881745975644936) q[15];
rz(2.9122840143674305) q[15];
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
ry(3.132307841418208) q[0];
rz(-0.5581408480920728) q[0];
ry(-0.15915326285924095) q[1];
rz(-3.0182352269774135) q[1];
ry(3.0161821244451903) q[2];
rz(2.8741422447296) q[2];
ry(-3.036325855915871) q[3];
rz(-2.1820798835261317) q[3];
ry(-1.8607295333912448) q[4];
rz(-2.736816410033963) q[4];
ry(1.2819993889905872) q[5];
rz(0.587956307518928) q[5];
ry(0.937935094259166) q[6];
rz(-0.6109865897138967) q[6];
ry(-1.974319240292179) q[7];
rz(0.20189036614528955) q[7];
ry(-3.138917626833024) q[8];
rz(-0.36805878387517854) q[8];
ry(0.028876101428176426) q[9];
rz(-0.7761736920545902) q[9];
ry(0.5589724500927913) q[10];
rz(2.4595422937851663) q[10];
ry(-1.8252869246071737) q[11];
rz(2.7921212789053516) q[11];
ry(0.006159639162081919) q[12];
rz(-1.8810992436821374) q[12];
ry(-3.13649360325868) q[13];
rz(-0.4858721358284633) q[13];
ry(-0.9238307460120909) q[14];
rz(0.9173528679935828) q[14];
ry(-2.7764294692046363) q[15];
rz(-2.6268708715605205) q[15];
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
ry(-1.0269059863232421) q[0];
rz(0.6014711060375499) q[0];
ry(-2.3806718833625906) q[1];
rz(-2.0968148032022143) q[1];
ry(-1.1514853581001208) q[2];
rz(0.42533093415589107) q[2];
ry(2.1368389200908497) q[3];
rz(2.9252265018814096) q[3];
ry(-3.069033031273419) q[4];
rz(-0.5032486316922054) q[4];
ry(3.04869087496414) q[5];
rz(2.831560195246811) q[5];
ry(-0.5190224535116208) q[6];
rz(0.5230370642651225) q[6];
ry(-0.0721621518929721) q[7];
rz(3.04653951389782) q[7];
ry(-2.7534240972312993) q[8];
rz(0.27804526313106875) q[8];
ry(-2.7558599244959403) q[9];
rz(2.698597984443993) q[9];
ry(0.8774058510620506) q[10];
rz(0.23251608396283888) q[10];
ry(0.8649195245481679) q[11];
rz(2.3970448544335334) q[11];
ry(1.4580013685014963) q[12];
rz(2.2847991786998607) q[12];
ry(1.2069872323700142) q[13];
rz(2.2815823841200666) q[13];
ry(2.6609135669823347) q[14];
rz(-2.5883713745267163) q[14];
ry(-2.8807333790956773) q[15];
rz(-0.7388056432214798) q[15];
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
ry(-0.0227799036950449) q[0];
rz(0.918780919490443) q[0];
ry(-0.034211118222800395) q[1];
rz(2.1602314237535265) q[1];
ry(0.8402512623986125) q[2];
rz(1.4096414916400757) q[2];
ry(1.5656110671165717) q[3];
rz(-2.7511898737728844) q[3];
ry(-2.280514124397921) q[4];
rz(0.04158615825750189) q[4];
ry(0.8114645430160667) q[5];
rz(-2.2901345687240884) q[5];
ry(-2.098626416893093) q[6];
rz(-0.3948814446190485) q[6];
ry(-1.0619549202984393) q[7];
rz(2.7794092301369604) q[7];
ry(-2.45892654499417) q[8];
rz(-1.4891887094517742) q[8];
ry(0.6846880029216837) q[9];
rz(1.6393766559559284) q[9];
ry(-3.0599428466200482) q[10];
rz(-2.4421857809309273) q[10];
ry(3.135341213228444) q[11];
rz(-0.8762400944895513) q[11];
ry(-1.3800245513476814) q[12];
rz(1.1497793482951995) q[12];
ry(1.8038490785535515) q[13];
rz(-0.911382089297201) q[13];
ry(-1.633228890814976) q[14];
rz(-0.2961254139795866) q[14];
ry(1.6357208870899864) q[15];
rz(-0.15245715335032806) q[15];
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
ry(-2.701486897269193) q[0];
rz(2.1767224119032935) q[0];
ry(0.4391272409508682) q[1];
rz(0.15816075889195394) q[1];
ry(1.7068609358056266) q[2];
rz(0.9936997821022784) q[2];
ry(0.4277349352713582) q[3];
rz(0.9585240636229377) q[3];
ry(3.1344393486921454) q[4];
rz(0.958566515337588) q[4];
ry(3.123671864900331) q[5];
rz(3.127900772376793) q[5];
ry(-1.1153129310514673) q[6];
rz(1.2635898534939909) q[6];
ry(1.969986081170882) q[7];
rz(0.8309591830336079) q[7];
ry(-1.8586890548684416) q[8];
rz(0.15978601262303396) q[8];
ry(-1.2656864597586228) q[9];
rz(2.9872323205882934) q[9];
ry(2.9041377753240796) q[10];
rz(2.2129758785230558) q[10];
ry(3.129055262440382) q[11];
rz(2.1130741998749505) q[11];
ry(-0.8814402078740071) q[12];
rz(1.393734220149864) q[12];
ry(2.1728713620745657) q[13];
rz(-0.8596148596042701) q[13];
ry(1.4886428753680105) q[14];
rz(-0.8042820828747512) q[14];
ry(-1.5172007470714837) q[15];
rz(0.7940690298132013) q[15];
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
ry(3.1312117378431217) q[0];
rz(0.37690345109894524) q[0];
ry(-0.0065579748037167965) q[1];
rz(-0.885381021369704) q[1];
ry(2.2800751036623206) q[2];
rz(1.5020218398210512) q[2];
ry(-1.9602548962138193) q[3];
rz(-1.5464555226082075) q[3];
ry(2.803702183393715) q[4];
rz(1.7464804954423827) q[4];
ry(-0.29285413078696365) q[5];
rz(1.756042339798391) q[5];
ry(1.9609252097626941) q[6];
rz(0.10651099941544385) q[6];
ry(-0.9957848795668979) q[7];
rz(2.977964584417053) q[7];
ry(1.2714890279580602) q[8];
rz(-0.569557798726441) q[8];
ry(-1.2427063401146992) q[9];
rz(-0.450070924693998) q[9];
ry(-3.1394044561324983) q[10];
rz(0.6107490444157796) q[10];
ry(-0.00161258658649797) q[11];
rz(-2.9279321007230616) q[11];
ry(0.4813850321439919) q[12];
rz(3.0898149178506142) q[12];
ry(3.0345525173537724) q[13];
rz(-1.8981772102585595) q[13];
ry(0.0828425064352869) q[14];
rz(2.5258449427690772) q[14];
ry(0.04652689884344055) q[15];
rz(1.0446284384958482) q[15];
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
ry(-3.1091282673253056) q[0];
rz(-2.4236610906169798) q[0];
ry(-0.010253642809218803) q[1];
rz(2.90331964129575) q[1];
ry(-1.045108696471437) q[2];
rz(-1.7861919685090235) q[2];
ry(1.772864576799261) q[3];
rz(-0.7340411287209632) q[3];
ry(-0.21556533948002637) q[4];
rz(-0.8853840322035721) q[4];
ry(2.9213980803213437) q[5];
rz(-1.5454678584925892) q[5];
ry(0.11514909034308651) q[6];
rz(-2.9155322751136916) q[6];
ry(0.08743442486311448) q[7];
rz(-0.03742980465090506) q[7];
ry(3.130514850271229) q[8];
rz(2.523670852565156) q[8];
ry(-0.029403833743987917) q[9];
rz(3.0059347487232517) q[9];
ry(1.8937591395855735) q[10];
rz(2.9555361346561293) q[10];
ry(1.954721007097965) q[11];
rz(-0.4189515756708548) q[11];
ry(-0.05606160421090342) q[12];
rz(-0.03891099238752613) q[12];
ry(0.0403563115514336) q[13];
rz(2.5913220086737208) q[13];
ry(-3.1369070273476516) q[14];
rz(2.564315335906801) q[14];
ry(-0.01081031760371326) q[15];
rz(1.3450518419032644) q[15];
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
ry(1.4964535497598446) q[0];
rz(-1.1963999391299538) q[0];
ry(-2.694822452554995) q[1];
rz(-1.3137330142082506) q[1];
ry(-1.87712432031387) q[2];
rz(0.101211833565743) q[2];
ry(2.618677304235542) q[3];
rz(-3.058300790394945) q[3];
ry(2.865860950231281) q[4];
rz(2.15024429142405) q[4];
ry(-1.1155166914922365) q[5];
rz(-1.3332661073545906) q[5];
ry(-1.983021633354606) q[6];
rz(0.9137106653838133) q[6];
ry(1.3974256999289465) q[7];
rz(-1.6380717630922907) q[7];
ry(-0.014454613444186748) q[8];
rz(1.4352335449227762) q[8];
ry(3.1079432736742603) q[9];
rz(1.67113110006336) q[9];
ry(0.0036869718583775494) q[10];
rz(0.37258777386034403) q[10];
ry(0.0012631247890411146) q[11];
rz(-2.1586222313842094) q[11];
ry(-0.220891139153319) q[12];
rz(1.6090011162660494) q[12];
ry(2.7368330253427122) q[13];
rz(0.04932288993982837) q[13];
ry(-3.0947977983271144) q[14];
rz(-2.575016916704506) q[14];
ry(3.010060226698803) q[15];
rz(-2.893966141440311) q[15];
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
ry(-0.43321935222349395) q[0];
rz(-0.9355363539364361) q[0];
ry(-0.002518301695911901) q[1];
rz(2.004196205719232) q[1];
ry(-0.0023708378885527828) q[2];
rz(-2.4144177396720243) q[2];
ry(0.0018944128890900913) q[3];
rz(1.0052223973751868) q[3];
ry(0.13998528790461526) q[4];
rz(-2.921594217309577) q[4];
ry(-0.13981964633278032) q[5];
rz(-3.086693183745129) q[5];
ry(0.043165190177080426) q[6];
rz(-1.256738654366858) q[6];
ry(-3.1089637472411678) q[7];
rz(1.3293059872867614) q[7];
ry(0.03004157246665695) q[8];
rz(2.2048355260196377) q[8];
ry(0.01958332732008561) q[9];
rz(0.6760515894116804) q[9];
ry(2.290655372316189) q[10];
rz(-2.5804834480777794) q[10];
ry(0.8528128595230305) q[11];
rz(-1.49712064449132) q[11];
ry(-0.07453303118770371) q[12];
rz(1.4188018681908128) q[12];
ry(-2.4111913560429348) q[13];
rz(0.06811897879973483) q[13];
ry(3.1222187673903816) q[14];
rz(-0.13737215164350314) q[14];
ry(3.1205118066528392) q[15];
rz(0.022695471086192343) q[15];
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
ry(-2.039366970362183) q[0];
rz(0.7371119775933447) q[0];
ry(0.677707673605209) q[1];
rz(-2.7914304732658173) q[1];
ry(-2.5983713004731905) q[2];
rz(1.4697922515735122) q[2];
ry(-2.593610265456028) q[3];
rz(-1.6312199786014014) q[3];
ry(1.6840801875223579) q[4];
rz(2.3079295328362965) q[4];
ry(-1.6283547017241793) q[5];
rz(-3.059216207763579) q[5];
ry(-0.9546928674230967) q[6];
rz(0.702705140951613) q[6];
ry(-0.9126402152012285) q[7];
rz(-0.15011238516640543) q[7];
ry(-0.03491076791702697) q[8];
rz(0.22192273822463354) q[8];
ry(3.100278973390675) q[9];
rz(-2.6724547031423063) q[9];
ry(3.138800585077069) q[10];
rz(0.8493389151283621) q[10];
ry(0.00028967885144215444) q[11];
rz(-0.26032930033555796) q[11];
ry(3.0991162612688044) q[12];
rz(2.3217845268385857) q[12];
ry(1.406922713432239) q[13];
rz(-1.8072496566509473) q[13];
ry(-1.568805532451563) q[14];
rz(-0.06184592654755299) q[14];
ry(-1.5749936686341712) q[15];
rz(-0.11369717408455232) q[15];
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
ry(1.88333409780169) q[0];
rz(1.9235600733110232) q[0];
ry(1.260407131956783) q[1];
rz(-1.285957570091501) q[1];
ry(-0.9720635957572076) q[2];
rz(-3.0654791334781355) q[2];
ry(-1.3661544793442821) q[3];
rz(0.046888850407387765) q[3];
ry(1.4670834219031998) q[4];
rz(-0.2609419266088352) q[4];
ry(-1.453198881540891) q[5];
rz(2.913461841311246) q[5];
ry(2.8872681510130613) q[6];
rz(-2.22695965430344) q[6];
ry(-2.9608673908667735) q[7];
rz(1.6107948357304018) q[7];
ry(0.6517541762554724) q[8];
rz(-2.888881924295332) q[8];
ry(-0.6707081037683729) q[9];
rz(0.2080553675639711) q[9];
ry(-2.8802774332152636) q[10];
rz(-0.3912422192312869) q[10];
ry(0.3576310466968078) q[11];
rz(2.9509472982627027) q[11];
ry(-0.9392089454023003) q[12];
rz(-3.100400214726802) q[12];
ry(0.648830868106689) q[13];
rz(-2.7900871611645974) q[13];
ry(-2.1718700440277727) q[14];
rz(0.03111907194577471) q[14];
ry(2.1711618449965524) q[15];
rz(-3.1106863974978616) q[15];