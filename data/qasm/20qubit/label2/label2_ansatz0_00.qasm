OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.5600280738463461) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.4469103867950376) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.019057385586155356) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-1.6355551668065194) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.6747236173197643) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.3995716197220917) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.7242408941267519) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-1.0563654286723285) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(1.1646406676050476) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(-7.615961195583105e-05) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(-0.0004143079808634723) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(0.4405473457516378) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(0.6338840432675645) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(-0.9370244179196637) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(0.015919512915689084) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(-0.0005828634540655412) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(-0.00045866282542309455) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(-0.2642227796679416) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(-1.1722975090079313) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(0.3985131690258978) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(0.3239484468116969) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(1.5725427419659064) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(-1.5778896978994252) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(0.6981683739919791) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(-0.0010994135199921978) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(0.0002549635317470459) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(-1.6890915334659162) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(-1.8283277849364965) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(1.2341480955115582) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(-0.3723007481248662) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(0.0010666586408031244) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(0.11677605289954103) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(-0.19541003685837374) q[11];
cx q[10],q[11];
h q[11];
h q[12];
cx q[11],q[12];
rz(-0.48876180377985123) q[12];
cx q[11],q[12];
h q[11];
h q[12];
sdg q[11];
h q[11];
sdg q[12];
h q[12];
cx q[11],q[12];
rz(-0.5233932278808167) q[12];
cx q[11],q[12];
h q[11];
s q[11];
h q[12];
s q[12];
cx q[11],q[12];
rz(0.9947105411447823) q[12];
cx q[11],q[12];
h q[12];
h q[13];
cx q[12],q[13];
rz(1.0562087570823167) q[13];
cx q[12],q[13];
h q[12];
h q[13];
sdg q[12];
h q[12];
sdg q[13];
h q[13];
cx q[12],q[13];
rz(-0.1721385307716683) q[13];
cx q[12],q[13];
h q[12];
s q[12];
h q[13];
s q[13];
cx q[12],q[13];
rz(-0.585285819118214) q[13];
cx q[12],q[13];
h q[13];
h q[14];
cx q[13],q[14];
rz(0.021862184441223967) q[14];
cx q[13],q[14];
h q[13];
h q[14];
sdg q[13];
h q[13];
sdg q[14];
h q[14];
cx q[13],q[14];
rz(-0.6773888296491002) q[14];
cx q[13],q[14];
h q[13];
s q[13];
h q[14];
s q[14];
cx q[13],q[14];
rz(0.7795499726922812) q[14];
cx q[13],q[14];
h q[14];
h q[15];
cx q[14],q[15];
rz(0.07407591318636972) q[15];
cx q[14],q[15];
h q[14];
h q[15];
sdg q[14];
h q[14];
sdg q[15];
h q[15];
cx q[14],q[15];
rz(0.07268418313949075) q[15];
cx q[14],q[15];
h q[14];
s q[14];
h q[15];
s q[15];
cx q[14],q[15];
rz(0.0024886305070724087) q[15];
cx q[14],q[15];
h q[15];
h q[16];
cx q[15],q[16];
rz(0.7629224652228045) q[16];
cx q[15],q[16];
h q[15];
h q[16];
sdg q[15];
h q[15];
sdg q[16];
h q[16];
cx q[15],q[16];
rz(-0.0014420571093087982) q[16];
cx q[15],q[16];
h q[15];
s q[15];
h q[16];
s q[16];
cx q[15],q[16];
rz(-1.03778795101965) q[16];
cx q[15],q[16];
h q[16];
h q[17];
cx q[16],q[17];
rz(-0.06334296769832554) q[17];
cx q[16],q[17];
h q[16];
h q[17];
sdg q[16];
h q[16];
sdg q[17];
h q[17];
cx q[16],q[17];
rz(-0.03297293608379992) q[17];
cx q[16],q[17];
h q[16];
s q[16];
h q[17];
s q[17];
cx q[16],q[17];
rz(-0.4467463200877181) q[17];
cx q[16],q[17];
h q[17];
h q[18];
cx q[17],q[18];
rz(-0.030624870357762707) q[18];
cx q[17],q[18];
h q[17];
h q[18];
sdg q[17];
h q[17];
sdg q[18];
h q[18];
cx q[17],q[18];
rz(-0.030628861668659885) q[18];
cx q[17],q[18];
h q[17];
s q[17];
h q[18];
s q[18];
cx q[17],q[18];
rz(0.5293401509224878) q[18];
cx q[17],q[18];
h q[18];
h q[19];
cx q[18],q[19];
rz(0.17708961573203538) q[19];
cx q[18],q[19];
h q[18];
h q[19];
sdg q[18];
h q[18];
sdg q[19];
h q[19];
cx q[18],q[19];
rz(0.1817364578670395) q[19];
cx q[18],q[19];
h q[18];
s q[18];
h q[19];
s q[19];
cx q[18],q[19];
rz(-0.35467981426239975) q[19];
cx q[18],q[19];
rx(1.074208167031192) q[0];
rz(-1.204677629582506) q[0];
rx(-1.6179786680664) q[1];
rz(-0.7583568319053315) q[1];
rx(-1.819570045776977) q[2];
rz(-1.797799275623814) q[2];
rx(0.4948109729844222) q[3];
rz(-0.9665035759599176) q[3];
rx(-0.6615307755791702) q[4];
rz(-1.6241846039211458) q[4];
rx(1.9678434414953336) q[5];
rz(-0.7276097268853591) q[5];
rx(0.8146024643992331) q[6];
rz(1.5459345405584477) q[6];
rx(-0.0016312029802022518) q[7];
rz(0.36231875497927934) q[7];
rx(-0.8208684428602379) q[8];
rz(-0.1640892525453914) q[8];
rx(0.00999340485030241) q[9];
rz(-0.2926317982473136) q[9];
rx(0.004333947885477942) q[10];
rz(0.5030792239907265) q[10];
rx(0.00012942599513140335) q[11];
rz(-1.106967596158687) q[11];
rx(0.004941384002377696) q[12];
rz(-0.15834730652164003) q[12];
rx(-2.3572150887483585) q[13];
rz(-0.01750964810634187) q[13];
rx(-0.7874706813760963) q[14];
rz(-0.0004242476073261727) q[14];
rx(-1.9310998278006413) q[15];
rz(-0.00018903957724492016) q[15];
rx(-1.4858878051922915) q[16];
rz(-0.05500252886197436) q[16];
rx(-0.44376505840317604) q[17];
rz(0.009981104498777846) q[17];
rx(-1.5694679775291163) q[18];
rz(-0.0016639882038817586) q[18];
rx(1.5417571081141968) q[19];
rz(0.04606081633881188) q[19];
h q[0];
h q[1];
cx q[0],q[1];
rz(-1.4578876142063797) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.9455127973718658) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.19590649260337026) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.37512247476056804) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(2.047698291468508) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-1.1519554811592845) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-1.766839946728171) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.8167573315276936) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.9240329155904717) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(0.0656012633244914) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(-0.010264628780911831) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(-0.01070516517450966) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(0.12518388260587685) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(-1.092788514820284) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(1.1129096280578281) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(0.049578381915924265) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(-0.0034802277026302225) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(-0.0028874983925262065) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(0.0004987215967548508) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(0.005952095207209965) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(-0.008130026373016103) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(1.5689501250873712) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(-1.572489743229583) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(0.29422988576760695) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(0.4395723251964879) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(1.1519156884407906) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(-0.08107905068462411) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(-0.03292992502603861) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(0.07035854770403512) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(-0.021132866921118133) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(0.3696462509525494) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(1.020204369836387) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(0.18875711684860594) q[11];
cx q[10],q[11];
h q[11];
h q[12];
cx q[11],q[12];
rz(0.06950921992790053) q[12];
cx q[11],q[12];
h q[11];
h q[12];
sdg q[11];
h q[11];
sdg q[12];
h q[12];
cx q[11],q[12];
rz(-0.010134757347274499) q[12];
cx q[11],q[12];
h q[11];
s q[11];
h q[12];
s q[12];
cx q[11],q[12];
rz(0.00213921901807761) q[12];
cx q[11],q[12];
h q[12];
h q[13];
cx q[12],q[13];
rz(-0.003507231580700431) q[13];
cx q[12],q[13];
h q[12];
h q[13];
sdg q[12];
h q[12];
sdg q[13];
h q[13];
cx q[12],q[13];
rz(-0.00526636331644811) q[13];
cx q[12],q[13];
h q[12];
s q[12];
h q[13];
s q[13];
cx q[12],q[13];
rz(-0.005920518449545529) q[13];
cx q[12],q[13];
h q[13];
h q[14];
cx q[13],q[14];
rz(-0.02198319216179167) q[14];
cx q[13],q[14];
h q[13];
h q[14];
sdg q[13];
h q[13];
sdg q[14];
h q[14];
cx q[13],q[14];
rz(-0.6659999167299148) q[14];
cx q[13],q[14];
h q[13];
s q[13];
h q[14];
s q[14];
cx q[13],q[14];
rz(0.6654701284180361) q[14];
cx q[13],q[14];
h q[14];
h q[15];
cx q[14],q[15];
rz(0.9091820145762491) q[15];
cx q[14],q[15];
h q[14];
h q[15];
sdg q[14];
h q[14];
sdg q[15];
h q[15];
cx q[14],q[15];
rz(0.004095352904197725) q[15];
cx q[14],q[15];
h q[14];
s q[14];
h q[15];
s q[15];
cx q[14],q[15];
rz(-0.0020976479110471845) q[15];
cx q[14],q[15];
h q[15];
h q[16];
cx q[15],q[16];
rz(1.3792415474563429) q[16];
cx q[15],q[16];
h q[15];
h q[16];
sdg q[15];
h q[15];
sdg q[16];
h q[16];
cx q[15],q[16];
rz(0.0008647470201865747) q[16];
cx q[15],q[16];
h q[15];
s q[15];
h q[16];
s q[16];
cx q[15],q[16];
rz(0.0023458189503006393) q[16];
cx q[15],q[16];
h q[16];
h q[17];
cx q[16],q[17];
rz(0.021581717717194297) q[17];
cx q[16],q[17];
h q[16];
h q[17];
sdg q[16];
h q[16];
sdg q[17];
h q[17];
cx q[16],q[17];
rz(-0.9833764437806204) q[17];
cx q[16],q[17];
h q[16];
s q[16];
h q[17];
s q[17];
cx q[16],q[17];
rz(1.4818351212828178) q[17];
cx q[16],q[17];
h q[17];
h q[18];
cx q[17],q[18];
rz(-0.010303898019036245) q[18];
cx q[17],q[18];
h q[17];
h q[18];
sdg q[17];
h q[17];
sdg q[18];
h q[18];
cx q[17],q[18];
rz(-3.0444503347706595) q[18];
cx q[17],q[18];
h q[17];
s q[17];
h q[18];
s q[18];
cx q[17],q[18];
rz(-0.003635194582844752) q[18];
cx q[17],q[18];
h q[18];
h q[19];
cx q[18],q[19];
rz(-0.007802452900730858) q[19];
cx q[18],q[19];
h q[18];
h q[19];
sdg q[18];
h q[18];
sdg q[19];
h q[19];
cx q[18],q[19];
rz(0.7475760188177987) q[19];
cx q[18],q[19];
h q[18];
s q[18];
h q[19];
s q[19];
cx q[18],q[19];
rz(-1.6656357969035298) q[19];
cx q[18],q[19];
rx(1.3840265582515248) q[0];
rz(-0.2676807396215899) q[0];
rx(-0.7807220827349179) q[1];
rz(0.06960664796796465) q[1];
rx(-0.575009053468985) q[2];
rz(1.500405704132923) q[2];
rx(2.5640178139197665) q[3];
rz(8.011635696520872e-05) q[3];
rx(-0.5406622585320765) q[4];
rz(-1.23125838882432e-05) q[4];
rx(0.5598719086002794) q[5];
rz(-0.00029045302575882474) q[5];
rx(0.5758062364267029) q[6];
rz(0.0004231716591461511) q[6];
rx(-1.229875994470733) q[7];
rz(0.1472688274988511) q[7];
rx(0.0018655639515566828) q[8];
rz(0.5695116479405956) q[8];
rx(-0.004827484464331442) q[9];
rz(2.1195735061903567) q[9];
rx(-0.0016566194713531451) q[10];
rz(0.6151843377615366) q[10];
rx(-0.0030706825288885306) q[11];
rz(-0.9016856580431439) q[11];
rx(0.0027732792603704357) q[12];
rz(1.139658755191383) q[12];
rx(-0.7930174065220824) q[13];
rz(2.683866629974281) q[13];
rx(0.7860924468684765) q[14];
rz(0.03745382427221692) q[14];
rx(-1.1029732490443944) q[15];
rz(1.5662633383133544) q[15];
rx(-0.5435798214983449) q[16];
rz(0.022814802650667723) q[16];
rx(1.4960467043486234) q[17];
rz(-1.6664305664865882) q[17];
rx(-1.62434007290886) q[18];
rz(0.247500505583751) q[18];
rx(1.5730217563696365) q[19];
rz(-1.8213618000007619) q[19];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.18371127644156582) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.9930869747707343) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(2.1008693296061556) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-1.905515467152927) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.6985270889277272) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-1.1645497185086742) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.9913076638368642) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.08122008577327822) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.06465452852549317) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(-0.018741816899899545) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(-0.05791105022205942) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(0.05074423006763217) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(1.0014691119453782) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(-2.3922619072375593) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(0.6025314678033201) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(0.010813689972684486) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(-0.046461645434322046) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(0.04685055982223197) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(-0.06054130464387082) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(-1.5716343883052262) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(1.5716564504093982) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(-0.04681094898573758) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(-0.04864332053304295) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(-0.04769026816000185) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(0.7591100557761417) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(0.7867626872432144) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(0.7427580760053174) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(0.05068199438602419) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(0.04311362782669025) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(0.046596967738512414) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(-0.7528606766971692) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(-0.7539620151336512) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(-0.7579654089853066) q[11];
cx q[10],q[11];
h q[11];
h q[12];
cx q[11],q[12];
rz(-0.06742679866985264) q[12];
cx q[11],q[12];
h q[11];
h q[12];
sdg q[11];
h q[11];
sdg q[12];
h q[12];
cx q[11],q[12];
rz(0.021462864292880676) q[12];
cx q[11],q[12];
h q[11];
s q[11];
h q[12];
s q[12];
cx q[11],q[12];
rz(0.051731371303048855) q[12];
cx q[11],q[12];
h q[12];
h q[13];
cx q[12],q[13];
rz(0.29250264676329174) q[13];
cx q[12],q[13];
h q[12];
h q[13];
sdg q[12];
h q[12];
sdg q[13];
h q[13];
cx q[12],q[13];
rz(-1.8444933239061423) q[13];
cx q[12],q[13];
h q[12];
s q[12];
h q[13];
s q[13];
cx q[12],q[13];
rz(-2.786517456833217) q[13];
cx q[12],q[13];
h q[13];
h q[14];
cx q[13],q[14];
rz(0.06093012012172172) q[14];
cx q[13],q[14];
h q[13];
h q[14];
sdg q[13];
h q[13];
sdg q[14];
h q[14];
cx q[13],q[14];
rz(-0.010038955094925233) q[14];
cx q[13],q[14];
h q[13];
s q[13];
h q[14];
s q[14];
cx q[13],q[14];
rz(0.07138415382777911) q[14];
cx q[13],q[14];
h q[14];
h q[15];
cx q[14],q[15];
rz(-0.0077479855926432945) q[15];
cx q[14],q[15];
h q[14];
h q[15];
sdg q[14];
h q[14];
sdg q[15];
h q[15];
cx q[14],q[15];
rz(-1.6404762408001812) q[15];
cx q[14],q[15];
h q[14];
s q[14];
h q[15];
s q[15];
cx q[14],q[15];
rz(-0.008883428051071566) q[15];
cx q[14],q[15];
h q[15];
h q[16];
cx q[15],q[16];
rz(0.056460952039803836) q[16];
cx q[15],q[16];
h q[15];
h q[16];
sdg q[15];
h q[15];
sdg q[16];
h q[16];
cx q[15],q[16];
rz(-2.073703905746044) q[16];
cx q[15],q[16];
h q[15];
s q[15];
h q[16];
s q[16];
cx q[15],q[16];
rz(-0.04891263721715331) q[16];
cx q[15],q[16];
h q[16];
h q[17];
cx q[16],q[17];
rz(3.0597292282251343) q[17];
cx q[16],q[17];
h q[16];
h q[17];
sdg q[16];
h q[16];
sdg q[17];
h q[17];
cx q[16],q[17];
rz(-1.4953749172030857) q[17];
cx q[16],q[17];
h q[16];
s q[16];
h q[17];
s q[17];
cx q[16],q[17];
rz(-0.06897984229674681) q[17];
cx q[16],q[17];
h q[17];
h q[18];
cx q[17],q[18];
rz(0.04831604728147459) q[18];
cx q[17],q[18];
h q[17];
h q[18];
sdg q[17];
h q[17];
sdg q[18];
h q[18];
cx q[17],q[18];
rz(-0.04090911951367548) q[18];
cx q[17],q[18];
h q[17];
s q[17];
h q[18];
s q[18];
cx q[17],q[18];
rz(-0.04794054268432924) q[18];
cx q[17],q[18];
h q[18];
h q[19];
cx q[18],q[19];
rz(-0.6808018738206717) q[19];
cx q[18],q[19];
h q[18];
h q[19];
sdg q[18];
h q[18];
sdg q[19];
h q[19];
cx q[18],q[19];
rz(0.8580221532617598) q[19];
cx q[18],q[19];
h q[18];
s q[18];
h q[19];
s q[19];
cx q[18],q[19];
rz(0.7032079117831306) q[19];
cx q[18],q[19];
rx(1.8304751011100433) q[0];
rz(-0.6113184531561047) q[0];
rx(-0.8151159563547774) q[1];
rz(0.8615533759803167) q[1];
rx(2.674710611247633) q[2];
rz(0.08780485723494118) q[2];
rx(-0.03386032937482726) q[3];
rz(0.04698379902069088) q[3];
rx(-0.04379567656031682) q[4];
rz(0.04389058612226215) q[4];
rx(-0.025364036676725613) q[5];
rz(0.0443869358032319) q[5];
rx(1.8451802179113048) q[6];
rz(0.044693860939817774) q[6];
rx(-0.020745314198500002) q[7];
rz(0.04371582294577803) q[7];
rx(-0.023725500789847146) q[8];
rz(0.08335503236063385) q[8];
rx(-0.014320046777842616) q[9];
rz(0.068415919127893) q[9];
rx(-0.01420366908125472) q[10];
rz(-0.02426162411233579) q[10];
rx(-0.01106173276923771) q[11];
rz(0.05499866148602202) q[11];
rx(-0.017835643627915593) q[12];
rz(0.08613683294942741) q[12];
rx(-0.01242266002436402) q[13];
rz(-0.3065513917540878) q[13];
rx(-0.007476943799854312) q[14];
rz(0.3098039522201295) q[14];
rx(0.01793976595444816) q[15];
rz(-0.3575715770003198) q[15];
rx(-0.5327607476378944) q[16];
rz(-0.3563394115549674) q[16];
rx(-0.0032010096285651937) q[17];
rz(-0.38254188197199634) q[17];
rx(3.1311623347517705) q[18];
rz(-0.39505719162899305) q[18];
rx(-0.0004178013939907231) q[19];
rz(-0.2968365531987726) q[19];