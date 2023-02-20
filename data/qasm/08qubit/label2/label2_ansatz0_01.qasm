OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.03359461610622826) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.11974758959249909) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.07360588480791846) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.329751373862978) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.029957271307373603) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.4525798998247873) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.5933021367520371) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.6158309504617541) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.48651702439557454) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(0.2745241378650335) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(-0.25554574834458643) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(-1.2345472553405015) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(-0.08163426801441935) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(-0.07709254474712521) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(0.747370765969036) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(1.4533904311751733) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(-0.11011290860294845) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(-0.9569273295667509) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(-0.12031265614195895) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(0.12227455939961439) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(0.9828010154498957) q[7];
cx q[6],q[7];
rx(6.143578243570929e-05) q[0];
rz(-0.08534762699442663) q[0];
rx(0.00014939352676781456) q[1];
rz(0.46118596381399496) q[1];
rx(6.751103375711342e-06) q[2];
rz(0.5542141099982881) q[2];
rx(-0.00024560144136479917) q[3];
rz(0.1681895628312281) q[3];
rx(-0.00013123514469586187) q[4];
rz(-1.2054806919171956) q[4];
rx(-0.019633338459634976) q[5];
rz(0.0651458323786446) q[5];
rx(-0.0009233739466863121) q[6];
rz(-0.5445840451800147) q[6];
rx(-0.003999318372425842) q[7];
rz(0.5737431359402033) q[7];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.07567736916440468) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.13480332937906503) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.17405162200962182) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.36154728838410577) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.042741975577874985) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.3287148384290967) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.07528786722756517) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.054376334999628954) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.003055824343623991) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(1.0629553289062015) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(-0.5317836345512972) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(-0.8986758615187362) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(0.09312663259925155) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(0.004360160467672427) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(-0.0005508564295751628) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(1.0484801432880277) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(0.02616314206051402) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(-0.07992000262713204) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(0.16647020816114658) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(-0.3004459545575204) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(-0.09924193215897814) q[7];
cx q[6],q[7];
rx(-5.568667276394742e-05) q[0];
rz(0.16715945161427342) q[0];
rx(-0.00014158849556841717) q[1];
rz(-0.2929874826134818) q[1];
rx(-3.0439252223201537e-05) q[2];
rz(0.4456060643082474) q[2];
rx(0.00026702034992970254) q[3];
rz(-0.25148943190529427) q[3];
rx(0.00018897045632741748) q[4];
rz(0.48056527577789526) q[4];
rx(0.019539522773739867) q[5];
rz(-1.1635873539337607) q[5];
rx(-0.00031811254608504806) q[6];
rz(0.46241290409992813) q[6];
rx(0.0026889843265987274) q[7];
rz(-1.1176415217329898) q[7];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.08418007873218071) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.16350899763082594) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.45608202035839474) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.7455128674800506) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(1.7353763816080012) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.20593694106782884) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.3786853112728414) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.30674128089264563) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.00044057436352912586) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(0.9436283525362957) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(-0.8387655052323396) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(-0.03248142818657099) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(0.015327745653105464) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(-1.2148078875514068) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(-0.000614327618057875) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(0.03696718224213749) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(-0.739947840939222) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(-0.06043665300807286) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(-1.6901920338007788) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(0.2808664370441424) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(0.16412353028598897) q[7];
cx q[6],q[7];
rx(2.754584015565812e-05) q[0];
rz(0.6477513498106166) q[0];
rx(1.1233219253057257e-05) q[1];
rz(-0.08382706645424275) q[1];
rx(0.00010559249264625495) q[2];
rz(1.4599422915985738) q[2];
rx(-4.198069081780786e-06) q[3];
rz(0.16341374210084308) q[3];
rx(-0.0001283764660625007) q[4];
rz(-0.08052031627182298) q[4];
rx(0.00017293213601698496) q[5];
rz(-1.4841025549613303) q[5];
rx(-0.0005488777345878386) q[6];
rz(-1.1542702637139288) q[6];
rx(0.0024861376605190273) q[7];
rz(0.626399658802066) q[7];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.6229669193388875) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.9840292858606396) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.11291481925816486) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.1350650872582349) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.00992362457968414) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.0762032903222033) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.021093019067868108) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-1.5998421672390737) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.009761006510166191) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(0.0527414150867951) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(-2.427245691245467) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(0.05305591315035142) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(-0.34624518669760207) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(-1.2040573300343227) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(-0.3459979375707066) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(0.05705191600054681) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(0.03933703380217263) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(0.06208923621858953) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(-0.8163980325381422) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(-0.70189459162659) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(-0.6740040122518638) q[7];
cx q[6],q[7];
rx(-2.8130469661222555e-05) q[0];
rz(0.24174081694104305) q[0];
rx(-4.546987589416202e-05) q[1];
rz(-1.4533234219109163) q[1];
rx(0.00010155981285623852) q[2];
rz(-1.8668431601531148) q[2];
rx(-1.6870172634357971e-06) q[3];
rz(0.37552152199726296) q[3];
rx(0.0018722077384907194) q[4];
rz(0.8508663452849833) q[4];
rx(-0.0002141124235727045) q[5];
rz(0.5420359832222428) q[5];
rx(-5.20586483759248e-05) q[6];
rz(0.40765202865047995) q[6];
rx(-0.00039849806983119156) q[7];
rz(0.607094793180394) q[7];