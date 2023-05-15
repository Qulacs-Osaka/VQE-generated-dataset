OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(1.5640741783668848) q[0];
rz(-1.3329121682399744e-05) q[0];
ry(3.009070866289979) q[1];
rz(-1.5709053857235424) q[1];
ry(-0.8470562644839752) q[2];
rz(3.1411358532478304) q[2];
ry(3.123579839626023) q[3];
rz(-1.566245702261958) q[3];
ry(-0.028092692633736847) q[4];
rz(-3.137751626374651) q[4];
ry(-0.31747483459438725) q[5];
rz(0.006893228343108681) q[5];
ry(-3.1289347361224076) q[6];
rz(-3.1332536513858926) q[6];
ry(1.6040204628506747) q[7];
rz(-3.141120393448922) q[7];
ry(0.031044660676409208) q[8];
rz(3.1363871095435876) q[8];
ry(0.5415254081244316) q[9];
rz(0.003947778722481294) q[9];
ry(0.4700916558017312) q[10];
rz(-0.002771222453541) q[10];
ry(-3.1025813092957604) q[11];
rz(0.0006457143103570929) q[11];
ry(1.5271044568336594) q[12];
rz(6.883058726980806e-05) q[12];
ry(-0.04798005206319532) q[13];
rz(0.0023063195067837532) q[13];
ry(-2.664287453616356) q[14];
rz(3.14087984178245) q[14];
ry(3.113956323524787) q[15];
rz(-0.0016040041029530272) q[15];
ry(-1.107606156402971) q[16];
rz(3.1405556147880715) q[16];
ry(3.0531108583553483) q[17];
rz(-0.0006937918619316363) q[17];
ry(-1.790445269742613) q[18];
rz(1.5707249006059647) q[18];
ry(-0.0379844327665034) q[19];
rz(1.5698863883592622) q[19];
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
ry(-1.3928558863188223) q[0];
rz(0.0021009967316629736) q[0];
ry(1.5710726273740008) q[1];
rz(2.7871324142682186) q[1];
ry(2.9570558403458853) q[2];
rz(-3.138306161951685) q[2];
ry(-1.5708302691592575) q[3];
rz(-1.022155917068737) q[3];
ry(-2.444782370222031) q[4];
rz(-1.5650855141536224) q[4];
ry(-3.0832128347364267) q[5];
rz(-1.571851995075031) q[5];
ry(-1.9179426470595555) q[6];
rz(-0.0010073195846205252) q[6];
ry(-1.4941270108568903) q[7];
rz(-0.01512434244172999) q[7];
ry(-0.6054267752125901) q[8];
rz(3.136860693921366) q[8];
ry(-0.04995581135069127) q[9];
rz(-3.0091893077856016) q[9];
ry(-0.035065654109128985) q[10];
rz(0.0015813566904746468) q[10];
ry(2.5595587703791836) q[11];
rz(-3.1386886517558072) q[11];
ry(-1.5279740155095147) q[12];
rz(-3.1415475679979856) q[12];
ry(1.5694969399935053) q[13];
rz(-3.1415101985703386) q[13];
ry(-0.046039642808079546) q[14];
rz(3.140455616539212) q[14];
ry(0.7121746618807236) q[15];
rz(-3.139190692485204) q[15];
ry(0.07117566467514269) q[16];
rz(-3.1401773166659104) q[16];
ry(-1.4049678081994497) q[17];
rz(0.00012412628958592) q[17];
ry(1.5704963674878523) q[18];
rz(-3.100248143651402) q[18];
ry(1.5709344306596236) q[19];
rz(3.106037785004693) q[19];
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
ry(-3.04234821637544) q[0];
rz(-3.133968340338888) q[0];
ry(-1.8586572853628713) q[1];
rz(-1.692133375841209) q[1];
ry(0.14865166612689712) q[2];
rz(-1.2534560711682152) q[2];
ry(0.01691399862114069) q[3];
rz(2.518612984879548) q[3];
ry(0.025263048731142664) q[4];
rz(-1.5738907992612585) q[4];
ry(-3.040368471403735) q[5];
rz(1.5730645277750206) q[5];
ry(1.9016951586686157) q[6];
rz(2.998961673995178) q[6];
ry(-0.031184106164109607) q[7];
rz(0.004752485251303007) q[7];
ry(3.0218587144363367) q[8];
rz(-3.0166976229560087) q[8];
ry(-3.1413894450790725) q[9];
rz(0.12614797573170924) q[9];
ry(3.010527893114084) q[10];
rz(3.130233049005426) q[10];
ry(-3.1325215446394417) q[11];
rz(0.005247454950008341) q[11];
ry(2.39440471111546) q[12];
rz(0.0042863704446709505) q[12];
ry(-1.5970817865810956) q[13];
rz(-3.139806504847899) q[13];
ry(3.134876303637066) q[14];
rz(-3.138421573832303) q[14];
ry(-3.1058166138529075) q[15];
rz(-3.1407238529809094) q[15];
ry(-3.004423577237248) q[16];
rz(0.0015874906058403076) q[16];
ry(-3.016305721755342) q[17];
rz(0.0007952033945874958) q[17];
ry(1.366090386106178) q[18];
rz(-0.028479998213462924) q[18];
ry(0.569591256646234) q[19];
rz(2.615820672672082) q[19];
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
ry(0.1185211147301297) q[0];
rz(-3.1058066552134345) q[0];
ry(-1.5725028696741754) q[1];
rz(-1.5301952240888579) q[1];
ry(3.140724914858246) q[2];
rz(-1.2084543214648698) q[2];
ry(1.572581942875868) q[3];
rz(1.6116892671191385) q[3];
ry(0.31544615902998546) q[4];
rz(-3.0972271638159548) q[4];
ry(-0.7639325119822987) q[5];
rz(0.03940444601427018) q[5];
ry(3.1392599037540236) q[6];
rz(-0.09567774591664067) q[6];
ry(2.4108172374591446) q[7];
rz(0.039357637992118555) q[7];
ry(-0.013801545602526666) q[8];
rz(3.0587448874345147) q[8];
ry(2.4308050732099393) q[9];
rz(0.03953936021884894) q[9];
ry(-3.0288790200034055) q[10];
rz(0.030674705136749303) q[10];
ry(-0.5600167972229286) q[11];
rz(0.039515144085117165) q[11];
ry(-2.808702185294124) q[12];
rz(-3.0967463923199383) q[12];
ry(0.522524890220253) q[13];
rz(-3.1015509336555427) q[13];
ry(-0.2782586675196968) q[14];
rz(-3.10576037441775) q[14];
ry(2.7083933383840386) q[15];
rz(0.04031382248089592) q[15];
ry(-1.4810003036676793) q[16];
rz(-3.101207331711303) q[16];
ry(-0.9279433843679055) q[17];
rz(-3.1022634174415984) q[17];
ry(-1.5722950518756054) q[18];
rz(1.611312533995583) q[18];
ry(1.5702250553356532) q[19];
rz(1.610538600254971) q[19];