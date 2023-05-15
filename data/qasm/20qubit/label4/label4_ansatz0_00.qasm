OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.3659918800248727) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.2680719642710909) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-1.1469674199782494) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.573916246231007) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.4695550631100458) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
h q[2];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.8673869471272057) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[4];
sdg q[2];
h q[2];
sdg q[4];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.8630505534914273) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[3];
rz(2.560818432710365) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(0.3436884121307379) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[5];
sdg q[3];
h q[3];
sdg q[5];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.15181629257790671) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[5];
s q[5];
h q[4];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(0.5981148463254563) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.5701711938971652) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(0.03633161904873916) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.8007478467559507) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-1.0463574636015784) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
h q[6];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.690516361398489) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
h q[8];
sdg q[6];
h q[6];
sdg q[8];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(0.7045084589103836) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
s q[6];
h q[8];
s q[8];
cx q[6],q[7];
rz(-0.03866750713930936) q[7];
cx q[6],q[7];
h q[7];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(-1.203781514341994) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
h q[9];
sdg q[7];
h q[7];
sdg q[9];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(2.058179365234776) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
s q[7];
h q[9];
s q[9];
h q[8];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(0.850378330158555) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
h q[10];
sdg q[8];
h q[8];
sdg q[10];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.8628327294139317) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
s q[8];
h q[10];
s q[10];
cx q[8],q[9];
rz(-0.03978828893273547) q[9];
cx q[8],q[9];
h q[9];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(0.03298997544973139) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
h q[11];
sdg q[9];
h q[9];
sdg q[11];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-1.5824359754909523) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
s q[9];
h q[11];
s q[11];
h q[10];
h q[12];
cx q[10],q[11];
cx q[11],q[12];
rz(-0.23332851007707575) q[12];
cx q[11],q[12];
cx q[10],q[11];
h q[10];
h q[12];
sdg q[10];
h q[10];
sdg q[12];
h q[12];
cx q[10],q[11];
cx q[11],q[12];
rz(-0.23298346806790604) q[12];
cx q[11],q[12];
cx q[10],q[11];
h q[10];
s q[10];
h q[12];
s q[12];
cx q[10],q[11];
rz(-0.12310533640444135) q[11];
cx q[10],q[11];
h q[11];
h q[13];
cx q[11],q[12];
cx q[12],q[13];
rz(-0.7915295663262722) q[13];
cx q[12],q[13];
cx q[11],q[12];
h q[11];
h q[13];
sdg q[11];
h q[11];
sdg q[13];
h q[13];
cx q[11],q[12];
cx q[12],q[13];
rz(-0.00792395366866069) q[13];
cx q[12],q[13];
cx q[11],q[12];
h q[11];
s q[11];
h q[13];
s q[13];
h q[12];
h q[14];
cx q[12],q[13];
cx q[13],q[14];
rz(0.9362824821918878) q[14];
cx q[13],q[14];
cx q[12],q[13];
h q[12];
h q[14];
sdg q[12];
h q[12];
sdg q[14];
h q[14];
cx q[12],q[13];
cx q[13],q[14];
rz(-0.9914681338372276) q[14];
cx q[13],q[14];
cx q[12],q[13];
h q[12];
s q[12];
h q[14];
s q[14];
cx q[12],q[13];
rz(0.02911471992613893) q[13];
cx q[12],q[13];
h q[13];
h q[15];
cx q[13],q[14];
cx q[14],q[15];
rz(1.079344963301433) q[15];
cx q[14],q[15];
cx q[13],q[14];
h q[13];
h q[15];
sdg q[13];
h q[13];
sdg q[15];
h q[15];
cx q[13],q[14];
cx q[14],q[15];
rz(-1.9892227126531639) q[15];
cx q[14],q[15];
cx q[13],q[14];
h q[13];
s q[13];
h q[15];
s q[15];
h q[14];
h q[16];
cx q[14],q[15];
cx q[15],q[16];
rz(-0.17717371871944762) q[16];
cx q[15],q[16];
cx q[14],q[15];
h q[14];
h q[16];
sdg q[14];
h q[14];
sdg q[16];
h q[16];
cx q[14],q[15];
cx q[15],q[16];
rz(-0.16532338788335785) q[16];
cx q[15],q[16];
cx q[14],q[15];
h q[14];
s q[14];
h q[16];
s q[16];
cx q[14],q[15];
rz(0.01242915812979323) q[15];
cx q[14],q[15];
h q[15];
h q[17];
cx q[15],q[16];
cx q[16],q[17];
rz(0.32705288927588944) q[17];
cx q[16],q[17];
cx q[15],q[16];
h q[15];
h q[17];
sdg q[15];
h q[15];
sdg q[17];
h q[17];
cx q[15],q[16];
cx q[16],q[17];
rz(-0.4862924049660813) q[17];
cx q[16],q[17];
cx q[15],q[16];
h q[15];
s q[15];
h q[17];
s q[17];
h q[16];
h q[18];
cx q[16],q[17];
cx q[17],q[18];
rz(-0.8869528175196425) q[18];
cx q[17],q[18];
cx q[16],q[17];
h q[16];
h q[18];
sdg q[16];
h q[16];
sdg q[18];
h q[18];
cx q[16],q[17];
cx q[17],q[18];
rz(0.8432585334623506) q[18];
cx q[17],q[18];
cx q[16],q[17];
h q[16];
s q[16];
h q[18];
s q[18];
cx q[16],q[17];
rz(-0.07124409224789871) q[17];
cx q[16],q[17];
h q[17];
h q[19];
cx q[17],q[18];
cx q[18],q[19];
rz(-0.012982704448012762) q[19];
cx q[18],q[19];
cx q[17],q[18];
h q[17];
h q[19];
sdg q[17];
h q[17];
sdg q[19];
h q[19];
cx q[17],q[18];
cx q[18],q[19];
rz(1.5600276261030248) q[19];
cx q[18],q[19];
cx q[17],q[18];
h q[17];
s q[17];
h q[19];
s q[19];
cx q[18],q[19];
rz(-0.006097601126837818) q[19];
cx q[18],q[19];
rx(0.0049036356070216855) q[0];
rz(0.0022409962301349214) q[0];
rx(0.0011737409138025839) q[1];
rz(1.8963959025349444) q[1];
rx(-3.1413185045531717) q[2];
rz(0.9074247906151396) q[2];
rx(0.0013673189803203392) q[3];
rz(-1.5345929278179296) q[3];
rx(0.0013357117596140568) q[4];
rz(-1.2125020920838903) q[4];
rx(1.5131205592837691e-05) q[5];
rz(1.6553705610893057) q[5];
rx(0.0009317548020931618) q[6];
rz(-0.22901091686652447) q[6];
rx(-9.602427196862175e-05) q[7];
rz(-1.7361617076110547) q[7];
rx(-3.141584142211777) q[8];
rz(0.21917453511010243) q[8];
rx(3.1415625652325985) q[9];
rz(-1.5524254216942281) q[9];
rx(-3.141194713029296) q[10];
rz(-1.3394054846982204) q[10];
rx(3.1407551659637503) q[11];
rz(2.8921444033818573) q[11];
rx(0.0003322636686412357) q[12];
rz(-0.40228703081204614) q[12];
rx(-0.0001382722507085665) q[13];
rz(-0.9832860398572222) q[13];
rx(0.0011176993654101008) q[14];
rz(-0.41577526658204483) q[14];
rx(0.0006890626524462372) q[15];
rz(-3.137380268146034) q[15];
rx(-3.140917266123295) q[16];
rz(0.4129492783972971) q[16];
rx(0.00011410440409266289) q[17];
rz(-1.49896067774118) q[17];
rx(-0.0002517126989327285) q[18];
rz(1.3161092544834965) q[18];
rx(0.000307916183118013) q[19];
rz(0.015093831621281334) q[19];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.00760271436334875) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.09612356973771982) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-2.923394584050871) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.9391621128373963) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(1.4539814581696147) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
h q[2];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(1.0131312353324786) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[4];
sdg q[2];
h q[2];
sdg q[4];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(-1.0672020701567226) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[3];
rz(-0.2832435823956248) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(3.073151389796641) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[5];
sdg q[3];
h q[3];
sdg q[5];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(0.504076485621855) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[5];
s q[5];
h q[4];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(0.3653824020225252) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.37687548216421857) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(0.02629877472976103) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-1.8697643965064923) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.46534943659383593) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
h q[6];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(0.8650175611558392) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
h q[8];
sdg q[6];
h q[6];
sdg q[8];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(0.8904011373303475) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
s q[6];
h q[8];
s q[8];
cx q[6],q[7];
rz(-0.11545720883168632) q[7];
cx q[6],q[7];
h q[7];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(0.4399985088232622) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
h q[9];
sdg q[7];
h q[7];
sdg q[9];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(-1.028398874726605) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
s q[7];
h q[9];
s q[9];
h q[8];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.6483227112528527) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
h q[10];
sdg q[8];
h q[8];
sdg q[10];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(0.607402996584524) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
s q[8];
h q[10];
s q[10];
cx q[8],q[9];
rz(0.07348198685931259) q[9];
cx q[8],q[9];
h q[9];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-2.3121456569502126) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
h q[11];
sdg q[9];
h q[9];
sdg q[11];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.01606629365833428) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
s q[9];
h q[11];
s q[11];
h q[10];
h q[12];
cx q[10],q[11];
cx q[11],q[12];
rz(-0.9034510169080134) q[12];
cx q[11],q[12];
cx q[10],q[11];
h q[10];
h q[12];
sdg q[10];
h q[10];
sdg q[12];
h q[12];
cx q[10],q[11];
cx q[11],q[12];
rz(0.8910755159589012) q[12];
cx q[11],q[12];
cx q[10],q[11];
h q[10];
s q[10];
h q[12];
s q[12];
cx q[10],q[11];
rz(0.07249602135530629) q[11];
cx q[10],q[11];
h q[11];
h q[13];
cx q[11],q[12];
cx q[12],q[13];
rz(-0.277939916238751) q[13];
cx q[12],q[13];
cx q[11],q[12];
h q[11];
h q[13];
sdg q[11];
h q[11];
sdg q[13];
h q[13];
cx q[11],q[12];
cx q[12],q[13];
rz(-1.5580987526512111) q[13];
cx q[12],q[13];
cx q[11],q[12];
h q[11];
s q[11];
h q[13];
s q[13];
h q[12];
h q[14];
cx q[12],q[13];
cx q[13],q[14];
rz(1.0271348119224153) q[14];
cx q[13],q[14];
cx q[12],q[13];
h q[12];
h q[14];
sdg q[12];
h q[12];
sdg q[14];
h q[14];
cx q[12],q[13];
cx q[13],q[14];
rz(-0.9953041683553544) q[14];
cx q[13],q[14];
cx q[12],q[13];
h q[12];
s q[12];
h q[14];
s q[14];
cx q[12],q[13];
rz(-0.057489287019259656) q[13];
cx q[12],q[13];
h q[13];
h q[15];
cx q[13],q[14];
cx q[14],q[15];
rz(-3.1279930577712407) q[15];
cx q[14],q[15];
cx q[13],q[14];
h q[13];
h q[15];
sdg q[13];
h q[13];
sdg q[15];
h q[15];
cx q[13],q[14];
cx q[14],q[15];
rz(2.160080029268385) q[15];
cx q[14],q[15];
cx q[13],q[14];
h q[13];
s q[13];
h q[15];
s q[15];
h q[14];
h q[16];
cx q[14],q[15];
cx q[15],q[16];
rz(0.5261010118368397) q[16];
cx q[15],q[16];
cx q[14],q[15];
h q[14];
h q[16];
sdg q[14];
h q[14];
sdg q[16];
h q[16];
cx q[14],q[15];
cx q[15],q[16];
rz(-0.5320595866992308) q[16];
cx q[15],q[16];
cx q[14],q[15];
h q[14];
s q[14];
h q[16];
s q[16];
cx q[14],q[15];
rz(0.09268519548515315) q[15];
cx q[14],q[15];
h q[15];
h q[17];
cx q[15],q[16];
cx q[16],q[17];
rz(-1.5021663266768883) q[17];
cx q[16],q[17];
cx q[15],q[16];
h q[15];
h q[17];
sdg q[15];
h q[15];
sdg q[17];
h q[17];
cx q[15],q[16];
cx q[16],q[17];
rz(-0.15806325692034504) q[17];
cx q[16],q[17];
cx q[15],q[16];
h q[15];
s q[15];
h q[17];
s q[17];
h q[16];
h q[18];
cx q[16],q[17];
cx q[17],q[18];
rz(0.10687578348591974) q[18];
cx q[17],q[18];
cx q[16],q[17];
h q[16];
h q[18];
sdg q[16];
h q[16];
sdg q[18];
h q[18];
cx q[16],q[17];
cx q[17],q[18];
rz(-3.0314288162102354) q[18];
cx q[17],q[18];
cx q[16],q[17];
h q[16];
s q[16];
h q[18];
s q[18];
cx q[16],q[17];
rz(0.14381774390613267) q[17];
cx q[16],q[17];
h q[17];
h q[19];
cx q[17],q[18];
cx q[18],q[19];
rz(-1.582347957581575) q[19];
cx q[18],q[19];
cx q[17],q[18];
h q[17];
h q[19];
sdg q[17];
h q[17];
sdg q[19];
h q[19];
cx q[17],q[18];
cx q[18],q[19];
rz(2.0767970950450403) q[19];
cx q[18],q[19];
cx q[17],q[18];
h q[17];
s q[17];
h q[19];
s q[19];
cx q[18],q[19];
rz(-0.09192013181305218) q[19];
cx q[18],q[19];
rx(0.00537029471779835) q[0];
rz(-0.6990628162782156) q[0];
rx(0.0009197944106956987) q[1];
rz(-1.6584040525005674) q[1];
rx(-0.0004011816805499415) q[2];
rz(-3.000906822239862) q[2];
rx(0.00015541195500475306) q[3];
rz(0.03701894672558294) q[3];
rx(0.000978918967208155) q[4];
rz(-0.44204961276332944) q[4];
rx(-3.1415426489041876) q[5];
rz(1.5106328592767282) q[5];
rx(3.1414580277138184) q[6];
rz(-1.0926136511421725) q[6];
rx(-0.0001226411916153778) q[7];
rz(-3.0930978882595013) q[7];
rx(-0.0012591785778009303) q[8];
rz(-2.467958941965648) q[8];
rx(0.0008960393647121406) q[9];
rz(-0.018842691817103817) q[9];
rx(-3.141384254690509) q[10];
rz(-1.5315718999259171) q[10];
rx(6.29914096734251e-05) q[11];
rz(1.254857185514332) q[11];
rx(-0.00026023139592485907) q[12];
rz(-1.0328796373486482) q[12];
rx(-3.1409056779868156) q[13];
rz(-0.8414949227665177) q[13];
rx(-0.0003415046796375092) q[14];
rz(0.2656335641645381) q[14];
rx(-0.0002531473118944557) q[15];
rz(0.3234080910700315) q[15];
rx(0.00017610019581664632) q[16];
rz(0.3512550425248109) q[16];
rx(-3.140550881245414) q[17];
rz(-0.09332678497882194) q[17];
rx(-0.00037705136808374213) q[18];
rz(2.2747647632305523) q[18];
rx(-0.004163739891402061) q[19];
rz(-0.004161193592635451) q[19];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.978711584963586) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-1.0277845041069478) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.016660991399786492) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.00626665509234341) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-1.5944466261771169) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
h q[2];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(0.7019143691298659) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[4];
sdg q[2];
h q[2];
sdg q[4];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.8023141976535373) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[3];
rz(0.008797899564966323) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.2715058854453072) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[5];
sdg q[3];
h q[3];
sdg q[5];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(0.0983243093748162) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[5];
s q[5];
h q[4];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(2.9081108401260356) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.2261098195920087) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(-0.0797776727231502) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(0.18655405318638957) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-1.5507924785975193) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
h q[6];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(0.13923021618310621) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
h q[8];
sdg q[6];
h q[6];
sdg q[8];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(0.15160897900376644) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
s q[6];
h q[8];
s q[8];
cx q[6],q[7];
rz(0.07001365296447927) q[7];
cx q[6],q[7];
h q[7];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(0.022903178860681966) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
h q[9];
sdg q[7];
h q[7];
sdg q[9];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(0.9248188158386663) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
s q[7];
h q[9];
s q[9];
h q[8];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.20716066128403995) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
h q[10];
sdg q[8];
h q[8];
sdg q[10];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.20762863374126844) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
s q[8];
h q[10];
s q[10];
cx q[8],q[9];
rz(3.1321909423414076) q[9];
cx q[8],q[9];
h q[9];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.14751115650867422) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
h q[11];
sdg q[9];
h q[9];
sdg q[11];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(0.21551354766774924) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
s q[9];
h q[11];
s q[11];
h q[10];
h q[12];
cx q[10],q[11];
cx q[11],q[12];
rz(0.626165730333046) q[12];
cx q[11],q[12];
cx q[10],q[11];
h q[10];
h q[12];
sdg q[10];
h q[10];
sdg q[12];
h q[12];
cx q[10],q[11];
cx q[11],q[12];
rz(0.6610044012820235) q[12];
cx q[11],q[12];
cx q[10],q[11];
h q[10];
s q[10];
h q[12];
s q[12];
cx q[10],q[11];
rz(0.02883955433011406) q[11];
cx q[10],q[11];
h q[11];
h q[13];
cx q[11],q[12];
cx q[12],q[13];
rz(-1.535897351302391) q[13];
cx q[12],q[13];
cx q[11],q[12];
h q[11];
h q[13];
sdg q[11];
h q[11];
sdg q[13];
h q[13];
cx q[11],q[12];
cx q[12],q[13];
rz(-0.3892270137529649) q[13];
cx q[12],q[13];
cx q[11],q[12];
h q[11];
s q[11];
h q[13];
s q[13];
h q[12];
h q[14];
cx q[12],q[13];
cx q[13],q[14];
rz(-0.30572499335104864) q[14];
cx q[13],q[14];
cx q[12],q[13];
h q[12];
h q[14];
sdg q[12];
h q[12];
sdg q[14];
h q[14];
cx q[12],q[13];
cx q[13],q[14];
rz(0.33233739939036344) q[14];
cx q[13],q[14];
cx q[12],q[13];
h q[12];
s q[12];
h q[14];
s q[14];
cx q[12],q[13];
rz(-0.13880976126980124) q[13];
cx q[12],q[13];
h q[13];
h q[15];
cx q[13],q[14];
cx q[14],q[15];
rz(-1.5747325965527355) q[15];
cx q[14],q[15];
cx q[13],q[14];
h q[13];
h q[15];
sdg q[13];
h q[13];
sdg q[15];
h q[15];
cx q[13],q[14];
cx q[14],q[15];
rz(0.4891474336879959) q[15];
cx q[14],q[15];
cx q[13],q[14];
h q[13];
s q[13];
h q[15];
s q[15];
h q[14];
h q[16];
cx q[14],q[15];
cx q[15],q[16];
rz(-0.11909994711285879) q[16];
cx q[15],q[16];
cx q[14],q[15];
h q[14];
h q[16];
sdg q[14];
h q[14];
sdg q[16];
h q[16];
cx q[14],q[15];
cx q[15],q[16];
rz(-0.10988493290976892) q[16];
cx q[15],q[16];
cx q[14],q[15];
h q[14];
s q[14];
h q[16];
s q[16];
cx q[14],q[15];
rz(-0.1297831541129734) q[15];
cx q[14],q[15];
h q[15];
h q[17];
cx q[15],q[16];
cx q[16],q[17];
rz(-1.4882565866496615) q[17];
cx q[16],q[17];
cx q[15],q[16];
h q[15];
h q[17];
sdg q[15];
h q[15];
sdg q[17];
h q[17];
cx q[15],q[16];
cx q[16],q[17];
rz(-0.5935474738594959) q[17];
cx q[16],q[17];
cx q[15],q[16];
h q[15];
s q[15];
h q[17];
s q[17];
h q[16];
h q[18];
cx q[16],q[17];
cx q[17],q[18];
rz(1.5682070337084106) q[18];
cx q[17],q[18];
cx q[16],q[17];
h q[16];
h q[18];
sdg q[16];
h q[16];
sdg q[18];
h q[18];
cx q[16],q[17];
cx q[17],q[18];
rz(-1.5778956076932769) q[18];
cx q[17],q[18];
cx q[16],q[17];
h q[16];
s q[16];
h q[18];
s q[18];
cx q[16],q[17];
rz(0.19730078952006405) q[17];
cx q[16],q[17];
h q[17];
h q[19];
cx q[17],q[18];
cx q[18],q[19];
rz(-1.5625423056738177) q[19];
cx q[18],q[19];
cx q[17],q[18];
h q[17];
h q[19];
sdg q[17];
h q[17];
sdg q[19];
h q[19];
cx q[17],q[18];
cx q[18],q[19];
rz(-0.03016432215226176) q[19];
cx q[18],q[19];
cx q[17],q[18];
h q[17];
s q[17];
h q[19];
s q[19];
cx q[18],q[19];
rz(0.006824136219625014) q[19];
cx q[18],q[19];
rx(-0.0015355413630352285) q[0];
rz(0.36785188267823293) q[0];
rx(-0.0003756536030694608) q[1];
rz(-0.5024512825779344) q[1];
rx(8.809746207946324e-05) q[2];
rz(1.8682301896429874) q[2];
rx(-0.0005348409888217225) q[3];
rz(1.5225347774670204) q[3];
rx(-0.001244811640109985) q[4];
rz(0.5573297371152912) q[4];
rx(0.00011875868863711207) q[5];
rz(-0.8178142776116453) q[5];
rx(0.0003638504949689046) q[6];
rz(-0.9822351566929203) q[6];
rx(-0.0009109119920013605) q[7];
rz(2.0677464515114683) q[7];
rx(0.0009086484380681707) q[8];
rz(0.40474192304697937) q[8];
rx(0.0011453235338361825) q[9];
rz(2.076476634008044) q[9];
rx(0.0004001257465890736) q[10];
rz(-0.2262413728787229) q[10];
rx(-0.00026638562512160233) q[11];
rz(-0.015613596708834655) q[11];
rx(-0.000322224918162747) q[12];
rz(-0.17893543682091811) q[12];
rx(0.00018756476250774096) q[13];
rz(-1.0181020240391194) q[13];
rx(-3.1413887551942716) q[14];
rz(-1.713052567795568) q[14];
rx(-0.00025506746360443077) q[15];
rz(-1.2585684542132622) q[15];
rx(0.0005782471122010441) q[16];
rz(1.1083900000624622) q[16];
rx(-0.0011799974948616576) q[17];
rz(-1.2828741241857957) q[17];
rx(-0.0004160036276067653) q[18];
rz(-1.5417241980291352) q[18];
rx(0.005664398518989802) q[19];
rz(-1.3942261406916705) q[19];