OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.15751038921084495) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.0033325565937775184) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.0164707262446399) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.19230886974505104) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.6013193145420219) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.06263249767983323) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.05167227322074148) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.01062787912054964) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.2378460071572276) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(-0.041031593326664305) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(-0.056476836083253275) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(-0.2717031440512884) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(-0.07038942530112925) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(-0.06622758388414374) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(-0.4949410324177158) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(-0.3538049579915607) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(-0.504288627093626) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(0.6804041677410827) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(0.022541924965704133) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(0.014629276907649683) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(-0.15670734013202098) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(-0.38828471166083867) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(0.539405630726829) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(0.16469936532067636) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(0.2158401312064061) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(0.026702786860102495) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(0.050911831951422626) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(0.02123274041763282) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(0.003604223508075922) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(-0.3928034135524416) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(-0.22546109245067383) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(-0.05711192966068614) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(0.3405473285199034) q[11];
cx q[10],q[11];
rx(-1.241087830975132) q[0];
rz(-0.23688622133659176) q[0];
rx(1.4008301876931586) q[1];
rz(-0.42165747824481686) q[1];
rx(0.38802563880597235) q[2];
rz(0.09765370307538622) q[2];
rx(-0.029102987423765447) q[3];
rz(-0.5842864114518406) q[3];
rx(-0.009451813205939153) q[4];
rz(-0.09718818511729709) q[4];
rx(-0.7354204434677689) q[5];
rz(0.15380452193422478) q[5];
rx(0.9851865285982155) q[6];
rz(-0.16708311102294462) q[6];
rx(-1.4800652472738043) q[7];
rz(-0.08163289623707491) q[7];
rx(0.2723798359056833) q[8];
rz(0.4385602692351066) q[8];
rx(-0.14623977291560886) q[9];
rz(-0.03351461610430712) q[9];
rx(0.6928524046783646) q[10];
rz(0.12159503238916582) q[10];
rx(-0.13241402521934686) q[11];
rz(-0.30214406827972506) q[11];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.015426578303244566) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.4142800064225007) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.12794491952367645) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(1.19600287217003) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-1.4660147126176082) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.2886813303469209) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.24734528245942813) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.15187644239107906) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.1463264981741398) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(-0.023945532651640296) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(0.0076897211466430675) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(0.31220719083779636) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(-0.04086630376729241) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(-0.00873540083751415) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(0.014502582245205118) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(-0.34259346373081456) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(-0.17041969539891208) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(0.2107936113461876) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(-0.02256148890549586) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(-0.11591211270985022) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(0.03639280335683265) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(-1.008555557343941) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(1.3153606020909818) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(0.16151403933822492) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(-0.030401015305736582) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(0.2927502247397165) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(0.2698510907726785) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(-0.2614920803448805) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(0.15162117243468565) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(0.10022080636852719) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(0.020859577143282827) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(0.05017541867825353) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(0.15548604948666497) q[11];
cx q[10],q[11];
rx(-0.9467332148691208) q[0];
rz(-0.8740294147621299) q[0];
rx(0.8903113127709813) q[1];
rz(0.29634342115708406) q[1];
rx(0.35940126099996456) q[2];
rz(0.16341876099106695) q[2];
rx(-0.19969340609356837) q[3];
rz(0.5281737796624659) q[3];
rx(0.8615883556112344) q[4];
rz(-0.015095674328383213) q[4];
rx(-0.860478939951786) q[5];
rz(-0.07255247043058353) q[5];
rx(0.5490089462315785) q[6];
rz(-0.2025544895910951) q[6];
rx(-1.081036806037145) q[7];
rz(0.3087181643067838) q[7];
rx(0.12336523082698374) q[8];
rz(0.2813905545751358) q[8];
rx(-0.0012318117421228848) q[9];
rz(0.885601254927046) q[9];
rx(0.6841852931451878) q[10];
rz(-0.09722803092829076) q[10];
rx(-0.1011317988443227) q[11];
rz(0.1584082011038931) q[11];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.36153664870887514) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.06917886645508055) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.3806175255023889) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(1.2757752888036573) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-1.6773358619784515) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.281299713928104) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.20510860562313565) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.3773628274053715) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.09023207111764997) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(-0.02713808553997143) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(-0.0004980334395023949) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(0.03128378002848516) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(0.059222276045272505) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(0.007802535075529662) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(-0.04492413760248693) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(-0.014420852990535318) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(-0.6847880975612222) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(-0.0022235924938886755) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(0.6046698742263533) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(0.03013365426899815) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(-0.03250476469274761) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(-0.49056197101431154) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(0.007618067218620462) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(0.03885035639202431) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(-0.18281132490104857) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(-0.12470322394214205) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(0.17277676914700288) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(-0.5772227152257547) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(0.12000223929563107) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(-0.020437269148943383) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(-0.16774997381412896) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(0.14878405694557942) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(-0.1578387789575127) q[11];
cx q[10],q[11];
rx(-1.04430924644628) q[0];
rz(-0.47784164064253687) q[0];
rx(1.2302241776070157) q[1];
rz(-0.15020858901018724) q[1];
rx(0.36081167657242375) q[2];
rz(-0.43105638317072714) q[2];
rx(0.010748104712093039) q[3];
rz(0.9248009137281339) q[3];
rx(0.5484785876664292) q[4];
rz(0.4866120983179485) q[4];
rx(-0.7285319714909662) q[5];
rz(0.341584256067965) q[5];
rx(0.759974569922313) q[6];
rz(0.052917314686638475) q[6];
rx(-0.6381670870726712) q[7];
rz(0.47360436988792215) q[7];
rx(0.1413327481721026) q[8];
rz(-0.3066743708923107) q[8];
rx(-0.22846897650760584) q[9];
rz(-0.12008069652470284) q[9];
rx(0.37789614453191406) q[10];
rz(-0.5399838321487903) q[10];
rx(-0.10563108732250806) q[11];
rz(0.118610945771233) q[11];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.3039565336675611) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.49750594124903963) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.5561702636451589) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.32470905960777185) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.594222529866097) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.26981726358440405) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.01714801776889327) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.15068332774006246) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.02739561864837323) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(-0.11334121314934294) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(0.013103344621511453) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(-0.016821099956943683) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(0.18975276037159647) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(0.07232743748175385) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(0.04070284572362609) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(0.027361636313218918) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(-0.02964445632196792) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(-0.027405359833550174) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(0.7819869451630125) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(0.305268895128482) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(0.22363287240780316) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(-0.03449474644422918) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(0.004180647065373553) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(0.0006498142462300625) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(0.7057877178001348) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(-0.0759768641869863) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(-0.0130346892918322) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(-1.113726054232421) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(0.13401312882811153) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(-0.029767967694496814) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(0.4833841676237348) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(-0.09730520904870851) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(0.17272991225107254) q[11];
cx q[10],q[11];
rx(-0.6972740363372198) q[0];
rz(0.5897766229520338) q[0];
rx(0.9397626656313143) q[1];
rz(-0.30741810893287874) q[1];
rx(-0.14459090289058116) q[2];
rz(-0.22894437279854915) q[2];
rx(0.02028412709745594) q[3];
rz(1.158133077050857) q[3];
rx(0.024389551794578936) q[4];
rz(1.493743990674303) q[4];
rx(0.2578932363963219) q[5];
rz(1.2080194959334147) q[5];
rx(0.6161743810037027) q[6];
rz(1.0909103973958996) q[6];
rx(-0.4980821591815732) q[7];
rz(-0.5560744438443491) q[7];
rx(0.11863367977492177) q[8];
rz(-0.23460091193946592) q[8];
rx(0.132683740914825) q[9];
rz(-0.3138794438157665) q[9];
rx(-0.0678603750100893) q[10];
rz(-0.8593651008011629) q[10];
rx(-1.0220494889925147) q[11];
rz(-0.24191274704956145) q[11];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.2496846222497773) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.40094867993076605) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.08064736088564452) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.4187700983834151) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.8032133702335954) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.05838715054357873) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.01134497012861052) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.7460589127949061) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.009908621369914494) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(-0.07347065719394118) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(0.1458964227585484) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(-0.00025892519135597807) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(-0.2870326208178806) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(-0.15357642513893147) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(-1.2103655065416223) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(0.010599854003812439) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(-0.007488318145058556) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(0.009903718023461345) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(0.0005457700596230939) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(0.6552889835201698) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(0.610697048279068) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(0.021748775709890995) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(0.020336523226446704) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(-0.02104403540917821) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(0.16788059788359816) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(-0.05592337411898126) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(0.06920191937909016) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(-0.9998240348666274) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(0.7364234237300603) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(-0.13846820088506706) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(-0.4806730176351097) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(0.6583712694320865) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(0.05185751307332752) q[11];
cx q[10],q[11];
rx(-0.726946935542133) q[0];
rz(0.37615438694444453) q[0];
rx(0.6686461533214181) q[1];
rz(0.004183972853055924) q[1];
rx(-0.03517217279414116) q[2];
rz(-0.11228783260802375) q[2];
rx(-0.03296023974204778) q[3];
rz(-0.17191853978355726) q[3];
rx(1.1639167533263701) q[4];
rz(-0.0748093122983636) q[4];
rx(0.11472303906359546) q[5];
rz(0.47229396947637997) q[5];
rx(0.22587935265268452) q[6];
rz(0.07859088172586032) q[6];
rx(-0.1967440780497009) q[7];
rz(-0.05199085275009957) q[7];
rx(0.2057728430548199) q[8];
rz(-0.06337494374548909) q[8];
rx(-0.40016699483262136) q[9];
rz(-0.591171643065767) q[9];
rx(0.9347049707142135) q[10];
rz(0.11218565435618405) q[10];
rx(-0.9774886927395109) q[11];
rz(0.32659774591419805) q[11];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.27142203961974004) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.06374792972632992) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.2661393165910831) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.47325061829536064) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.4490601000935663) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.1840177067849932) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.07302262800103415) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.4495227313059803) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.06865994297075662) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(0.022564787825563922) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(-0.01958976446394528) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(-0.00919475146366888) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(0.2320542370403921) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(-0.5398057847376009) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(0.7811703040965895) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(0.003279260517931141) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(0.0017281357580457267) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(0.011983779755545874) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(0.4568098991685398) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(0.31022277200280257) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(0.21663648664670246) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(-0.01624692818144846) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(-0.0034830711311340094) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(-0.00409651167252633) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(0.3070921399207064) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(1.2359548475863393) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(-0.18580310565281863) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(-0.05441676966115324) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(-0.035451464756181404) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(-0.031021162790887118) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(-0.2105342076912509) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(-0.7012903916243403) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(0.32878008029684325) q[11];
cx q[10],q[11];
rx(-0.9983773920772018) q[0];
rz(-0.321989135713847) q[0];
rx(0.2110664848463088) q[1];
rz(0.3870019099112561) q[1];
rx(0.3240978417311871) q[2];
rz(-0.49604737578221436) q[2];
rx(-0.3741929738326819) q[3];
rz(0.35599364321516025) q[3];
rx(0.5560503139245155) q[4];
rz(0.3439104451280701) q[4];
rx(0.11364706875208673) q[5];
rz(-0.44390324519164903) q[5];
rx(-0.5113061043975468) q[6];
rz(-0.07101214260972896) q[6];
rx(0.12513944974093072) q[7];
rz(-0.12535457435641625) q[7];
rx(-0.8068091190301003) q[8];
rz(0.17343704733711932) q[8];
rx(0.4908305382950557) q[9];
rz(0.18134527526204283) q[9];
rx(0.6554276243125564) q[10];
rz(0.7760970450122239) q[10];
rx(-0.42196434386752996) q[11];
rz(-0.9275864979768551) q[11];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.33848575825907007) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.7657614958885572) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.7816028014537729) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.050761128999496904) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.031972387133166046) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.016257367226693485) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.2382899402793737) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.013509911459847021) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.02520498417574058) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(0.021338472032413923) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(0.029761150403243203) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(0.05393760851178691) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(-0.38882758922208854) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(-1.2872336240855862) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(-0.40238832136799807) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(-0.06040809667535067) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(0.05318175403957542) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(-0.0582196203716465) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(-0.7062166149787211) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(2.3216732127391215) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(-0.6948131939872078) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(-0.05986721594664911) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(-0.05392489793788451) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(-0.0569326129033953) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(0.8143790458531767) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(0.8328005396671988) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(0.7387454470425463) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(-0.07315638904970849) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(-0.08635891573558846) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(0.0662389390032956) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(-0.7694787435582195) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(-0.8067603830834494) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(2.1325229913633903) q[11];
cx q[10],q[11];
rx(-0.818789162520814) q[0];
rz(-0.1312579288664654) q[0];
rx(0.5241685278097389) q[1];
rz(0.08204700543501647) q[1];
rx(0.43604998469573375) q[2];
rz(-0.7034090614618572) q[2];
rx(0.05628557863977248) q[3];
rz(0.06247652569713083) q[3];
rx(0.43302809006253073) q[4];
rz(0.3831410338939651) q[4];
rx(-0.045937789743790154) q[5];
rz(-0.040196609732240665) q[5];
rx(-0.09193938226994539) q[6];
rz(-0.03226182966807845) q[6];
rx(-0.027938217324596695) q[7];
rz(-0.0693021834335028) q[7];
rx(-0.0733120058203179) q[8];
rz(-0.11833672776723511) q[8];
rx(-0.01171694986889493) q[9];
rz(-0.106997482497108) q[9];
rx(0.1338831129582553) q[10];
rz(0.3215288374110851) q[10];
rx(-0.1696746282930117) q[11];
rz(-0.0876708830221351) q[11];