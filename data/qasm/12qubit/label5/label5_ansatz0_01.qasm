OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.9511814863706974) q[2];
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
rz(0.868504865246038) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
h q[0];
h q[4];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.011149664256284489) q[4];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[4];
sdg q[0];
h q[0];
sdg q[4];
h q[4];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.04806243276099055) q[4];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[4];
s q[4];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.04291931519353156) q[3];
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
rz(-0.04347510323269904) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
h q[1];
h q[5];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.026898049266406772) q[5];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[5];
sdg q[1];
h q[1];
sdg q[5];
h q[5];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.003167417948902931) q[5];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[5];
s q[5];
h q[2];
h q[6];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
rz(0.011943354833746322) q[6];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[6];
sdg q[2];
h q[2];
sdg q[6];
h q[6];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.027298317713595154) q[6];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[6];
s q[6];
h q[3];
h q[7];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
rz(0.6286101343126699) q[7];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[7];
sdg q[3];
h q[3];
sdg q[7];
h q[7];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
rz(0.6297471315553765) q[7];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[7];
s q[7];
h q[4];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.40904672646956997) q[6];
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
rz(-0.4093869824492979) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-1.347669542465229) q[7];
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
rz(-0.11540514184780191) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
cx q[0],q[1];
rz(-0.01565218710270249) q[1];
cx q[0],q[1];
cx q[2],q[3];
rz(-1.120189912444114) q[3];
cx q[2],q[3];
cx q[4],q[5];
rz(-0.0043603978265777206) q[5];
cx q[4],q[5];
cx q[6],q[7];
rz(1.5977312731968167) q[7];
cx q[6],q[7];
h q[4];
h q[8];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.5057404734225817) q[8];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[8];
sdg q[4];
h q[4];
sdg q[8];
h q[8];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
rz(0.4405532445254523) q[8];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[8];
s q[8];
h q[5];
h q[9];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
rz(1.9036212473828988) q[9];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[9];
sdg q[5];
h q[5];
sdg q[9];
h q[9];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
rz(0.6898862929436819) q[9];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[9];
s q[9];
h q[6];
h q[10];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
rz(-1.1892285125678377) q[10];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
h q[10];
sdg q[6];
h q[6];
sdg q[10];
h q[10];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
rz(-2.124795014262475) q[10];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
s q[6];
h q[10];
s q[10];
h q[7];
h q[11];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[10],q[11];
rz(-1.2340452392411183) q[11];
cx q[10],q[11];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
h q[11];
sdg q[7];
h q[7];
sdg q[11];
h q[11];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[10],q[11];
rz(0.6961913532273576) q[11];
cx q[10],q[11];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
s q[7];
h q[11];
s q[11];
h q[8];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.125016991722793) q[10];
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
rz(1.440797493093167) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
s q[8];
h q[10];
s q[10];
h q[9];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(1.6831456811199008) q[11];
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
rz(-0.08043778816487603) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
s q[9];
h q[11];
s q[11];
cx q[8],q[9];
rz(-0.03901177271670875) q[9];
cx q[8],q[9];
cx q[10],q[11];
rz(-0.4601039765375313) q[11];
cx q[10],q[11];
rx(0.0019148453507704538) q[0];
rz(0.09093273889885393) q[0];
rx(1.4761409471640304e-05) q[1];
rz(0.2111152396075038) q[1];
rx(-0.0013152971030539321) q[2];
rz(-0.35797856300822384) q[2];
rx(-0.00010064825806466942) q[3];
rz(0.10541880273044882) q[3];
rx(0.0005308893225868959) q[4];
rz(0.3786872563750991) q[4];
rx(-5.4580142290979915e-06) q[5];
rz(0.471908117898312) q[5];
rx(0.0007949724530624723) q[6];
rz(0.469549830245488) q[6];
rx(-6.5663762359281175e-06) q[7];
rz(-0.34166492523206615) q[7];
rx(-0.3886104054024821) q[8];
rz(-0.00065593285595271) q[8];
rx(1.0461770910933289e-05) q[9];
rz(1.7751427734903666) q[9];
rx(-0.000785822911960838) q[10];
rz(-1.4026323838089516) q[10];
rx(-1.6583142989543874e-06) q[11];
rz(-1.4283364674555323) q[11];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.09784586254630634) q[2];
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
rz(2.042999522845225) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
h q[0];
h q[4];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.0005375078376205432) q[4];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[4];
sdg q[0];
h q[0];
sdg q[4];
h q[4];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.017730814734963116) q[4];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[4];
s q[4];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-1.9133638357947647) q[3];
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
rz(-0.33552568228405194) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
h q[1];
h q[5];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
rz(0.017097768236564257) q[5];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[5];
sdg q[1];
h q[1];
sdg q[5];
h q[5];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
rz(0.033690534916153335) q[5];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[5];
s q[5];
h q[2];
h q[6];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.010201195035688667) q[6];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[6];
sdg q[2];
h q[2];
sdg q[6];
h q[6];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.011389850581055606) q[6];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[6];
s q[6];
h q[3];
h q[7];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
rz(0.03662326799743064) q[7];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[7];
sdg q[3];
h q[3];
sdg q[7];
h q[7];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.02171030485248138) q[7];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[7];
s q[7];
h q[4];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-1.7002283401948097) q[6];
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
rz(-0.1274283488690456) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(0.224464731186049) q[7];
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
rz(1.6556490375039832) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
cx q[0],q[1];
rz(0.017232572627787445) q[1];
cx q[0],q[1];
cx q[2],q[3];
rz(0.019252048775392612) q[3];
cx q[2],q[3];
cx q[4],q[5];
rz(-0.12478255299286425) q[5];
cx q[4],q[5];
cx q[6],q[7];
rz(0.12603296022906765) q[7];
cx q[6],q[7];
h q[4];
h q[8];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
rz(0.04694834748787616) q[8];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[8];
sdg q[4];
h q[4];
sdg q[8];
h q[8];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
rz(0.003314272250766661) q[8];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[8];
s q[8];
h q[5];
h q[9];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
rz(0.003935895198974846) q[9];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[9];
sdg q[5];
h q[5];
sdg q[9];
h q[9];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
rz(0.005035754884949019) q[9];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[9];
s q[9];
h q[6];
h q[10];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.0009404258051140871) q[10];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
h q[10];
sdg q[6];
h q[6];
sdg q[10];
h q[10];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.000742884138492335) q[10];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
s q[6];
h q[10];
s q[10];
h q[7];
h q[11];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[10],q[11];
rz(0.002183513760421907) q[11];
cx q[10],q[11];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
h q[11];
sdg q[7];
h q[7];
sdg q[11];
h q[11];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[10],q[11];
rz(0.0018744932667859953) q[11];
cx q[10],q[11];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
s q[7];
h q[11];
s q[11];
h q[8];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(3.1264639707241395) q[10];
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
rz(0.005833707669709345) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
s q[8];
h q[10];
s q[10];
h q[9];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.014515940974486068) q[11];
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
rz(-0.01368326363850441) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
s q[9];
h q[11];
s q[11];
cx q[8],q[9];
rz(0.0015806304302711425) q[9];
cx q[8],q[9];
cx q[10],q[11];
rz(-0.506595440946671) q[11];
cx q[10],q[11];
rx(0.0011482722645231533) q[0];
rz(0.4818848496562509) q[0];
rx(1.1094155641582617e-05) q[1];
rz(-0.6437129236446719) q[1];
rx(-3.431782725388753e-06) q[2];
rz(-0.1836357959977922) q[2];
rx(-3.141497742853059) q[3];
rz(0.864400387486413) q[3];
rx(0.0007292304190487715) q[4];
rz(-1.5725701626919446) q[4];
rx(7.196309257190255e-06) q[5];
rz(0.34482646544417384) q[5];
rx(-8.136390201649187e-05) q[6];
rz(0.0017903675843887138) q[6];
rx(1.3041268058212917e-06) q[7];
rz(-0.9974170994003141) q[7];
rx(-2.7537474689179984) q[8];
rz(-1.5599340548258929) q[8];
rx(9.960450262559245e-06) q[9];
rz(-0.10396213135152847) q[9];
rx(-0.00022895306439181667) q[10];
rz(-1.387266984140225) q[10];
rx(-0.00014605466451727485) q[11];
rz(-0.0983748655566376) q[11];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.6456005041427125) q[2];
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
rz(1.9308172967383288) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
h q[0];
h q[4];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.8906755958400034) q[4];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[4];
sdg q[0];
h q[0];
sdg q[4];
h q[4];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.9012332822517675) q[4];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[4];
s q[4];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.853891678089065) q[3];
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
rz(0.9284136952068437) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
h q[1];
h q[5];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
rz(0.8883868585309946) q[5];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[5];
sdg q[1];
h q[1];
sdg q[5];
h q[5];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
rz(0.8786095944722031) q[5];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[5];
s q[5];
h q[2];
h q[6];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.8819687510880543) q[6];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[6];
sdg q[2];
h q[2];
sdg q[6];
h q[6];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.8992440087961148) q[6];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[6];
s q[6];
h q[3];
h q[7];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.8851931000637121) q[7];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[7];
sdg q[3];
h q[3];
sdg q[7];
h q[7];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.8729049818582554) q[7];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[7];
s q[7];
h q[4];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-1.5560150132988844) q[6];
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
rz(1.6761550628111113) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-1.620684103078413) q[7];
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
rz(-1.6081420065129812) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
cx q[0],q[1];
rz(-0.09867202226316996) q[1];
cx q[0],q[1];
cx q[2],q[3];
rz(0.09666812328770817) q[3];
cx q[2],q[3];
cx q[4],q[5];
rz(-0.1476492781077909) q[5];
cx q[4],q[5];
cx q[6],q[7];
rz(0.14850410797199604) q[7];
cx q[6],q[7];
h q[4];
h q[8];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.22087965301197454) q[8];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[8];
sdg q[4];
h q[4];
sdg q[8];
h q[8];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.2486680845161726) q[8];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[8];
s q[8];
h q[5];
h q[9];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
rz(0.24619443006252584) q[9];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[9];
sdg q[5];
h q[5];
sdg q[9];
h q[9];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
rz(0.23457708143088865) q[9];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[9];
s q[9];
h q[6];
h q[10];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.2392718418327157) q[10];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
h q[10];
sdg q[6];
h q[6];
sdg q[10];
h q[10];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
rz(0.21785909633832143) q[10];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
s q[6];
h q[10];
s q[10];
h q[7];
h q[11];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.23096949440619632) q[11];
cx q[10],q[11];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
h q[11];
sdg q[7];
h q[7];
sdg q[11];
h q[11];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.23688604111226474) q[11];
cx q[10],q[11];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
s q[7];
h q[11];
s q[11];
h q[8];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(1.0048161308613213) q[10];
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
rz(1.0127987252642265) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
s q[8];
h q[10];
s q[10];
h q[9];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-2.1304633285267) q[11];
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
rz(1.0255807940314086) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
s q[9];
h q[11];
s q[11];
cx q[8],q[9];
rz(0.05580224637042428) q[9];
cx q[8],q[9];
cx q[10],q[11];
rz(0.06386638838991258) q[11];
cx q[10],q[11];
rx(0.0013978424816333968) q[0];
rz(-0.0009287158652437669) q[0];
rx(-6.238573148085899e-06) q[1];
rz(-0.20157910773861148) q[1];
rx(0.00012563324816834234) q[2];
rz(0.010227526603342748) q[2];
rx(-8.068046788032505e-07) q[3];
rz(-0.12900906307330068) q[3];
rx(-2.9342038410431883e-05) q[4];
rz(-0.0012649833469600118) q[4];
rx(2.5999376815998553e-06) q[5];
rz(2.9934234009477136) q[5];
rx(-8.495916832583234e-05) q[6];
rz(-0.0003280858956417715) q[6];
rx(9.893892039124777e-06) q[7];
rz(-0.20675780003187622) q[7];
rx(0.00016459318761413176) q[8];
rz(-0.00969814113893934) q[8];
rx(7.235431313329776e-06) q[9];
rz(0.2029717471052418) q[9];
rx(0.00041321418564478976) q[10];
rz(-0.010269727798428606) q[10];
rx(0.00026250245913679116) q[11];
rz(0.1812246044524109) q[11];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.6530109326932734) q[2];
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
rz(-2.3842339451372814) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
h q[0];
h q[4];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
rz(-1.3489693161195275) q[4];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[4];
sdg q[0];
h q[0];
sdg q[4];
h q[4];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
rz(-1.783505690534101) q[4];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[4];
s q[4];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.713651729009867) q[3];
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
rz(-0.6967625528466862) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
h q[1];
h q[5];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
rz(-1.3490047020064413) q[5];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[5];
sdg q[1];
h q[1];
sdg q[5];
h q[5];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
rz(-1.3444873838059843) q[5];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[5];
s q[5];
h q[2];
h q[6];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
rz(1.3589099044763082) q[6];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[6];
sdg q[2];
h q[2];
sdg q[6];
h q[6];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
rz(-1.3384379495580172) q[6];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[6];
s q[6];
h q[3];
h q[7];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
rz(-1.3417691729478873) q[7];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[7];
sdg q[3];
h q[3];
sdg q[7];
h q[7];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
rz(-1.347578374901857) q[7];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[7];
s q[7];
h q[4];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-2.431988895469446) q[6];
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
rz(0.6291076705886763) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(0.66333348093934) q[7];
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
rz(0.6747598691594225) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
cx q[0],q[1];
rz(0.07674941159604765) q[1];
cx q[0],q[1];
cx q[2],q[3];
rz(0.07674710396760345) q[3];
cx q[2],q[3];
cx q[4],q[5];
rz(0.064619179708444) q[5];
cx q[4],q[5];
cx q[6],q[7];
rz(0.05993255719019368) q[7];
cx q[6],q[7];
h q[4];
h q[8];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.2736862817903732) q[8];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[8];
sdg q[4];
h q[4];
sdg q[8];
h q[8];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.2546563662912051) q[8];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[8];
s q[8];
h q[5];
h q[9];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.25693724598324713) q[9];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[9];
sdg q[5];
h q[5];
sdg q[9];
h q[9];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
rz(0.2655617778298783) q[9];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[9];
s q[9];
h q[6];
h q[10];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.27086303465454564) q[10];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
h q[10];
sdg q[6];
h q[6];
sdg q[10];
h q[10];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.2728147819204732) q[10];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
s q[6];
h q[10];
s q[10];
h q[7];
h q[11];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[10],q[11];
rz(0.2578481427429396) q[11];
cx q[10],q[11];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
h q[11];
sdg q[7];
h q[7];
sdg q[11];
h q[11];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.25240090793596576) q[11];
cx q[10],q[11];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
s q[7];
h q[11];
s q[11];
h q[8];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(-2.662344380345832) q[10];
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
rz(-2.623748583755272) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
s q[8];
h q[10];
s q[10];
h q[9];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(0.499746337695425) q[11];
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
rz(-2.6541371007528896) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
s q[9];
h q[11];
s q[11];
cx q[8],q[9];
rz(-0.02621782068118851) q[9];
cx q[8],q[9];
cx q[10],q[11];
rz(-0.02121367011959032) q[11];
cx q[10],q[11];
rx(0.00042676016247100755) q[0];
rz(-0.024761667335011006) q[0];
rx(-1.3087062266591455e-05) q[1];
rz(-0.05085678805417167) q[1];
rx(-4.653972479141979e-05) q[2];
rz(-0.019724910417560866) q[2];
rx(2.019360854582749e-06) q[3];
rz(-0.0489438074849011) q[3];
rx(2.9749021696208817e-05) q[4];
rz(-0.0285753612863823) q[4];
rx(-2.1646477668789503e-06) q[5];
rz(-0.0693210769527573) q[5];
rx(-0.00021122375014365536) q[6];
rz(-0.011621627264742416) q[6];
rx(-1.0801139867457704e-05) q[7];
rz(-0.06340119804542783) q[7];
rx(0.0003188332186085276) q[8];
rz(-0.023978683233998356) q[8];
rx(3.329614333680826e-07) q[9];
rz(-0.07604413088147081) q[9];
rx(0.00026380787441758946) q[10];
rz(-0.014793756622173807) q[10];
rx(0.00023140809725428146) q[11];
rz(-0.050513227144216924) q[11];