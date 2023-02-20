OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.1731684356531413) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.05657924044411307) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.084162084809572) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.1599262429901374) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.002762443809878746) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.07139235807767655) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.08686623446355006) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.0190581664527281) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.04958074857087838) q[3];
cx q[2],q[3];
rx(0.44236458416444313) q[0];
rz(-0.00913215155539711) q[0];
rx(-0.5193304711326945) q[1];
rz(-0.039037933095812015) q[1];
rx(-0.02121003804364577) q[2];
rz(-0.07694078636238935) q[2];
rx(0.283931301204001) q[3];
rz(-0.043509014745034644) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.05071173200396245) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.11339256865344276) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.0040678388460181365) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.140288780986996) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.059084794232187106) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.12000561423022947) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.1412637693552474) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.1047547728424312) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.11046574227828732) q[3];
cx q[2],q[3];
rx(0.3111449846236263) q[0];
rz(0.021238737581511367) q[0];
rx(-0.4265951592709985) q[1];
rz(0.015404696031030975) q[1];
rx(0.0982940575971436) q[2];
rz(-0.22105605058236805) q[2];
rx(0.20605108185519824) q[3];
rz(-0.07095956921761197) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.03528435023386377) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.1773226614365916) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.004206612028479612) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.2269998359978585) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.03600182596877936) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.2997488889025923) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.1477965995465158) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.015080138988735883) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.11904956235296181) q[3];
cx q[2],q[3];
rx(0.12813732015841908) q[0];
rz(0.09916485743717232) q[0];
rx(-0.3476208227885178) q[1];
rz(-0.1259768731998035) q[1];
rx(0.22834869287455972) q[2];
rz(-0.3241471165235349) q[2];
rx(0.028064133234790612) q[3];
rz(0.004862978273512799) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.21729947600912095) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.23053969227858698) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.06612586804634947) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.29225486806719525) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.08283834599137739) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.34422470134507965) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.17401449132275448) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.05200996448628402) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.17869418045145916) q[3];
cx q[2],q[3];
rx(0.03872579927322405) q[0];
rz(0.14933533970512486) q[0];
rx(-0.21236111629824572) q[1];
rz(-0.07514681050969088) q[1];
rx(0.3235187366461274) q[2];
rz(-0.35390903786363204) q[2];
rx(-0.021768412694503716) q[3];
rz(0.014803982577746054) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.3977868825626456) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.31556508804316397) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.2527137187212252) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.299399746785973) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.3684634156060956) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.11330468226153657) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.21179748074139634) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.05999968277826079) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.2780214367336268) q[3];
cx q[2],q[3];
rx(-0.09337928410075028) q[0];
rz(0.36933293388905253) q[0];
rx(-0.05356235864946554) q[1];
rz(0.057314152819700113) q[1];
rx(0.48383075480253696) q[2];
rz(-0.40306655034292105) q[2];
rx(-0.015681705221782236) q[3];
rz(-0.007776252587973849) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.3588074396607379) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.29012582545709104) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.3233462440887539) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.23595569358443128) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.40781642982898125) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.007423823714094752) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.20727261740317213) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.07314933561977133) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.30749329107982687) q[3];
cx q[2],q[3];
rx(-0.10323567093838082) q[0];
rz(0.35689710889162046) q[0];
rx(-0.019535667033301132) q[1];
rz(-0.045784947412488175) q[1];
rx(0.5345936778288891) q[2];
rz(-0.38488323875052605) q[2];
rx(-0.1845400510283477) q[3];
rz(0.07146447540471566) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.43690566988224155) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.4272179839393653) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.2764652126998863) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.11838338897948968) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.37170832159909345) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.08478078920192537) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.1459495553120376) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.08825413715094106) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.3633768523270041) q[3];
cx q[2],q[3];
rx(-0.22667001426309763) q[0];
rz(0.34241102237695353) q[0];
rx(0.06566425893248339) q[1];
rz(-0.22295767162333344) q[1];
rx(0.5118152407178055) q[2];
rz(-0.33614428345553016) q[2];
rx(-0.2604633466117094) q[3];
rz(0.08896112641415059) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.4272595322566201) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.46490729578174256) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.2911632621098855) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.018025557940304734) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.31367303862702683) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.2744530329823162) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.16997481753612398) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.018351012536104966) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.2829283044260635) q[3];
cx q[2],q[3];
rx(-0.4153093591875273) q[0];
rz(0.37842177346345457) q[0];
rx(0.24090697845273187) q[1];
rz(-0.34049935988540686) q[1];
rx(0.5226646260075851) q[2];
rz(-0.3754164835812217) q[2];
rx(-0.3603975376907628) q[3];
rz(0.21933754073918926) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.3232038411990353) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.35258491350289245) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.3660632115150213) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.03855056964774064) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.31517667397836563) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.3976054109547779) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.22196289088424975) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.1455667712253963) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.07653730290533921) q[3];
cx q[2],q[3];
rx(-0.4571026430656509) q[0];
rz(0.5374635336843054) q[0];
rx(0.33118877856482126) q[1];
rz(-0.33406357092821176) q[1];
rx(0.5381550317189512) q[2];
rz(-0.24284333748394574) q[2];
rx(-0.4119413333284298) q[3];
rz(0.2285684385352385) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.1253120790364252) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.3196848283178509) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.3336061901029807) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.009452555752398699) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.32476781902232194) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.5202381873846423) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.19606725619796547) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.11609549643356684) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.21725011244787037) q[3];
cx q[2],q[3];
rx(-0.5331640225090086) q[0];
rz(0.5474871370816877) q[0];
rx(0.3985250396114291) q[1];
rz(-0.21111804395771097) q[1];
rx(0.37252897507311294) q[2];
rz(-0.13794373386684372) q[2];
rx(-0.4074962274094377) q[3];
rz(0.2752312262953669) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.17851638661551558) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.10328918871222312) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.17012057816881715) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.0017449259605185506) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.3565899795286663) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.49575480881204764) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.19071083502571673) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.0503932913478714) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.21145643423175278) q[3];
cx q[2],q[3];
rx(-0.7491419262390984) q[0];
rz(0.46568334889294893) q[0];
rx(0.5090203805808781) q[1];
rz(-0.2480338995777671) q[1];
rx(0.3449890592166726) q[2];
rz(-0.05094274397009167) q[2];
rx(-0.4879420166198723) q[3];
rz(0.19212170158661282) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.36239647057962854) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.06597292084187853) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.14367227264413288) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.05583844811748054) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.22064265249434348) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.4029638875792679) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.1562870983486705) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.106283300930355) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.12994495724769664) q[3];
cx q[2],q[3];
rx(-0.9121856340172034) q[0];
rz(0.3509633021553528) q[0];
rx(0.6328447061512764) q[1];
rz(-0.23765443445492881) q[1];
rx(0.2270403190615442) q[2];
rz(-0.03248074176635377) q[2];
rx(-0.4460254514831437) q[3];
rz(0.11867159866514523) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.38332634009867017) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.11843492497522706) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.09523185419335975) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.06947915257673644) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.0458101205435834) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.2455329066889565) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.0781009101797005) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.2385512993249349) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.15560279531848964) q[3];
cx q[2],q[3];
rx(-1.0193065144909008) q[0];
rz(0.2548253168242378) q[0];
rx(0.6934333343384259) q[1];
rz(-0.15193075521691016) q[1];
rx(0.22917401086204017) q[2];
rz(-0.07453662088937116) q[2];
rx(-0.4711199954958174) q[3];
rz(0.011982766555863581) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.262483688988069) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.27248971717898635) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.1587916221396073) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.060739742229213196) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.35089199259779436) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.019150388927951022) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.048785242409471734) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.4557967578457773) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.027188051942098314) q[3];
cx q[2],q[3];
rx(-0.9707235897632115) q[0];
rz(0.34256726710667423) q[0];
rx(0.7373870250582449) q[1];
rz(-0.10058839474648545) q[1];
rx(0.11514255467192323) q[2];
rz(-0.10821914979622935) q[2];
rx(-0.3364454246157857) q[3];
rz(-0.05990462651298669) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.10144705698313398) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.20843648376481327) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.2527160860620539) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.19704878074886895) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.5720855775570903) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.07475218631880269) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.025404172817372523) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.35565733872617594) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.11249530680912276) q[3];
cx q[2],q[3];
rx(-0.8817968054082622) q[0];
rz(0.29409161323253497) q[0];
rx(0.755327387157885) q[1];
rz(0.06541373678586102) q[1];
rx(0.1411973646114751) q[2];
rz(-0.026651156621260035) q[2];
rx(-0.26850398465153574) q[3];
rz(-0.08749476567416584) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.011256091184718993) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.17459010507197073) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.21649473608768163) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.31672434183696674) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.5351070887726602) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.02333203203909534) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.0232358572536433) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.20246133230303823) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.0705752636628246) q[3];
cx q[2],q[3];
rx(-0.7130034207064654) q[0];
rz(0.1946006854801086) q[0];
rx(0.7219632957345249) q[1];
rz(0.1380393250926303) q[1];
rx(0.2307469438445875) q[2];
rz(0.1596485373274884) q[2];
rx(-0.2546293006276183) q[3];
rz(-0.2109275510414457) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.11849767658118236) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.3628601547917231) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.0749470863320453) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.4492341039069849) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.5095148331759819) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.10475353271612992) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.011530475296969883) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.06045277274233432) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.03717230988275367) q[3];
cx q[2],q[3];
rx(-0.61653024261127) q[0];
rz(0.04587636851275695) q[0];
rx(0.6486350352450073) q[1];
rz(-0.016023730150850415) q[1];
rx(0.2543453434202295) q[2];
rz(0.24559366621431222) q[2];
rx(-0.15095216367404132) q[3];
rz(-0.2664755092858382) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.22484692291620753) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.3843996736274068) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.20064762923030327) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.4809345116625099) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.6592878112848807) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.0027732361403050325) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.22322807164553837) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.0843396873487832) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.19538688784316088) q[3];
cx q[2],q[3];
rx(-0.5606792324798199) q[0];
rz(-0.10237410493255276) q[0];
rx(0.6423931827049569) q[1];
rz(-0.14064229949437201) q[1];
rx(0.21210546771111372) q[2];
rz(0.07202252223890386) q[2];
rx(-0.08159890477593255) q[3];
rz(-0.4491897487919202) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.16133593785517988) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.0585222134582758) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.1658433766978382) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.5378534330132723) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.7467046082047899) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.22011244074453834) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.26451774860549976) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.20542900557407376) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.3668045952929686) q[3];
cx q[2],q[3];
rx(-0.4935288082556987) q[0];
rz(-0.315238514570127) q[0];
rx(0.5373182079851683) q[1];
rz(-0.02351313193840215) q[1];
rx(0.12386468494824254) q[2];
rz(0.15881946629008636) q[2];
rx(0.12710572238473267) q[3];
rz(-0.45781669203184544) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.1326516587089162) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.1343470161165855) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.26830661978660214) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.34783005877191797) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.7808467038092275) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.2875716598678225) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.07347455910855545) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.20797431389275614) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.18553665934756242) q[3];
cx q[2],q[3];
rx(-0.26421390874954254) q[0];
rz(-0.4142217477385186) q[0];
rx(0.2979727048231977) q[1];
rz(0.3924858119345803) q[1];
rx(0.07193765406614915) q[2];
rz(0.3620965276455026) q[2];
rx(0.07412861054391502) q[3];
rz(-0.5500463109265724) q[3];