OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.15626682569676975) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.34447974546458376) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.08877016646598054) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.6526300712561729) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.5023840741373572) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.21605970452463807) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.3421968889894012) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.5002105731838311) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.24051545634320332) q[3];
cx q[2],q[3];
rx(-0.31554258502956806) q[0];
rz(-0.15653785986352606) q[0];
rx(0.15686133171075858) q[1];
rz(0.02421218960886741) q[1];
rx(0.08840863120316982) q[2];
rz(0.2034103648332182) q[2];
rx(0.39674880212543484) q[3];
rz(-0.32571469226046545) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.1830712385003135) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.10605117328942597) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.18019440632618677) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.8417401260867711) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.6363399527291861) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.3992345827204006) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.11930400622424454) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.0773920320946787) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.11391983670862126) q[3];
cx q[2],q[3];
rx(-0.44293933701417215) q[0];
rz(-0.057583188265247935) q[0];
rx(-0.22388103925200065) q[1];
rz(0.11387767950233219) q[1];
rx(-0.4511874158019173) q[2];
rz(0.17742750694211235) q[2];
rx(0.519848562124343) q[3];
rz(-0.6136537755147248) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.23338103569818122) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.3710796777208854) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.36144204845763517) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.7736369027666272) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.6118542344558282) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.4609100128338729) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.586199097769017) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.5166930821240758) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.3945014009609442) q[3];
cx q[2],q[3];
rx(-0.4090118860527474) q[0];
rz(-0.09359665758465119) q[0];
rx(-0.2956603465818946) q[1];
rz(-0.16971913769490313) q[1];
rx(-0.6783110749816618) q[2];
rz(0.03552444864814621) q[2];
rx(0.5831256027379459) q[3];
rz(-0.3764060073093139) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.2903684245543319) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.5264080203694694) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.18564281747878386) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.9180761798963345) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.7235542752186999) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.21203733893007234) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.47112546569541985) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.10020931856231245) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.1417866851399689) q[3];
cx q[2],q[3];
rx(-0.3965276007574658) q[0];
rz(-0.30923975650253516) q[0];
rx(-0.1329174069645145) q[1];
rz(-0.08056111966934709) q[1];
rx(-0.41888676056566687) q[2];
rz(0.2717811278768117) q[2];
rx(0.40645928848722646) q[3];
rz(-0.310801278946104) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.18492477388874767) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.0074155600930839935) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.9862692374946059) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-1.2508632450387913) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.772627770982241) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.15548237383837754) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.19820961834961742) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.009890110105860012) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.09607091119440987) q[3];
cx q[2],q[3];
rx(-0.5372552294586798) q[0];
rz(-0.210362705906266) q[0];
rx(0.213018259151186) q[1];
rz(-0.2386920879194496) q[1];
rx(-0.34279365116637267) q[2];
rz(0.2410213685410856) q[2];
rx(0.34033214275421453) q[3];
rz(-0.3822309010968645) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.35331467455654714) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.5686810749967923) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.2782976772231773) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.9948419407581793) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.8040703713646896) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.2828020123799833) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.04450153168864825) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.2931318364824621) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.042234604365589304) q[3];
cx q[2],q[3];
rx(-0.14014803742634452) q[0];
rz(-0.15989864810097887) q[0];
rx(0.4456797119246573) q[1];
rz(-0.13467654583032915) q[1];
rx(-0.27089145306740886) q[2];
rz(0.4350707287575119) q[2];
rx(0.34767962797334945) q[3];
rz(-0.2859444496872682) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.16150832510198762) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.04147302765236349) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.6797945101532413) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.751426875411773) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.8147379839521088) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.2862707242542536) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.5279168407335141) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.16136446780897473) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.07886550178259703) q[3];
cx q[2],q[3];
rx(-0.580784053124865) q[0];
rz(-0.1468986773200348) q[0];
rx(0.13698591597709017) q[1];
rz(0.05750396877607032) q[1];
rx(-0.08237706172460615) q[2];
rz(0.1607100895386864) q[2];
rx(0.4512887268650019) q[3];
rz(0.027532278510291233) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.3795007737379633) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.24108997983687017) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.09007505427661235) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.3274883095540786) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.7456517813000979) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.10142473803361747) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.5509206805539907) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.725104461819304) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.2733364908133309) q[3];
cx q[2],q[3];
rx(-0.49208098906863834) q[0];
rz(0.07730132437462639) q[0];
rx(0.19377610944301574) q[1];
rz(-0.18970776270669742) q[1];
rx(-0.3770557429485032) q[2];
rz(0.05435536224856639) q[2];
rx(0.4952905927882184) q[3];
rz(-0.10749034221220616) q[3];