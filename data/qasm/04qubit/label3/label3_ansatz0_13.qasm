OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.1552409041421584) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.2699036718765503) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.06373902074987227) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.004607156043168262) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.031906146663103006) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.07349542279377139) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.5650250103161489) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.574041105349076) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.1321457337077672) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(0.3180637712236598) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(-0.42841145627682437) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(-0.174463499525079) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.12512048266631998) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(0.06385838492691621) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(-0.14873233982494644) q[3];
cx q[1],q[3];
rx(0.5140408378077042) q[0];
rz(-0.10130791235391619) q[0];
rx(-0.4258515824322881) q[1];
rz(-0.30796944098845785) q[1];
rx(0.03503909983461985) q[2];
rz(-0.1602564368915638) q[2];
rx(0.40432916134142766) q[3];
rz(-0.13428397512751766) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.04749443800921698) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.29316994195790275) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.3037423336053776) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.0314894745159515) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.008024987854468358) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.07450786933281935) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.34985561236935836) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.1351179919932851) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.36663093101782707) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(0.4102408909320274) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(-0.5528379009356844) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(0.019114044019646677) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.16406050307649644) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(0.04202104874897784) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(-0.09012071184099014) q[3];
cx q[1],q[3];
rx(0.42001008999128286) q[0];
rz(-0.1358753037364796) q[0];
rx(-0.4130983970107193) q[1];
rz(-0.3229459397557505) q[1];
rx(-0.026684632967126573) q[2];
rz(-0.22759704997122443) q[2];
rx(0.15335528415449404) q[3];
rz(0.01767690032112272) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.08928682676150769) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.23035278312687088) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.26460241265029893) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.07641746707259993) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.14637604896180734) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.0387443360408586) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.17345606863304677) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.1261789479213534) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.3346158771093625) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(0.36447337198046814) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(-0.4874749859168055) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(0.0543589279709745) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.11040330446093813) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(-0.02781906592757977) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(0.042167542253397236) q[3];
cx q[1],q[3];
rx(0.30027096961099997) q[0];
rz(-0.049233772975612274) q[0];
rx(-0.40900401621504184) q[1];
rz(-0.2531309631156823) q[1];
rx(0.02453107857371159) q[2];
rz(-0.1750480927313324) q[2];
rx(-0.0031680669000753513) q[3];
rz(0.1027348302879039) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.0318083343698824) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.20321238428331745) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.16093673471348036) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.07948750042302384) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.24950877627637316) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.1058792546183037) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.014986017486433658) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.4099217418787768) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.2954579375526738) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(0.3508011323010463) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(-0.3808111147159092) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(0.14986400286971877) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.0980245110919484) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(-0.13756249498439846) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(0.16102904153175016) q[3];
cx q[1],q[3];
rx(0.23827232115504676) q[0];
rz(0.026558213494623097) q[0];
rx(-0.39677126310619476) q[1];
rz(-0.19046444060133266) q[1];
rx(0.10200875408653799) q[2];
rz(-0.24377484106163327) q[2];
rx(-0.07498519336209984) q[3];
rz(0.2538200394753417) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.1344089557392868) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.08625755584099604) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.11434481165175256) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.13314499333573876) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.18306163587430022) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.005117709398408886) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.1755536274573857) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.47467938451191605) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.3186681950221319) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(0.24931763893490058) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(-0.14858075529494388) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(0.06438669949153054) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.0705857093252786) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(-0.3463901327211698) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(0.19356210314982608) q[3];
cx q[1],q[3];
rx(0.23095074850519803) q[0];
rz(-0.02024294327528695) q[0];
rx(-0.27098884694289355) q[1];
rz(-0.027392665860197597) q[1];
rx(0.29987948846340007) q[2];
rz(-0.27056032519646456) q[2];
rx(-0.1762283806913667) q[3];
rz(0.2907399019721103) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.1284717736986603) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.19567077700457167) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.12036142262065831) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.2219363618802309) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.07627383759834873) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.11291946819567825) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.41768407291416815) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.5056605185030146) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.18323626252858508) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(0.12158578907798734) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(-0.011490018030684866) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(0.07714301338972562) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.07814857548019871) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(-0.2893706918460699) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(0.24526529078619508) q[3];
cx q[1],q[3];
rx(0.09747278904884993) q[0];
rz(0.1101957384638839) q[0];
rx(-0.23880017447587754) q[1];
rz(-0.06210770037606318) q[1];
rx(0.3669171384053728) q[2];
rz(-0.27537126268986295) q[2];
rx(-0.22449610512882942) q[3];
rz(0.26060374348616694) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.23050983457090377) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.11952819256354746) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.002764533524782275) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.09980208790730533) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.0965134632145658) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.2609066443547841) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.5085897144172459) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.4171341067825305) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.027855878329321803) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(0.06656484172394592) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(0.02039893941289288) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(0.02479231010171573) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.044278955826867826) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(-0.1385694935674352) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(0.14175404633663158) q[3];
cx q[1],q[3];
rx(0.02924233786651632) q[0];
rz(0.14105732687469819) q[0];
rx(-0.07655633127880028) q[1];
rz(-0.05707925793491959) q[1];
rx(0.3114274860927265) q[2];
rz(-0.33207899501199467) q[2];
rx(-0.2772492427871814) q[3];
rz(0.16117131923153322) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.3082270729244741) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.14869913724538092) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.04338049999076842) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.06134370354204847) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.17538749404999626) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.2393095296924377) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.5548428713286849) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.3276260657324495) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.12108022872348873) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.011929905219429103) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(0.007221133095801389) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(-0.041755566738906805) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.07075929554670438) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(0.03140156058237968) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(-0.0693765950553138) q[3];
cx q[1],q[3];
rx(-0.03645033960452734) q[0];
rz(0.24621302590858507) q[0];
rx(-0.019214773892000887) q[1];
rz(-0.2101989369912549) q[1];
rx(0.3484323260688022) q[2];
rz(-0.29296401140784245) q[2];
rx(-0.25872737209192054) q[3];
rz(0.05849247325933135) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.33643768817860287) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.04159593658074976) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.0230324631284264) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.14401887649146694) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.18391378088449162) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.2655362958263915) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.6288837127658106) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.29912685793359434) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.17548861158728807) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.05638537400922988) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(-0.2230197995863264) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(0.03836224991918455) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.031490147928487686) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(0.04712233485025031) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(-0.25451154560776973) q[3];
cx q[1],q[3];
rx(-0.046631977471467204) q[0];
rz(0.2852751679138206) q[0];
rx(0.05482927059313596) q[1];
rz(-0.2598424813998336) q[1];
rx(0.3156203563372348) q[2];
rz(-0.27152132299999515) q[2];
rx(-0.26751572520683414) q[3];
rz(-0.06474276009091634) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.39579931099593146) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.040022259212422744) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.03984297538609175) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.06666903252826721) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.1881501092368237) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.4482024322486175) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.4850788184253225) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.2548803787794588) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.1614513789655455) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.2400454478504077) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(-0.4048693741980064) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(0.13314303222528368) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.01933634979439722) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(0.11471511992079123) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(-0.2506123575058101) q[3];
cx q[1],q[3];
rx(-0.03835764224378081) q[0];
rz(0.18583626725864985) q[0];
rx(0.10363381757794533) q[1];
rz(-0.3158958585723084) q[1];
rx(0.23089171715364445) q[2];
rz(0.0253985268867621) q[2];
rx(-0.24068236825678374) q[3];
rz(-0.062433658391570244) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.21069948366040678) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.027068434752457435) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.1573021030539572) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.04400414133028586) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.056308442176018034) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.36310663760409284) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.46640537106946506) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.2729887273204348) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.2450525354034607) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.25722147420166663) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(-0.3907175922074433) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(0.22202487204272972) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(0.03031382319357584) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(0.01250627949257838) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(-0.3487100809929218) q[3];
cx q[1],q[3];
rx(-0.021550606063827797) q[0];
rz(0.15133828778532277) q[0];
rx(0.15805403223750344) q[1];
rz(-0.21320284958302524) q[1];
rx(0.08178582919476217) q[2];
rz(0.13340712240906435) q[2];
rx(-0.2510413938243354) q[3];
rz(-0.1869800197433766) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.16710969299429956) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.11794860317256353) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.16317941531733288) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.05104164312332615) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.22793626348491228) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.2884339556430075) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.378805038048346) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.15084126654832372) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.31458513214551814) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.259436930277776) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(-0.4197570966913386) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(0.19759273284226062) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(0.15380358002774994) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(-0.04043035673518595) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(-0.3243162750622824) q[3];
cx q[1],q[3];
rx(0.00416452130213277) q[0];
rz(0.07129097091030809) q[0];
rx(0.15321043580964636) q[1];
rz(-0.22820534844786736) q[1];
rx(-0.07733261760410073) q[2];
rz(0.188261869380104) q[2];
rx(-0.1640294271952979) q[3];
rz(-0.17554738197304282) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.09211960973760469) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.1396533617908954) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.1018926004986615) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.009429310245472354) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.4310281952920074) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.20739572370712095) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.30621608855378957) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.1160534157050945) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.3810628670841218) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.15368836860151017) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(-0.3407147813125993) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(0.042471010092607744) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(0.1845245120314475) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(-0.10506143961188795) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(-0.3316876327895581) q[3];
cx q[1],q[3];
rx(0.04450339027223863) q[0];
rz(0.018786126289626927) q[0];
rx(0.12433661843568715) q[1];
rz(-0.15779857909096173) q[1];
rx(-0.22284287037399655) q[2];
rz(0.1439002070498123) q[2];
rx(-0.08881485885120648) q[3];
rz(-0.2692182992365252) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.07212758460005202) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.04793255910997013) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.1874234652017116) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.023175182889547275) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.3505959701149092) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.210614906840185) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.18958651391211062) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.14083713897481262) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.32120132140129504) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.14742096992929427) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(-0.2082446882695042) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(-0.05017483581301006) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(0.01843030657329922) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(-0.0914557960532463) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(-0.3206955969828583) q[3];
cx q[1],q[3];
rx(0.019411951253162318) q[0];
rz(0.02820502445346936) q[0];
rx(0.10977529122813169) q[1];
rz(-0.059131518431466905) q[1];
rx(-0.2988353696608924) q[2];
rz(0.13879465325180354) q[2];
rx(-0.08192435072456973) q[3];
rz(-0.3519461738536806) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.06546429071608931) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.02497559332523669) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.22905081371182884) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.07347630912763307) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.2906175522032373) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.09335701874834443) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.048533792152154184) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.231440716047339) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.07474103024760696) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.059782572677590975) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(-0.15227424550670512) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(-0.061550023802441525) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.03877135653200296) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(0.0582136140337808) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(-0.27945197926951554) q[3];
cx q[1],q[3];
rx(0.046974724969144545) q[0];
rz(-0.01910009467779772) q[0];
rx(0.14376971128261656) q[1];
rz(0.028140132702578223) q[1];
rx(-0.4089528625049614) q[2];
rz(0.13625821076738065) q[2];
rx(-0.02430897615344702) q[3];
rz(-0.39696012529346314) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.05646802669645149) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.029173938760386396) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.2950257631599103) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.07873617666992878) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.24497838636819777) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.034225161969717546) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.365376066119092) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.40214120489313243) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.15469129061507805) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.012738068622237446) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(-0.021810380089893527) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(-0.06392719995172962) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.2728890825048822) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(0.1023475878886673) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(-0.11646248275401326) q[3];
cx q[1],q[3];
rx(0.03739050115323576) q[0];
rz(-0.06938142429619289) q[0];
rx(0.24396891681975105) q[1];
rz(0.07014433325300555) q[1];
rx(-0.48469411173568056) q[2];
rz(0.18835560493444917) q[2];
rx(0.018284572270607393) q[3];
rz(-0.37286017944247524) q[3];