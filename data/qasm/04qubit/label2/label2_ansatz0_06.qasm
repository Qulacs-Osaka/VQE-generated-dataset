OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.1354518818177289) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.23608615445969108) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.028809113541546503) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.7880094760417107) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-1.091596040216789) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.12266986640849208) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.23580003793839535) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.07930445556143698) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.012725115871685101) q[3];
cx q[2],q[3];
rx(0.9408510801578192) q[0];
rz(-0.00023191562652907977) q[0];
rx(-0.305330257328368) q[1];
rz(-0.15723720431434998) q[1];
rx(0.36167154337665935) q[2];
rz(-0.12901922777600167) q[2];
rx(-0.06551835175139145) q[3];
rz(-0.1862659919922851) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.08021772235413682) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.49107753752953) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.03634706621419443) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.20039170459750696) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.6452707906405879) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.6341092686439598) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.13273354802502332) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.08389413570858124) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.2224041406520328) q[3];
cx q[2],q[3];
rx(0.7309202704262433) q[0];
rz(-0.23873365689963558) q[0];
rx(-0.2850507163189782) q[1];
rz(0.04404632927596176) q[1];
rx(0.22813876178321296) q[2];
rz(-0.08608210915159353) q[2];
rx(-0.016900530531094336) q[3];
rz(-0.3444096760716711) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.0723809330432973) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.4894172139893746) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.26808421253335857) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.15313751157093022) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.19082497761857847) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.7008808521688038) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.18840293330177132) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.05401078087990299) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.265499710439322) q[3];
cx q[2],q[3];
rx(0.20393913726270407) q[0];
rz(-0.2502434748642862) q[0];
rx(-0.5725466110609468) q[1];
rz(0.29490926287623537) q[1];
rx(0.15688219217286542) q[2];
rz(-0.40944411599436387) q[2];
rx(0.047874882572598226) q[3];
rz(-0.36363766187524493) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.3369234000585343) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.14103167915258152) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.3505820148990667) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.208806831952004) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.2371195577156777) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.7425589089540745) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.1871317488163046) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.11264575557534834) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.3841072098539621) q[3];
cx q[2],q[3];
rx(0.021308259567071684) q[0];
rz(0.3046146424192669) q[0];
rx(-0.5328450997523823) q[1];
rz(0.6393912865333521) q[1];
rx(-0.07690382142728232) q[2];
rz(-0.4893988139926559) q[2];
rx(0.26635006572946857) q[3];
rz(-0.47775287687134566) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.22949587785111517) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.056633779610760306) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.2727832707755856) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.31388504927080807) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.06516001602335565) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.6750716800238448) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.16801768425267435) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.1710012679565952) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.4146607245859085) q[3];
cx q[2],q[3];
rx(-0.014628118867842663) q[0];
rz(0.48513182396650767) q[0];
rx(-0.3353968628976026) q[1];
rz(0.3862151621269632) q[1];
rx(-0.37194461736464235) q[2];
rz(-0.2139402821402886) q[2];
rx(0.49591144591811526) q[3];
rz(-0.05837646208327841) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.235657557795486) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.4997473035062332) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.2646365917984497) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.4799505900066492) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.11929693721640004) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.2643844066933582) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.09079773094498365) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.022326499788497104) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.024746704852038365) q[3];
cx q[2],q[3];
rx(-0.10118976020668777) q[0];
rz(0.6565720152215194) q[0];
rx(-0.29167950886006744) q[1];
rz(-0.09052516853549686) q[1];
rx(-0.5215079433881769) q[2];
rz(0.2286961676581863) q[2];
rx(0.545638663679811) q[3];
rz(0.18390518948633744) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.4946311212489435) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.429802306005263) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.10812884575194279) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.20770913878463013) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.3572007003607398) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.09945279215212825) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.38180358392257113) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.008823129146270807) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.04273052116192357) q[3];
cx q[2],q[3];
rx(-0.27382050485399295) q[0];
rz(0.67282049278372) q[0];
rx(0.06457798798145953) q[1];
rz(-0.4882436269610892) q[1];
rx(-0.5624937927358474) q[2];
rz(0.0623348738163184) q[2];
rx(0.5797191400780272) q[3];
rz(0.31441078741102796) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.08735243620058977) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.17764081543724378) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.07259133626570788) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.039078355984043106) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.24540245484892476) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.15683468010725246) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.6290174778352462) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.22524736842297308) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.3235466149001217) q[3];
cx q[2],q[3];
rx(-0.2730327589980528) q[0];
rz(0.9287353883086803) q[0];
rx(-0.067383582195863) q[1];
rz(-0.9416419805062175) q[1];
rx(-0.6282713081653465) q[2];
rz(-0.27684188234584656) q[2];
rx(0.5727674871333581) q[3];
rz(0.2593685795230123) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.10032896838472498) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.009669251096368422) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.014390407385890962) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.01530958289842753) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.15711364001912098) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.025882903029490683) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.38172686011265033) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.12267663298922803) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.4773646801901761) q[3];
cx q[2],q[3];
rx(0.03133111240579004) q[0];
rz(1.1025915918356781) q[0];
rx(-0.3525033518212298) q[1];
rz(-1.2507802551188543) q[1];
rx(-0.5876573766115747) q[2];
rz(-0.5088163497502293) q[2];
rx(0.6738346727713792) q[3];
rz(0.5054801804902427) q[3];