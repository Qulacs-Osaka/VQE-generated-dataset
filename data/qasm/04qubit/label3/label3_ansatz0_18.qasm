OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.014912947832122728) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.1686849024777988) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.01149496067473733) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.26986523733888235) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.08550146109354623) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.01041017504239411) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.1336048109981783) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.014170467364656356) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.06317980831332824) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.41241354183189466) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(0.2439855644072268) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(-0.044895129245773206) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.012879818830810307) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(-0.053389600453216925) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(-0.06102272689404419) q[3];
cx q[1],q[3];
rx(-0.2316709721859828) q[0];
rz(-0.10084177513667754) q[0];
rx(0.1790005476945053) q[1];
rz(-0.029690855096599772) q[1];
rx(-0.010868472961326857) q[2];
rz(-0.16919998077497198) q[2];
rx(0.018545801882951733) q[3];
rz(-0.02375967744668343) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.012973675167623534) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.09433306218653778) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.14778041442965859) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.1729829631943845) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.10495906466744036) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.05267907716243048) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.12586037621992396) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.03396100281666954) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.18384443140645185) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.37988786618832765) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(0.1420837935510809) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(-0.0991382861060691) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.0051276822042546156) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(-0.004098761918116733) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(-0.044059011730168164) q[3];
cx q[1],q[3];
rx(-0.22263210781815976) q[0];
rz(-0.0856698082107422) q[0];
rx(0.10335837475549398) q[1];
rz(-0.11236730628419711) q[1];
rx(0.09475426688982982) q[2];
rz(-0.20287560721223638) q[2];
rx(-0.08026447329127555) q[3];
rz(-0.004952358484708382) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.06114793782644299) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.11016116881019698) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.0877676760570641) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.03430618043500836) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.07357064975613628) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.15400284826475363) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.25959456582577983) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.07835430590919969) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.13518982576007707) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.312520869636174) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(-0.05270669230216652) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(-0.1671672356131372) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.03281833202714866) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(-0.08846522497269742) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(0.04878972778467697) q[3];
cx q[1],q[3];
rx(-0.21852527872530916) q[0];
rz(-0.05010229788648901) q[0];
rx(0.04453782219819259) q[1];
rz(-0.06824443292667287) q[1];
rx(0.17018335419306896) q[2];
rz(-0.142698754662364) q[2];
rx(-0.10862482178828223) q[3];
rz(0.016117926652265913) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.026031852903891792) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.13339441911989877) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.035250578009431405) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.004187328459899307) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.007479476732183233) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.11711221432936597) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.18970612689504465) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.14728411681185036) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.16677209242742014) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.19328489537302612) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(-0.06302855850822764) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(-0.3000823329294036) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.08056788969319424) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(-0.11412840073005963) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(0.06204259192028034) q[3];
cx q[1],q[3];
rx(-0.24404832405656302) q[0];
rz(-0.11936795148953594) q[0];
rx(-0.0006107940196909391) q[1];
rz(0.026582419736418537) q[1];
rx(0.2600479666114497) q[2];
rz(-0.155763065031151) q[2];
rx(-0.19939059858270491) q[3];
rz(0.10639304552891644) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.05896063282700731) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.10033893809624288) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.1848166590131552) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.0036504164455395784) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.07779016819719922) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.14855599851356063) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.25640840805335274) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.12503505419169741) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.08334712764681292) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.19670371745301793) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(-0.11473216085242166) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(-0.28406523600236067) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.05489828124594108) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(-0.26477160468894484) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(0.1595122235806705) q[3];
cx q[1],q[3];
rx(-0.23435230356890124) q[0];
rz(-0.21023509258464337) q[0];
rx(-0.057284816675529975) q[1];
rz(0.009130080841640463) q[1];
rx(0.27093250950401854) q[2];
rz(-0.1622852392866005) q[2];
rx(-0.14300662941305847) q[3];
rz(0.09393308146611445) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.02184046458702279) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.09706288524727974) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.2898845341242225) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.023213655906622447) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.17027168681325386) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.20393431029358358) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.26215170692672624) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.04000362596396456) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.11504030285598366) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.0500018133805221) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(-0.14263636692608012) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(-0.36553126897932037) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.06495113043846504) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(-0.2879265887071275) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(0.18157150150461698) q[3];
cx q[1],q[3];
rx(-0.31779253618146297) q[0];
rz(-0.24139586138427355) q[0];
rx(-0.12389025268598895) q[1];
rz(0.05842591239872613) q[1];
rx(0.4507597250535957) q[2];
rz(-0.129841226439399) q[2];
rx(-0.17046539029155916) q[3];
rz(0.03899409855718943) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.0030393091573331353) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.13912600626041877) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.2845713376513939) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.08131157319805894) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.24753580784868132) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.16104954505314853) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.2504511112984675) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.05755302133148568) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.061233691392119184) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.0877831028370069) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(-0.11429106920175695) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(-0.2965601235917885) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.09615677339787317) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(-0.31350962194487214) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(0.17175825078670157) q[3];
cx q[1],q[3];
rx(-0.23218324316638406) q[0];
rz(-0.3202865585296456) q[0];
rx(-0.1598251434070021) q[1];
rz(0.07785984861642822) q[1];
rx(0.4478817531078092) q[2];
rz(-0.04873908326573596) q[2];
rx(-0.1778363040025883) q[3];
rz(0.0620201056686098) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.05005992749786085) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.11660035062199155) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.2971139647330033) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.14362152306950282) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.2715824786757948) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.10317689166074495) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.18735191489236522) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.009215256995767475) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.023396567094967076) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.1484503982627372) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(0.0175881777932139) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(-0.2916787809757843) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.12503450889021492) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(-0.25940815855375765) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(0.1495005719118459) q[3];
cx q[1],q[3];
rx(-0.20002577057954354) q[0];
rz(-0.3472472018354021) q[0];
rx(-0.27546664697323314) q[1];
rz(0.09025386792754667) q[1];
rx(0.4054859091634252) q[2];
rz(0.05857903194526723) q[2];
rx(-0.10617167139252837) q[3];
rz(0.04939410107445848) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.03453576893447425) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.02280037229785306) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.23578991437572128) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.18056432067100653) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.3335019398013917) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.0007819133277629869) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.10556855222505669) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.006626935493120667) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.024584580825718096) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.1414072435247121) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(-0.029236066534260502) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(-0.2119218809594557) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.12634698753140564) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(-0.15865174512556762) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(0.09997973843710102) q[3];
cx q[1],q[3];
rx(-0.10444363181909203) q[0];
rz(-0.2454881364806846) q[0];
rx(-0.28752043197234073) q[1];
rz(0.08606771886535325) q[1];
rx(0.3837928895870417) q[2];
rz(0.06326667018609491) q[2];
rx(-0.09743655189593468) q[3];
rz(-0.027928629266011622) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.0012468768108948103) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.07751940989187449) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.18883827057502964) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.2207920923967246) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.32532059258572754) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.16344012804575045) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.11945023444694645) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.12751269675087865) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.03488644529807906) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.19769656348305856) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(-0.025972740332410406) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(-0.11309090466730673) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.1899809905084455) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(-0.08221698889963441) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(0.053414501637444156) q[3];
cx q[1],q[3];
rx(-0.08864879020087545) q[0];
rz(-0.21566119923541482) q[0];
rx(-0.2794521600664108) q[1];
rz(0.09350085727414771) q[1];
rx(0.24410526593751994) q[2];
rz(0.12676825460443047) q[2];
rx(0.018649586266229794) q[3];
rz(-0.11816470119851062) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.05764506399790418) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.06664454402689636) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.01287056480697225) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.26415786038189765) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.31480746782232444) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.20310884452225697) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.09497477163086468) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.17569661407107698) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.06079312031390293) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.1766413567902077) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(0.0037353317595605804) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(-0.10369476035901837) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.16717194335490557) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(0.10010195929453372) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(-0.04177096048272972) q[3];
cx q[1],q[3];
rx(-0.060076616961063) q[0];
rz(-0.14716642561055027) q[0];
rx(-0.32505788352013787) q[1];
rz(0.10803154196232037) q[1];
rx(0.1397694458849462) q[2];
rz(0.07172995261566875) q[2];
rx(0.01290333129773711) q[3];
rz(-0.064975024912596) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.11713332535584982) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.012373830307925961) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.07747464788419443) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.11536538704949509) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.35683404855071893) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.2353735513240616) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.10734801128328277) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.36842896286971494) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.0663409360632759) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.11383322343960464) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(-0.04998240705450343) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(-0.015644491912384796) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.10286765158775542) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(0.24681710929864234) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(-0.15347635525924738) q[3];
cx q[1],q[3];
rx(-0.11952566486805662) q[0];
rz(-0.1947187003790014) q[0];
rx(-0.2873838529529926) q[1];
rz(0.031851906331736325) q[1];
rx(-0.023335971424400085) q[2];
rz(0.15903030006872332) q[2];
rx(0.0646803264551994) q[3];
rz(-0.20542785140323455) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.07793100798630408) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.0610556483042043) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.15951670934330342) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.08187251330533617) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.2415715669182917) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.27874513611442864) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.022047991164215415) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.4322572721711805) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.15415606685021788) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.10502065775392003) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(-0.046877413825526924) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(0.09456816622150108) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.08755252741917902) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(0.22228966056535138) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(-0.3174198845365611) q[3];
cx q[1],q[3];
rx(-0.1894433163403673) q[0];
rz(-0.18572389530949637) q[0];
rx(-0.23156329099155654) q[1];
rz(-0.006945866619101385) q[1];
rx(0.01713038732077451) q[2];
rz(0.07983354156615018) q[2];
rx(0.09595900465036192) q[3];
rz(-0.20736378544361808) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.09747679358385533) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.057556162534457944) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.19625324756619195) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.02429386630880354) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.042635223184146434) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.2183382287406802) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.017370944831413944) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.46175275269290966) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.11361526998752894) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(0.025827908684623493) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(-0.20251383361164974) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(0.11359586059113762) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.08656828334454866) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(0.05783564583963953) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(-0.32514558374283126) q[3];
cx q[1],q[3];
rx(-0.2520061531326575) q[0];
rz(-0.1793700869221754) q[0];
rx(-0.2911183467082737) q[1];
rz(-0.11708105697719541) q[1];
rx(-0.13356694886535925) q[2];
rz(0.19329217661991746) q[2];
rx(0.15206518113095013) q[3];
rz(-0.20562504867331) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.1643033499983219) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.06867929848593902) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.04141711618112416) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.0004780813781390835) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.11921606600843122) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.1648501453931703) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.06804581978124466) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.4290498878011431) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.19313586604567473) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.034262413072794776) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(-0.2700906935699504) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(0.07543064632848455) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.12120583397357086) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(0.0035291028738130502) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(-0.4048348823122567) q[3];
cx q[1],q[3];
rx(-0.2612538042067036) q[0];
rz(-0.1804881055812887) q[0];
rx(-0.24145922587195556) q[1];
rz(-0.0812593067598079) q[1];
rx(-0.1333769498073444) q[2];
rz(0.17195122235156987) q[2];
rx(0.18100275034782995) q[3];
rz(-0.1517959367377168) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.24278774554804622) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.13523564984855188) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.04642246940280598) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.10826440950830717) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.11764026164339451) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.03259149300598458) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.04105266031124217) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.22741891565371675) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.1777914927196209) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.010133394860868943) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(-0.3682746998482151) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(0.05649132217381135) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.08028888785037723) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(-0.07652398454714217) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(-0.28565384455670423) q[3];
cx q[1],q[3];
rx(-0.24137845356676482) q[0];
rz(-0.03409343039569316) q[0];
rx(-0.2240955324271045) q[1];
rz(-0.10175926332787774) q[1];
rx(-0.30381883360037126) q[2];
rz(0.21228563413660997) q[2];
rx(0.1652122621212634) q[3];
rz(-0.11112930017146708) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.2889065768641802) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.11456103558164416) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.045151205159210304) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.14478794863597144) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.0300774846905284) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.08267519356848711) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.013044616208908604) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.07158475620765523) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.10698858612347988) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.036570994436946884) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(-0.37225681495702156) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(-0.008639788891530454) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.05934395561240012) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(-0.09176164850781314) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(-0.17091632848720292) q[3];
cx q[1],q[3];
rx(-0.2571342449399271) q[0];
rz(-0.05567880129263516) q[0];
rx(-0.21094473465475364) q[1];
rz(-0.17235664807533485) q[1];
rx(-0.4487834670947664) q[2];
rz(0.15246224080122617) q[2];
rx(0.17448974083464902) q[3];
rz(-0.10031337869926492) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.23026098636660242) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.12450724545016006) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.023403151164802915) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.22567347488178255) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.023954433666321155) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.005375295911542869) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.017811902930732537) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.1355959217651197) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.05709233778697417) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.09921304321546208) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(-0.3493127412015008) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(-0.033630942789457836) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(0.005529130908209974) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(-0.17023844133692542) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(-0.12221230988494475) q[3];
cx q[1],q[3];
rx(-0.16398378091005386) q[0];
rz(-0.02595801554443582) q[0];
rx(-0.11308611710612916) q[1];
rz(-0.23218684274417878) q[1];
rx(-0.47950462842585384) q[2];
rz(0.1342480774703907) q[2];
rx(0.1551248198448471) q[3];
rz(-0.10238205821039432) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.21626670079070887) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.03932273544653979) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.022291180032457763) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.2585044403932756) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.04159981412086218) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.05145495360915541) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.11305240683466673) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.2639459633318408) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.08957534266298457) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.12867838737940654) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(-0.303125084441386) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(-0.08670701154296973) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(0.17467078457450433) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(-0.34764317000047806) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(-0.009063604997822789) q[3];
cx q[1],q[3];
rx(-0.10889681053894211) q[0];
rz(0.023365778413408794) q[0];
rx(0.0947160494058149) q[1];
rz(-0.22082646872249723) q[1];
rx(-0.40232774827616413) q[2];
rz(0.0389905700515563) q[2];
rx(0.19902930090058255) q[3];
rz(-0.012307069161091977) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.2248715595709205) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.004774388301269941) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.027955093393719623) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.1647556134109612) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.03814200950329541) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.1448650170957018) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.15548578567357432) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.294408521921538) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.17141891065818693) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.16911200416176525) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(-0.24119281615877808) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(-0.15304287268351752) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(0.20786162013975545) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(-0.3637421803835598) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(-0.1311996016741214) q[3];
cx q[1],q[3];
rx(-0.11798452627944177) q[0];
rz(0.09543321738778786) q[0];
rx(0.24939129597785) q[1];
rz(-0.23861953587554519) q[1];
rx(-0.4963786044757728) q[2];
rz(-0.17269621797164966) q[2];
rx(0.06980192153072544) q[3];
rz(-0.0007067368505164693) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.2815589828974854) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.18024228600132347) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.11919378890696906) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.11026992643552168) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.21962982108351095) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.15286272521516256) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.34817458893015235) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.3946249824621745) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.08907695195263023) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.1122266779143805) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(0.02810443074313241) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(-0.15142838441203096) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(0.3584552821148199) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(-0.23772374144566225) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(-0.07295797344279806) q[3];
cx q[1],q[3];
rx(-0.07226764586322963) q[0];
rz(0.20164265559192307) q[0];
rx(0.36703906915415724) q[1];
rz(-0.19637008272996725) q[1];
rx(-0.5049552907042512) q[2];
rz(-0.11850800759504607) q[2];
rx(0.03181709556629283) q[3];
rz(-0.01791186556453005) q[3];