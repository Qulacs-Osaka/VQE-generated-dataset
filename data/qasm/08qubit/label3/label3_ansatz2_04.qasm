OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-3.1415446188277145) q[0];
rz(-0.16166681015874573) q[0];
ry(-0.025496330827319724) q[1];
rz(-2.258256611303268) q[1];
ry(-2.7295299790019967) q[2];
rz(-0.861605735451799) q[2];
ry(-3.1122782406758907) q[3];
rz(0.6440509474288333) q[3];
ry(-1.394275516123174) q[4];
rz(-3.0474251554613203) q[4];
ry(-1.991679983580358e-05) q[5];
rz(-0.09634816663744061) q[5];
ry(-1.6927984723724991) q[6];
rz(-0.06672548028602152) q[6];
ry(-0.12504886251285363) q[7];
rz(-2.2006950665105274) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(6.916826806779852e-05) q[0];
rz(0.15864212958989385) q[0];
ry(-1.8318257190589444) q[1];
rz(-2.9900893265530257) q[1];
ry(-3.0529064510271064) q[2];
rz(2.023493860762794) q[2];
ry(-1.569042447544752) q[3];
rz(-1.6475392679191323) q[3];
ry(2.4625108021034774) q[4];
rz(-2.0961739668404613) q[4];
ry(3.1415924998444567) q[5];
rz(2.4988804701274363) q[5];
ry(0.8638223982118349) q[6];
rz(-2.26787462276138) q[6];
ry(1.5748811824124012) q[7];
rz(-3.12829160193236) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-0.0006534957112595381) q[0];
rz(-2.5357867103215237) q[0];
ry(-0.08501520297348088) q[1];
rz(1.4797794944367597) q[1];
ry(3.0044797788961755) q[2];
rz(-0.08297193646293836) q[2];
ry(0.03838077928272701) q[3];
rz(-3.091749828787033) q[3];
ry(-1.6353607565857968) q[4];
rz(0.40569841308751986) q[4];
ry(-0.0002879656982182864) q[5];
rz(2.088934539037833) q[5];
ry(1.705789994890746) q[6];
rz(-3.001369115178603) q[6];
ry(-1.6038088667472212) q[7];
rz(1.6754223215302728) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-0.00013995751035444925) q[0];
rz(1.5406521723990219) q[0];
ry(0.0431717854346525) q[1];
rz(-1.6218361242056745) q[1];
ry(3.028852218674039) q[2];
rz(-1.9770324122696543) q[2];
ry(2.3874143152902887) q[3];
rz(-2.9478222578407145) q[3];
ry(-2.453006627701781) q[4];
rz(-2.9540856342060624) q[4];
ry(3.14153558461467) q[5];
rz(0.8621310940280984) q[5];
ry(-0.92253283478031) q[6];
rz(-2.2670162061804575) q[6];
ry(2.501512892726839) q[7];
rz(0.25543890362173194) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-1.5712304247614128) q[0];
rz(0.02999044942721858) q[0];
ry(-0.4987559546594949) q[1];
rz(2.1951013125108196) q[1];
ry(-1.6927365937480179) q[2];
rz(1.800718797993505) q[2];
ry(-0.38298140468043806) q[3];
rz(2.927431971166879) q[3];
ry(2.503202337828031) q[4];
rz(-2.565319573292969) q[4];
ry(1.5709257769892713) q[5];
rz(-1.5710813991227779) q[5];
ry(-1.442967519314582) q[6];
rz(-0.8585722570436474) q[6];
ry(-1.8483396774323042) q[7];
rz(0.4841842703048313) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(3.135492447404061) q[0];
rz(-3.111406916171787) q[0];
ry(-3.141505999225467) q[1];
rz(2.200403771758917) q[1];
ry(2.303372372569612e-05) q[2];
rz(1.4722196496424775) q[2];
ry(-1.5708672392616898) q[3];
rz(1.570801353977144) q[3];
ry(3.14152429312867) q[4];
rz(0.6879229496064739) q[4];
ry(-1.5708093766764353) q[5];
rz(-2.7138609436279353) q[5];
ry(-3.1415171824717185) q[6];
rz(-2.4219448568653337) q[6];
ry(-1.5707988444849332) q[7];
rz(1.574569968604182) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.5628801140225574) q[0];
rz(-2.7725826981070596) q[0];
ry(1.5708481936578604) q[1];
rz(-0.7639445978196326) q[1];
ry(-1.571024684310852) q[2];
rz(-2.413272607338032) q[2];
ry(-1.5708555611678936) q[3];
rz(-1.570056243082883) q[3];
ry(1.5709840826395984) q[4];
rz(2.7658687911216795) q[4];
ry(0.01232824306015404) q[5];
rz(1.1485538799796562) q[5];
ry(-1.5708233408925407) q[6];
rz(0.68604689111299) q[6];
ry(-1.5707943700983649) q[7];
rz(1.5713933835779579) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-0.00011088948468174208) q[0];
rz(2.8664315713246893) q[0];
ry(-0.0009549179338392833) q[1];
rz(2.4291681730269885) q[1];
ry(-3.141464055372116) q[2];
rz(-0.7485233805906323) q[2];
ry(1.5714403127749454) q[3];
rz(0.0944513883815068) q[3];
ry(3.1414209944242155) q[4];
rz(1.2893875964632011) q[4];
ry(1.5713768243363848) q[5];
rz(-1.476277730423055) q[5];
ry(-0.00018719941091723175) q[6];
rz(0.9790668789151268) q[6];
ry(-1.571302796217906) q[7];
rz(-3.0470926955748725) q[7];