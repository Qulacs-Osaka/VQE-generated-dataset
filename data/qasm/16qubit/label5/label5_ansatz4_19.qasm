OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(0.7429058398630639) q[0];
rz(-2.5307118327106566) q[0];
ry(-1.2799649682372742) q[1];
rz(1.1383156019831653) q[1];
ry(-2.5347279354131094) q[2];
rz(-0.07500495028847086) q[2];
ry(-1.987313169492285) q[3];
rz(-2.464993632907246) q[3];
ry(3.0883074563001625) q[4];
rz(0.39970681322454377) q[4];
ry(-3.0907588863554603) q[5];
rz(-1.9418975982485227) q[5];
ry(1.5569903799252787) q[6];
rz(3.0976252362595647) q[6];
ry(1.5662250135158695) q[7];
rz(-3.077241866854242) q[7];
ry(1.5784632033013581) q[8];
rz(-0.6998058477594168) q[8];
ry(3.114435633305558) q[9];
rz(-2.912472109360957) q[9];
ry(-1.577105239314825) q[10];
rz(1.5553063586563933) q[10];
ry(-0.0038961897259907484) q[11];
rz(-0.6306192685110625) q[11];
ry(-3.1292450764072197) q[12];
rz(2.9276140325806805) q[12];
ry(-2.659898693855856) q[13];
rz(-1.6401912123151365) q[13];
ry(0.708640069567835) q[14];
rz(-1.5324926648891397) q[14];
ry(-2.9521244268868316) q[15];
rz(2.4445314105545286) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(1.389254801597188) q[0];
rz(-1.1895872271576389) q[0];
ry(-1.4860351905364642) q[1];
rz(-2.3176815773411183) q[1];
ry(2.666374171115372) q[2];
rz(1.433204783341825) q[2];
ry(-1.5702213022968694) q[3];
rz(-0.2724384765643988) q[3];
ry(-2.1148201820212376) q[4];
rz(-2.1468705054027986) q[4];
ry(-2.4401345680839155) q[5];
rz(0.5514245742284133) q[5];
ry(-0.4010152103262552) q[6];
rz(-1.5436725945534624) q[6];
ry(-2.7379366748454266) q[7];
rz(0.37548492860843563) q[7];
ry(-3.1302960800225765) q[8];
rz(0.8666550177864007) q[8];
ry(0.00032282040637543474) q[9];
rz(0.14048966403699636) q[9];
ry(1.56611034955605) q[10];
rz(-0.8250375629950102) q[10];
ry(2.0562532034167873) q[11];
rz(2.5697852859403336) q[11];
ry(-0.001810358712798776) q[12];
rz(-0.17145331026330246) q[12];
ry(0.04055849652587984) q[13];
rz(1.4812874916306615) q[13];
ry(-0.5724063553863057) q[14];
rz(-2.3953332070208413) q[14];
ry(-1.7553851005760803) q[15];
rz(0.9344085511681097) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(2.1475062738586663) q[0];
rz(2.6053205117959077) q[0];
ry(0.6243603650101983) q[1];
rz(0.8863110891714263) q[1];
ry(-0.6784531787806545) q[2];
rz(-0.21074638944188856) q[2];
ry(-1.1718034597348024) q[3];
rz(3.0767490207481587) q[3];
ry(-0.04354661404292421) q[4];
rz(2.7969945905810887) q[4];
ry(3.0679646772076077) q[5];
rz(0.2815577182150726) q[5];
ry(2.9979525273364893) q[6];
rz(3.120208857726118) q[6];
ry(-3.1390164745019504) q[7];
rz(0.3091610571036844) q[7];
ry(1.5679310797467108) q[8];
rz(2.5856352387019443) q[8];
ry(3.1375097792911677) q[9];
rz(-0.09008963736261144) q[9];
ry(-3.039821396718772) q[10];
rz(-2.8181694078291746) q[10];
ry(0.5307722277539155) q[11];
rz(-3.125702246173629) q[11];
ry(-3.0392497064006485) q[12];
rz(-2.40869673095236) q[12];
ry(-0.002724275773629087) q[13];
rz(0.029236872718445377) q[13];
ry(1.3460185057323157) q[14];
rz(-0.9032964256526496) q[14];
ry(-0.4916670347953156) q[15];
rz(-0.9849803480434673) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-2.665920669550584) q[0];
rz(3.1001008096942573) q[0];
ry(3.0755428084071466) q[1];
rz(-3.0668703641501485) q[1];
ry(0.13108670559987556) q[2];
rz(-2.4238632959338178) q[2];
ry(-2.430976021023403) q[3];
rz(-2.6473137379242146) q[3];
ry(0.36353672475785537) q[4];
rz(0.9946355127653712) q[4];
ry(-2.3889227665728305) q[5];
rz(-1.6078346333222306) q[5];
ry(1.572887773422412) q[6];
rz(-2.0382923469073955) q[6];
ry(1.9930402288343565) q[7];
rz(1.9372761377918977) q[7];
ry(-1.0732058003560878) q[8];
rz(-3.1139195936342254) q[8];
ry(-1.8911712991015044) q[9];
rz(3.138124785749777) q[9];
ry(0.09755675014703966) q[10];
rz(0.42138563808426976) q[10];
ry(-3.1040704705830735) q[11];
rz(3.100904085590023) q[11];
ry(0.004771801232112409) q[12];
rz(-0.7491927609750393) q[12];
ry(-3.1342128575323462) q[13];
rz(-3.1365737539432756) q[13];
ry(3.071483302909568) q[14];
rz(-2.461108304269332) q[14];
ry(-1.7518423901258093) q[15];
rz(-1.6587638716452566) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(1.526757114580077) q[0];
rz(-1.0535771531471883) q[0];
ry(1.1834406298125337) q[1];
rz(-0.5726719945983669) q[1];
ry(2.068665981579952) q[2];
rz(-3.0155238584354946) q[2];
ry(0.5713612083057893) q[3];
rz(-2.925494424849526) q[3];
ry(3.1376919480941172) q[4];
rz(-2.532230293027855) q[4];
ry(0.0011257713923047783) q[5];
rz(-3.052319149978556) q[5];
ry(3.138827864563553) q[6];
rz(-2.41897147667324) q[6];
ry(3.1322869438285106) q[7];
rz(1.5016468744344038) q[7];
ry(0.8626518010444206) q[8];
rz(1.5707291269217245) q[8];
ry(-1.5854199023523332) q[9];
rz(-3.141569826951855) q[9];
ry(-1.5637771156262967) q[10];
rz(2.5657678646928126) q[10];
ry(2.715688728958884) q[11];
rz(0.011969430204814516) q[11];
ry(-3.050317380616405) q[12];
rz(-1.3472752934820542) q[12];
ry(0.0006043882348976126) q[13];
rz(2.2026975706788874) q[13];
ry(3.0925989159180443) q[14];
rz(1.302679168381772) q[14];
ry(-0.19136628803701644) q[15];
rz(0.1407979529885874) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(1.2667954415765161) q[0];
rz(1.9418345002940063) q[0];
ry(-0.6849949025923892) q[1];
rz(-1.0261591830021382) q[1];
ry(1.8406561887352588) q[2];
rz(-0.6724256674142144) q[2];
ry(-2.7062976963457093) q[3];
rz(-0.4144840110720839) q[3];
ry(-2.6534643672868405) q[4];
rz(-1.5663290934427243) q[4];
ry(1.9034162683628102) q[5];
rz(2.5350275572205674) q[5];
ry(-3.1386633473680057) q[6];
rz(1.0314268230881674) q[6];
ry(-3.137660394420753) q[7];
rz(1.2422763034119892) q[7];
ry(-1.5745397562053272) q[8];
rz(-0.5345361733888954) q[8];
ry(1.569931955057544) q[9];
rz(1.5719014303021244) q[9];
ry(0.0015497718115782412) q[10];
rz(1.3176758560679194) q[10];
ry(0.3550180383954959) q[11];
rz(-1.0318978706653121) q[11];
ry(-1.4891298980527514) q[12];
rz(2.0421394521966425) q[12];
ry(-1.0092898824376553) q[13];
rz(-0.7156659953029817) q[13];
ry(-1.1922153960223238) q[14];
rz(1.1957546077508079) q[14];
ry(-2.913541415112917) q[15];
rz(-0.5391358940522137) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-1.9929144206934533) q[0];
rz(2.301258019189151) q[0];
ry(-1.883713899282422) q[1];
rz(-1.2713649336378978) q[1];
ry(-1.0731269380462125) q[2];
rz(-0.33818721490604003) q[2];
ry(-1.674570526224537) q[3];
rz(-1.0398669439421018) q[3];
ry(0.0011518378216832659) q[4];
rz(3.139445206685915) q[4];
ry(-3.14089867840167) q[5];
rz(2.4142834645430558) q[5];
ry(3.1399018726700882) q[6];
rz(0.04383753867297858) q[6];
ry(0.003635289195396574) q[7];
rz(-1.9861306893793031) q[7];
ry(3.1404917311085105) q[8];
rz(2.6186476408507984) q[8];
ry(-2.2395019459627923) q[9];
rz(-0.1852502086100548) q[9];
ry(3.14096603146639) q[10];
rz(-0.8083750105919515) q[10];
ry(-3.139362056256105) q[11];
rz(0.5707138533697469) q[11];
ry(3.141366159826351) q[12];
rz(-1.098782561897604) q[12];
ry(0.0005416593858198482) q[13];
rz(0.7162886231707319) q[13];
ry(0.018342473606212058) q[14];
rz(1.3376784437099971) q[14];
ry(0.004420989645518958) q[15];
rz(0.9161512388242735) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-0.2495238415893013) q[0];
rz(-0.8757456193083158) q[0];
ry(0.22017829289739932) q[1];
rz(2.2439661912461957) q[1];
ry(1.4736442009565058) q[2];
rz(-0.36840303034507293) q[2];
ry(1.1840806927386232) q[3];
rz(-1.866232234492359) q[3];
ry(1.940264102298602) q[4];
rz(1.8545281336601107) q[4];
ry(0.36386201459588025) q[5];
rz(2.426338718494625) q[5];
ry(0.05516578172826646) q[6];
rz(2.098663354278842) q[6];
ry(1.5919508217973286) q[7];
rz(-0.4184670516996576) q[7];
ry(-1.570685932445917) q[8];
rz(1.6975891501931226) q[8];
ry(3.138851286978156) q[9];
rz(1.7977657391219253) q[9];
ry(-1.5731416459637022) q[10];
rz(-0.6789552870553432) q[10];
ry(-1.5684317141728368) q[11];
rz(1.8217347189886899) q[11];
ry(1.655465654131486) q[12];
rz(0.13175586666714345) q[12];
ry(2.131325745343789) q[13];
rz(-0.3775131497631229) q[13];
ry(-0.43976573045265305) q[14];
rz(2.613569480806393) q[14];
ry(2.32539037405647) q[15];
rz(-0.06652417822234113) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-1.9468724139628841) q[0];
rz(-2.2353080227355138) q[0];
ry(-2.2990839377290877) q[1];
rz(-2.019031780759833) q[1];
ry(1.770338299758536) q[2];
rz(2.3840665154972864) q[2];
ry(-0.4104825024959279) q[3];
rz(0.4956555332811181) q[3];
ry(3.1396180220703087) q[4];
rz(-0.9885317898352664) q[4];
ry(3.140994820437596) q[5];
rz(2.3849315145521315) q[5];
ry(3.128158831326701) q[6];
rz(-1.9610087049143337) q[6];
ry(-3.1384951676225077) q[7];
rz(-2.323137068122774) q[7];
ry(3.0886114394454722) q[8];
rz(2.6592733902769266) q[8];
ry(2.6662878683580967e-05) q[9];
rz(0.7039481223216336) q[9];
ry(3.1380316746360513) q[10];
rz(0.8718237995992952) q[10];
ry(-1.9496770365152374) q[11];
rz(-0.13073688926676752) q[11];
ry(2.1714182667486552) q[12];
rz(-0.5485658006540106) q[12];
ry(1.7560188812056161) q[13];
rz(0.8904967297032105) q[13];
ry(-0.416769513879899) q[14];
rz(-2.2231527529157216) q[14];
ry(1.356702664116757) q[15];
rz(-0.127450080905624) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-1.539113542427544) q[0];
rz(-2.6208741922011622) q[0];
ry(2.734647421535554) q[1];
rz(0.712568592587469) q[1];
ry(1.2845105247261557) q[2];
rz(0.44851120859662735) q[2];
ry(-2.8386153578282203) q[3];
rz(0.397308816468202) q[3];
ry(1.434176006257057) q[4];
rz(2.204341232619235) q[4];
ry(-0.9512566533146227) q[5];
rz(-2.9493402571379534) q[5];
ry(-1.5747453343582498) q[6];
rz(0.7200789132083817) q[6];
ry(0.3240376359220374) q[7];
rz(1.8101170659520653) q[7];
ry(-3.1368964679094344) q[8];
rz(-1.0281002523421927) q[8];
ry(-3.0494112193619185) q[9];
rz(-2.0025447532459104) q[9];
ry(0.22476484485772197) q[10];
rz(0.1506854329147398) q[10];
ry(3.13203306291174) q[11];
rz(-0.36568683727583406) q[11];
ry(-3.1231350501152995) q[12];
rz(2.554541455212853) q[12];
ry(-3.1415091015880563) q[13];
rz(0.2861782064837399) q[13];
ry(0.009197274038635683) q[14];
rz(0.7839051568308512) q[14];
ry(-0.004419548176596955) q[15];
rz(0.07152743015011787) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(1.9335697326327521) q[0];
rz(-1.3738039580703352) q[0];
ry(-0.5003188384985425) q[1];
rz(1.0440662678030603) q[1];
ry(1.3666225303542712) q[2];
rz(-0.8124562694067868) q[2];
ry(-0.9862155056312413) q[3];
rz(2.654693382597526) q[3];
ry(0.043720889020260084) q[4];
rz(-1.743535350018881) q[4];
ry(-3.139437373630199) q[5];
rz(-2.4939955457072394) q[5];
ry(-0.00892547452701642) q[6];
rz(-2.319215127981375) q[6];
ry(3.13912548621401) q[7];
rz(0.06865447218518206) q[7];
ry(-3.141087595131315) q[8];
rz(-1.0585228801774944) q[8];
ry(0.07826345453240381) q[9];
rz(1.9695135525949974) q[9];
ry(-0.005654209892489881) q[10];
rz(-0.12498422913263861) q[10];
ry(0.06463488478603906) q[11];
rz(-1.331655514912245) q[11];
ry(-2.5365017738163402) q[12];
rz(3.107585713521134) q[12];
ry(-3.111637029712332) q[13];
rz(-2.6927957269108465) q[13];
ry(-0.08106643189448537) q[14];
rz(1.5322646175688917) q[14];
ry(-0.3581338895675268) q[15];
rz(-1.6695947716777624) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(1.9585834875964512) q[0];
rz(-2.0051575661809466) q[0];
ry(-1.2176046853107287) q[1];
rz(0.5622222775342987) q[1];
ry(0.9641338355483148) q[2];
rz(-1.6902121658659395) q[2];
ry(0.7743059569120511) q[3];
rz(-2.0848985976120247) q[3];
ry(-1.3079865069884735) q[4];
rz(-2.0736213744470637) q[4];
ry(-0.8656110263071032) q[5];
rz(1.578462072761841) q[5];
ry(0.42496933582086577) q[6];
rz(0.07879495348599441) q[6];
ry(-3.1317530027620903) q[7];
rz(1.7248643857255457) q[7];
ry(0.013585320711640057) q[8];
rz(-0.9459877685235684) q[8];
ry(3.0532073673747355) q[9];
rz(-2.718121097845375) q[9];
ry(1.751703823924373) q[10];
rz(0.004130233379445514) q[10];
ry(1.5726372576689132) q[11];
rz(0.007482567320434673) q[11];
ry(2.2131446429354398) q[12];
rz(-0.9787544753780248) q[12];
ry(1.6954459931042862) q[13];
rz(2.493503414625368) q[13];
ry(-2.8143917807647503) q[14];
rz(-1.1438408377678455) q[14];
ry(-1.1836867977918262) q[15];
rz(-1.0178176388012992) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-1.1440511377493134) q[0];
rz(-3.048440694957183) q[0];
ry(1.9277398539758257) q[1];
rz(-1.5346211058740309) q[1];
ry(-2.984276181454976) q[2];
rz(-1.8026330930195718) q[2];
ry(-0.5157767071612678) q[3];
rz(0.9494487175964962) q[3];
ry(-3.110139690647412) q[4];
rz(0.01707303776785274) q[4];
ry(-0.028935883334365364) q[5];
rz(-2.127424699320697) q[5];
ry(1.5697744852325257) q[6];
rz(1.5757849005772284) q[6];
ry(-1.5826750044765623) q[7];
rz(-3.1365583910175197) q[7];
ry(1.5834123170013361) q[8];
rz(-0.019579247479024234) q[8];
ry(-1.5762745383048333) q[9];
rz(-2.945965895190043) q[9];
ry(-1.573385394122301) q[10];
rz(1.446686867319281) q[10];
ry(-1.5771814843592837) q[11];
rz(-0.4712687877716263) q[11];
ry(0.0076514848263891716) q[12];
rz(1.3195276149797712) q[12];
ry(3.139949782916894) q[13];
rz(-1.048492007659908) q[13];
ry(-1.9172193659207437) q[14];
rz(-2.8046137082710825) q[14];
ry(-1.3317225020236663) q[15];
rz(3.080986828610592) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(0.007025672274925121) q[0];
rz(-2.805709681376814) q[0];
ry(-1.4080804768625979) q[1];
rz(2.4086992476392206) q[1];
ry(2.0054259019783833) q[2];
rz(-0.6364823283031269) q[2];
ry(1.6764306284740806) q[3];
rz(-0.3977138621127677) q[3];
ry(-1.5710705928564384) q[4];
rz(-0.0038299797621776936) q[4];
ry(1.5677596671546272) q[5];
rz(2.9644377298275173) q[5];
ry(1.4504844408452326) q[6];
rz(-3.128066567351316) q[6];
ry(1.5687600174979313) q[7];
rz(-1.5728774497929552) q[7];
ry(-0.2935767542800802) q[8];
rz(1.5928706539273858) q[8];
ry(-0.004529305095398507) q[9];
rz(1.3910918145984201) q[9];
ry(-1.390359651771884) q[10];
rz(-0.7049428238815288) q[10];
ry(3.1415431707329478) q[11];
rz(-0.08334807363654184) q[11];
ry(3.1333704746338804) q[12];
rz(2.572996560641272) q[12];
ry(-3.055633135571468) q[13];
rz(-1.4490775555857143) q[13];
ry(-1.8346676824009953) q[14];
rz(-0.5807262472283874) q[14];
ry(2.267298324960911) q[15];
rz(2.002135506323992) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(1.3523517576478288) q[0];
rz(1.1752059654454199) q[0];
ry(2.565940034257937) q[1];
rz(0.20990342639736687) q[1];
ry(-0.1796913035076093) q[2];
rz(1.6399801401488228) q[2];
ry(1.8913386379796415) q[3];
rz(1.991927906419554) q[3];
ry(-0.8000847547388127) q[4];
rz(-1.363718917852542) q[4];
ry(0.0007359330737646468) q[5];
rz(1.084872054397608) q[5];
ry(1.5709771691554562) q[6];
rz(0.0010003346195484042) q[6];
ry(-1.5712538740196158) q[7];
rz(-3.13914679661313) q[7];
ry(-1.5700574033740633) q[8];
rz(-1.020661828912166) q[8];
ry(1.5656101789452945) q[9];
rz(-1.5689377427861213) q[9];
ry(-0.0005409982991686491) q[10];
rz(2.498397087092147) q[10];
ry(-0.0012575642362318005) q[11];
rz(2.757751529700095) q[11];
ry(0.0017225163333716154) q[12];
rz(-0.6706288948671791) q[12];
ry(3.140129084620325) q[13];
rz(-0.026770565244050058) q[13];
ry(1.3758416426715658) q[14];
rz(-0.9742978308379886) q[14];
ry(-1.5831229830268436) q[15];
rz(1.0268467692645098) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-1.8334656110583458) q[0];
rz(-1.0391993092381795) q[0];
ry(0.8765615862325609) q[1];
rz(0.3090829332016831) q[1];
ry(-0.017098799397021303) q[2];
rz(-0.02436537373825498) q[2];
ry(-0.005039058439139899) q[3];
rz(-0.22362552433185057) q[3];
ry(-0.000993558699152075) q[4];
rz(2.9804245949091523) q[4];
ry(-0.0011387239575268993) q[5];
rz(-0.9093509934094869) q[5];
ry(1.0987024235089544) q[6];
rz(2.6249373242284997) q[6];
ry(1.5733551996183355) q[7];
rz(1.5645092937874507) q[7];
ry(-0.0036093326351575275) q[8];
rz(-0.6552784675264505) q[8];
ry(-0.5445736449541752) q[9];
rz(-0.3002864750926245) q[9];
ry(-1.566436108316812) q[10];
rz(-0.003984311101638659) q[10];
ry(-1.5663101127661738) q[11];
rz(0.005153031731801743) q[11];
ry(1.5725767395351804) q[12];
rz(-1.5750590215482054) q[12];
ry(-1.5698614677799798) q[13];
rz(1.0374166691287812) q[13];
ry(-2.8911461377013272) q[14];
rz(3.1052736499903313) q[14];
ry(-1.0803472292742466) q[15];
rz(0.4020578789088418) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(0.7353952287079161) q[0];
rz(2.086840634739174) q[0];
ry(0.3619557889394517) q[1];
rz(-0.5999990877588236) q[1];
ry(2.4370430175154616) q[2];
rz(0.44919086008529524) q[2];
ry(-2.3548174645168323) q[3];
rz(-2.808813781186157) q[3];
ry(0.08983401028878947) q[4];
rz(-2.5456275098341035) q[4];
ry(1.5685984623507867) q[5];
rz(0.011881671604186374) q[5];
ry(-0.0030042557804537493) q[6];
rz(3.0993138594191962) q[6];
ry(-1.5572411240621653) q[7];
rz(-2.9364067881817535) q[7];
ry(-3.1127894951878132) q[8];
rz(1.4662829337374745) q[8];
ry(-3.1401181683661807) q[9];
rz(1.2728217143688332) q[9];
ry(1.5668476048482054) q[10];
rz(-2.8191340733243018) q[10];
ry(-1.565195983800301) q[11];
rz(-2.367941290364088) q[11];
ry(1.5673176888574654) q[12];
rz(2.913268606257023) q[12];
ry(0.010506074917183916) q[13];
rz(0.5284360108095197) q[13];
ry(0.011784690500355135) q[14];
rz(-0.6963666456270001) q[14];
ry(1.571502125690344) q[15];
rz(-0.01411659581896707) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(1.7018089989009155) q[0];
rz(-1.6246577851040507) q[0];
ry(1.3581961686436872) q[1];
rz(-1.6561524442382347) q[1];
ry(-1.0942061335824906) q[2];
rz(-1.9905991321465661) q[2];
ry(-1.5759999505411546) q[3];
rz(1.4480152466509197) q[3];
ry(0.002230777525716483) q[4];
rz(-2.2078873242329182) q[4];
ry(0.1377280031322325) q[5];
rz(0.034190360357808196) q[5];
ry(3.1415357105576875) q[6];
rz(-2.1296436638854077) q[6];
ry(3.141554438252175) q[7];
rz(0.20556325130521458) q[7];
ry(-1.581508089150772) q[8];
rz(-2.9027470039591754) q[8];
ry(-1.5719958762793624) q[9];
rz(0.0830090564898296) q[9];
ry(-3.141041085969342) q[10];
rz(-0.24690343491336705) q[10];
ry(-0.002709190246367566) q[11];
rz(0.8007993785451619) q[11];
ry(3.1365437098983016) q[12];
rz(1.3454315072105376) q[12];
ry(-1.5726306601878042) q[13];
rz(-0.6561096551370279) q[13];
ry(0.004159968473105735) q[14];
rz(-1.2056056269709687) q[14];
ry(-1.525861161341794) q[15];
rz(-2.550172947909307) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(1.549183461952393) q[0];
rz(0.020546176757844314) q[0];
ry(-3.1125365081782737) q[1];
rz(0.5717458091177844) q[1];
ry(-3.1227581117290995) q[2];
rz(-1.9970267513113278) q[2];
ry(1.5538260371929717) q[3];
rz(-1.559862137549921) q[3];
ry(-1.57573787757031) q[4];
rz(-1.3648788629288546) q[4];
ry(-1.5690169336748614) q[5];
rz(-2.52922924858103) q[5];
ry(1.5681108420327947) q[6];
rz(-0.018991973348326554) q[6];
ry(-1.5701646587808797) q[7];
rz(-3.1081394558092965) q[7];
ry(3.1398848724309656) q[8];
rz(1.4050347473827243) q[8];
ry(-1.5735768447671843) q[9];
rz(1.5983165888178494) q[9];
ry(0.001368062068215535) q[10];
rz(-1.795570789836616) q[10];
ry(1.5731549996584417) q[11];
rz(2.659670214842474) q[11];
ry(-1.5779030299439283) q[12];
rz(0.4420465560172907) q[12];
ry(-0.004829554242285138) q[13];
rz(-0.9083671422514863) q[13];
ry(1.5782740311311778) q[14];
rz(-1.375166719468679) q[14];
ry(-0.009177980703610977) q[15];
rz(-1.075339438859343) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-1.5825163473707864) q[0];
rz(0.48997960651053774) q[0];
ry(-1.668881035001424) q[1];
rz(0.030715287800274993) q[1];
ry(1.5602888621366262) q[2];
rz(-2.1109071771192767) q[2];
ry(-1.5186822924097223) q[3];
rz(0.003785040907480663) q[3];
ry(8.383246161782203e-05) q[4];
rz(-2.626258228650751) q[4];
ry(-5.2466528355665794e-05) q[5];
rz(0.9868703274647967) q[5];
ry(0.07747177230540107) q[6];
rz(-1.5753858775972382) q[6];
ry(1.577540144210715) q[7];
rz(1.5748600167907416) q[7];
ry(-3.1378926900514954) q[8];
rz(-2.139507411194473) q[8];
ry(-1.5736339598394018) q[9];
rz(3.139090326075375) q[9];
ry(3.1410998874911535) q[10];
rz(-0.7928325440365782) q[10];
ry(3.1414910165706185) q[11];
rz(-0.48436318104716214) q[11];
ry(-3.14064861926308) q[12];
rz(-2.9028928473853446) q[12];
ry(-0.5923772461105333) q[13];
rz(-0.814862054809446) q[13];
ry(1.4962024739624349) q[14];
rz(2.191356777023208) q[14];
ry(0.15236746920483046) q[15];
rz(1.9564006602476318) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-0.03945379343894717) q[0];
rz(-2.2277227806023787) q[0];
ry(-1.9001817818989704) q[1];
rz(2.2931153130832787) q[1];
ry(-3.1259518102588513) q[2];
rz(-0.5396590117565297) q[2];
ry(-1.5635969921750794) q[3];
rz(0.044549958360956765) q[3];
ry(-3.1401746627247737) q[4];
rz(0.49461364619556836) q[4];
ry(0.07065391903234097) q[5];
rz(-0.7031229847590899) q[5];
ry(3.128164961095374) q[6];
rz(3.1172301348192306) q[6];
ry(0.006700850557384896) q[7];
rz(-0.1414920786430507) q[7];
ry(-3.136657839599981) q[8];
rz(2.977271649243931) q[8];
ry(1.5878281418702755) q[9];
rz(-1.6001969904015712) q[9];
ry(-1.570897521540148) q[10];
rz(-0.6902827076303453) q[10];
ry(-1.5704310483060775) q[11];
rz(2.3097625657794456) q[11];
ry(-3.13391558201168) q[12];
rz(1.3515190109569026) q[12];
ry(0.00017686224575275133) q[13];
rz(1.5359534879116883) q[13];
ry(-0.0038616776889787473) q[14];
rz(-1.0595581315876235) q[14];
ry(-0.23359911198545988) q[15];
rz(-0.5371567228800247) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-1.5478948504531163) q[0];
rz(0.8908654677308112) q[0];
ry(0.07360761144547824) q[1];
rz(2.6272333020800516) q[1];
ry(-1.5692799267865967) q[2];
rz(-1.5354675354890035) q[2];
ry(-1.5614862177750428) q[3];
rz(0.0008722333862429822) q[3];
ry(3.1409207011941587) q[4];
rz(-1.7463133605499914) q[4];
ry(-3.138533082927759) q[5];
rz(-0.6769884021211992) q[5];
ry(-1.5729601123908754) q[6];
rz(2.907569380966928) q[6];
ry(-1.5286264840444252) q[7];
rz(0.08735849753035071) q[7];
ry(1.5711067586855805) q[8];
rz(2.36015127208071) q[8];
ry(-1.6042014872428694) q[9];
rz(-1.5682198155032794) q[9];
ry(-0.0001605085258835133) q[10];
rz(0.4810368632196209) q[10];
ry(3.14147599890427) q[11];
rz(2.3121047338230265) q[11];
ry(3.140504262077677) q[12];
rz(-0.013992664809650038) q[12];
ry(3.138985140195164) q[13];
rz(2.7954894576396465) q[13];
ry(-2.8544808222032545) q[14];
rz(0.5313446230649935) q[14];
ry(0.14203713318767885) q[15];
rz(0.4114803814805237) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-0.0011210157268630033) q[0];
rz(1.9092977836104366) q[0];
ry(-2.9888248529476953) q[1];
rz(-0.8587389542125905) q[1];
ry(3.121474175542827) q[2];
rz(-1.4075639627993608) q[2];
ry(1.569992722140813) q[3];
rz(-1.5051720672809743) q[3];
ry(-3.1410748764830387) q[4];
rz(-3.0577368964584464) q[4];
ry(1.5705167391690829) q[5];
rz(-2.8218945367306567) q[5];
ry(-3.140420758987389) q[6];
rz(-0.19035808601345125) q[6];
ry(1.5648729268285673) q[7];
rz(1.3556967389368415) q[7];
ry(0.0003520428986547586) q[8];
rz(-2.320419648842886) q[8];
ry(-1.5671203860526284) q[9];
rz(1.6972719364565716) q[9];
ry(3.1410852257212967) q[10];
rz(1.4079585164578117) q[10];
ry(-1.5718660516769736) q[11];
rz(1.0899544932843828) q[11];
ry(-1.572223135265801) q[12];
rz(0.034977299120035) q[12];
ry(-0.004842079338691185) q[13];
rz(-0.5805625386587554) q[13];
ry(-3.1339909949884395) q[14];
rz(-2.160073111519081) q[14];
ry(0.6859717088651547) q[15];
rz(3.1082004667986207) q[15];