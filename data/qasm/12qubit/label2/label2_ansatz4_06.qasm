OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(3.141489618859028) q[0];
rz(-1.226596377484193) q[0];
ry(-3.1413979537733874) q[1];
rz(-2.6578420706955654) q[1];
ry(-1.5712216930092486) q[2];
rz(-2.192057119283011) q[2];
ry(1.5710400122985784) q[3];
rz(2.251677313275736) q[3];
ry(7.190113475296995e-05) q[4];
rz(2.9830255154345595) q[4];
ry(-3.1411654058087883) q[5];
rz(-0.44801483547023807) q[5];
ry(-1.5577562515179195) q[6];
rz(0.0023523153648743744) q[6];
ry(-3.1412603692727643) q[7];
rz(-1.7600509604517764) q[7];
ry(-3.0933308760665224) q[8];
rz(-1.099720218492549) q[8];
ry(-3.1413745537999866) q[9];
rz(-3.10392303863996) q[9];
ry(1.5721304426561717) q[10];
rz(0.37849009117603843) q[10];
ry(1.5720708485088226) q[11];
rz(-2.5572390735503983) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-1.5706591697441297) q[0];
rz(1.2456629140753226) q[0];
ry(-1.5707983253917215) q[1];
rz(-0.019129323626756397) q[1];
ry(2.1528687473872274) q[2];
rz(1.8672243543140352) q[2];
ry(-2.0790509810519526) q[3];
rz(-1.3977075421976455) q[3];
ry(1.5702466200528487) q[4];
rz(0.048275953931406164) q[4];
ry(1.617356108708222) q[5];
rz(-0.0005366575939429231) q[5];
ry(1.5578801562838236) q[6];
rz(1.7267336435901157) q[6];
ry(1.5628945384118598) q[7];
rz(2.012552889637597) q[7];
ry(-2.448574862697415) q[8];
rz(1.484479131066033) q[8];
ry(-2.2844249260502467) q[9];
rz(-1.5433809181471059) q[9];
ry(2.081455761094216) q[10];
rz(2.4427943184281764) q[10];
ry(-2.513741737239852) q[11];
rz(1.4352631674318468) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-0.020197037370698112) q[0];
rz(-1.5318685639494696) q[0];
ry(1.5772759049838003) q[1];
rz(0.2850507562978119) q[1];
ry(1.6127351179301934) q[2];
rz(0.2200194303190779) q[2];
ry(-0.08919254711896298) q[3];
rz(0.4998463171547502) q[3];
ry(-1.3819307017073985) q[4];
rz(1.5592897605944822) q[4];
ry(-1.3905016362513285) q[5];
rz(3.11267250689815) q[5];
ry(-3.1414579832418363) q[6];
rz(1.7443998965852687) q[6];
ry(-3.1405101378399003) q[7];
rz(1.086657541075816) q[7];
ry(0.04654636243726094) q[8];
rz(2.246659354944696) q[8];
ry(-0.04632586929392694) q[9];
rz(-0.330502007568759) q[9];
ry(3.132711325142012) q[10];
rz(-1.1527980238634234) q[10];
ry(3.1333928080185265) q[11];
rz(0.22109445814133213) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(1.528123947441296) q[0];
rz(-1.0224518752611975) q[0];
ry(1.5274167363661548) q[1];
rz(-0.6114188420932503) q[1];
ry(-3.1378635387705445) q[2];
rz(2.631568362277363) q[2];
ry(1.523971092117339) q[3];
rz(-3.0309538610016316) q[3];
ry(-0.0018771698067424148) q[4];
rz(-1.7064248225898415) q[4];
ry(-0.06677363026648475) q[5];
rz(-3.1124428924403613) q[5];
ry(1.5709193170589622) q[6];
rz(0.4757345854585058) q[6];
ry(3.127613071431771) q[7];
rz(-1.645138915277857) q[7];
ry(2.124249011934613) q[8];
rz(1.6304962629897053) q[8];
ry(-1.085335596293019) q[9];
rz(-2.0949936603702266) q[9];
ry(-3.1309883743230635) q[10];
rz(-0.059833865003984237) q[10];
ry(-1.4602606743772493) q[11];
rz(2.033700383914784) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-2.4311355220411834) q[0];
rz(0.20742191852418213) q[0];
ry(-1.5875512778654466) q[1];
rz(2.5951149475630566) q[1];
ry(1.4977461705053683) q[2];
rz(0.4236196754821071) q[2];
ry(-2.3775333173204727) q[3];
rz(0.6106605549473931) q[3];
ry(-0.0001814283345549228) q[4];
rz(0.44891281646658404) q[4];
ry(-1.5709077449322792) q[5];
rz(2.1557088621636438) q[5];
ry(-0.0005417912735898867) q[6];
rz(-0.6674549097494556) q[6];
ry(3.1413094470497986) q[7];
rz(-1.5586891851069522) q[7];
ry(0.0006268569479324481) q[8];
rz(-2.8220099334919193) q[8];
ry(-0.0001360886140910822) q[9];
rz(1.7414178929744466) q[9];
ry(-0.1239792164845639) q[10];
rz(1.418845265711127) q[10];
ry(-0.057121784471953596) q[11];
rz(-0.6147029043362408) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-1.5927793983242875) q[0];
rz(2.457414331999486) q[0];
ry(-2.8350009621971473) q[1];
rz(-0.8156911208988556) q[1];
ry(-0.06571541226999451) q[2];
rz(1.472604514440973) q[2];
ry(0.007719696052953067) q[3];
rz(0.9471468538597689) q[3];
ry(-3.0732379279919155) q[4];
rz(-2.7343019320241435) q[4];
ry(-0.6495995089350743) q[5];
rz(-3.0081167784319054) q[5];
ry(-3.0837315899266073) q[6];
rz(2.1109060956511487) q[6];
ry(-0.0036900136092106466) q[7];
rz(0.8374878579238194) q[7];
ry(-1.549026787409721) q[8];
rz(-3.0760236779486823) q[8];
ry(-3.076868252738733) q[9];
rz(-0.15674633087227186) q[9];
ry(-0.3324812117145698) q[10];
rz(1.7580079621563187) q[10];
ry(-0.7444404561033684) q[11];
rz(0.5326607122995659) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(2.053960193677313) q[0];
rz(-2.5626040576428992) q[0];
ry(1.019780877186657) q[1];
rz(0.6831211155487473) q[1];
ry(-0.005918584286393802) q[2];
rz(-0.21842584237185098) q[2];
ry(-3.1396211704273242) q[3];
rz(-1.965140148873414) q[3];
ry(-0.00011048986171591225) q[4];
rz(-2.2611946671581373) q[4];
ry(0.000332900310353601) q[5];
rz(2.516826157506874) q[5];
ry(-6.814730590676277e-05) q[6];
rz(0.7924519384883532) q[6];
ry(3.141581511367099) q[7];
rz(-1.17521800038539) q[7];
ry(0.04741020624407177) q[8];
rz(-2.648793624957933) q[8];
ry(0.006630056657490553) q[9];
rz(0.1251729809190502) q[9];
ry(-1.8649211961350933) q[10];
rz(1.5389359220810586) q[10];
ry(-2.4866117950500337) q[11];
rz(0.4860193804557306) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-1.3660821940523231) q[0];
rz(0.3063018774048599) q[0];
ry(1.3719367597895042) q[1];
rz(-2.9114260520787347) q[1];
ry(-3.0417285915545724) q[2];
rz(1.3392948562444558) q[2];
ry(-1.4767636136722393) q[3];
rz(0.7901616896179375) q[3];
ry(-2.6471733599261627) q[4];
rz(-1.0596776162288135) q[4];
ry(1.2162511582011786) q[5];
rz(-0.09137761386779708) q[5];
ry(-2.7792101954077157) q[6];
rz(-1.6768292097850839) q[6];
ry(-1.5785851435515759) q[7];
rz(-1.1474961457423505) q[7];
ry(-3.0455374274487905) q[8];
rz(-0.9661586647001111) q[8];
ry(3.1125035154846623) q[9];
rz(-0.48026599474381726) q[9];
ry(-0.0015248701883390225) q[10];
rz(0.7033825884374476) q[10];
ry(-3.1366038832033682) q[11];
rz(-2.0290580991304026) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-1.5675514155381203) q[0];
rz(0.16103846900423768) q[0];
ry(-1.5741071833080833) q[1];
rz(-1.4111431702807842) q[1];
ry(-0.0007873675774048294) q[2];
rz(-3.1257713620223555) q[2];
ry(-3.1389580100190866) q[3];
rz(-0.764762167277519) q[3];
ry(3.1390152916174316) q[4];
rz(2.962261422481376) q[4];
ry(3.136570847024614) q[5];
rz(-0.012481471604510295) q[5];
ry(1.0019891215972757e-05) q[6];
rz(2.466176856351609) q[6];
ry(3.141581721787442) q[7];
rz(-1.6828915343132413) q[7];
ry(-1.5709045819396552) q[8];
rz(1.5486084739516734) q[8];
ry(-1.5715772778391086) q[9];
rz(-1.5203217237806212) q[9];
ry(1.571463182703452) q[10];
rz(2.0412992754085924) q[10];
ry(-3.1220518831822326) q[11];
rz(0.6563911637530442) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-0.013503398037601144) q[0];
rz(3.10151998378372) q[0];
ry(3.057500757257397) q[1];
rz(1.8604123395954342) q[1];
ry(-1.5634111339783083) q[2];
rz(1.5861354189969579) q[2];
ry(-1.5871214179700235) q[3];
rz(-2.9341695417889553) q[3];
ry(-1.844525494745529) q[4];
rz(-1.0255240005847799) q[4];
ry(0.6885952277269152) q[5];
rz(-1.3459221571955304) q[5];
ry(0.01673586349045808) q[6];
rz(-0.7240165992439331) q[6];
ry(3.1135069458412237) q[7];
rz(1.1493794476726888) q[7];
ry(1.881884177809968) q[8];
rz(-3.0353311646547163) q[8];
ry(-2.68132278070297) q[9];
rz(-2.9830481544628924) q[9];
ry(-1.5830062941199519) q[10];
rz(1.6826600165294883) q[10];
ry(0.7256858054523008) q[11];
rz(0.09938411041311923) q[11];