OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-2.7418971119760025) q[0];
rz(-0.4753294616109294) q[0];
ry(2.156100841299863) q[1];
rz(2.569863399677783) q[1];
ry(-1.467817535343005) q[2];
rz(1.1117915106341254) q[2];
ry(-2.709224096173907) q[3];
rz(2.9212964777838635) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.6753824006089912) q[0];
rz(-2.7974071036475063) q[0];
ry(-1.2905981331648309) q[1];
rz(1.1096659141003506) q[1];
ry(1.7886529795092727) q[2];
rz(-2.1109265953272116) q[2];
ry(-1.624896519049025) q[3];
rz(-0.15360684617598225) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.4791815321092026) q[0];
rz(0.32368338034285227) q[0];
ry(-1.787129978157175) q[1];
rz(-2.6649339088547257) q[1];
ry(-3.0109562500470255) q[2];
rz(-0.5166480602133303) q[2];
ry(1.6911751092135618) q[3];
rz(-3.106080662542027) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.0391135772551738) q[0];
rz(-1.9166562020178768) q[0];
ry(1.063906475531792) q[1];
rz(1.6225855399689384) q[1];
ry(0.2197400104110665) q[2];
rz(2.025286697941752) q[2];
ry(0.16550365719827004) q[3];
rz(-1.8590401989467544) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.8267616050171425) q[0];
rz(-1.8275265791407025) q[0];
ry(0.5829356464730172) q[1];
rz(0.9925221889298337) q[1];
ry(2.923689788238323) q[2];
rz(0.12946490119453327) q[2];
ry(-2.3987375570893974) q[3];
rz(0.9018748572334809) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.2401623968902813) q[0];
rz(0.044643907471294715) q[0];
ry(1.376123257805708) q[1];
rz(1.007786251599873) q[1];
ry(0.33359366803809193) q[2];
rz(-0.20127719745629413) q[2];
ry(-2.657558586986048) q[3];
rz(-2.716824149102018) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.804719032823568) q[0];
rz(0.5632529944380619) q[0];
ry(-2.2242340657851574) q[1];
rz(2.982527712056755) q[1];
ry(2.023891023519137) q[2];
rz(-1.724592180236465) q[2];
ry(0.7039513975316708) q[3];
rz(1.4483583699465827) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-3.053685239512168) q[0];
rz(-2.457236815463133) q[0];
ry(0.15792618001667066) q[1];
rz(-0.9044641782174176) q[1];
ry(2.977031016859452) q[2];
rz(-2.015263264753257) q[2];
ry(-0.4697797546941946) q[3];
rz(-1.3076989976992541) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.2862562031100917) q[0];
rz(2.879870555731809) q[0];
ry(0.6666712155150626) q[1];
rz(1.8963231978565438) q[1];
ry(-0.5439530199852172) q[2];
rz(0.9239435323922313) q[2];
ry(2.7972026544382103) q[3];
rz(0.7632879954022567) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.2531819798676418) q[0];
rz(0.2562547235599608) q[0];
ry(-0.3974491108423548) q[1];
rz(-1.6568240041746503) q[1];
ry(-0.8843785701060742) q[2];
rz(-3.1304191623234137) q[2];
ry(-0.06877759343923116) q[3];
rz(2.450608631144716) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.109912511148301) q[0];
rz(0.6870573263487544) q[0];
ry(-0.8212391259421168) q[1];
rz(1.0756663914356386) q[1];
ry(-1.599931989779235) q[2];
rz(-2.7786170775615444) q[2];
ry(-1.8180537849329554) q[3];
rz(-0.22132261870352996) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.8865486282634585) q[0];
rz(2.9962378595726205) q[0];
ry(-2.038224960942223) q[1];
rz(-0.050556144331147) q[1];
ry(-0.33700896906702393) q[2];
rz(-2.1056303353417887) q[2];
ry(0.23346733661527086) q[3];
rz(-0.4097523107477832) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.7451369557944867) q[0];
rz(-1.5006926056425751) q[0];
ry(2.6994764533292654) q[1];
rz(2.111841986696799) q[1];
ry(2.4600613370335807) q[2];
rz(-0.2899724913611758) q[2];
ry(0.9731751170348666) q[3];
rz(2.530829206977119) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.134875045707754) q[0];
rz(1.9100003880057184) q[0];
ry(-2.908897127602612) q[1];
rz(-1.3179314859321098) q[1];
ry(-1.997798408543515) q[2];
rz(0.7467505860146045) q[2];
ry(-1.7132483967844137) q[3];
rz(-2.5910739552859177) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.39579415232366) q[0];
rz(1.1507584863366052) q[0];
ry(1.4574576222998232) q[1];
rz(-0.39719166287533286) q[1];
ry(1.549069720824905) q[2];
rz(-0.34959382269743244) q[2];
ry(-2.045066463937463) q[3];
rz(-2.895834301651833) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.1404566462119963) q[0];
rz(2.9290239167503085) q[0];
ry(2.6995859525543557) q[1];
rz(1.002758387968676) q[1];
ry(2.4708308818612665) q[2];
rz(0.7375811693721577) q[2];
ry(-2.2119296905950097) q[3];
rz(-1.443423297466466) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.2531934765702217) q[0];
rz(-0.5937351541753804) q[0];
ry(-0.10789913087456018) q[1];
rz(-2.3903083251487733) q[1];
ry(-0.08464156173029334) q[2];
rz(-1.6495560347498148) q[2];
ry(1.5577835459235285) q[3];
rz(-3.0532873258635815) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.406508950336349) q[0];
rz(2.747258984736785) q[0];
ry(0.06011481428099806) q[1];
rz(2.2252547443986055) q[1];
ry(0.31086482821131683) q[2];
rz(-3.1402400337274035) q[2];
ry(-0.8628730603303757) q[3];
rz(-0.29322412681628585) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.7195513781025706) q[0];
rz(-2.529576070877796) q[0];
ry(0.5772594113451996) q[1];
rz(-1.7713857201937715) q[1];
ry(-0.025866216016912313) q[2];
rz(0.8292205330343945) q[2];
ry(-0.6250784886435056) q[3];
rz(1.193945866336084) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.5505165083194417) q[0];
rz(-2.234739815647874) q[0];
ry(-2.514195732624407) q[1];
rz(2.606665556581823) q[1];
ry(-1.806847931438714) q[2];
rz(1.3531587752691765) q[2];
ry(-1.3743980092215864) q[3];
rz(-0.8182535815018913) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.9787253559246917) q[0];
rz(0.11567297116824275) q[0];
ry(1.4647015999146769) q[1];
rz(-0.4867359484538625) q[1];
ry(-1.1633029616605093) q[2];
rz(1.9202934290448903) q[2];
ry(0.6922335449846111) q[3];
rz(-2.4832534892032863) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.7318894123709887) q[0];
rz(-0.6452125880947781) q[0];
ry(-1.0968994052015997) q[1];
rz(-1.8266776394679143) q[1];
ry(2.35808332945763) q[2];
rz(1.3145010080226143) q[2];
ry(1.9187823981406562) q[3];
rz(2.05250610193995) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.2385653696852879) q[0];
rz(-0.46719176600333606) q[0];
ry(-0.12257876032728633) q[1];
rz(0.04595722917433154) q[1];
ry(-2.188415064702027) q[2];
rz(-3.010047171205795) q[2];
ry(2.1887842590076034) q[3];
rz(0.8721928066136916) q[3];