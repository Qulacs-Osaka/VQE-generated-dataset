OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(0.5901430211019951) q[0];
ry(-2.584236098747793) q[1];
cx q[0],q[1];
ry(-1.1584678456352906) q[0];
ry(-1.9266356371577382) q[1];
cx q[0],q[1];
ry(-0.3093184202453911) q[0];
ry(-1.4416928408680219) q[2];
cx q[0],q[2];
ry(-0.21489343348142942) q[0];
ry(1.639176090552316) q[2];
cx q[0],q[2];
ry(-2.1996628996724343) q[0];
ry(-2.6790363842565315) q[3];
cx q[0],q[3];
ry(-1.963632777334181) q[0];
ry(-0.9586260542715506) q[3];
cx q[0],q[3];
ry(0.43078593387421993) q[1];
ry(2.0997057754217616) q[2];
cx q[1],q[2];
ry(-1.1386089718046506) q[1];
ry(-2.1821132122927773) q[2];
cx q[1],q[2];
ry(-2.055144959008052) q[1];
ry(2.637130806260376) q[3];
cx q[1],q[3];
ry(-1.2391780597560884) q[1];
ry(1.1933858156989476) q[3];
cx q[1],q[3];
ry(0.37773521090454415) q[2];
ry(-0.037924136472785204) q[3];
cx q[2],q[3];
ry(-1.239754394164734) q[2];
ry(2.857381796900065) q[3];
cx q[2],q[3];
ry(-0.9138764461114894) q[0];
ry(-2.2869672684937723) q[1];
cx q[0],q[1];
ry(2.0121825583808004) q[0];
ry(-0.5114253089529246) q[1];
cx q[0],q[1];
ry(-2.2210009652526193) q[0];
ry(0.29059142419403194) q[2];
cx q[0],q[2];
ry(2.6907018487151104) q[0];
ry(-0.5333252316985799) q[2];
cx q[0],q[2];
ry(-2.941253684293027) q[0];
ry(2.3198061625508095) q[3];
cx q[0],q[3];
ry(-2.6045425143490193) q[0];
ry(1.2190950937728755) q[3];
cx q[0],q[3];
ry(-1.2334538754971813) q[1];
ry(0.11564991874162356) q[2];
cx q[1],q[2];
ry(-1.5079121533709177) q[1];
ry(1.957835348446829) q[2];
cx q[1],q[2];
ry(-1.5738652242963218) q[1];
ry(-1.2154450594271928) q[3];
cx q[1],q[3];
ry(0.7580293014458563) q[1];
ry(-2.774314286325741) q[3];
cx q[1],q[3];
ry(-1.3323536329355918) q[2];
ry(-3.065805583366862) q[3];
cx q[2],q[3];
ry(1.3618156912150894) q[2];
ry(0.38242312450447163) q[3];
cx q[2],q[3];
ry(1.2146836304276394) q[0];
ry(-2.0330596376939445) q[1];
cx q[0],q[1];
ry(-2.7946888589470658) q[0];
ry(0.5383369935351539) q[1];
cx q[0],q[1];
ry(2.9086778827227557) q[0];
ry(-1.9285482655044621) q[2];
cx q[0],q[2];
ry(-1.9187928333424282) q[0];
ry(-0.09829330181328633) q[2];
cx q[0],q[2];
ry(-1.114068362605078) q[0];
ry(-2.8001890794487094) q[3];
cx q[0],q[3];
ry(-0.38745430236461065) q[0];
ry(-0.5742856308472906) q[3];
cx q[0],q[3];
ry(-2.871587880793463) q[1];
ry(0.010064948477640279) q[2];
cx q[1],q[2];
ry(-2.31268995088938) q[1];
ry(-0.24772802154008253) q[2];
cx q[1],q[2];
ry(2.002434725234046) q[1];
ry(-1.1641027930146863) q[3];
cx q[1],q[3];
ry(-2.0455673696826033) q[1];
ry(-2.4791108684680836) q[3];
cx q[1],q[3];
ry(2.6326681987187253) q[2];
ry(2.035824295109351) q[3];
cx q[2],q[3];
ry(1.83336292691871) q[2];
ry(-1.8465982186877223) q[3];
cx q[2],q[3];
ry(0.696202444691661) q[0];
ry(2.2736825125675626) q[1];
cx q[0],q[1];
ry(2.20714385863375) q[0];
ry(-0.9422217240405049) q[1];
cx q[0],q[1];
ry(1.774203964948079) q[0];
ry(-0.4937323824216069) q[2];
cx q[0],q[2];
ry(2.084785296693435) q[0];
ry(-2.0253661125319167) q[2];
cx q[0],q[2];
ry(-2.9242130788633967) q[0];
ry(1.3312621319618465) q[3];
cx q[0],q[3];
ry(1.1208253689214613) q[0];
ry(2.7224032925535044) q[3];
cx q[0],q[3];
ry(-1.5300471482697209) q[1];
ry(0.11196208492618873) q[2];
cx q[1],q[2];
ry(-2.8421846754751714) q[1];
ry(0.9081776769319035) q[2];
cx q[1],q[2];
ry(-1.7513892011677212) q[1];
ry(-3.0281288001168467) q[3];
cx q[1],q[3];
ry(2.033679296015238) q[1];
ry(1.4515711202568102) q[3];
cx q[1],q[3];
ry(-1.5451164541576112) q[2];
ry(-2.3138064692623455) q[3];
cx q[2],q[3];
ry(0.9718977409451197) q[2];
ry(-2.80055158770904) q[3];
cx q[2],q[3];
ry(2.970096676368498) q[0];
ry(-0.37342516059714714) q[1];
cx q[0],q[1];
ry(-2.733376391605542) q[0];
ry(-1.7140840527132921) q[1];
cx q[0],q[1];
ry(3.025316119424812) q[0];
ry(-2.390519663990515) q[2];
cx q[0],q[2];
ry(-3.0400830281322775) q[0];
ry(-0.6938531268604837) q[2];
cx q[0],q[2];
ry(-0.9365902692869613) q[0];
ry(-2.720390360025585) q[3];
cx q[0],q[3];
ry(-1.6162544243070558) q[0];
ry(0.5301538689629949) q[3];
cx q[0],q[3];
ry(1.7056835606513492) q[1];
ry(-2.696121607182948) q[2];
cx q[1],q[2];
ry(-1.6363115432369575) q[1];
ry(2.5333536005977026) q[2];
cx q[1],q[2];
ry(1.5295977966930403) q[1];
ry(2.225477587190456) q[3];
cx q[1],q[3];
ry(-2.638807485885117) q[1];
ry(2.050173312738532) q[3];
cx q[1],q[3];
ry(-1.3820556163365652) q[2];
ry(-1.018045036865197) q[3];
cx q[2],q[3];
ry(-0.87805218417828) q[2];
ry(-1.744698636173272) q[3];
cx q[2],q[3];
ry(0.29804373786972244) q[0];
ry(0.3975037269582318) q[1];
cx q[0],q[1];
ry(-0.07264334460780456) q[0];
ry(0.4758510188277503) q[1];
cx q[0],q[1];
ry(-1.789894895257218) q[0];
ry(-2.3849116072363845) q[2];
cx q[0],q[2];
ry(2.6897359162171424) q[0];
ry(3.0605051743672638) q[2];
cx q[0],q[2];
ry(0.16368152051759388) q[0];
ry(0.9658921562664468) q[3];
cx q[0],q[3];
ry(-2.2249650398571186) q[0];
ry(-1.829228867347891) q[3];
cx q[0],q[3];
ry(-2.6511648954241003) q[1];
ry(-0.4564457991199724) q[2];
cx q[1],q[2];
ry(0.5746948776516836) q[1];
ry(2.81016038170724) q[2];
cx q[1],q[2];
ry(0.1672975183168974) q[1];
ry(-1.266421830602054) q[3];
cx q[1],q[3];
ry(-0.12757311184501052) q[1];
ry(-0.5391773329096301) q[3];
cx q[1],q[3];
ry(1.6335004927706485) q[2];
ry(-1.8423226945746918) q[3];
cx q[2],q[3];
ry(-0.5348798067177052) q[2];
ry(0.7327934999179027) q[3];
cx q[2],q[3];
ry(2.939305991014686) q[0];
ry(-1.9887939303146465) q[1];
ry(-3.0828029592087227) q[2];
ry(-1.1596580019358456) q[3];