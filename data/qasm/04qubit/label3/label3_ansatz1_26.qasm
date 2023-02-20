OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(1.696352464953306) q[0];
rz(2.540552434640572) q[0];
ry(-1.6765463705339014) q[1];
rz(-0.80000424632604) q[1];
ry(2.545444800710801) q[2];
rz(-1.3377874850787677) q[2];
ry(0.9995306378713558) q[3];
rz(1.8819780899004401) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.675578840780142) q[0];
rz(0.7351749660014004) q[0];
ry(2.4578018340150862) q[1];
rz(1.0306943890458018) q[1];
ry(1.3236139374778784) q[2];
rz(0.02265960500997084) q[2];
ry(-1.2837772218363215) q[3];
rz(-2.861663006218548) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.0190765945054536) q[0];
rz(0.5330667693316894) q[0];
ry(-0.9438809743650571) q[1];
rz(0.694795160702613) q[1];
ry(-1.0358676217127967) q[2];
rz(-2.141341401557492) q[2];
ry(-0.9494944725080545) q[3];
rz(-2.3679036768416917) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.11179967842207356) q[0];
rz(2.3470363154536416) q[0];
ry(0.7100690223453394) q[1];
rz(0.8108405438189008) q[1];
ry(1.8098135528370172) q[2];
rz(0.09630796690847543) q[2];
ry(2.875451736116638) q[3];
rz(2.3668328806298176) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.36120299150892143) q[0];
rz(-0.024826458126923967) q[0];
ry(-1.9210238068683194) q[1];
rz(-2.313407025582112) q[1];
ry(2.7726142831902227) q[2];
rz(-2.4474257730270077) q[2];
ry(0.006846684118861467) q[3];
rz(-0.616128803947779) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.7880789749404552) q[0];
rz(0.8963807664616468) q[0];
ry(2.193079443634719) q[1];
rz(2.461520772401762) q[1];
ry(-0.8421865030809148) q[2];
rz(3.047857147255751) q[2];
ry(-2.9622080477736916) q[3];
rz(-0.8420949962928379) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.527910162420884) q[0];
rz(-1.435492866568401) q[0];
ry(3.0281729293916206) q[1];
rz(-0.5182630345150798) q[1];
ry(-0.12052974858834747) q[2];
rz(3.0725547144126746) q[2];
ry(-2.3095809017755426) q[3];
rz(0.8620806360664572) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.3213011481196015) q[0];
rz(-0.6624234100307088) q[0];
ry(2.2341738726233142) q[1];
rz(1.4056896197758946) q[1];
ry(0.08182590081016983) q[2];
rz(1.4983818677459437) q[2];
ry(2.6587507796180794) q[3];
rz(0.5480819227941244) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.1416869714708697) q[0];
rz(-2.0849438458278913) q[0];
ry(1.9912901042077715) q[1];
rz(1.8005321837008033) q[1];
ry(-3.036273430431181) q[2];
rz(1.4834999245398228) q[2];
ry(-0.4483775131152692) q[3];
rz(-0.8521591233467712) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.651242284760138) q[0];
rz(1.0857144067251148) q[0];
ry(-2.5924845795478686) q[1];
rz(2.2272922205721097) q[1];
ry(-2.708430411101691) q[2];
rz(-0.9984902378974515) q[2];
ry(1.1981241464295262) q[3];
rz(0.9764302992689132) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.1459366050314217) q[0];
rz(-1.9324880734728311) q[0];
ry(2.263968317548832) q[1];
rz(-1.1546789644756437) q[1];
ry(-1.5681924891432293) q[2];
rz(2.308643354457029) q[2];
ry(-3.0150051748265208) q[3];
rz(2.101527704931528) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.897433169514227) q[0];
rz(-3.1337635688699974) q[0];
ry(2.7723033673415163) q[1];
rz(-0.8278763366299273) q[1];
ry(-1.7255823044419683) q[2];
rz(0.8583866422584651) q[2];
ry(1.9794290454192585) q[3];
rz(1.6462235833275605) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.353460915530407) q[0];
rz(2.5934569247729367) q[0];
ry(-2.6789941380896733) q[1];
rz(2.6481804880820436) q[1];
ry(2.161285441776403) q[2];
rz(0.8662347937736454) q[2];
ry(-2.591020060363072) q[3];
rz(-1.1515168348751395) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.0349218242582294) q[0];
rz(-2.819156862062394) q[0];
ry(2.4382264590709943) q[1];
rz(-1.6132209837890912) q[1];
ry(1.1825601122816687) q[2];
rz(-0.5191486475459234) q[2];
ry(-0.6867743290902517) q[3];
rz(-1.7987248824039492) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.8220018928489203) q[0];
rz(0.867042251615839) q[0];
ry(-1.1518851740039904) q[1];
rz(-0.7280885778282444) q[1];
ry(0.7496767732448215) q[2];
rz(-0.8650982948241563) q[2];
ry(-2.758792396676093) q[3];
rz(0.8352624254346432) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.5793576536515412) q[0];
rz(1.7248981516254327) q[0];
ry(-0.1771398331101084) q[1];
rz(1.3063674279962925) q[1];
ry(3.0515550646389724) q[2];
rz(3.035943376709422) q[2];
ry(2.861511193953959) q[3];
rz(-0.4743757260816858) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.11990556933848363) q[0];
rz(2.3351418692078973) q[0];
ry(0.9271508491490827) q[1];
rz(-2.128615489275452) q[1];
ry(-2.5303544451682276) q[2];
rz(-0.4731960707238976) q[2];
ry(1.3210999197527946) q[3];
rz(-2.8420766352737274) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.7106901921554176) q[0];
rz(0.7065590230374399) q[0];
ry(2.855785469393997) q[1];
rz(-2.6528532777133003) q[1];
ry(2.9820335501105344) q[2];
rz(2.6778500480969822) q[2];
ry(2.0892610854039875) q[3];
rz(1.15075328631883) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.687801224889111) q[0];
rz(1.8887963833730532) q[0];
ry(1.4595314056862698) q[1];
rz(2.613779849245704) q[1];
ry(-1.1385294555338579) q[2];
rz(-2.3694611075052343) q[2];
ry(-1.0674531304836545) q[3];
rz(-2.37594373809427) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.806717134255436) q[0];
rz(-2.0144507371356757) q[0];
ry(2.8203233236451073) q[1];
rz(2.4635248192441863) q[1];
ry(-2.2404250371584657) q[2];
rz(-0.3666243473745085) q[2];
ry(2.401296106138736) q[3];
rz(0.38458583822185327) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.21097852578849072) q[0];
rz(-0.41568532979281964) q[0];
ry(-2.1672090244635713) q[1];
rz(-0.9679036237093506) q[1];
ry(-2.9308766440506577) q[2];
rz(0.2852472305674281) q[2];
ry(-1.255020128084988) q[3];
rz(1.714309218285024) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.4610696534137686) q[0];
rz(1.3413146892441143) q[0];
ry(-2.7292237067971277) q[1];
rz(1.4387500080894267) q[1];
ry(-1.3432146746031561) q[2];
rz(-0.9199058646529797) q[2];
ry(2.4889566200622877) q[3];
rz(-1.8042076918306122) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-3.050668842015441) q[0];
rz(-1.4202942533261407) q[0];
ry(-0.3865703185958216) q[1];
rz(0.04392188987447021) q[1];
ry(-0.978814488420837) q[2];
rz(1.560454515153266) q[2];
ry(-2.7621047197154653) q[3];
rz(0.959170181861388) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.2130668065997945) q[0];
rz(-1.1403576299404004) q[0];
ry(-0.6856901428633158) q[1];
rz(-2.568765877090614) q[1];
ry(-3.11815846878863) q[2];
rz(-2.990930242507572) q[2];
ry(-1.8214141975378704) q[3];
rz(0.009279304328965843) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.7049069253772346) q[0];
rz(-1.7716338926519948) q[0];
ry(-2.592514078844851) q[1];
rz(-0.2565416988298382) q[1];
ry(-2.830413020057219) q[2];
rz(2.84601076891243) q[2];
ry(-2.9366893105263014) q[3];
rz(-1.805051449753277) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.6352395392959012) q[0];
rz(-2.2793862043708764) q[0];
ry(1.7871361323005477) q[1];
rz(2.3447289965431586) q[1];
ry(-1.0540972536501991) q[2];
rz(0.20232833734971134) q[2];
ry(-1.3904460352555779) q[3];
rz(2.2122026983764833) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.2274377440087436) q[0];
rz(-0.7403255061547793) q[0];
ry(-0.8952856352891203) q[1];
rz(-0.9952848313866828) q[1];
ry(-1.45430344912245) q[2];
rz(-1.362437201211115) q[2];
ry(0.910666905759153) q[3];
rz(-0.02533880381363221) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.2857156563970738) q[0];
rz(-1.2170169024641337) q[0];
ry(-0.3021838551662763) q[1];
rz(0.20500267402481323) q[1];
ry(-1.6488199557336065) q[2];
rz(1.366998889477914) q[2];
ry(-1.1575222328331183) q[3];
rz(-2.955476710929911) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.0632764273935988) q[0];
rz(1.7595056814990144) q[0];
ry(-0.1578807468598109) q[1];
rz(2.71339226260587) q[1];
ry(-1.2145050691012969) q[2];
rz(1.2737786780155191) q[2];
ry(-2.96373209754573) q[3];
rz(2.2354017930758645) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.406498932156553) q[0];
rz(-3.034680917524865) q[0];
ry(1.0820971088598175) q[1];
rz(-0.6613001009262067) q[1];
ry(3.1090289050571616) q[2];
rz(2.9626942693517715) q[2];
ry(-2.9857287548204385) q[3];
rz(-0.6054250899313379) q[3];