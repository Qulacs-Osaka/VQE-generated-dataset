OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(1.570796323248155) q[0];
rz(1.915210015254713e-06) q[0];
ry(3.087567508330835e-07) q[1];
rz(2.3724679598343164) q[1];
ry(1.5707967405028542) q[2];
rz(5.158241126086783e-09) q[2];
ry(0.003983620108579283) q[3];
rz(1.5028723598933738) q[3];
ry(1.5707957239602472) q[4];
rz(1.0923267188545083e-06) q[4];
ry(1.55008221675852) q[5];
rz(1.17117099959827) q[5];
ry(-1.5707968430703554) q[6];
rz(0.5002968583605489) q[6];
ry(-3.003839093174928) q[7];
rz(1.5712664914833203) q[7];
ry(-3.1415925757426377) q[8];
rz(2.6578048579315023) q[8];
ry(-1.5560691566080531) q[9];
rz(1.5707814251584793) q[9];
ry(3.1415924312665475) q[10];
rz(-2.6310347082662977) q[10];
ry(1.570685940620898) q[11];
rz(1.5687598354647774) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(1.0897159161501921) q[0];
rz(-1.5707985759124155) q[0];
ry(3.141590783027308) q[1];
rz(0.7529855947514151) q[1];
ry(-0.8268882664729373) q[2];
rz(2.9917766652962756) q[2];
ry(0.6880780732286164) q[3];
rz(1.5703263081002836) q[3];
ry(1.6260618227298584) q[4];
rz(-1.757558114916002) q[4];
ry(0.16877086040976944) q[5];
rz(0.4048322236796329) q[5];
ry(9.947880129423936e-06) q[6];
rz(1.0704964979989557) q[6];
ry(-3.1023505324301928) q[7];
rz(1.585038284945801) q[7];
ry(-3.478653907059445e-07) q[8];
rz(-0.3561834538109352) q[8];
ry(-1.5706668466610818) q[9];
rz(3.027585789793736) q[9];
ry(3.141592641965405) q[10];
rz(0.29685723411576537) q[10];
ry(-1.5717918023066406) q[11];
rz(-2.8861006631227313) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-1.0646368722434965) q[0];
rz(-3.1415918994569596) q[0];
ry(1.5707971755467456) q[1];
rz(1.5707944216466627) q[1];
ry(-6.038939197594573e-07) q[2];
rz(-1.4209798658937294) q[2];
ry(2.643075078611084) q[3];
rz(-0.17578908989944253) q[3];
ry(-3.1415708362196497) q[4];
rz(1.3840342323345654) q[4];
ry(-1.5717313361899858) q[5];
rz(1.5688817684456868) q[5];
ry(2.634483350050944) q[6];
rz(-1.0657061296616428) q[6];
ry(0.01816805130849503) q[7];
rz(-0.16694280933814407) q[7];
ry(-3.1415918658209634) q[8];
rz(1.631447333427836) q[8];
ry(-3.141448793224122) q[9];
rz(0.22941238155575674) q[9];
ry(-3.141592338807336) q[10];
rz(1.9956252624720694) q[10];
ry(0.00631994698799776) q[11];
rz(-1.6686321671185034) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-1.570796404569017) q[0];
rz(2.3548473545853557) q[0];
ry(-2.119656818423277) q[1];
rz(-1.694822515829932) q[1];
ry(1.570797431730817) q[2];
rz(1.3037508493088779) q[2];
ry(-3.1415915912257293) q[3];
rz(-1.2576995420855286) q[3];
ry(-0.5197047445353192) q[4];
rz(1.5707964952452673) q[4];
ry(-3.869089812980064e-06) q[5];
rz(1.5810738479450626) q[5];
ry(3.141591323315672) q[6];
rz(2.0757175703938033) q[6];
ry(3.141592470997814) q[7];
rz(-2.5078385635841762) q[7];
ry(-3.1415926301625006) q[8];
rz(-2.7715334979483903) q[8];
ry(3.1415926428062773) q[9];
rz(-2.79972965946544) q[9];
ry(1.4365426158824653e-07) q[10];
rz(-2.4786706748445924) q[10];
ry(-3.141592432347779) q[11];
rz(-1.8448163935104493) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-1.570796513724475) q[0];
rz(1.7409465221673879) q[0];
ry(-9.529519795451136e-05) q[1];
rz(-1.1707244659709932) q[1];
ry(1.5707953491067563) q[2];
rz(0.28141578570453507) q[2];
ry(6.547107179954992e-06) q[3];
rz(0.2850024250040928) q[3];
ry(1.5340282001202201) q[4];
rz(1.5680827892491473) q[4];
ry(-1.5707849332305326) q[5];
rz(1.1265196910719744) q[5];
ry(-0.03677907621815368) q[6];
rz(1.499629239114289) q[6];
ry(2.795471143457462e-08) q[7];
rz(-0.3746784110830192) q[7];
ry(1.5707966653855596) q[8];
rz(1.570796335721151) q[8];
ry(3.70465727258546e-07) q[9];
rz(1.6197448080404513) q[9];
ry(-1.5707963950657273) q[10];
rz(1.5707963782241023) q[10];
ry(1.5707960740666156) q[11];
rz(2.821029032416088) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(1.67393091388135e-07) q[0];
rz(-1.7409467197282247) q[0];
ry(4.6499398020927174e-07) q[1];
rz(1.2947486665934238) q[1];
ry(3.1415925520946306) q[2];
rz(-2.8601780772136283) q[2];
ry(8.346353540744644e-06) q[3];
rz(-2.3439380305820197) q[3];
ry(6.374042785761702e-05) q[4];
rz(-3.138878576936831) q[4];
ry(8.107269622215998e-07) q[5];
rz(-2.696848998822228) q[5];
ry(6.225745029997398e-05) q[6];
rz(0.07133899921146726) q[6];
ry(-3.1415915348006718) q[7];
rz(0.4116380054758677) q[7];
ry(-1.5707963516745265) q[8];
rz(-0.5289081121358983) q[8];
ry(-1.5707980782713005) q[9];
rz(-3.0518621180230707) q[9];
ry(-1.5707963342027949) q[10];
rz(-1.0911508817542934) q[10];
ry(1.0146411765586144e-05) q[11];
rz(-0.9903867803135761) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-1.5707948542807693) q[0];
rz(1.3776653833413697) q[0];
ry(1.459271635541822) q[1];
rz(2.1256881778618983) q[1];
ry(-1.5707965597978844) q[2];
rz(-0.19313139356049636) q[2];
ry(1.5707956679094057) q[3];
rz(2.125688344441256) q[3];
ry(-1.5707976411634301) q[4];
rz(-1.8870642071937551) q[4];
ry(-2.991285351488024) q[5];
rz(2.1261577834943806) q[5];
ry(1.57079674590952) q[6];
rz(2.8252542783718546) q[6];
ry(-1.570796608468697) q[7];
rz(2.125686467712409) q[7];
ry(-3.1415920665439288) q[8];
rz(-1.9042042657751477) q[8];
ry(1.570796451261761) q[9];
rz(0.5548897390450003) q[9];
ry(3.141592287797689) q[10];
rz(2.245941717810724) q[10];
ry(-1.0924304568387844e-06) q[11];
rz(-2.846549267638922) q[11];