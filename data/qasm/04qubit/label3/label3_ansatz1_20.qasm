OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(1.3028966030705023) q[0];
rz(1.5801841715961178) q[0];
ry(-2.084085913644234) q[1];
rz(-2.6772535114967417) q[1];
ry(1.0067166424762908) q[2];
rz(1.842194975509722) q[2];
ry(-0.9404688748071361) q[3];
rz(1.1064841077606609) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.528918535078919) q[0];
rz(-0.5307304361137276) q[0];
ry(-0.7387955392928234) q[1];
rz(0.10951519362885286) q[1];
ry(1.3699320155589954) q[2];
rz(-0.10623544394002717) q[2];
ry(0.48818286051014215) q[3];
rz(2.4625647357798504) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.2069417349920304) q[0];
rz(0.5728070319569518) q[0];
ry(2.6169015310710146) q[1];
rz(1.4434726867371754) q[1];
ry(-0.05650235340568788) q[2];
rz(2.7021862087834325) q[2];
ry(-0.4035014947617883) q[3];
rz(1.2470663165656086) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.86779803658608) q[0];
rz(-2.57320504830641) q[0];
ry(0.10301084467208987) q[1];
rz(-1.1295566126918422) q[1];
ry(-0.21806830857554665) q[2];
rz(-0.04705810435451882) q[2];
ry(-0.897044196771967) q[3];
rz(0.12381716075248672) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.25986171259076385) q[0];
rz(-1.2132519977971368) q[0];
ry(-1.7593798669704785) q[1];
rz(2.5879371350675395) q[1];
ry(0.5520464396731217) q[2];
rz(-1.6106735622035482) q[2];
ry(0.49466671701212084) q[3];
rz(2.501126339974073) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.2074665267529543) q[0];
rz(2.7577498332898784) q[0];
ry(-2.2771342065861475) q[1];
rz(2.05952134814071) q[1];
ry(2.7626073148095145) q[2];
rz(-2.210536503391565) q[2];
ry(-0.272583143488329) q[3];
rz(-0.7671503526707872) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.6984156404947304) q[0];
rz(2.5828818464581693) q[0];
ry(2.55049420118582) q[1];
rz(-0.058197102062217676) q[1];
ry(1.7786615927262304) q[2];
rz(-0.3526117337420888) q[2];
ry(0.5233136854135577) q[3];
rz(-0.5331532296681) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.343191736360587) q[0];
rz(2.617117168633545) q[0];
ry(-1.2574881308285686) q[1];
rz(-0.7812401103165776) q[1];
ry(-2.5064887144520003) q[2];
rz(1.1181945008175287) q[2];
ry(1.287803260429298) q[3];
rz(1.1370270911770408) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.4193117848336748) q[0];
rz(3.0173429418098143) q[0];
ry(-1.0214678686748693) q[1];
rz(1.9391607595722462) q[1];
ry(0.039746378369324376) q[2];
rz(-2.2651827951844754) q[2];
ry(0.42253906626974563) q[3];
rz(-1.1721130539467133) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.8056724160026327) q[0];
rz(2.307473710908577) q[0];
ry(-1.5480251641885683) q[1];
rz(2.301613796461776) q[1];
ry(0.9510222443460038) q[2];
rz(-0.3038741302533605) q[2];
ry(1.6704642960873042) q[3];
rz(2.384885478950754) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.0677426337066063) q[0];
rz(-1.8379693076452812) q[0];
ry(-1.0126874922911033) q[1];
rz(-1.4871454754307356) q[1];
ry(1.6235418082771922) q[2];
rz(2.917360253583379) q[2];
ry(-2.3466565197576656) q[3];
rz(-2.9803531201577216) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.4694692684876063) q[0];
rz(-2.6338361168555426) q[0];
ry(-1.7313989615262573) q[1];
rz(0.5942319405261031) q[1];
ry(-2.067094401032786) q[2];
rz(3.0884649481048974) q[2];
ry(2.6770403644146596) q[3];
rz(1.3471361745477541) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.0999766218421657) q[0];
rz(-0.17395028145902192) q[0];
ry(1.4892726327390422) q[1];
rz(-2.7426531603348225) q[1];
ry(-2.6397208096695013) q[2];
rz(-1.5759619883556875) q[2];
ry(-2.2051539844791437) q[3];
rz(-0.5613403252123278) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.8489699056497835) q[0];
rz(0.8656751957498069) q[0];
ry(-0.589158393800373) q[1];
rz(-2.157715272588216) q[1];
ry(-3.018358152216103) q[2];
rz(-2.1876055058744783) q[2];
ry(2.7555412399427284) q[3];
rz(0.7250227517580744) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.13987790997646243) q[0];
rz(-2.371404286513101) q[0];
ry(0.8937049506413759) q[1];
rz(0.6514633639340658) q[1];
ry(-2.559918510695126) q[2];
rz(2.306398426468602) q[2];
ry(-2.1090188228666147) q[3];
rz(1.1948091298871022) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.748099231764017) q[0];
rz(2.1218503369866504) q[0];
ry(2.217574918505136) q[1];
rz(-2.2482414584381796) q[1];
ry(0.34318503003459894) q[2];
rz(-1.4345376197941517) q[2];
ry(0.45997042087228923) q[3];
rz(-1.0191942702230035) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.945567292971348) q[0];
rz(2.395034813111685) q[0];
ry(-0.11241842188327134) q[1];
rz(-0.6083583109805516) q[1];
ry(-2.1182841374912416) q[2];
rz(0.7915144364834095) q[2];
ry(-1.9533926497131853) q[3];
rz(0.0921633137272222) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.0662092194286146) q[0];
rz(2.252719506690372) q[0];
ry(3.0480338211118925) q[1];
rz(-1.7961933048722984) q[1];
ry(-3.1216892182246947) q[2];
rz(-2.78217157305639) q[2];
ry(0.5303192406977573) q[3];
rz(2.2203965083540904) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(3.002152638559044) q[0];
rz(2.7640537481484753) q[0];
ry(-2.3633243724647994) q[1];
rz(2.914903381305771) q[1];
ry(0.6447488657983519) q[2];
rz(1.265676463360239) q[2];
ry(-2.1247674999994013) q[3];
rz(-1.0761857921530975) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.9095723509817866) q[0];
rz(1.3515051464926522) q[0];
ry(0.6559264846070603) q[1];
rz(-3.0399476051543886) q[1];
ry(-0.7296541531013494) q[2];
rz(-0.8415467856341223) q[2];
ry(2.594319588453151) q[3];
rz(-1.8143581465306182) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.5951715740233865) q[0];
rz(-2.6118684776864636) q[0];
ry(1.6184826028985664) q[1];
rz(-2.4953010139278047) q[1];
ry(2.372173777864028) q[2];
rz(3.084963735868726) q[2];
ry(1.271397714625705) q[3];
rz(2.0185102143394227) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.9312861632563607) q[0];
rz(-2.7640597869392383) q[0];
ry(2.3965057385138304) q[1];
rz(-0.5039195686400317) q[1];
ry(2.5583550158602297) q[2];
rz(-0.2794639601112093) q[2];
ry(0.8929019479822333) q[3];
rz(-2.0674863212824635) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.35924575989837937) q[0];
rz(0.21609604226983414) q[0];
ry(-0.531061828972904) q[1];
rz(-0.6615664047086095) q[1];
ry(0.9530396171691455) q[2];
rz(-2.4473402666660786) q[2];
ry(2.6837022452333255) q[3];
rz(-1.385351659909932) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.5348050731914888) q[0];
rz(2.933550347570115) q[0];
ry(2.7474357451163494) q[1];
rz(-2.9171349851530737) q[1];
ry(-3.0439787803653973) q[2];
rz(-2.6836852079348397) q[2];
ry(1.9914263403886965) q[3];
rz(2.9713586311625155) q[3];