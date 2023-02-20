OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(2.6437790810513757) q[0];
rz(2.735548648461896) q[0];
ry(-0.042424126370237694) q[1];
rz(1.5686024091663848) q[1];
ry(2.458900793059314) q[2];
rz(0.6424652129921214) q[2];
ry(2.2135715462355527) q[3];
rz(-1.302752345332121) q[3];
ry(3.1397106361462477) q[4];
rz(-1.779090955617566) q[4];
ry(-3.13721984894354) q[5];
rz(2.7920710675039393) q[5];
ry(1.54819616814782) q[6];
rz(-0.5471012661544812) q[6];
ry(-1.4538302206501879) q[7];
rz(0.1944589224963549) q[7];
ry(0.07957049983780906) q[8];
rz(1.6129212276238478) q[8];
ry(-3.0691903598418) q[9];
rz(1.5261427980415334) q[9];
ry(3.133828513396279) q[10];
rz(-2.9593936602972573) q[10];
ry(0.31422335794956435) q[11];
rz(-0.7843769653573873) q[11];
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
ry(0.6985037929110813) q[0];
rz(-1.9065854373072586) q[0];
ry(-1.168074400667889) q[1];
rz(-0.4990187345089909) q[1];
ry(1.9675716845760767) q[2];
rz(1.9475481534280943) q[2];
ry(-0.9127097744154415) q[3];
rz(-1.0023297959967596) q[3];
ry(-3.025793821892734) q[4];
rz(0.684564414110084) q[4];
ry(-0.023094184422689956) q[5];
rz(2.4761864952468406) q[5];
ry(-0.3877805989673613) q[6];
rz(0.029615370862547508) q[6];
ry(1.2424504627410926) q[7];
rz(2.969578335894595) q[7];
ry(-1.5617188027118865) q[8];
rz(-0.04140010800819027) q[8];
ry(1.5881122100723024) q[9];
rz(-0.06982757446556802) q[9];
ry(-0.06961149265125588) q[10];
rz(2.6005226738409903) q[10];
ry(-0.45906741763263614) q[11];
rz(0.18117486346544265) q[11];
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
ry(1.5755381754774085) q[0];
rz(1.6897164214126847) q[0];
ry(-1.4766799028172035) q[1];
rz(-0.4516981300816125) q[1];
ry(0.9563428531068263) q[2];
rz(-2.248500333754502) q[2];
ry(1.1546000769522315) q[3];
rz(-1.0138059021684933) q[3];
ry(-1.8707970119569604) q[4];
rz(-0.06350010443446294) q[4];
ry(2.5446256954500064) q[5];
rz(-0.056226743865414826) q[5];
ry(2.0595887217898934) q[6];
rz(1.84010353892175) q[6];
ry(-1.7791174725380172) q[7];
rz(0.6859866267115242) q[7];
ry(-1.4638138259240971) q[8];
rz(0.3924405293501385) q[8];
ry(1.6286822972579333) q[9];
rz(0.8194965094667651) q[9];
ry(0.7084649469321729) q[10];
rz(-1.4289779251371624) q[10];
ry(1.2270315509190315) q[11];
rz(2.0706196065025293) q[11];
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
ry(0.9303115411453636) q[0];
rz(2.4712021849707893) q[0];
ry(2.2680222428665497) q[1];
rz(-2.400193124056383) q[1];
ry(-0.357921779814924) q[2];
rz(2.725975367112354) q[2];
ry(2.995803125850898) q[3];
rz(-3.009198217306206) q[3];
ry(2.702985028428988) q[4];
rz(2.86188633268267) q[4];
ry(0.1847822959112311) q[5];
rz(-3.0276913479539256) q[5];
ry(3.086248737162645) q[6];
rz(-0.17462331974734238) q[6];
ry(1.954218724520694) q[7];
rz(-2.9603661813954383) q[7];
ry(-3.129808481498814) q[8];
rz(-1.141473990009407) q[8];
ry(-0.014923500406555767) q[9];
rz(0.7701736828269583) q[9];
ry(-1.4120712711592371) q[10];
rz(1.4912224235646583) q[10];
ry(1.9529741160068794) q[11];
rz(2.83401960991162) q[11];
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
ry(-2.5468589347109383) q[0];
rz(-0.6000502960641123) q[0];
ry(0.9883052818088822) q[1];
rz(0.613993345440676) q[1];
ry(1.04795199176338) q[2];
rz(-0.4955071075305453) q[2];
ry(-0.8929579619775871) q[3];
rz(0.7502119714598204) q[3];
ry(-3.127901608037808) q[4];
rz(-0.3128092172136965) q[4];
ry(-0.08300915693891575) q[5];
rz(0.7049551988320747) q[5];
ry(3.010584464166246) q[6];
rz(0.27644618580153857) q[6];
ry(-0.040122759346199466) q[7];
rz(1.3143867277666963) q[7];
ry(1.5711207102458484) q[8];
rz(0.2015062601913711) q[8];
ry(-1.5602953286277632) q[9];
rz(-2.7378732946012754) q[9];
ry(1.7130984964534035) q[10];
rz(-1.1303810722920637) q[10];
ry(1.44525901981033) q[11];
rz(-1.565522221102156) q[11];
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
ry(-1.5714223930730533) q[0];
rz(-0.6249928158498452) q[0];
ry(-0.6137860383918579) q[1];
rz(-0.5475884519130965) q[1];
ry(0.06367848551403643) q[2];
rz(-1.433533871233852) q[2];
ry(2.945230914306506) q[3];
rz(-1.9595913917189458) q[3];
ry(-0.006465114936689959) q[4];
rz(2.114844575185587) q[4];
ry(-3.1366401632657444) q[5];
rz(0.8524650578716013) q[5];
ry(1.6345420573865073) q[6];
rz(-0.48898114685754557) q[6];
ry(-1.4417079454252135) q[7];
rz(-0.43572875045523407) q[7];
ry(2.539171092006195) q[8];
rz(0.5523893453015462) q[8];
ry(2.500574446390057) q[9];
rz(-1.6524024598198328) q[9];
ry(1.0785249987213585) q[10];
rz(-0.7803734453487143) q[10];
ry(2.4399349705454276) q[11];
rz(-1.5702628561612597) q[11];
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
ry(-2.06691063468166) q[0];
rz(-0.7519935121423803) q[0];
ry(1.1633212688913215) q[1];
rz(1.5699993347956998) q[1];
ry(-1.2269739512203435) q[2];
rz(1.835620939523299) q[2];
ry(1.3444829446195004) q[3];
rz(-0.7983411507899435) q[3];
ry(-0.11397930170602016) q[4];
rz(1.7515638941701155) q[4];
ry(0.6863540624329452) q[5];
rz(-1.3276506567330526) q[5];
ry(-0.02281163370643124) q[6];
rz(-0.12386260879807498) q[6];
ry(3.091843948256294) q[7];
rz(-0.7340181185296349) q[7];
ry(1.9278528509243935) q[8];
rz(0.5886644204932345) q[8];
ry(0.30799685259634185) q[9];
rz(-2.2860728962577204) q[9];
ry(0.973442851650514) q[10];
rz(0.8263321438420997) q[10];
ry(1.5248436734342288) q[11];
rz(0.6625167390157093) q[11];
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
ry(-0.5841535489007933) q[0];
rz(-0.3247868433839452) q[0];
ry(2.8770966229329056) q[1];
rz(-1.947053846618429) q[1];
ry(-2.570802080594682) q[2];
rz(-1.4764004989964659) q[2];
ry(-0.80258166620318) q[3];
rz(-1.64180129255686) q[3];
ry(0.20872302899990458) q[4];
rz(-1.7927939644129296) q[4];
ry(0.0469714511377331) q[5];
rz(-2.5755993258439687) q[5];
ry(-0.008030124403260232) q[6];
rz(-2.555200871076747) q[6];
ry(-3.133745838025377) q[7];
rz(1.1290604999668394) q[7];
ry(-1.5028167551570042) q[8];
rz(-1.4050984760235228) q[8];
ry(1.6729017405840236) q[9];
rz(1.3625908124104547) q[9];
ry(-1.3157818468259546) q[10];
rz(-2.9606981788948477) q[10];
ry(1.2110801667529203) q[11];
rz(2.2453741416034623) q[11];
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
ry(0.3963888327461235) q[0];
rz(-0.9402796845001548) q[0];
ry(-2.827381718916211) q[1];
rz(1.0883916576514536) q[1];
ry(0.7995870828620539) q[2];
rz(1.9645023730044555) q[2];
ry(-2.2668367888528094) q[3];
rz(2.141483063109442) q[3];
ry(-3.0644523202487823) q[4];
rz(1.8676458626523615) q[4];
ry(3.0404835389133127) q[5];
rz(0.44418382239792903) q[5];
ry(-3.140566976033901) q[6];
rz(-0.9695043009094628) q[6];
ry(0.006015121290035452) q[7];
rz(-0.05319473350078387) q[7];
ry(2.5626942602770644) q[8];
rz(-3.0411362111897136) q[8];
ry(1.5751922754699805) q[9];
rz(0.003159290536055792) q[9];
ry(1.5689997982250898) q[10];
rz(-1.9609707040872415) q[10];
ry(1.1580153037176935) q[11];
rz(1.84068962366484) q[11];
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
ry(-2.081171964285214) q[0];
rz(0.1582665384012616) q[0];
ry(1.4177299657129987) q[1];
rz(0.751314510741646) q[1];
ry(-2.173188266819798) q[2];
rz(-2.953898539863711) q[2];
ry(-1.1994083175296666) q[3];
rz(2.775146099651289) q[3];
ry(3.035186438387982) q[4];
rz(-1.731235256066613) q[4];
ry(3.09846629121078) q[5];
rz(-2.1365549444072145) q[5];
ry(-0.017588527481938065) q[6];
rz(0.25778700463632526) q[6];
ry(0.009617011976803232) q[7];
rz(2.6827828125275706) q[7];
ry(0.9298051673539138) q[8];
rz(1.529959319331867) q[8];
ry(-1.7691926451198443) q[9];
rz(1.5209102176810212) q[9];
ry(1.6913669340996558) q[10];
rz(-1.4195645539367607) q[10];
ry(-2.8513588133855934) q[11];
rz(-2.7329961272404444) q[11];
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
ry(3.051985770799503) q[0];
rz(-2.086465501240079) q[0];
ry(1.7697763926280066) q[1];
rz(-2.251421684829774) q[1];
ry(1.5047409934255773) q[2];
rz(-2.7675573203023847) q[2];
ry(-1.6249041963862538) q[3];
rz(0.44563573246807847) q[3];
ry(0.01365495461850319) q[4];
rz(-2.992315934589796) q[4];
ry(3.052389597684043) q[5];
rz(-1.5986434098499656) q[5];
ry(-3.1394715644187916) q[6];
rz(-1.5820407561631256) q[6];
ry(-6.632115073124112e-05) q[7];
rz(-1.5485208664202976) q[7];
ry(-1.4328519730920644) q[8];
rz(1.055274809606346) q[8];
ry(-1.572337722675889) q[9];
rz(1.5335914128066461) q[9];
ry(2.311978185727119) q[10];
rz(1.2474298254508323) q[10];
ry(1.1166736065845624) q[11];
rz(2.9233217566560783) q[11];
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
ry(-1.8538579290960342) q[0];
rz(1.2453820542364964) q[0];
ry(-2.5036859669089138) q[1];
rz(1.8254097664330908) q[1];
ry(-1.2380541663691293) q[2];
rz(-2.729988866132351) q[2];
ry(2.647801921340522) q[3];
rz(-0.3579265101393725) q[3];
ry(-2.158315937522558) q[4];
rz(1.2442749966683062) q[4];
ry(0.17755791912311913) q[5];
rz(-1.267715512095096) q[5];
ry(-0.3935761187231881) q[6];
rz(-0.3038035900263747) q[6];
ry(1.0958529495643532) q[7];
rz(2.3502675808830733) q[7];
ry(1.8591392353410328) q[8];
rz(-2.920797289761144) q[8];
ry(2.0500811530104732) q[9];
rz(1.5144663424446811) q[9];
ry(1.5105110107902726) q[10];
rz(-2.8063167832696116) q[10];
ry(-2.594153183921224) q[11];
rz(-0.053131430164216724) q[11];
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
ry(-2.4980597586425435) q[0];
rz(-0.17237647153838329) q[0];
ry(0.12554097647196155) q[1];
rz(-2.726846186285518) q[1];
ry(2.744482745492118) q[2];
rz(1.3669650951833698) q[2];
ry(-0.2755271574637419) q[3];
rz(0.4646460915153545) q[3];
ry(-3.1385969646526903) q[4];
rz(2.629214719978223) q[4];
ry(-3.1393930966814345) q[5];
rz(-0.4140220102716974) q[5];
ry(-3.0367551030628364) q[6];
rz(0.30691632899645405) q[6];
ry(-3.1178553684454116) q[7];
rz(-2.8371890445553345) q[7];
ry(1.4644669588337453) q[8];
rz(-2.877252061084095) q[8];
ry(-0.35382048217781664) q[9];
rz(-2.8058959182726357) q[9];
ry(0.5437758800528298) q[10];
rz(0.3543646361941121) q[10];
ry(-1.4188184323965107) q[11];
rz(1.156396461717926) q[11];
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
ry(-2.989523209356464) q[0];
rz(-0.7094746527567857) q[0];
ry(0.3389012414839927) q[1];
rz(-0.9853459196180129) q[1];
ry(1.6358479940059527) q[2];
rz(1.8656217633151935) q[2];
ry(-1.1591497035141092) q[3];
rz(-1.2651095727571466) q[3];
ry(0.038422943494984985) q[4];
rz(2.7375940561640584) q[4];
ry(-3.004755369007739) q[5];
rz(-3.0401414614501236) q[5];
ry(-0.30563533405733345) q[6];
rz(2.9016094627208786) q[6];
ry(-2.909833205567029) q[7];
rz(-3.0199279789668636) q[7];
ry(3.130022824971838) q[8];
rz(-1.7141313069867679) q[8];
ry(0.06550932982236013) q[9];
rz(-0.08654360173455271) q[9];
ry(-0.1243698377640019) q[10];
rz(-1.1211653888110877) q[10];
ry(-0.029118953831574666) q[11];
rz(-2.2831878228177325) q[11];
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
ry(-0.59576033667964) q[0];
rz(-0.8629278504152077) q[0];
ry(-2.1470182393762993) q[1];
rz(1.4257572568830594) q[1];
ry(3.0263774484757437) q[2];
rz(-0.2376269199260941) q[2];
ry(0.30118467201778) q[3];
rz(0.7564110970827426) q[3];
ry(3.1322109293716696) q[4];
rz(2.492765188915322) q[4];
ry(-0.0030190500182378996) q[5];
rz(-2.6086087768850548) q[5];
ry(3.028035935934932) q[6];
rz(1.7468685245124926) q[6];
ry(-0.02825963831824344) q[7];
rz(-0.7039119861293763) q[7];
ry(-0.22849422005318498) q[8];
rz(0.34734764075860264) q[8];
ry(-0.3613477209058881) q[9];
rz(2.9048191872406686) q[9];
ry(0.2231388782114302) q[10];
rz(1.271830106907605) q[10];
ry(3.1377105538295607) q[11];
rz(-0.07950954678395955) q[11];
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
ry(-1.1559641994575833) q[0];
rz(-2.3516065479737263) q[0];
ry(-0.8870016166042043) q[1];
rz(2.089010582381376) q[1];
ry(-1.3960678813123906) q[2];
rz(-0.715045387292923) q[2];
ry(-2.2464635366950514) q[3];
rz(-0.4685777794481642) q[3];
ry(-1.6717769109700331) q[4];
rz(-0.6960406565687003) q[4];
ry(-2.9676736453593526) q[5];
rz(1.7740661518633256) q[5];
ry(1.3986080160059684) q[6];
rz(0.5332471791983008) q[6];
ry(-1.4079891654614254) q[7];
rz(-2.7681124787423594) q[7];
ry(-1.5579497518463885) q[8];
rz(-2.990430157007421) q[8];
ry(0.09308067123172961) q[9];
rz(1.9618943660826504) q[9];
ry(3.0699112219965023) q[10];
rz(-3.1386746661388063) q[10];
ry(3.120089072935463) q[11];
rz(1.141567080894256) q[11];