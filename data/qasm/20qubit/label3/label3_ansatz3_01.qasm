OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(1.570116228520743) q[0];
rz(1.571861122051273) q[0];
ry(-1.5655382903236061) q[1];
rz(1.5756755083598937) q[1];
ry(0.38881022747367044) q[2];
rz(-1.553436777013106) q[2];
ry(-0.3980570412955622) q[3];
rz(1.5537696463373916) q[3];
ry(1.56354869934983) q[4];
rz(0.67330451991183) q[4];
ry(-1.5644762520348336) q[5];
rz(-0.6320499785393802) q[5];
ry(0.0005856437066240616) q[6];
rz(-0.9977236357103925) q[6];
ry(-0.0015711605605028822) q[7];
rz(1.635551164563146) q[7];
ry(-0.12988905333470058) q[8];
rz(-0.05695326939354087) q[8];
ry(-3.1360428492350048) q[9];
rz(-0.5579342689948588) q[9];
ry(1.590414945834123) q[10];
rz(3.1299269763593807) q[10];
ry(-3.113238553732729) q[11];
rz(0.9567162733277845) q[11];
ry(-0.22771855681929165) q[12];
rz(-0.9124901215633928) q[12];
ry(1.601651825732237) q[13];
rz(-0.019823016158361458) q[13];
ry(-2.5039307646786746) q[14];
rz(1.6443406096371878) q[14];
ry(1.5762005130115513) q[15];
rz(-0.03848020570306892) q[15];
ry(-0.92432557097748) q[16];
rz(-2.9326785633998917) q[16];
ry(-3.0591286917021168) q[17];
rz(-2.931090631224593) q[17];
ry(-0.8601651925761322) q[18];
rz(0.04726618765808066) q[18];
ry(1.5779397548655356) q[19];
rz(1.8305579073361817) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(-1.525408703827534) q[0];
rz(0.034378728260696434) q[0];
ry(1.8904347117023388) q[1];
rz(0.1955180498166751) q[1];
ry(-1.5685123309201003) q[2];
rz(2.8424418208133324) q[2];
ry(1.5715826721010986) q[3];
rz(0.8050574507560659) q[3];
ry(-1.7597174011032926) q[4];
rz(1.4504149030201532) q[4];
ry(-1.3367452897795302) q[5];
rz(1.4128244587387035) q[5];
ry(-0.00220276070323866) q[6];
rz(0.23580172775723052) q[6];
ry(-0.0003951799909339007) q[7];
rz(-0.27674069329394957) q[7];
ry(-1.577489965357506) q[8];
rz(-1.1872131436082916) q[8];
ry(1.5593418929197274) q[9];
rz(-2.738081165047957) q[9];
ry(1.552763616204074) q[10];
rz(-0.07341478056151617) q[10];
ry(-2.7788515414729322) q[11];
rz(0.8591585235702761) q[11];
ry(-0.05025560635691294) q[12];
rz(-0.750101856351548) q[12];
ry(2.805676039413764) q[13];
rz(-0.0004249071979325336) q[13];
ry(3.119003116418999) q[14];
rz(1.6014643682534038) q[14];
ry(0.04040882807848333) q[15];
rz(0.1321502430777333) q[15];
ry(1.8107634353521433) q[16];
rz(-0.416646000798342) q[16];
ry(1.438944181734482) q[17];
rz(0.471820686615418) q[17];
ry(-2.3789503344852485) q[18];
rz(1.222601582907397) q[18];
ry(0.6824427306356774) q[19];
rz(-0.5577932560651018) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(0.3018142798478829) q[0];
rz(0.05954677256298527) q[0];
ry(2.8395416513155993) q[1];
rz(0.17896588942597447) q[1];
ry(0.041886472891444086) q[2];
rz(1.9501241673435008) q[2];
ry(0.03493107165906384) q[3];
rz(-2.341398803369164) q[3];
ry(-1.5623234420579513) q[4];
rz(2.3925847924504717) q[4];
ry(-1.558130654713795) q[5];
rz(0.8490256448851509) q[5];
ry(0.2863795588903848) q[6];
rz(-0.10395933570934196) q[6];
ry(-0.31778564717699265) q[7];
rz(0.10492925039977476) q[7];
ry(0.04255354989130973) q[8];
rz(1.1899875394755588) q[8];
ry(-3.111518215666946) q[9];
rz(-2.7750377939675537) q[9];
ry(0.4425287647682694) q[10];
rz(1.518237977933917) q[10];
ry(2.7897815331797386) q[11];
rz(1.5012649458941598) q[11];
ry(1.578719925972159) q[12];
rz(3.1266879766572293) q[12];
ry(1.581056602127478) q[13];
rz(2.9219626169699735) q[13];
ry(1.6595192490827522) q[14];
rz(-0.4209990816051938) q[14];
ry(-0.19435657726228153) q[15];
rz(2.5373870634305757) q[15];
ry(-3.0984254467056234) q[16];
rz(-0.8943371881252432) q[16];
ry(3.131597700967029) q[17];
rz(-1.0331468206853849) q[17];
ry(-1.1653013885289498) q[18];
rz(-0.7010228062878185) q[18];
ry(1.6210141433646177) q[19];
rz(0.024150847331680936) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(-1.5645808876157412) q[0];
rz(1.5247301202448034) q[0];
ry(1.246336881968706) q[1];
rz(-1.252329108538147) q[1];
ry(-1.5703049121587278) q[2];
rz(-3.1297199957357913) q[2];
ry(-1.5720639177921472) q[3];
rz(-3.1199098221454626) q[3];
ry(3.108233739562544) q[4];
rz(-2.5462192131180004) q[4];
ry(0.019322662220766063) q[5];
rz(-1.552856710664249) q[5];
ry(2.762266189241557) q[6];
rz(-0.012453848133536559) q[6];
ry(0.39901229517560594) q[7];
rz(-0.029452798518994964) q[7];
ry(1.5634423578952232) q[8];
rz(3.0421342389047012) q[8];
ry(-1.5878118105664356) q[9];
rz(0.07705601989636054) q[9];
ry(-0.21078998111193492) q[10];
rz(-1.4186846532558146) q[10];
ry(3.1275637600658532) q[11];
rz(1.5800662713660438) q[11];
ry(-2.782911802928071) q[12];
rz(1.4310142717133203) q[12];
ry(3.127668817225989) q[13];
rz(1.9797842649064636) q[13];
ry(0.0025260656739538724) q[14];
rz(1.6613613489779093) q[14];
ry(-3.095572044815975) q[15];
rz(1.1626336710663834) q[15];
ry(0.3392033762339244) q[16];
rz(2.3892186866557097) q[16];
ry(-1.3894120945141362) q[17];
rz(0.3939750466923586) q[17];
ry(2.9793051430167026) q[18];
rz(-0.5692409744443336) q[18];
ry(1.3346212929892511) q[19];
rz(2.905406863046305) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(0.07796841096009623) q[0];
rz(-1.5920576692229915) q[0];
ry(-3.1085505871320582) q[1];
rz(-1.6528008459460999) q[1];
ry(1.488823374445723) q[2];
rz(1.5211692813713755) q[2];
ry(1.5414907455498028) q[3];
rz(1.1372788564318645) q[3];
ry(0.09026464727712066) q[4];
rz(-1.3585472573208788) q[4];
ry(3.1011055672418406) q[5];
rz(2.037348062990267) q[5];
ry(-1.6673044458109487) q[6];
rz(1.6357938703982018) q[6];
ry(-1.5958465429372621) q[7];
rz(1.2022768413728073) q[7];
ry(1.461393241964321) q[8];
rz(1.7080460376100952) q[8];
ry(1.6085204741294659) q[9];
rz(-1.8889967345981962) q[9];
ry(1.687288149539446) q[10];
rz(-1.3825735406487065) q[10];
ry(-1.6174914946470178) q[11];
rz(1.30589993250215) q[11];
ry(-1.70615471304622) q[12];
rz(0.1903387267205101) q[12];
ry(1.5208459457044199) q[13];
rz(1.3980570664979226) q[13];
ry(-1.6973114625741748) q[14];
rz(0.15806710198732077) q[14];
ry(2.183197087873501) q[15];
rz(-0.04808600419155517) q[15];
ry(-1.9489115005510866) q[16];
rz(2.1341886359130253) q[16];
ry(-1.4957653573783496) q[17];
rz(-1.6670204388861514) q[17];
ry(-0.22081266596706595) q[18];
rz(-3.110973439751722) q[18];
ry(-2.775253566398201) q[19];
rz(-0.23308374429449935) q[19];