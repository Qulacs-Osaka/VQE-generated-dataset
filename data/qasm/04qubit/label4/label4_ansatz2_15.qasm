OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-1.9560316332923886) q[0];
rz(0.8681210236180884) q[0];
ry(-1.9418214022400377) q[1];
rz(1.8274125238849113) q[1];
ry(-0.9915335995829692) q[2];
rz(-2.8671035830950684) q[2];
ry(-2.2062332942616427) q[3];
rz(-1.531816637029457) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-3.02394335899191) q[0];
rz(-0.2403519967232953) q[0];
ry(-0.4485286133659485) q[1];
rz(0.479007749401962) q[1];
ry(1.231227546653315) q[2];
rz(0.6766261374258731) q[2];
ry(-0.9401589975623539) q[3];
rz(-2.037504778455124) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-0.9001682139812983) q[0];
rz(1.3885268533695712) q[0];
ry(-0.031508704376044756) q[1];
rz(0.12048135276070984) q[1];
ry(1.9768433730112391) q[2];
rz(0.5223234274844213) q[2];
ry(-2.961109362312502) q[3];
rz(2.131602368370113) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.6165953958843824) q[0];
rz(-0.45960679109624325) q[0];
ry(1.9031535581719579) q[1];
rz(2.8164214492352198) q[1];
ry(-3.078032145333594) q[2];
rz(0.08626646333143843) q[2];
ry(-0.3773485881853018) q[3];
rz(-2.0459240709810778) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(2.903622271133214) q[0];
rz(-1.450513503051713) q[0];
ry(-3.0765438369230753) q[1];
rz(-0.80167053828727) q[1];
ry(2.1996938777495965) q[2];
rz(1.135451337089012) q[2];
ry(-1.0847125539868794) q[3];
rz(-0.6032623234832862) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.6474409849273712) q[0];
rz(-0.6500556136608557) q[0];
ry(2.7592875130978283) q[1];
rz(-1.9637089581011153) q[1];
ry(1.8222721115199425) q[2];
rz(1.2469617331982659) q[2];
ry(0.8662042896775937) q[3];
rz(2.596246644545438) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.5751138205298858) q[0];
rz(-0.512526369903969) q[0];
ry(3.0549363423277494) q[1];
rz(-0.9764480457109084) q[1];
ry(-2.627499527989666) q[2];
rz(2.500064931850211) q[2];
ry(2.231497497687462) q[3];
rz(1.4576775015614216) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.723699546740886) q[0];
rz(0.12754357886107914) q[0];
ry(1.5492206156154094) q[1];
rz(1.250376029307919) q[1];
ry(2.4636881308904512) q[2];
rz(1.5322482950329088) q[2];
ry(-2.4902044534076357) q[3];
rz(1.4352345997283207) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.5162288137025854) q[0];
rz(-1.8871490511942621) q[0];
ry(-1.9393347754862542) q[1];
rz(-1.8498361005761992) q[1];
ry(-1.2721239499015482) q[2];
rz(-0.9052499506389804) q[2];
ry(0.5534668551734905) q[3];
rz(2.0926305759837183) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.9798675803333623) q[0];
rz(2.4259320972174034) q[0];
ry(2.5196388182637937) q[1];
rz(-1.7405761235498123) q[1];
ry(-1.673499088697305) q[2];
rz(2.763332334482961) q[2];
ry(2.8168077159224754) q[3];
rz(1.9860213812191845) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.315088451729121) q[0];
rz(1.0215821431966186) q[0];
ry(-3.048972794843726) q[1];
rz(2.4433104309983507) q[1];
ry(0.45030053992460484) q[2];
rz(1.9061953823969349) q[2];
ry(3.008573990446691) q[3];
rz(-0.3226346490569609) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.3748962019649213) q[0];
rz(1.7987522039798511) q[0];
ry(1.98967974802591) q[1];
rz(0.9564751459986838) q[1];
ry(2.874885716839497) q[2];
rz(-2.873454776792031) q[2];
ry(1.079782880545018) q[3];
rz(1.5860217321793708) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-3.087103339649934) q[0];
rz(0.4561386310030704) q[0];
ry(-2.7664936238013635) q[1];
rz(1.583706282258069) q[1];
ry(-2.1569824231644708) q[2];
rz(0.5020903187737032) q[2];
ry(-0.13247527035072662) q[3];
rz(-0.0005094121513344874) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.73006622853222) q[0];
rz(-1.8399128135226457) q[0];
ry(1.561415320097336) q[1];
rz(-1.6335973318295487) q[1];
ry(-1.3033352481360554) q[2];
rz(2.372830838547143) q[2];
ry(-0.18850179274995382) q[3];
rz(-0.14045610197980185) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.8866629996574114) q[0];
rz(-2.2797637114971767) q[0];
ry(-0.8179278901290756) q[1];
rz(-0.591354403481798) q[1];
ry(-2.9559667677307226) q[2];
rz(-0.677300325061812) q[2];
ry(0.5709671710464354) q[3];
rz(0.18161172387506452) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.113308405318822) q[0];
rz(2.9973702190649574) q[0];
ry(2.0910222863746095) q[1];
rz(2.7442176994843153) q[1];
ry(2.2251679541533713) q[2];
rz(-1.6270104997476302) q[2];
ry(0.4762505207019424) q[3];
rz(-2.1506451293155187) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.0420863591637763) q[0];
rz(1.7908749838236213) q[0];
ry(-0.1444992441920494) q[1];
rz(0.3861593611714103) q[1];
ry(1.1850966006115038) q[2];
rz(1.0085211764409436) q[2];
ry(-2.0711869068716267) q[3];
rz(2.583229898142914) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.401897579570329) q[0];
rz(-1.67914727609296) q[0];
ry(2.2302773599867987) q[1];
rz(-2.658319643990447) q[1];
ry(3.006455778240749) q[2];
rz(0.5028556227508592) q[2];
ry(1.2640687040280278) q[3];
rz(1.3022643253882702) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.0252016746946513) q[0];
rz(-0.7205157487347358) q[0];
ry(2.700269989651898) q[1];
rz(1.8774051564306458) q[1];
ry(1.4534475498775903) q[2];
rz(2.63110178879522) q[2];
ry(-2.5466994252972355) q[3];
rz(-2.556080135633134) q[3];