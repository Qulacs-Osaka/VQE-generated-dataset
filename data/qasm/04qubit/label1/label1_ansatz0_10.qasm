OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.41227375199442423) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.3027930989459282) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.013956461116374774) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.20356118400513942) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.13590267166823275) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.06588882003841341) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.2942562982155483) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.3803794763940722) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.15454373286927994) q[3];
cx q[2],q[3];
rz(0.02645118929997259) q[0];
rz(-0.0025323546192082087) q[1];
rz(-0.03320097459801344) q[2];
rz(0.07158134044676293) q[3];
rx(-0.40843023759648417) q[0];
rx(-0.3934630418215683) q[1];
rx(-0.8439546270854842) q[2];
rx(-0.3117850569839727) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.34526851632515215) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.28507397238295606) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.0952225811564202) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.18679945824974523) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.10613322306771053) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.0673027476811775) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.28335320565717925) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.5303566012044317) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.013650823245874326) q[3];
cx q[2],q[3];
rz(0.046999038558756236) q[0];
rz(-0.005235452894706829) q[1];
rz(0.2334713958541866) q[2];
rz(0.06234978500368107) q[3];
rx(-0.4033367401417426) q[0];
rx(-0.42499175776114856) q[1];
rx(-0.8775990073961554) q[2];
rx(-0.2888420687305251) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.44688478277339044) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.2722835576216012) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.05941172767936727) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.18982393749146503) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.10243641754591308) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.12716518406382674) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.3186445647230747) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.5687945333783282) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.17020659854432874) q[3];
cx q[2],q[3];
rz(0.0642063111554196) q[0];
rz(-0.06642085582667216) q[1];
rz(0.33918842297993534) q[2];
rz(0.051256212390454595) q[3];
rx(-0.45729190498004624) q[0];
rx(-0.43299246395851126) q[1];
rx(-0.8867361842972883) q[2];
rx(-0.28036547836704184) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.5652355090099095) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.2614324138527762) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.07707744892816709) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.09083582958190632) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.20051795502131628) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.2974693871778871) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.22124764094439917) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.30563464526699935) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.0005248959904879993) q[3];
cx q[2],q[3];
rz(-0.002980800122795574) q[0];
rz(-0.20673427038320033) q[1];
rz(0.270578899714202) q[2];
rz(0.03641330859346028) q[3];
rx(-0.4361619881278446) q[0];
rx(-0.37176305844422963) q[1];
rx(-0.8563076536629055) q[2];
rx(-0.2705029162624114) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.546654081264898) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.4335684593631623) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.16203487066652536) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.0118302704512986) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.11896632754313244) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.1427055394503654) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.3005751852154289) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.023377262251201042) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.14162039768539025) q[3];
cx q[2],q[3];
rz(-0.0673534061660986) q[0];
rz(-0.19286121535901957) q[1];
rz(0.16594923754163834) q[2];
rz(0.11407787217579353) q[3];
rx(-0.49682001859454394) q[0];
rx(-0.2691684382541137) q[1];
rx(-0.8425503467478209) q[2];
rx(-0.21141661684181054) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.4577355787071872) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.4597993349969498) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.35277770086305965) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.1154663696298097) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.17457515227458573) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.3756465041109419) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.319640012820289) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.14162107428337134) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.27377592954850755) q[3];
cx q[2],q[3];
rz(0.0909157378293782) q[0];
rz(0.06256649710460051) q[1];
rz(-0.06731037923372839) q[2];
rz(0.25506668624171713) q[3];
rx(-0.5008075958483493) q[0];
rx(-0.07038106774767511) q[1];
rx(-0.6349055937293185) q[2];
rx(-0.3136778935494219) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.25814293808396827) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.5061184899261716) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.4695998410596737) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.22860286694931808) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.3259370790698408) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.11227282834515072) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.21400216096260008) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.30756581212540696) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.5626424132370487) q[3];
cx q[2],q[3];
rz(0.01146886263190145) q[0];
rz(0.1315968314030418) q[1];
rz(-0.21133317082546565) q[2];
rz(0.30139158860413584) q[3];
rx(-0.40562612191546893) q[0];
rx(-0.2786685510802175) q[1];
rx(-0.5759603993312484) q[2];
rx(-0.3532385602981863) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.12115910419389878) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.4412544794907611) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.43410521131902746) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.1833164234556527) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.10201891761850616) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.1334623403938322) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.12031940641758115) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.35172147354698075) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.6716306560314346) q[3];
cx q[2],q[3];
rz(-0.014636007207761537) q[0];
rz(0.038880664296346006) q[1];
rz(-0.09826217839812536) q[2];
rz(0.2254729742139607) q[3];
rx(-0.4222026785637191) q[0];
rx(-0.3996368723674136) q[1];
rx(-0.5542571759192386) q[2];
rx(-0.49564349785681966) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.10205015130009461) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.3493237534018401) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.3157569843937196) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.23280390185743793) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.20242443338622793) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.06538845574913545) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.02938210070608363) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.3643709584157314) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.5474407399209475) q[3];
cx q[2],q[3];
rz(-0.12882194235536323) q[0];
rz(0.0056966579664553645) q[1];
rz(-0.18213112802380793) q[2];
rz(0.16744864552241173) q[3];
rx(-0.4206839928136801) q[0];
rx(-0.5056826296906684) q[1];
rx(-0.36385579160302) q[2];
rx(-0.5229365949441064) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.028506137271395508) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.3198593645467283) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.4205877776252283) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.10728920406120437) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.3353396341675495) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.059490761802862537) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.0183147726106635) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.3249121987629485) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.5801917181371711) q[3];
cx q[2],q[3];
rz(-0.1379230404885837) q[0];
rz(0.01714828609791985) q[1];
rz(-0.26819182049069806) q[2];
rz(-0.025412621741044354) q[3];
rx(-0.42346617316323) q[0];
rx(-0.7006281965182034) q[1];
rx(-0.30744219064823763) q[2];
rx(-0.5465116562887877) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.16758339004655196) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.2970078525618756) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.41118653477307116) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.18598725703574168) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.06373054055089153) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.12496110378296983) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.10323228636018682) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.0986038622492451) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.2942561136046782) q[3];
cx q[2],q[3];
rz(-0.1751005411790448) q[0];
rz(-0.08772757504499536) q[1];
rz(-0.07138295293389356) q[2];
rz(-0.058109893483919414) q[3];
rx(-0.48816944049136557) q[0];
rx(-0.7250178510308809) q[1];
rx(-0.33451575257834204) q[2];
rx(-0.5096356431024602) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.17027724259706065) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.009996107030867287) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.2010880770269344) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.4531286094289835) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.05097386466027308) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.12723793254011237) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.2004243299253552) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.04694185447265709) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.27816774705864233) q[3];
cx q[2],q[3];
rz(-0.09583300146005322) q[0];
rz(-0.16095616921247524) q[1];
rz(-0.005785321825462705) q[2];
rz(-0.058248065229718246) q[3];
rx(-0.4513985417293672) q[0];
rx(-0.6496911517107563) q[1];
rx(-0.4251965433031931) q[2];
rx(-0.4008715932637233) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.34612692416793767) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.10589279081019748) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.10603529852057865) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.6399605733263063) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.2127131750847882) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.09799790425736252) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.20982112990086557) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.03988209110491804) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.33809463387236627) q[3];
cx q[2],q[3];
rz(-0.08269593990496135) q[0];
rz(-0.1322995275218183) q[1];
rz(0.07195346956377954) q[2];
rz(-0.03760521251492669) q[3];
rx(-0.40787677889841595) q[0];
rx(-0.5549597141100975) q[1];
rx(-0.43073386978766426) q[2];
rx(-0.4777251663253352) q[3];