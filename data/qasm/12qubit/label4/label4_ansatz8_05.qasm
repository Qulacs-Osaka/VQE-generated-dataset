OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-1.958935497425126) q[0];
ry(2.007094172589643) q[1];
cx q[0],q[1];
ry(-0.6194501171348195) q[0];
ry(2.8561132838125856) q[1];
cx q[0],q[1];
ry(2.806314199015235) q[2];
ry(-3.1141040290189093) q[3];
cx q[2],q[3];
ry(2.430297891668156) q[2];
ry(1.2221291689004223) q[3];
cx q[2],q[3];
ry(-2.416049354464738) q[4];
ry(1.4129882617339637) q[5];
cx q[4],q[5];
ry(1.5653160214852166) q[4];
ry(-2.2545540469236567) q[5];
cx q[4],q[5];
ry(1.945556946220413) q[6];
ry(-1.0752879569573144) q[7];
cx q[6],q[7];
ry(-0.4851133116672904) q[6];
ry(1.6155418715155014) q[7];
cx q[6],q[7];
ry(-2.033624295565189) q[8];
ry(-3.030970501304999) q[9];
cx q[8],q[9];
ry(2.172019925269439) q[8];
ry(1.3544411847574678) q[9];
cx q[8],q[9];
ry(-1.1971169211559685) q[10];
ry(-2.9813486965274087) q[11];
cx q[10],q[11];
ry(-0.5434929050576942) q[10];
ry(2.4665771908978775) q[11];
cx q[10],q[11];
ry(1.0340684865935428) q[0];
ry(3.1365329621764526) q[2];
cx q[0],q[2];
ry(0.05966358395886129) q[0];
ry(-3.082200696518764) q[2];
cx q[0],q[2];
ry(0.7761999421541589) q[2];
ry(2.382398371968618) q[4];
cx q[2],q[4];
ry(-3.093441337079521) q[2];
ry(-1.217748510527647) q[4];
cx q[2],q[4];
ry(-1.880067870917351) q[4];
ry(2.0394923899054387) q[6];
cx q[4],q[6];
ry(2.716817455051464) q[4];
ry(-2.859952396537733) q[6];
cx q[4],q[6];
ry(1.0285507643672185) q[6];
ry(1.583065894846846) q[8];
cx q[6],q[8];
ry(3.0429820025417085) q[6];
ry(-3.002432740894002) q[8];
cx q[6],q[8];
ry(2.8486404270326258) q[8];
ry(0.916405795009041) q[10];
cx q[8],q[10];
ry(0.22155625898272024) q[8];
ry(-0.18540115117566006) q[10];
cx q[8],q[10];
ry(2.7626120371619356) q[1];
ry(-0.5758453928590167) q[3];
cx q[1],q[3];
ry(1.9850919321044527) q[1];
ry(2.122342337170918) q[3];
cx q[1],q[3];
ry(-0.6261579383837315) q[3];
ry(-0.48955014144060455) q[5];
cx q[3],q[5];
ry(1.0208110043608416) q[3];
ry(3.065162898512188) q[5];
cx q[3],q[5];
ry(0.5523372502531085) q[5];
ry(-1.7775936363996352) q[7];
cx q[5],q[7];
ry(2.5083760908518085) q[5];
ry(0.3970589342221582) q[7];
cx q[5],q[7];
ry(-0.8320265576802609) q[7];
ry(0.5673368914118126) q[9];
cx q[7],q[9];
ry(3.0225486615434143) q[7];
ry(0.08977452442795987) q[9];
cx q[7],q[9];
ry(-2.028553650166647) q[9];
ry(-1.2742436688540406) q[11];
cx q[9],q[11];
ry(0.8572214676758714) q[9];
ry(0.1178617939783119) q[11];
cx q[9],q[11];
ry(-0.6503133237052929) q[0];
ry(-1.6951020828476573) q[1];
cx q[0],q[1];
ry(-1.6397087849876133) q[0];
ry(-1.5604688283823684) q[1];
cx q[0],q[1];
ry(1.5660202134721153) q[2];
ry(-2.790636959834105) q[3];
cx q[2],q[3];
ry(1.7153421843245438) q[2];
ry(-1.6421918236762227) q[3];
cx q[2],q[3];
ry(-0.691769619455988) q[4];
ry(-2.436630879823657) q[5];
cx q[4],q[5];
ry(-2.5746916140017593) q[4];
ry(0.8722131293232525) q[5];
cx q[4],q[5];
ry(-2.0741813429118237) q[6];
ry(-0.8723323908330421) q[7];
cx q[6],q[7];
ry(2.134787273119607) q[6];
ry(0.4744317378174756) q[7];
cx q[6],q[7];
ry(1.902870374428432) q[8];
ry(-0.49587925650233644) q[9];
cx q[8],q[9];
ry(-1.600115753362508) q[8];
ry(1.8334411323625037) q[9];
cx q[8],q[9];
ry(0.7867292893424235) q[10];
ry(-2.1828116592507776) q[11];
cx q[10],q[11];
ry(0.2814898952137135) q[10];
ry(-2.5054913077022904) q[11];
cx q[10],q[11];
ry(-2.57276363982899) q[0];
ry(-1.928042488798922) q[2];
cx q[0],q[2];
ry(-0.0006846323595484094) q[0];
ry(0.016799046629574654) q[2];
cx q[0],q[2];
ry(2.9310394644561004) q[2];
ry(-2.804297249470305) q[4];
cx q[2],q[4];
ry(-0.9381471555695855) q[2];
ry(-2.692352510761967) q[4];
cx q[2],q[4];
ry(0.785389111959256) q[4];
ry(2.576645077207561) q[6];
cx q[4],q[6];
ry(2.6183813288057807) q[4];
ry(2.6303775261539393) q[6];
cx q[4],q[6];
ry(-2.1107818889405583) q[6];
ry(-2.130609772670722) q[8];
cx q[6],q[8];
ry(-0.012938132806537472) q[6];
ry(0.05337397724301152) q[8];
cx q[6],q[8];
ry(-2.8559723262117864) q[8];
ry(-2.3360753863158945) q[10];
cx q[8],q[10];
ry(1.5885106796976842) q[8];
ry(0.07573143633233759) q[10];
cx q[8],q[10];
ry(-2.4847895194873195) q[1];
ry(-1.1877872080906098) q[3];
cx q[1],q[3];
ry(1.003125034620657) q[1];
ry(1.70251745747424) q[3];
cx q[1],q[3];
ry(2.719290949110246) q[3];
ry(-0.5827815286674944) q[5];
cx q[3],q[5];
ry(3.1153269672773742) q[3];
ry(3.090899042525817) q[5];
cx q[3],q[5];
ry(2.355980821721967) q[5];
ry(-3.086274676051336) q[7];
cx q[5],q[7];
ry(2.9208839895595995) q[5];
ry(2.400415867243375) q[7];
cx q[5],q[7];
ry(-1.8513897221246944) q[7];
ry(-2.9622345793788445) q[9];
cx q[7],q[9];
ry(3.1038582611660286) q[7];
ry(2.89788949320403) q[9];
cx q[7],q[9];
ry(-0.30812063773155796) q[9];
ry(-0.9753433556208683) q[11];
cx q[9],q[11];
ry(0.35789847431701016) q[9];
ry(0.012048458778605388) q[11];
cx q[9],q[11];
ry(-1.8751311794165857) q[0];
ry(2.092517683377915) q[1];
cx q[0],q[1];
ry(0.01698456674796405) q[0];
ry(-1.529602028244195) q[1];
cx q[0],q[1];
ry(1.8598066244681322) q[2];
ry(-2.7165052222987716) q[3];
cx q[2],q[3];
ry(-0.15509042560370725) q[2];
ry(-3.1214845718098623) q[3];
cx q[2],q[3];
ry(2.64957401537606) q[4];
ry(2.103758369275653) q[5];
cx q[4],q[5];
ry(-1.7465801983622924) q[4];
ry(0.14373685846769213) q[5];
cx q[4],q[5];
ry(-1.737445567932199) q[6];
ry(1.8686826150350178) q[7];
cx q[6],q[7];
ry(-0.25439319677625294) q[6];
ry(2.9783982216594924) q[7];
cx q[6],q[7];
ry(-0.23263186638188849) q[8];
ry(0.9419035769573069) q[9];
cx q[8],q[9];
ry(-0.22786962334704164) q[8];
ry(3.0974980622473294) q[9];
cx q[8],q[9];
ry(2.7256313585164076) q[10];
ry(1.9820490938782687) q[11];
cx q[10],q[11];
ry(0.2439367809612429) q[10];
ry(0.7860961421525682) q[11];
cx q[10],q[11];
ry(-0.7012085794415178) q[0];
ry(1.9703803300729872) q[2];
cx q[0],q[2];
ry(-0.0007013060439270962) q[0];
ry(3.139698289573721) q[2];
cx q[0],q[2];
ry(-2.58513926327729) q[2];
ry(3.131736678106823) q[4];
cx q[2],q[4];
ry(-2.6002715069131277) q[2];
ry(-1.2191635540991639) q[4];
cx q[2],q[4];
ry(1.4520145358572676) q[4];
ry(-1.7273164311041462) q[6];
cx q[4],q[6];
ry(0.023632975635297367) q[4];
ry(1.11514963585242) q[6];
cx q[4],q[6];
ry(0.2541919930216399) q[6];
ry(1.6599538731711343) q[8];
cx q[6],q[8];
ry(3.119792653193332) q[6];
ry(0.024011580843469403) q[8];
cx q[6],q[8];
ry(3.121123030369049) q[8];
ry(-1.2139373637260595) q[10];
cx q[8],q[10];
ry(1.3144730375508331) q[8];
ry(1.221846230290513) q[10];
cx q[8],q[10];
ry(-3.0856522483604363) q[1];
ry(-1.3872363889812949) q[3];
cx q[1],q[3];
ry(2.1578237953033073) q[1];
ry(-2.008299956273702) q[3];
cx q[1],q[3];
ry(-1.7335471638636175) q[3];
ry(3.0993714346751573) q[5];
cx q[3],q[5];
ry(0.02842885910779369) q[3];
ry(2.565289857723801) q[5];
cx q[3],q[5];
ry(2.987415162455866) q[5];
ry(0.5402861380443782) q[7];
cx q[5],q[7];
ry(-0.41007689700508454) q[5];
ry(-2.8761049918249317) q[7];
cx q[5],q[7];
ry(1.2291257258825894) q[7];
ry(-0.025691014047570057) q[9];
cx q[7],q[9];
ry(-3.007502020866095) q[7];
ry(-0.236932840561868) q[9];
cx q[7],q[9];
ry(-1.0469085860067429) q[9];
ry(2.583248054728085) q[11];
cx q[9],q[11];
ry(-2.2340682957039215) q[9];
ry(-0.29222618097813235) q[11];
cx q[9],q[11];
ry(-0.7614627251852522) q[0];
ry(0.2648867339249996) q[1];
cx q[0],q[1];
ry(-0.0019361848729631808) q[0];
ry(2.5831252811811716) q[1];
cx q[0],q[1];
ry(2.8365087721596765) q[2];
ry(-1.9404112439263594) q[3];
cx q[2],q[3];
ry(2.6627148685334805) q[2];
ry(-1.426526099522381) q[3];
cx q[2],q[3];
ry(-0.3563755353779632) q[4];
ry(2.6078098778915635) q[5];
cx q[4],q[5];
ry(1.0899581763191941) q[4];
ry(-2.7956812938072977) q[5];
cx q[4],q[5];
ry(-2.8818236412810125) q[6];
ry(-0.20915632240911083) q[7];
cx q[6],q[7];
ry(-0.8254259772010102) q[6];
ry(1.4062578837120183) q[7];
cx q[6],q[7];
ry(-1.7863517291335254) q[8];
ry(2.5511637287516846) q[9];
cx q[8],q[9];
ry(-1.2623375417455742) q[8];
ry(1.38668405116598) q[9];
cx q[8],q[9];
ry(1.3235916883168874) q[10];
ry(-1.0242180291913972) q[11];
cx q[10],q[11];
ry(1.944240562905697) q[10];
ry(-1.4334545641169671) q[11];
cx q[10],q[11];
ry(-0.4281760394163193) q[0];
ry(1.3962197746019018) q[2];
cx q[0],q[2];
ry(-0.005771299809131491) q[0];
ry(-3.1400972087286823) q[2];
cx q[0],q[2];
ry(-1.4525989643374877) q[2];
ry(0.393822766617791) q[4];
cx q[2],q[4];
ry(3.022120411039459) q[2];
ry(-2.7786075241344554) q[4];
cx q[2],q[4];
ry(1.9551525852440939) q[4];
ry(-1.10876603629585) q[6];
cx q[4],q[6];
ry(-1.0136108756007531) q[4];
ry(-1.9208354840895574) q[6];
cx q[4],q[6];
ry(-0.9703129382370217) q[6];
ry(-1.4088535361581949) q[8];
cx q[6],q[8];
ry(3.1283569601278383) q[6];
ry(0.001516989099666688) q[8];
cx q[6],q[8];
ry(-2.213076615851531) q[8];
ry(0.07109392923930644) q[10];
cx q[8],q[10];
ry(1.7992433107292678) q[8];
ry(-0.7310572295640597) q[10];
cx q[8],q[10];
ry(2.2000424848678235) q[1];
ry(1.4079141716601864) q[3];
cx q[1],q[3];
ry(1.5706303122833993) q[1];
ry(0.15051003553600975) q[3];
cx q[1],q[3];
ry(2.6550887375842382) q[3];
ry(-2.5236421706493117) q[5];
cx q[3],q[5];
ry(-3.131574787159376) q[3];
ry(-3.1357396122975287) q[5];
cx q[3],q[5];
ry(-0.5095620484430894) q[5];
ry(2.254897426249609) q[7];
cx q[5],q[7];
ry(1.9633135227927216) q[5];
ry(0.8356789341276505) q[7];
cx q[5],q[7];
ry(1.389783378811857) q[7];
ry(0.34148875041557636) q[9];
cx q[7],q[9];
ry(0.01030338617707649) q[7];
ry(-3.136123253089154) q[9];
cx q[7],q[9];
ry(-1.8594574935479464) q[9];
ry(-2.1420280120481303) q[11];
cx q[9],q[11];
ry(-1.2556198486348864) q[9];
ry(1.0588736142476014) q[11];
cx q[9],q[11];
ry(0.8947098986666782) q[0];
ry(0.8691710885062589) q[1];
cx q[0],q[1];
ry(1.747442548475215) q[0];
ry(-0.0188455542124899) q[1];
cx q[0],q[1];
ry(1.6055653847843692) q[2];
ry(-0.4261899182805894) q[3];
cx q[2],q[3];
ry(-1.5058170473727577) q[2];
ry(2.0626430916129888) q[3];
cx q[2],q[3];
ry(-1.1217627711831766) q[4];
ry(-0.28064613025109786) q[5];
cx q[4],q[5];
ry(-1.8805887483047143) q[4];
ry(-0.5607367305419012) q[5];
cx q[4],q[5];
ry(1.061273293039136) q[6];
ry(2.977021260026272) q[7];
cx q[6],q[7];
ry(0.6680555364975629) q[6];
ry(-1.0513752218696888) q[7];
cx q[6],q[7];
ry(-2.59407805746937) q[8];
ry(-0.5325305453806086) q[9];
cx q[8],q[9];
ry(0.14539307524008738) q[8];
ry(-1.4089174905504287) q[9];
cx q[8],q[9];
ry(-3.0314984359036123) q[10];
ry(0.511685791492994) q[11];
cx q[10],q[11];
ry(-0.7453576007216284) q[10];
ry(0.8396500235724556) q[11];
cx q[10],q[11];
ry(-1.5825153595511772) q[0];
ry(-3.139786137605775) q[2];
cx q[0],q[2];
ry(-0.006081698804785346) q[0];
ry(-3.1392565020222953) q[2];
cx q[0],q[2];
ry(-0.18077474292389795) q[2];
ry(-2.4407264626938883) q[4];
cx q[2],q[4];
ry(2.8788788222829065) q[2];
ry(-0.21011186535626822) q[4];
cx q[2],q[4];
ry(1.5278394450548647) q[4];
ry(2.7148774883643783) q[6];
cx q[4],q[6];
ry(-2.9717934266025865) q[4];
ry(-1.5078449746121478) q[6];
cx q[4],q[6];
ry(1.497410337177741) q[6];
ry(-1.8512996486268818) q[8];
cx q[6],q[8];
ry(3.136193332393187) q[6];
ry(3.1170761288805227) q[8];
cx q[6],q[8];
ry(-0.39807065019799825) q[8];
ry(2.303323816261598) q[10];
cx q[8],q[10];
ry(2.0164282345670266) q[8];
ry(-1.4242084597739053) q[10];
cx q[8],q[10];
ry(1.565124502703388) q[1];
ry(-2.2028607329999437) q[3];
cx q[1],q[3];
ry(-0.016880280771850453) q[1];
ry(1.5305677244232663) q[3];
cx q[1],q[3];
ry(-2.95233898358238) q[3];
ry(-1.3867685792932365) q[5];
cx q[3],q[5];
ry(3.1283522697066846) q[3];
ry(3.113488074087914) q[5];
cx q[3],q[5];
ry(-2.1609823987872456) q[5];
ry(-0.7634964651958684) q[7];
cx q[5],q[7];
ry(1.8081601162252472) q[5];
ry(-2.453538317487313) q[7];
cx q[5],q[7];
ry(-2.332131879951836) q[7];
ry(0.7914443619357039) q[9];
cx q[7],q[9];
ry(-0.04738429801546318) q[7];
ry(-2.795764436485789) q[9];
cx q[7],q[9];
ry(1.0762584387343825) q[9];
ry(-0.928220246420234) q[11];
cx q[9],q[11];
ry(1.7464745194967524) q[9];
ry(-0.2888627901379168) q[11];
cx q[9],q[11];
ry(-1.883723614336921) q[0];
ry(2.487982345217262) q[1];
cx q[0],q[1];
ry(2.8675364110747337) q[0];
ry(-1.5716297193582698) q[1];
cx q[0],q[1];
ry(-0.6216671860060855) q[2];
ry(0.223310454998784) q[3];
cx q[2],q[3];
ry(2.9649762215272673) q[2];
ry(-0.00036707224579490827) q[3];
cx q[2],q[3];
ry(-1.5127518406671707) q[4];
ry(-2.1522746790063927) q[5];
cx q[4],q[5];
ry(-2.8761759881419615) q[4];
ry(-1.59659299174873) q[5];
cx q[4],q[5];
ry(-2.862157632811182) q[6];
ry(0.7498167687128413) q[7];
cx q[6],q[7];
ry(3.1320578319301378) q[6];
ry(1.5889267000746286) q[7];
cx q[6],q[7];
ry(1.8771939052089788) q[8];
ry(1.7894974834975317) q[9];
cx q[8],q[9];
ry(-1.5310197751111203) q[8];
ry(-2.839753160363485) q[9];
cx q[8],q[9];
ry(-2.3819009493814525) q[10];
ry(2.461486494181823) q[11];
cx q[10],q[11];
ry(0.16639842120661733) q[10];
ry(-1.6193748867160382) q[11];
cx q[10],q[11];
ry(0.49583164937904883) q[0];
ry(-0.106332605313241) q[2];
cx q[0],q[2];
ry(0.08654444025425434) q[0];
ry(0.508362424578933) q[2];
cx q[0],q[2];
ry(-0.2619342315163788) q[2];
ry(1.1410843845152652) q[4];
cx q[2],q[4];
ry(3.0656945743444397) q[2];
ry(-3.0879731222540125) q[4];
cx q[2],q[4];
ry(2.712321503044238) q[4];
ry(1.5574968966787646) q[6];
cx q[4],q[6];
ry(1.7721388788023487) q[4];
ry(-3.128758613178017) q[6];
cx q[4],q[6];
ry(1.5896213512342845) q[6];
ry(1.682343069296893) q[8];
cx q[6],q[8];
ry(1.3222054340710838) q[6];
ry(-2.8710090045448315) q[8];
cx q[6],q[8];
ry(-0.6919667316982052) q[8];
ry(-0.23365556723305403) q[10];
cx q[8],q[10];
ry(-3.079514779248547) q[8];
ry(-1.6031538118757456) q[10];
cx q[8],q[10];
ry(-2.263044105264175) q[1];
ry(-2.8281684204524957) q[3];
cx q[1],q[3];
ry(-0.0237865396823033) q[1];
ry(-0.8545201660373891) q[3];
cx q[1],q[3];
ry(1.1740591482759946) q[3];
ry(-1.7740223826792387) q[5];
cx q[3],q[5];
ry(-0.00668105524833873) q[3];
ry(0.0667018322313142) q[5];
cx q[3],q[5];
ry(2.9885254324604302) q[5];
ry(0.8003318411180551) q[7];
cx q[5],q[7];
ry(-2.8933553034138804) q[5];
ry(3.1076962079252577) q[7];
cx q[5],q[7];
ry(1.0949172733516574) q[7];
ry(0.5514441232044219) q[9];
cx q[7],q[9];
ry(3.131006066538629) q[7];
ry(3.1299883530474704) q[9];
cx q[7],q[9];
ry(-2.667916088486252) q[9];
ry(-1.243712123715544) q[11];
cx q[9],q[11];
ry(1.6290062666471012) q[9];
ry(2.0086396589227293) q[11];
cx q[9],q[11];
ry(-1.648306381203719) q[0];
ry(0.44128275881580153) q[1];
cx q[0],q[1];
ry(0.003704017274821254) q[0];
ry(-3.140322219616531) q[1];
cx q[0],q[1];
ry(-2.3773894724907376) q[2];
ry(0.3754695907248353) q[3];
cx q[2],q[3];
ry(0.2459664807648041) q[2];
ry(-3.040060748436789) q[3];
cx q[2],q[3];
ry(-2.603476206774962) q[4];
ry(-0.7472700290681766) q[5];
cx q[4],q[5];
ry(-1.3825160640925338) q[4];
ry(-1.5879351284187821) q[5];
cx q[4],q[5];
ry(1.878153836693616) q[6];
ry(2.580651244362683) q[7];
cx q[6],q[7];
ry(1.535548611766691) q[6];
ry(-2.3618721104661695) q[7];
cx q[6],q[7];
ry(1.5884654893138346) q[8];
ry(2.7687599162442584) q[9];
cx q[8],q[9];
ry(-0.1861645464563244) q[8];
ry(-1.5859090264407671) q[9];
cx q[8],q[9];
ry(-1.9052980298321827) q[10];
ry(2.7050705206861423) q[11];
cx q[10],q[11];
ry(1.7742989863734442) q[10];
ry(3.115553580239482) q[11];
cx q[10],q[11];
ry(1.8195488556829154) q[0];
ry(-0.2410726692052254) q[2];
cx q[0],q[2];
ry(0.005684813251858323) q[0];
ry(0.909814138817131) q[2];
cx q[0],q[2];
ry(-1.0455563635348073) q[2];
ry(2.7476076115858827) q[4];
cx q[2],q[4];
ry(0.36594352947120984) q[2];
ry(2.564115800932581) q[4];
cx q[2],q[4];
ry(-1.4066431469503735) q[4];
ry(0.5540138751928692) q[6];
cx q[4],q[6];
ry(3.14078268542347) q[4];
ry(-0.008254672492858677) q[6];
cx q[4],q[6];
ry(-1.7403347959278737) q[6];
ry(-2.364881970665648) q[8];
cx q[6],q[8];
ry(0.01254787151329051) q[6];
ry(-0.0015944094053614813) q[8];
cx q[6],q[8];
ry(-0.7789864403617244) q[8];
ry(-0.11972571067346847) q[10];
cx q[8],q[10];
ry(-2.7475385944714388) q[8];
ry(-2.236120436474401) q[10];
cx q[8],q[10];
ry(-2.1185075620066147) q[1];
ry(-1.3381968019684143) q[3];
cx q[1],q[3];
ry(3.137046935261651) q[1];
ry(-0.0055165588185793835) q[3];
cx q[1],q[3];
ry(-2.3764186701485714) q[3];
ry(-1.5026064632028566) q[5];
cx q[3],q[5];
ry(0.006896965859719819) q[3];
ry(0.0027367950575556505) q[5];
cx q[3],q[5];
ry(0.8141144844683907) q[5];
ry(-0.718109427436191) q[7];
cx q[5],q[7];
ry(3.1369471524379215) q[5];
ry(3.1302307677640178) q[7];
cx q[5],q[7];
ry(-1.4375582905608852) q[7];
ry(-1.8641461300428979) q[9];
cx q[7],q[9];
ry(-2.7533157704757123) q[7];
ry(3.125169383919964) q[9];
cx q[7],q[9];
ry(3.0733979663106923) q[9];
ry(-2.6045911499017986) q[11];
cx q[9],q[11];
ry(-0.00018907498152920255) q[9];
ry(0.01830754150021852) q[11];
cx q[9],q[11];
ry(-1.5578939778733127) q[0];
ry(0.13056782506045828) q[1];
cx q[0],q[1];
ry(-1.2860077675025217) q[0];
ry(3.1415465954900816) q[1];
cx q[0],q[1];
ry(-0.31018015998901627) q[2];
ry(0.9342535165735955) q[3];
cx q[2],q[3];
ry(0.4240253052585921) q[2];
ry(-0.0812652384258068) q[3];
cx q[2],q[3];
ry(0.06682888540866028) q[4];
ry(-3.1210700138032195) q[5];
cx q[4],q[5];
ry(1.0568709408396382) q[4];
ry(-0.5580968203209805) q[5];
cx q[4],q[5];
ry(2.302166979749513) q[6];
ry(-0.7081075092599515) q[7];
cx q[6],q[7];
ry(1.5838367364106354) q[6];
ry(0.7667122787546452) q[7];
cx q[6],q[7];
ry(-1.559746838782198) q[8];
ry(1.8636419612466806) q[9];
cx q[8],q[9];
ry(-3.046188811966589) q[8];
ry(1.5943171210844824) q[9];
cx q[8],q[9];
ry(1.2061283847964335) q[10];
ry(-0.5823793547476468) q[11];
cx q[10],q[11];
ry(-1.8261169776317265) q[10];
ry(1.5501962101587092) q[11];
cx q[10],q[11];
ry(1.3084137181300735) q[0];
ry(-1.2689550891868242) q[2];
cx q[0],q[2];
ry(1.7630857720913022) q[0];
ry(-1.6606188058860396) q[2];
cx q[0],q[2];
ry(1.083784026285745) q[2];
ry(-0.725594828285808) q[4];
cx q[2],q[4];
ry(-3.0730964703126205) q[2];
ry(-0.023795558370748537) q[4];
cx q[2],q[4];
ry(0.7618998258058542) q[4];
ry(2.9759489801184418) q[6];
cx q[4],q[6];
ry(0.010594519785613874) q[4];
ry(-0.009376879954969839) q[6];
cx q[4],q[6];
ry(1.4866922999644965) q[6];
ry(2.400879023969099) q[8];
cx q[6],q[8];
ry(3.127551622467684) q[6];
ry(-3.0506061693428097) q[8];
cx q[6],q[8];
ry(-2.4425965182343496) q[8];
ry(1.6436330960424481) q[10];
cx q[8],q[10];
ry(-1.5612548233065437) q[8];
ry(0.02986841139667984) q[10];
cx q[8],q[10];
ry(-1.3839740212526168) q[1];
ry(-1.3764456531207543) q[3];
cx q[1],q[3];
ry(1.5739677354154904) q[1];
ry(-0.013981527024050422) q[3];
cx q[1],q[3];
ry(1.427997684950995) q[3];
ry(-0.12331782945294166) q[5];
cx q[3],q[5];
ry(1.5723421615249666) q[3];
ry(-3.1363088973956104) q[5];
cx q[3],q[5];
ry(1.5662970298671142) q[5];
ry(1.5532338512025037) q[7];
cx q[5],q[7];
ry(-1.5706143831111197) q[5];
ry(-0.021084167411101973) q[7];
cx q[5],q[7];
ry(1.5707945443086067) q[7];
ry(0.16566615901995618) q[9];
cx q[7],q[9];
ry(-1.570891135113563) q[7];
ry(-2.3875555817820118) q[9];
cx q[7],q[9];
ry(-1.5707585969974562) q[9];
ry(2.0204192254278417) q[11];
cx q[9],q[11];
ry(-1.5707811420480031) q[9];
ry(1.7629307902605582) q[11];
cx q[9],q[11];
ry(0.5027523021542335) q[0];
ry(-1.7602913911056355) q[1];
ry(2.0550615654409934) q[2];
ry(1.4294415115947752) q[3];
ry(-1.3976315923238287) q[4];
ry(-1.5658223493213486) q[5];
ry(-1.5520806204610023) q[6];
ry(1.5709965754520667) q[7];
ry(-1.4452325597087903) q[8];
ry(1.57082256350051) q[9];
ry(1.5658605574564683) q[10];
ry(1.5709146102272995) q[11];