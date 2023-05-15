OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-2.1348797924247758) q[0];
rz(-2.9824410214327965) q[0];
ry(-1.461289828506475) q[1];
rz(2.198396582889559) q[1];
ry(3.141547109175069) q[2];
rz(1.6907307077658915) q[2];
ry(-3.141323260575496) q[3];
rz(0.09253511022479799) q[3];
ry(-3.003842751850618) q[4];
rz(-1.1864103903596057) q[4];
ry(-2.6462853316557515) q[5];
rz(-2.3505254464500136) q[5];
ry(3.1405440272567304) q[6];
rz(1.6905573908411562) q[6];
ry(-0.08884759647140975) q[7];
rz(-3.080325644755143) q[7];
ry(-1.6388946832343967) q[8];
rz(2.3860796050780615) q[8];
ry(3.023194535871226) q[9];
rz(1.9882040015186897) q[9];
ry(0.03987236091014238) q[10];
rz(-0.400438791135639) q[10];
ry(0.2321887018079495) q[11];
rz(1.171800082976377) q[11];
ry(0.000653329276341985) q[12];
rz(-1.7410083477446023) q[12];
ry(-3.1406807120464144) q[13];
rz(-2.08995409031376) q[13];
ry(2.8024943158065083) q[14];
rz(2.053109185197516) q[14];
ry(-2.626932325398166) q[15];
rz(1.9223811296053697) q[15];
ry(2.473785955146545) q[16];
rz(-1.3457615848024984) q[16];
ry(-0.30086404977663245) q[17];
rz(0.3797967458143061) q[17];
ry(-0.1799763641400114) q[18];
rz(-2.786410918357095) q[18];
ry(-0.37016553877204866) q[19];
rz(-1.09591677524483) q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(2.299312573872363) q[0];
rz(-2.6377626750040646) q[0];
ry(2.2737076844193376) q[1];
rz(0.6269841797856321) q[1];
ry(-3.1397790517640787) q[2];
rz(1.6709530340278866) q[2];
ry(-0.0023851201599640603) q[3];
rz(3.3924349360984e-05) q[3];
ry(2.615446311284095) q[4];
rz(1.631844392694144) q[4];
ry(-1.2719310308827598) q[5];
rz(-1.9669171530831224) q[5];
ry(1.5773276835766825) q[6];
rz(-1.7566059476930747) q[6];
ry(1.5913389889493308) q[7];
rz(1.4777537456765266) q[7];
ry(-0.5411497921301524) q[8];
rz(-2.539105616619634) q[8];
ry(-1.7210976630690547) q[9];
rz(-0.5856500591265994) q[9];
ry(-1.5523919293350836) q[10];
rz(-0.5196589022501996) q[10];
ry(-3.1394793254657016) q[11];
rz(2.5281222184468857) q[11];
ry(-3.1403675791472643) q[12];
rz(-2.228430233925105) q[12];
ry(-3.1409506383610766) q[13];
rz(-2.1767523387321583) q[13];
ry(-1.3807774936840467) q[14];
rz(2.377085970121462) q[14];
ry(1.3543951541296773) q[15];
rz(1.3164976229002425) q[15];
ry(-2.7714440372071185) q[16];
rz(-2.1554026999004607) q[16];
ry(-0.16781518638415352) q[17];
rz(2.172660894188772) q[17];
ry(0.0941059905848105) q[18];
rz(0.13543138440885907) q[18];
ry(1.3674249311849744) q[19];
rz(2.957924851815089) q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-2.435575062233278) q[0];
rz(2.3069811890386576) q[0];
ry(1.9233572640097627) q[1];
rz(-1.3769230506876917) q[1];
ry(-1.6209589949411605) q[2];
rz(3.091134779875822) q[2];
ry(1.601915314444777) q[3];
rz(3.1326379043848567) q[3];
ry(1.5976584239589053) q[4];
rz(0.33890191332853353) q[4];
ry(-1.5709861389633784) q[5];
rz(1.3483472760363024) q[5];
ry(-3.119828287168189) q[6];
rz(2.9613311634800223) q[6];
ry(-0.015381720579836231) q[7];
rz(0.10636614426960593) q[7];
ry(3.139744549670376) q[8];
rz(-0.35669240203395697) q[8];
ry(3.1395531425529173) q[9];
rz(-0.4523040555108195) q[9];
ry(2.8398003366868383) q[10];
rz(1.0903388664603035) q[10];
ry(0.0005787330566660032) q[11];
rz(0.19313935252929237) q[11];
ry(1.578739220056348) q[12];
rz(2.6741368065970015) q[12];
ry(-1.5702482005681035) q[13];
rz(-2.7326305937463435) q[13];
ry(-2.0356106357483297) q[14];
rz(0.8520128888716219) q[14];
ry(-2.1913945026860757) q[15];
rz(-1.2849161630047925) q[15];
ry(1.6687287981841004) q[16];
rz(1.6502346769919773) q[16];
ry(-2.6191736006756954) q[17];
rz(-2.1014630869810906) q[17];
ry(0.14602371502485756) q[18];
rz(0.006783193341829277) q[18];
ry(0.027899592700084245) q[19];
rz(-2.5908945618720307) q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.5212890374299564) q[0];
rz(2.007772079296484) q[0];
ry(1.2960451312286225) q[1];
rz(1.5166589821637835) q[1];
ry(2.7694612816795012) q[2];
rz(-2.8068407038788346) q[2];
ry(-2.7952950801618073) q[3];
rz(1.6257600723274892) q[3];
ry(-0.27673749197915143) q[4];
rz(-1.7373332268679185) q[4];
ry(3.076054769602874) q[5];
rz(-1.7499050212455067) q[5];
ry(1.6131373139686813) q[6];
rz(0.4380369303707062) q[6];
ry(-3.117530463504566) q[7];
rz(2.219421165518142) q[7];
ry(-2.7423270515472895) q[8];
rz(0.36555196583285376) q[8];
ry(2.8934067383409814) q[9];
rz(-2.7160413534116845) q[9];
ry(0.7773581033462804) q[10];
rz(0.046521125271920205) q[10];
ry(-3.1343095349590233) q[11];
rz(1.0176695722181943) q[11];
ry(-0.0011874376679743407) q[12];
rz(-2.6773336250785347) q[12];
ry(-3.1412661152112134) q[13];
rz(0.4013843599612388) q[13];
ry(-0.04463550765893801) q[14];
rz(0.0428092283843542) q[14];
ry(3.0995016714838006) q[15];
rz(-0.22672467843508895) q[15];
ry(-0.19374439019068654) q[16];
rz(-0.7925127146306572) q[16];
ry(0.744832472178977) q[17];
rz(-1.9658773694711433) q[17];
ry(2.27555267226316) q[18];
rz(-2.743822073684276) q[18];
ry(1.8106838108253713) q[19];
rz(-1.7095470646121336) q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.5535290827162584) q[0];
rz(1.6230803318171763) q[0];
ry(1.7878600741742599) q[1];
rz(2.2964023280251333) q[1];
ry(-0.04363378553139265) q[2];
rz(-0.35384204611778447) q[2];
ry(-0.08427489546573064) q[3];
rz(-1.679521307425805) q[3];
ry(-1.6318669480838588) q[4];
rz(0.844907283840307) q[4];
ry(1.566167839288953) q[5];
rz(2.0505736260036453) q[5];
ry(2.815349852770812) q[6];
rz(0.48041887242456066) q[6];
ry(0.17295578036468803) q[7];
rz(-2.129199841013723) q[7];
ry(-0.0004964701245197141) q[8];
rz(2.6487115480389463) q[8];
ry(-3.13817361291122) q[9];
rz(3.0683941793090868) q[9];
ry(1.8747831111340822) q[10];
rz(-2.3674325765340862) q[10];
ry(-0.2679268157291973) q[11];
rz(1.5599304794581004) q[11];
ry(-1.5602988911509232) q[12];
rz(1.9473874078696203) q[12];
ry(1.5688858126368281) q[13];
rz(-2.2052929061502535) q[13];
ry(-1.791071099099289) q[14];
rz(-0.8102100927125352) q[14];
ry(1.4729584542950542) q[15];
rz(0.7695952214976782) q[15];
ry(2.2768811779538947) q[16];
rz(0.6537273616146484) q[16];
ry(-0.233508553848651) q[17];
rz(0.003906164414077651) q[17];
ry(0.6189434992759342) q[18];
rz(0.2854161090445872) q[18];
ry(1.6590412884702253) q[19];
rz(-2.412746862346989) q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-2.824485004586518) q[0];
rz(0.4749094439566681) q[0];
ry(-2.2502917786821115) q[1];
rz(-0.5091180503995054) q[1];
ry(1.5400658097497653) q[2];
rz(0.002735243885079619) q[2];
ry(1.5440065919394002) q[3];
rz(-3.13759190667364) q[3];
ry(0.03568163655921541) q[4];
rz(1.7522320932148068) q[4];
ry(1.8840310442375672) q[5];
rz(1.505658424130056) q[5];
ry(1.2210153574900273) q[6];
rz(0.6598368208913704) q[6];
ry(-0.2995946398588618) q[7];
rz(1.9786108732538572) q[7];
ry(-1.4905533711273184) q[8];
rz(-1.6557752865086641) q[8];
ry(3.1384287296044193) q[9];
rz(2.9255750205983393) q[9];
ry(-0.06739512210446692) q[10];
rz(2.797502687565894) q[10];
ry(1.561319612647487) q[11];
rz(2.063555125906814) q[11];
ry(-3.139651175174376) q[12];
rz(0.9965538756136282) q[12];
ry(-3.1415633625271546) q[13];
rz(-2.2876828126233897) q[13];
ry(-1.501664025728365) q[14];
rz(-2.2830890351828685) q[14];
ry(1.4701558438579294) q[15];
rz(2.6904542674399203) q[15];
ry(1.8374403144035387) q[16];
rz(1.4357196211763883) q[16];
ry(-2.9721745788928384) q[17];
rz(1.2189663232968124) q[17];
ry(-1.1716494242996471) q[18];
rz(-0.8205063159114806) q[18];
ry(-0.27276772474648414) q[19];
rz(-1.659431627859954) q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(3.0865741554702986) q[0];
rz(2.089202063671054) q[0];
ry(2.964406247575258) q[1];
rz(-0.17815413593612453) q[1];
ry(1.5962801687152535) q[2];
rz(-2.5068462311647224) q[2];
ry(1.5605603403652397) q[3];
rz(1.7730729430618313) q[3];
ry(-2.0217247477052864) q[4];
rz(-1.6965577392218103) q[4];
ry(2.789331961960858) q[5];
rz(-0.1811402199610505) q[5];
ry(3.124454383585294) q[6];
rz(-2.591628298276831) q[6];
ry(3.1332127235449585) q[7];
rz(1.7110446894867068) q[7];
ry(-1.5193984221071712) q[8];
rz(-1.5770826455846914) q[8];
ry(1.5789817773291528) q[9];
rz(-1.7563537832460308) q[9];
ry(-1.5226696911997983) q[10];
rz(-1.124972273670611) q[10];
ry(3.13171723508481) q[11];
rz(2.0886976120408285) q[11];
ry(-3.1401652756622274) q[12];
rz(-1.834858247666872) q[12];
ry(-0.00010995878097500017) q[13];
rz(-2.969681303095048) q[13];
ry(2.914266768467013) q[14];
rz(-1.5262043451579457) q[14];
ry(2.9415676839435356) q[15];
rz(-1.163363710101415) q[15];
ry(0.7627538795829691) q[16];
rz(1.4394022840368494) q[16];
ry(-1.2936676184374871) q[17];
rz(0.9237255482632718) q[17];
ry(-0.06233053772186725) q[18];
rz(2.7987133637981336) q[18];
ry(-2.291472463907687) q[19];
rz(0.6609648419419962) q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.6061878469163036) q[0];
rz(2.8634856321062103) q[0];
ry(-1.6320198425679806) q[1];
rz(-1.0492268675318988) q[1];
ry(2.3263957118558123) q[2];
rz(2.0341148968770164) q[2];
ry(-1.9054373809762684) q[3];
rz(2.7429834249191334) q[3];
ry(-1.507643304745681) q[4];
rz(1.6332945740352545) q[4];
ry(1.9493494470007757) q[5];
rz(-2.129718716266316) q[5];
ry(1.238456520420096) q[6];
rz(1.051474280319856) q[6];
ry(1.5591991910666225) q[7];
rz(-1.3339301065809916) q[7];
ry(-2.132605672438399) q[8];
rz(-3.0472579199056047) q[8];
ry(-0.01002909413296571) q[9];
rz(0.402622384023987) q[9];
ry(1.7379880030838522) q[10];
rz(-0.3570749485806166) q[10];
ry(2.9864038934265116) q[11];
rz(1.6303219287453987) q[11];
ry(-0.09840206659966408) q[12];
rz(-2.2839608962527036) q[12];
ry(0.025689607226945018) q[13];
rz(1.7286364212094711) q[13];
ry(-3.0971165415610837) q[14];
rz(1.2258246436999842) q[14];
ry(-3.0420165777097035) q[15];
rz(2.5303545780245003) q[15];
ry(0.038063599144181566) q[16];
rz(0.6010094612123097) q[16];
ry(1.352622125093344) q[17];
rz(-2.810240884564905) q[17];
ry(1.3020758272437403) q[18];
rz(0.31598592729812763) q[18];
ry(2.1539531844777757) q[19];
rz(-1.4971989653062723) q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-0.007125789931671456) q[0];
rz(2.069580520254499) q[0];
ry(0.011165667364622323) q[1];
rz(1.0380819646035206) q[1];
ry(0.10487495034363015) q[2];
rz(-0.2911401105161265) q[2];
ry(2.798211225492256) q[3];
rz(2.8552650824726586) q[3];
ry(-3.1265761167927915) q[4];
rz(0.7352063152825954) q[4];
ry(-3.136018638950779) q[5];
rz(2.572127266964521) q[5];
ry(3.1413123058673555) q[6];
rz(1.9927251355599926) q[6];
ry(-0.0018415812691054967) q[7];
rz(2.687192692560428) q[7];
ry(-0.10404343863895804) q[8];
rz(1.4866636233223667) q[8];
ry(0.007001322719238515) q[9];
rz(-1.7856471996188361) q[9];
ry(-1.810627404023662) q[10];
rz(1.2798387979093606) q[10];
ry(1.5994266886421664) q[11];
rz(-1.499773614684594) q[11];
ry(0.002895486512287975) q[12];
rz(1.3380734285570515) q[12];
ry(0.007460003273009442) q[13];
rz(1.5696897312321374) q[13];
ry(1.5837540135264816) q[14];
rz(1.674003981264196) q[14];
ry(-1.5733871891051567) q[15];
rz(1.562457679493146) q[15];
ry(3.137313982148994) q[16];
rz(1.0467075313611265) q[16];
ry(-1.9004103351407942) q[17];
rz(2.3359031536993866) q[17];
ry(-1.0526882019193884) q[18];
rz(2.422039940261934) q[18];
ry(1.9505415845433034) q[19];
rz(0.5771488470153827) q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-3.1299420395874744) q[0];
rz(2.983731102955355) q[0];
ry(3.1283660185700866) q[1];
rz(-1.9121450533558653) q[1];
ry(1.715983748877096) q[2];
rz(2.1577237326409953) q[2];
ry(-1.0813106478095036) q[3];
rz(-0.5366257460127563) q[3];
ry(2.947361643453931) q[4];
rz(0.1505527759221541) q[4];
ry(3.0528238219188415) q[5];
rz(-2.2539196216214465) q[5];
ry(1.8626977363431978) q[6];
rz(-0.7110255221101462) q[6];
ry(-2.687453731739255) q[7];
rz(-2.5722155547172574) q[7];
ry(1.5658082654412446) q[8];
rz(0.08600335406996672) q[8];
ry(1.5837158867652998) q[9];
rz(3.0960121880107874) q[9];
ry(0.010224542695010313) q[10];
rz(-2.645798732292228) q[10];
ry(-1.0912450059461323) q[11];
rz(1.1035772512652529) q[11];
ry(-3.136237064756987) q[12];
rz(0.813908786593919) q[12];
ry(-3.137494101585616) q[13];
rz(3.0251923530202034) q[13];
ry(2.472309575155886) q[14];
rz(1.6445158831448534) q[14];
ry(-0.5780335942165007) q[15];
rz(-1.701578381701112) q[15];
ry(-1.524771371505361) q[16];
rz(2.2633875016274905) q[16];
ry(2.527062573578733) q[17];
rz(-0.4239955953892195) q[17];
ry(-2.1342235831344736) q[18];
rz(-2.8620269611506415) q[18];
ry(-3.0748429608098524) q[19];
rz(-3.0833293686557934) q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.6360556975768876) q[0];
rz(-2.560383223531505) q[0];
ry(-1.9324418393644835) q[1];
rz(3.1063475319041145) q[1];
ry(-0.07482319752460903) q[2];
rz(0.9873378132359496) q[2];
ry(-1.4143589586097487) q[3];
rz(-1.519761552124212) q[3];
ry(-3.140280527768995) q[4];
rz(-2.6096013894863543) q[4];
ry(0.002270103503937193) q[5];
rz(-1.2682017843535907) q[5];
ry(0.06300849952931337) q[6];
rz(2.50401167009969) q[6];
ry(-0.051809900591771894) q[7];
rz(1.2168583200039476) q[7];
ry(2.955572570354602) q[8];
rz(0.01969723271486643) q[8];
ry(3.128327796109529) q[9];
rz(1.5512978001112834) q[9];
ry(-1.610572009206711) q[10];
rz(1.5495056709469086) q[10];
ry(3.1295466012416164) q[11];
rz(-0.4264693911213442) q[11];
ry(0.0006737310738538227) q[12];
rz(1.2552025604116814) q[12];
ry(3.141007853882749) q[13];
rz(-2.684506904515142) q[13];
ry(1.6240646001307217) q[14];
rz(-0.9127205474154205) q[14];
ry(-1.6412589147042826) q[15];
rz(-0.5977858240919582) q[15];
ry(0.21468079836736284) q[16];
rz(2.113591897752017) q[16];
ry(2.025858865091031) q[17];
rz(-1.412998193178213) q[17];
ry(1.4872043613810408) q[18];
rz(0.9108504384282144) q[18];
ry(-0.6357136216758353) q[19];
rz(-0.11843004379224027) q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-2.7164289868923817) q[0];
rz(-1.2081878570963034) q[0];
ry(1.5923294696680237) q[1];
rz(-1.4938052077220565) q[1];
ry(2.7840091396572233) q[2];
rz(-1.190948937130499) q[2];
ry(-0.9941696156660118) q[3];
rz(1.066563094781382) q[3];
ry(-2.9523514981490595) q[4];
rz(-0.6394538414662011) q[4];
ry(2.9588659326034965) q[5];
rz(-2.5549364632945246) q[5];
ry(0.7915171726880768) q[6];
rz(-2.8122162329620446) q[6];
ry(1.718692172244073) q[7];
rz(-2.2671467030618437) q[7];
ry(1.5831016969204776) q[8];
rz(0.013427158279915459) q[8];
ry(1.5878364775905736) q[9];
rz(-0.03234421693144629) q[9];
ry(1.552022219187971) q[10];
rz(0.060049530051034306) q[10];
ry(1.5693127091071553) q[11];
rz(1.5476154349852302) q[11];
ry(2.5809907569183426) q[12];
rz(0.6491246805932391) q[12];
ry(1.7985251923710968) q[13];
rz(-0.033717824354840616) q[13];
ry(1.5573133361869549) q[14];
rz(2.697354865418115) q[14];
ry(-1.0608188615460277) q[15];
rz(-1.4093120968375576) q[15];
ry(-0.13738485671468278) q[16];
rz(3.0904425434465037) q[16];
ry(-3.1266815613152223) q[17];
rz(2.4699913193427516) q[17];
ry(2.2132055862393236) q[18];
rz(-2.6125132466460794) q[18];
ry(-1.321446606783928) q[19];
rz(0.44269450696419965) q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-2.154840967774664) q[0];
rz(2.36650695213666) q[0];
ry(1.6414471820979877) q[1];
rz(-3.0090927330549637) q[1];
ry(0.04941412398470723) q[2];
rz(-1.7787537378112637) q[2];
ry(0.04729001516160203) q[3];
rz(-0.9247960821589843) q[3];
ry(-0.0002993562703596168) q[4];
rz(1.7167171724090613) q[4];
ry(-0.007814941361601789) q[5];
rz(-2.76094114747128) q[5];
ry(1.5561720324598518) q[6];
rz(2.343516253578947) q[6];
ry(1.5841748538567737) q[7];
rz(1.6144669937387264) q[7];
ry(1.3568130245561463) q[8];
rz(3.1094623940342974) q[8];
ry(-1.5589836105465447) q[9];
rz(1.5765419520021542) q[9];
ry(-1.4295471728605431) q[10];
rz(3.1389182961227733) q[10];
ry(1.5587022506857018) q[11];
rz(-2.5273275293902406) q[11];
ry(-3.138311524707261) q[12];
rz(-1.016941012229089) q[12];
ry(0.004277305211392099) q[13];
rz(-1.6244087617306462) q[13];
ry(-3.117529303801312) q[14];
rz(-0.14852104737406258) q[14];
ry(-0.002595929097680552) q[15];
rz(0.19634977681885957) q[15];
ry(2.3279969519299706) q[16];
rz(-3.0565066093031685) q[16];
ry(1.4780719001272324) q[17];
rz(2.6977392950351073) q[17];
ry(1.7530391991813374) q[18];
rz(-2.7294819461851922) q[18];
ry(-0.5811603727776253) q[19];
rz(0.9743248065675516) q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-0.2587579643276305) q[0];
rz(-2.7067377704690125) q[0];
ry(-1.242133107301136) q[1];
rz(2.9535757688915125) q[1];
ry(2.6077254227812245) q[2];
rz(-0.7984064314114814) q[2];
ry(2.172283461950044) q[3];
rz(-2.6487441199820765) q[3];
ry(0.03550257851511838) q[4];
rz(-0.537243914693029) q[4];
ry(3.005784313427984) q[5];
rz(-1.4493008608381777) q[5];
ry(-0.007311193807960131) q[6];
rz(2.391128300170124) q[6];
ry(-0.10039315504109148) q[7];
rz(-0.0030211890995847563) q[7];
ry(2.9370697907907317) q[8];
rz(-0.15321164193811662) q[8];
ry(-1.5336268533029953) q[9];
rz(-3.0573848416076213) q[9];
ry(1.5720765012286892) q[10];
rz(-2.9853756534253657) q[10];
ry(0.01344810588684009) q[11];
rz(-0.8892224629608083) q[11];
ry(1.566752869810987) q[12];
rz(3.0787356873846923) q[12];
ry(-1.6139599432670233) q[13];
rz(0.2248732376060625) q[13];
ry(1.482716535417009) q[14];
rz(-2.1200008871708116) q[14];
ry(-2.7976851475153603) q[15];
rz(-2.5011394439656534) q[15];
ry(0.3690394757619204) q[16];
rz(1.4848802808122554) q[16];
ry(0.706714091139168) q[17];
rz(2.363144717524157) q[17];
ry(2.386307663110019) q[18];
rz(-1.3563962190381196) q[18];
ry(0.36742750219829867) q[19];
rz(-0.5159488434061484) q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.3796729363634688) q[0];
rz(-2.510617509652344) q[0];
ry(2.904197725456428) q[1];
rz(2.0207578810938207) q[1];
ry(-0.018359813714022667) q[2];
rz(-1.4002754087533344) q[2];
ry(-3.1397133495347056) q[3];
rz(2.1060813897625588) q[3];
ry(-3.1383365947182758) q[4];
rz(-0.29305580011053245) q[4];
ry(3.129313587798931) q[5];
rz(-1.1717405880905227) q[5];
ry(-2.562007650408937) q[6];
rz(-3.135832732639584) q[6];
ry(0.8738819050542619) q[7];
rz(2.855726624721595) q[7];
ry(0.04541200143252446) q[8];
rz(-0.022608099087208622) q[8];
ry(3.110069501240564) q[9];
rz(3.0095984659198733) q[9];
ry(3.1396512895162103) q[10];
rz(3.121057542219763) q[10];
ry(0.006851929608386876) q[11];
rz(2.513321600268791) q[11];
ry(-1.5761058052004246) q[12];
rz(0.5844954135198703) q[12];
ry(-1.5749837431982225) q[13];
rz(1.536742156161858) q[13];
ry(-2.9963608765594922) q[14];
rz(1.9781813624133056) q[14];
ry(-0.06294756324232702) q[15];
rz(0.23414232709382257) q[15];
ry(-2.0317631653388384) q[16];
rz(-3.1239188425306708) q[16];
ry(1.485660791110984) q[17];
rz(0.10754862114617686) q[17];
ry(1.3966175142998327) q[18];
rz(-2.758477647729495) q[18];
ry(0.059783743572110734) q[19];
rz(-2.3737843743702784) q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.2177919766829126) q[0];
rz(1.3150479779545856) q[0];
ry(-3.068287373485237) q[1];
rz(1.479914680756635) q[1];
ry(-1.4165683744066206) q[2];
rz(2.5429134118835384) q[2];
ry(3.1203277847468947) q[3];
rz(0.8739953276545229) q[3];
ry(-0.008424906537556067) q[4];
rz(-2.7421302180605114) q[4];
ry(-0.320195392343079) q[5];
rz(-0.9112434064508133) q[5];
ry(-0.09046450653998139) q[6];
rz(-1.5328476105110698) q[6];
ry(0.015884164518235263) q[7];
rz(1.8193685418667247) q[7];
ry(2.7784996009554472) q[8];
rz(2.5521079740064576) q[8];
ry(-0.04026242766589482) q[9];
rz(0.06846074684183899) q[9];
ry(3.115605148038405) q[10];
rz(-1.744244348765914) q[10];
ry(-3.1026264611303294) q[11];
rz(-2.346572002616671) q[11];
ry(-1.589774481781487) q[12];
rz(1.5977518463405416) q[12];
ry(3.1232755508813583) q[13];
rz(1.0700167301432144) q[13];
ry(-1.5762939005769308) q[14];
rz(-1.5615139742634259) q[14];
ry(1.5774969173945106) q[15];
rz(1.766746232643398) q[15];
ry(-1.6224027218243664) q[16];
rz(-2.0202865909487935) q[16];
ry(-0.01996829438823422) q[17];
rz(1.9918146891416795) q[17];
ry(2.550145755603476) q[18];
rz(2.7786597395518187) q[18];
ry(2.9292059020656906) q[19];
rz(2.5697833844461297) q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(2.44382085329614) q[0];
rz(-0.8532762468919216) q[0];
ry(1.6978828685656957) q[1];
rz(-1.8708111751527519) q[1];
ry(-0.014057313401725224) q[2];
rz(1.0544129388218737) q[2];
ry(-0.025278530959282897) q[3];
rz(-1.7285377275526823) q[3];
ry(-3.1309806541775136) q[4];
rz(2.649777262523133) q[4];
ry(0.019526548179352916) q[5];
rz(-0.6256744165015009) q[5];
ry(-1.5699383191822724) q[6];
rz(2.567316730573404) q[6];
ry(-1.573657121643004) q[7];
rz(-2.1017075905532128) q[7];
ry(-0.02549970828113324) q[8];
rz(0.4835166800105047) q[8];
ry(-3.100758074446475) q[9];
rz(2.7362690302755857) q[9];
ry(0.05083493435935751) q[10];
rz(1.5664130057770074) q[10];
ry(-0.002476933858332802) q[11];
rz(3.0744672495716445) q[11];
ry(0.004710311969725922) q[12];
rz(-0.054876557033177245) q[12];
ry(0.0001888242175933641) q[13];
rz(0.25637506763593865) q[13];
ry(0.27670418471045544) q[14];
rz(0.7308453293115741) q[14];
ry(-3.141358452377909) q[15];
rz(2.1840638566659756) q[15];
ry(-1.864757430392035) q[16];
rz(0.5594937972378569) q[16];
ry(1.557325598731188) q[17];
rz(-0.9811348315345593) q[17];
ry(-2.340451703353786) q[18];
rz(1.7058720361464064) q[18];
ry(-0.21090070676809436) q[19];
rz(2.598583018527462) q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.4883850532411529) q[0];
rz(0.19093264244119101) q[0];
ry(2.929041885928469) q[1];
rz(0.060934662724754673) q[1];
ry(0.8520758757802191) q[2];
rz(0.9731336515363617) q[2];
ry(2.955429966771097) q[3];
rz(-2.5254232168749358) q[3];
ry(-1.56411722649189) q[4];
rz(0.9158942453339785) q[4];
ry(-2.732443251313766) q[5];
rz(-0.7644589718953226) q[5];
ry(-1.2900135465618412) q[6];
rz(-0.2999757470191673) q[6];
ry(2.0760150131116024) q[7];
rz(0.8290615155386005) q[7];
ry(-0.1931371322522315) q[8];
rz(0.35311443646157487) q[8];
ry(1.7253114174109658) q[9];
rz(-0.7501296227129374) q[9];
ry(-1.6043118907313056) q[10];
rz(2.760312721588185) q[10];
ry(0.051107253532517556) q[11];
rz(1.4150762460009998) q[11];
ry(2.5834444693900465) q[12];
rz(-0.4914631485693964) q[12];
ry(-1.57403743322784) q[13];
rz(-1.3964150588372424) q[13];
ry(3.1223589445789344) q[14];
rz(-0.8402507144982083) q[14];
ry(-0.006442955788615201) q[15];
rz(-0.40896240690925634) q[15];
ry(3.098253794959039) q[16];
rz(2.0099633050280534) q[16];
ry(3.0543236749492437) q[17];
rz(1.4415791833418172) q[17];
ry(-1.870751026506213) q[18];
rz(-2.7667126650718212) q[18];
ry(-0.19588512694772472) q[19];
rz(1.775171657381) q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-0.5812402710861673) q[0];
rz(0.5054426488203416) q[0];
ry(-1.8369308842023964) q[1];
rz(0.12974825925278446) q[1];
ry(-0.03889709924143885) q[2];
rz(2.1667814899282116) q[2];
ry(0.00232743350405583) q[3];
rz(1.4327296816693416) q[3];
ry(-3.129020328967089) q[4];
rz(2.879442330652957) q[4];
ry(-3.1413870589960395) q[5];
rz(-2.3892579655687904) q[5];
ry(-3.1385243770506572) q[6];
rz(2.746346961519707) q[6];
ry(-3.139166772046745) q[7];
rz(-2.609512190864131) q[7];
ry(-3.109615195914334) q[8];
rz(-2.412328080000352) q[8];
ry(-3.13203234606824) q[9];
rz(2.3994394302227326) q[9];
ry(-3.0725492320535333) q[10];
rz(1.336759951928614) q[10];
ry(3.1171755245272545) q[11];
rz(-2.054379954005956) q[11];
ry(0.0011667552129059897) q[12];
rz(1.5813575471797456) q[12];
ry(-3.1219719393734757) q[13];
rz(-2.4833404321684385) q[13];
ry(-1.579724351777886) q[14];
rz(-2.7635976921364054) q[14];
ry(1.5742334033287348) q[15];
rz(0.011960562722940968) q[15];
ry(-1.55135322569439) q[16];
rz(2.453824716247116) q[16];
ry(2.003540935185117) q[17];
rz(0.9430969405885277) q[17];
ry(2.3912354505191313) q[18];
rz(0.020809314610558283) q[18];
ry(3.031676507078558) q[19];
rz(-2.4953270808573658) q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-2.5322987486851156) q[0];
rz(2.416199234402897) q[0];
ry(-1.3358049194980266) q[1];
rz(-2.942695362706954) q[1];
ry(-2.084302326706176) q[2];
rz(1.8265408429309673) q[2];
ry(-0.9743935259414188) q[3];
rz(-0.24189401425235624) q[3];
ry(-1.0326050147519918) q[4];
rz(-0.20127023223488555) q[4];
ry(-1.4145749890040848) q[5];
rz(2.0547699071575014) q[5];
ry(-0.3773574943553992) q[6];
rz(0.04075364987544544) q[6];
ry(2.62651071675192) q[7];
rz(-0.2385488321765843) q[7];
ry(-3.0816749921820574) q[8];
rz(1.9515040988282983) q[8];
ry(-1.6881007042557172) q[9];
rz(-1.813524272590743) q[9];
ry(2.8889674795633258e-05) q[10];
rz(3.008155649908747) q[10];
ry(0.0009903204432188681) q[11];
rz(0.4290184526284784) q[11];
ry(3.139227251239871) q[12];
rz(-0.30463807008415) q[12];
ry(-3.138508061907573) q[13];
rz(2.219978952450586) q[13];
ry(-3.141090892856819) q[14];
rz(2.0948017556818996) q[14];
ry(1.56227624361387) q[15];
rz(-1.5695163624918005) q[15];
ry(-2.511534986884957) q[16];
rz(0.3081175407146688) q[16];
ry(-1.9130775279973546) q[17];
rz(0.7421249981043497) q[17];
ry(2.9566256153732957) q[18];
rz(2.439803008545983) q[18];
ry(-2.985337297559337) q[19];
rz(1.695277058881047) q[19];