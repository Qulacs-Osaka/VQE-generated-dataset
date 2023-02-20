OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.07175655532105651) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.013503230261572298) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.025615676633255386) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.06158077166521727) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.04562837541771514) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.08587124061438511) q[3];
cx q[2],q[3];
rx(-0.2352776891715753) q[0];
rz(-0.043285810904147126) q[0];
rx(-0.05994998685451717) q[1];
rz(-0.10174152383914767) q[1];
rx(-0.043831332663513654) q[2];
rz(-0.12247918307989364) q[2];
rx(-0.25791070087856244) q[3];
rz(-0.12963987291469523) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.037830610197982055) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.012460686051159192) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.0342000392391897) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.04411293600285927) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.025969554859310108) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.15331431196620895) q[3];
cx q[2],q[3];
rx(-0.23455849472033466) q[0];
rz(-0.06271068116449785) q[0];
rx(0.013845296602750528) q[1];
rz(-0.07394980972504901) q[1];
rx(-0.033575191892225666) q[2];
rz(-0.0739694605179175) q[2];
rx(-0.3299541185800182) q[3];
rz(-0.1186402381599295) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.11610494892994075) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.010655096627703286) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.1322283824919513) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.06434737689229902) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.011722166096460575) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.2174643031903744) q[3];
cx q[2],q[3];
rx(-0.296368894376829) q[0];
rz(-0.05901041946176867) q[0];
rx(-0.03943162744925028) q[1];
rz(-0.09000320474018009) q[1];
rx(-0.01121578206185615) q[2];
rz(-0.09421919702020078) q[2];
rx(-0.3133947522695719) q[3];
rz(-0.16661588847923323) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.1500366140768711) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.045677369260047826) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.10305331800058909) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.07580104879286383) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.03702339094559386) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.1009194091352475) q[3];
cx q[2],q[3];
rx(-0.36341114989915035) q[0];
rz(-0.03343636412649324) q[0];
rx(-0.07168524292024765) q[1];
rz(-0.1397757759816372) q[1];
rx(0.012951181605622435) q[2];
rz(-0.11416702075231727) q[2];
rx(-0.22585099021337646) q[3];
rz(-0.11945826539659243) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.21986497241887124) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.04063173286900673) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.08877539149905264) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.19802021468100528) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.03712861778210104) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.057690878630224825) q[3];
cx q[2],q[3];
rx(-0.3398095421433441) q[0];
rz(0.03579285159424323) q[0];
rx(-0.0532813918450679) q[1];
rz(-0.09359578681888689) q[1];
rx(-0.008286000251287428) q[2];
rz(-0.10212101067635751) q[2];
rx(-0.25448601654819736) q[3];
rz(-0.12739243789269936) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.2735454599756001) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.06663310266227036) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.03692576717187174) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.2884134801598177) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.10884647397393518) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.007074847534610582) q[3];
cx q[2],q[3];
rx(-0.24588963446573242) q[0];
rz(0.005087227694520098) q[0];
rx(-0.08711290351288793) q[1];
rz(-0.13232497964974288) q[1];
rx(0.025897808970202055) q[2];
rz(-0.0497448305366004) q[2];
rx(-0.2545336126519152) q[3];
rz(-0.037637620731626534) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.3353604910282853) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.09420726798504903) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.0698403704792073) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.33439181586915034) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.10849702253004954) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.056083945531843496) q[3];
cx q[2],q[3];
rx(-0.27410136432222326) q[0];
rz(0.04176499999279495) q[0];
rx(-0.07100960453360323) q[1];
rz(-0.18216067774227732) q[1];
rx(0.02446565486035208) q[2];
rz(-0.08367189707033224) q[2];
rx(-0.2475676313259429) q[3];
rz(-0.0757692665845778) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.3351042738633938) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.020731051510677813) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.06791488011708532) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.3054884020792584) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.08528982309392363) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.12644911732474196) q[3];
cx q[2],q[3];
rx(-0.22154813052291938) q[0];
rz(0.025935719947208908) q[0];
rx(-0.046171734262225636) q[1];
rz(-0.15152978451436344) q[1];
rx(-0.03665887493598445) q[2];
rz(-0.1205197436641944) q[2];
rx(-0.22041127880487987) q[3];
rz(-0.041485669911310454) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.3207360237406788) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.07126882178483816) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.08420873301366728) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.22703623350007412) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.021474663619460096) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.11006048975799436) q[3];
cx q[2],q[3];
rx(-0.20748606918429535) q[0];
rz(0.04401835430650882) q[0];
rx(-0.1700506384756159) q[1];
rz(-0.24956691442033654) q[1];
rx(-0.10601649262215629) q[2];
rz(-0.1473683166583964) q[2];
rx(-0.1768470631460795) q[3];
rz(0.023078207937281447) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.23847771895008274) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.1439406816205995) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.005600428732320775) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.19779588330214373) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.17480934620733185) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.07566612976850565) q[3];
cx q[2],q[3];
rx(-0.22481167903215601) q[0];
rz(0.025421543781018158) q[0];
rx(-0.08224112024063554) q[1];
rz(-0.19066274471292322) q[1];
rx(-0.07985542439761102) q[2];
rz(-0.145644666768909) q[2];
rx(-0.22040905006303335) q[3];
rz(0.040975321258670705) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.2383070254239736) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.104341202906294) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.030950184550819503) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.1646528473354045) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.19455558194977576) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.043288929958733395) q[3];
cx q[2],q[3];
rx(-0.18669910262063383) q[0];
rz(-0.02413597260356491) q[0];
rx(-0.0938294336966304) q[1];
rz(-0.12920537850748579) q[1];
rx(-0.006141584040332857) q[2];
rz(-0.12830028415199535) q[2];
rx(-0.20351986711390088) q[3];
rz(0.05031144922469606) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.2465278559572361) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.20524916092897733) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.12435448876816416) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.18228990058410593) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.2806612416719587) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.04642160660376065) q[3];
cx q[2],q[3];
rx(-0.11101331860948872) q[0];
rz(-0.047620552222685304) q[0];
rx(-0.056806455379601654) q[1];
rz(-0.07985754683034829) q[1];
rx(0.0849098798467498) q[2];
rz(-0.16461768058594314) q[2];
rx(-0.2690154525016058) q[3];
rz(0.0007249771283095338) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.20055097544628167) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.23446165292373122) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.09523492976740995) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.18121773456704737) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.24606979557060457) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.15380780275371195) q[3];
cx q[2],q[3];
rx(-0.1479493079043325) q[0];
rz(-0.05985285859157425) q[0];
rx(0.05415529266917971) q[1];
rz(-0.08735319296030442) q[1];
rx(-0.057392693646520034) q[2];
rz(-0.08563356385285481) q[2];
rx(-0.24162502921143666) q[3];
rz(-0.04401030868598633) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.2394136674194516) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.14739768327913336) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.07442700362839812) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.24016515579073988) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.2635291123034628) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.07446574238314296) q[3];
cx q[2],q[3];
rx(-0.12735014767787944) q[0];
rz(-0.18454496214170163) q[0];
rx(-0.009570401726777025) q[1];
rz(-0.029071202676308004) q[1];
rx(0.03537604987429985) q[2];
rz(0.06451876485097974) q[2];
rx(-0.24239760942180155) q[3];
rz(-0.12785703521750616) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.2311400809681943) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.1582470429393417) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.003833315471431854) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.2038192209372291) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.20202416881616977) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.0010770712197143437) q[3];
cx q[2],q[3];
rx(-0.10460522503459695) q[0];
rz(-0.19557640799625986) q[0];
rx(-0.05517952243821798) q[1];
rz(0.040673581648515415) q[1];
rx(0.0056029012424050045) q[2];
rz(0.13844128623510196) q[2];
rx(-0.30028680096423854) q[3];
rz(-0.21471031885651654) q[3];