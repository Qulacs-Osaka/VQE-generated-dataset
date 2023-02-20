OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(0.8586104404864319) q[0];
rz(-3.141439477531031) q[0];
ry(-1.5709270556831036) q[1];
rz(-3.1365033789962666) q[1];
ry(-0.7733304156322995) q[2];
rz(-1.5677857059739577) q[2];
ry(1.570824557828862) q[3];
rz(1.6085201761308117) q[3];
ry(-6.503069785442212e-05) q[4];
rz(3.0712036047697766) q[4];
ry(-2.250579473474126) q[5];
rz(3.137414056729373) q[5];
ry(2.941811737675308) q[6];
rz(0.02282304192715401) q[6];
ry(1.5734955589959863) q[7];
rz(2.389946933233728) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.1332000380864928) q[0];
rz(-0.8871550007992138) q[0];
ry(1.5812798627664055) q[1];
rz(1.5729743935641032) q[1];
ry(-1.571352193917038) q[2];
rz(2.734958415439778) q[2];
ry(-2.3909902114408452) q[3];
rz(0.027894258114839837) q[3];
ry(-3.141496401641947) q[4];
rz(-0.13238958005742443) q[4];
ry(-0.03878752077728953) q[5];
rz(-1.5694223686532434) q[5];
ry(1.5718672712726576) q[6];
rz(0.8697806699751401) q[6];
ry(-0.00042993090154119784) q[7];
rz(0.3791944889568297) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.002411015050081465) q[0];
rz(0.8856181164239276) q[0];
ry(1.5713807358287084) q[1];
rz(-1.1300941128752735) q[1];
ry(3.1407873076013137) q[2];
rz(-0.3415704032385189) q[2];
ry(1.5548959382331722) q[3];
rz(-3.1159629303294136) q[3];
ry(-3.1352544322040528) q[4];
rz(0.7173675094023747) q[4];
ry(-1.5718316263794803) q[5];
rz(-2.1773739639450183) q[5];
ry(0.3649866576036116) q[6];
rz(2.1667205851130564) q[6];
ry(-2.3702409525521952) q[7];
rz(2.884762195607207) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.2520892561795387) q[0];
rz(-1.3264947531190403) q[0];
ry(-2.715998697514196) q[1];
rz(-0.1495067462129491) q[1];
ry(2.8403298214779995) q[2];
rz(-1.518165295035879) q[2];
ry(-3.1414086759162023) q[3];
rz(1.3763463638891589) q[3];
ry(1.8914736946268373e-06) q[4];
rz(-0.4998083910563701) q[4];
ry(-0.00031137229103078137) q[5];
rz(0.5885567747012271) q[5];
ry(-0.6573002458949793) q[6];
rz(-0.5019997644093221) q[6];
ry(-1.815532380160473) q[7];
rz(1.861712583441747) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(3.1296357897593174) q[0];
rz(-2.890418261484623) q[0];
ry(3.1320549086968197) q[1];
rz(-2.7588303029029637) q[1];
ry(-1.5712689397032968) q[2];
rz(3.1174417585166996) q[2];
ry(1.6016276349136875) q[3];
rz(-1.5328120738045665) q[3];
ry(-1.5633824436731993) q[4];
rz(2.529700423720943) q[4];
ry(0.0034236376847092136) q[5];
rz(2.633997729773615) q[5];
ry(-0.2280910958245581) q[6];
rz(0.20935608084230964) q[6];
ry(-1.5527644301075245) q[7];
rz(-0.6896350683757317) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.3772413815079796) q[0];
rz(2.0662724199528038) q[0];
ry(-1.5711080199590888) q[1];
rz(-1.5707264239416032) q[1];
ry(-3.141337883075301) q[2];
rz(-1.62156651717192) q[2];
ry(1.5694115192459597) q[3];
rz(-1.5709243987058494) q[3];
ry(0.00028892178819555546) q[4];
rz(-2.5382318184507917) q[4];
ry(0.00011949860676008228) q[5];
rz(0.49023221520459614) q[5];
ry(-1.5695657201928128) q[6];
rz(-3.101417819399955) q[6];
ry(-1.5708820743880159) q[7];
rz(-3.128172734020895) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-3.1414680283158676) q[0];
rz(2.0716371779500102) q[0];
ry(1.571065906330662) q[1];
rz(-8.672552409080101e-05) q[1];
ry(7.051509571351864e-05) q[2];
rz(-0.0007692810172077458) q[2];
ry(1.5707634336403382) q[3];
rz(-0.01220187665204754) q[3];
ry(1.4219351018929733) q[4];
rz(3.0504917612518927) q[4];
ry(1.5711310795854248) q[5];
rz(-2.8898575978957566) q[5];
ry(0.0001121270440427076) q[6];
rz(-0.381388548782696) q[6];
ry(-1.5471783838663518) q[7];
rz(-0.4935867505371947) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.9374286423434688) q[0];
rz(0.0012593175117547872) q[0];
ry(-1.571245802010342) q[1];
rz(1.5836552009887463) q[1];
ry(0.0005454277749882912) q[2];
rz(-3.1083258951087487) q[2];
ry(0.0032750934645466856) q[3];
rz(-3.1305897402887415) q[3];
ry(3.129638478383021) q[4];
rz(-1.6651036267518433) q[4];
ry(-3.1414012399838804) q[5];
rz(-0.8727940898799948) q[5];
ry(3.1413459956611263) q[6];
rz(-0.34602915242519483) q[6];
ry(-1.5702206298982588) q[7];
rz(0.680590971027161) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.455677185997005) q[0];
rz(-1.1201185044567712) q[0];
ry(-1.5773818559238881) q[1];
rz(0.0004090725690980435) q[1];
ry(0.7411906582561533) q[2];
rz(0.02611309175663212) q[2];
ry(1.29785331466692) q[3];
rz(3.1266494752929344) q[3];
ry(-1.5734930647383012) q[4];
rz(-3.118268724578775) q[4];
ry(3.129154414699848) q[5];
rz(-1.683762878381787) q[5];
ry(1.5739028685046828) q[6];
rz(-1.5855299204051523) q[6];
ry(0.024126821091920547) q[7];
rz(1.0266133458015432) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.00044644132002652904) q[0];
rz(-1.2874087145587183) q[0];
ry(1.570727893171058) q[1];
rz(0.0775222229216198) q[1];
ry(3.114518682017076) q[2];
rz(1.8569785888868162) q[2];
ry(-0.07753419871957115) q[3];
rz(-2.5457979913404487) q[3];
ry(0.02464521183241075) q[4];
rz(1.5688912140085067) q[4];
ry(-3.1414934632827993) q[5];
rz(-0.5590635810339526) q[5];
ry(2.633292801369429) q[6];
rz(-0.49380625392820554) q[6];
ry(3.1393697933257725) q[7];
rz(-0.3713243993708888) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(3.1415671693202722) q[0];
rz(-0.802865489203942) q[0];
ry(-0.006232863809928241) q[1];
rz(1.5279756800448387) q[1];
ry(0.0004898474987724555) q[2];
rz(2.918681763396002) q[2];
ry(0.001294755622090804) q[3];
rz(1.0196827701217366) q[3];
ry(-3.139908999627955) q[4];
rz(-3.0741458270857307) q[4];
ry(-1.5766887056940664) q[5];
rz(-1.5082869781100032) q[5];
ry(3.1356149068488053) q[6];
rz(-2.001215273933161) q[6];
ry(-0.019008259574901487) q[7];
rz(-2.5893175114580154) q[7];