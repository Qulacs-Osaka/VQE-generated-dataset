OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-2.627741347334648) q[0];
rz(-2.42035128798955) q[0];
ry(-1.0532156089970528) q[1];
rz(1.4493836753933946) q[1];
ry(1.561269952203645) q[2];
rz(-0.26060678420605654) q[2];
ry(2.4468084540482846) q[3];
rz(2.520957034205281) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.7828291327829273) q[0];
rz(2.07205182786946) q[0];
ry(-1.7078765445454946) q[1];
rz(1.189707154001177) q[1];
ry(-0.202386011392516) q[2];
rz(-0.1399355477652655) q[2];
ry(-0.6839743053711844) q[3];
rz(2.669424692436088) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.8010594947542944) q[0];
rz(-0.8002065689346604) q[0];
ry(0.4239888375934722) q[1];
rz(-1.2308622337126465) q[1];
ry(2.3172192630750907) q[2];
rz(-0.24048821263580497) q[2];
ry(-2.9004526409032065) q[3];
rz(1.5825575070274907) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.23835508173441503) q[0];
rz(1.8524927121822046) q[0];
ry(1.2079127682949546) q[1];
rz(0.605785227759327) q[1];
ry(-0.6069385547867503) q[2];
rz(1.5743928385237753) q[2];
ry(0.3914480160978604) q[3];
rz(-0.41878116945875704) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.452908000687763) q[0];
rz(-2.924491121024525) q[0];
ry(2.7559835001402178) q[1];
rz(-2.981214442726781) q[1];
ry(0.7316910996912055) q[2];
rz(0.6262603888603273) q[2];
ry(-1.2879292544230652) q[3];
rz(0.0589790733598383) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.382305933977772) q[0];
rz(-2.2715981012802144) q[0];
ry(-1.3278485871748191) q[1];
rz(-0.55388736281321) q[1];
ry(2.7265577414951276) q[2];
rz(-0.8981498117608367) q[2];
ry(-0.018596211840588184) q[3];
rz(0.16957807679222725) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.47239854439069) q[0];
rz(-0.02719339888774985) q[0];
ry(1.9280652835077818) q[1];
rz(2.024868839385467) q[1];
ry(-0.6975696611194898) q[2];
rz(-1.8303513471203514) q[2];
ry(-0.9266685072287034) q[3];
rz(1.2393533318144945) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.6650603489049223) q[0];
rz(-0.4082566729113269) q[0];
ry(0.025120287907971495) q[1];
rz(0.7509162433753007) q[1];
ry(2.932699007568114) q[2];
rz(-1.0014477893886111) q[2];
ry(0.9665522496071626) q[3];
rz(-0.7606039284190941) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-3.003298457081019) q[0];
rz(3.0221530835683588) q[0];
ry(0.8431416779687789) q[1];
rz(1.4745427135768416) q[1];
ry(0.01672002870619327) q[2];
rz(-1.7454965166140428) q[2];
ry(2.8898720638713846) q[3];
rz(1.0845081423761354) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.662317429410412) q[0];
rz(-0.8985070732630801) q[0];
ry(1.9788033860711316) q[1];
rz(0.9977099440042478) q[1];
ry(-2.5074973198309056) q[2];
rz(-1.5890143162689343) q[2];
ry(0.7106766057502157) q[3];
rz(2.376337369507846) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.667985768681369) q[0];
rz(0.7502155462344486) q[0];
ry(1.350891445153994) q[1];
rz(1.7117174300014442) q[1];
ry(-0.6587119781732406) q[2];
rz(-0.8643996958374818) q[2];
ry(-1.7024679274091952) q[3];
rz(2.2541155162338833) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.797129671490076) q[0];
rz(-1.3116818827062984) q[0];
ry(-2.070063852596501) q[1];
rz(0.013132910825735087) q[1];
ry(1.9295174453220831) q[2];
rz(-3.1026107469591864) q[2];
ry(2.620294252096004) q[3];
rz(2.062288816039656) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.72486986890533) q[0];
rz(-0.9750119597408621) q[0];
ry(0.5938700208752197) q[1];
rz(1.8406117732898002) q[1];
ry(0.6178782652727826) q[2];
rz(2.4991546501310182) q[2];
ry(2.5689848365118877) q[3];
rz(1.5966273178070622) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.1410124618034048) q[0];
rz(3.0246620024600586) q[0];
ry(-0.6449967021856889) q[1];
rz(2.0621207229200826) q[1];
ry(-2.895396406071967) q[2];
rz(-0.6230045061228103) q[2];
ry(2.878824781927306) q[3];
rz(2.1503641328112475) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.2929491055393161) q[0];
rz(0.9645404917392016) q[0];
ry(2.1936508490060804) q[1];
rz(3.107048883072385) q[1];
ry(-1.684287440966803) q[2];
rz(2.7321499076885227) q[2];
ry(1.9416761171334629) q[3];
rz(-0.6161682615640256) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.5238946529088295) q[0];
rz(2.8857190401254695) q[0];
ry(1.1489540072562041) q[1];
rz(0.3226905573392509) q[1];
ry(0.5256475393183068) q[2];
rz(-0.9206025699614108) q[2];
ry(2.937265067097004) q[3];
rz(0.3881349966987324) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.057573548271289) q[0];
rz(0.8576252172771959) q[0];
ry(-2.657994464672844) q[1];
rz(-0.11482731910556067) q[1];
ry(0.28948214439845277) q[2];
rz(3.0734885765422884) q[2];
ry(2.656158538288486) q[3];
rz(-0.02095590411673065) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.8857043840261164) q[0];
rz(-0.05445334619830078) q[0];
ry(-1.6798615538070552) q[1];
rz(-0.3913864184098495) q[1];
ry(0.583591813612605) q[2];
rz(1.0738992325316996) q[2];
ry(-2.741531102606193) q[3];
rz(-0.2259978327937553) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.4505314086569232) q[0];
rz(1.6060951510608044) q[0];
ry(-1.7875046032792188) q[1];
rz(0.41491100190824) q[1];
ry(-2.379784478168744) q[2];
rz(-1.260052617640893) q[2];
ry(2.909401636546229) q[3];
rz(1.022982402917938) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-3.090087770441666) q[0];
rz(1.0427398521421098) q[0];
ry(-1.8185967453582836) q[1];
rz(0.31424498509495535) q[1];
ry(0.2173374714484695) q[2];
rz(1.6811260281726503) q[2];
ry(1.4824099776525062) q[3];
rz(-2.1104125695599847) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.3624657771762565) q[0];
rz(1.433804058817021) q[0];
ry(-2.3243163998294722) q[1];
rz(-0.08117587265015391) q[1];
ry(-2.086667977872655) q[2];
rz(-0.9054550438828205) q[2];
ry(2.6472036787513065) q[3];
rz(-0.5519846010930628) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.9723897611724563) q[0];
rz(3.0281730152524116) q[0];
ry(-1.5160968179784664) q[1];
rz(1.564192181113612) q[1];
ry(2.367382757977881) q[2];
rz(1.6658715451792574) q[2];
ry(2.143783756653572) q[3];
rz(-2.8282815362371845) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.538788093707035) q[0];
rz(0.10520803010639099) q[0];
ry(0.17222366226439156) q[1];
rz(1.2105620809523703) q[1];
ry(-1.5617694881480693) q[2];
rz(0.3023038833202183) q[2];
ry(-0.9459763033304541) q[3];
rz(-0.0754969425769603) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.08512733962731563) q[0];
rz(1.7347516877440767) q[0];
ry(-2.2030631008923613) q[1];
rz(-2.561210363772061) q[1];
ry(2.312719406455261) q[2];
rz(-1.764454831347153) q[2];
ry(-2.7595675052105464) q[3];
rz(-2.0344158248338653) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.9290332736241669) q[0];
rz(-1.4788770421720434) q[0];
ry(0.7857105113653953) q[1];
rz(-1.0195335749713101) q[1];
ry(-0.7453111533126283) q[2];
rz(0.31511385459339714) q[2];
ry(2.2177214077314793) q[3];
rz(-0.06469974425868406) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.2682887504549205) q[0];
rz(0.2698400151698275) q[0];
ry(2.3824771439075323) q[1];
rz(2.0903699015263353) q[1];
ry(-0.704677650886528) q[2];
rz(1.6085761334131945) q[2];
ry(-1.1204222010449212) q[3];
rz(0.5398539756650713) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.1050868206815365) q[0];
rz(-1.3898810862468085) q[0];
ry(-0.6777323494144785) q[1];
rz(0.5406413469420212) q[1];
ry(2.4708599613317923) q[2];
rz(1.2388515485378009) q[2];
ry(-1.0408686696141387) q[3];
rz(2.1738650224699083) q[3];