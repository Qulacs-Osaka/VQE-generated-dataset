OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(1.6412164916834637) q[0];
rz(-0.21649403792856112) q[0];
ry(0.8950732113242719) q[1];
rz(0.3750685698147675) q[1];
ry(-0.8495808316774163) q[2];
rz(0.5931190777171853) q[2];
ry(-2.430985478224169) q[3];
rz(2.8287288495295346) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.7669328451602113) q[0];
rz(0.4991718169024954) q[0];
ry(-2.13222411892962) q[1];
rz(-3.020868074787064) q[1];
ry(1.8798147006216048) q[2];
rz(-1.1676290811069796) q[2];
ry(-2.052469626203708) q[3];
rz(2.3794143824126275) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.3282571044093596) q[0];
rz(-0.3030513922960781) q[0];
ry(-1.3303162754688314) q[1];
rz(-2.3937475817745604) q[1];
ry(-0.7062060814865173) q[2];
rz(-0.8956149380440918) q[2];
ry(1.685703465658543) q[3];
rz(1.3344492008666384) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.463226097878264) q[0];
rz(-2.7454664891305245) q[0];
ry(0.7773031669468535) q[1];
rz(2.1595323836028113) q[1];
ry(2.332036851068922) q[2];
rz(-1.817002309911787) q[2];
ry(-0.5755823678900686) q[3];
rz(-0.4699653345656705) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.7711711576426437) q[0];
rz(1.7801423368658746) q[0];
ry(1.2491338742379847) q[1];
rz(0.6649076110918414) q[1];
ry(0.7807958518900049) q[2];
rz(-1.0568814481714541) q[2];
ry(2.257457330425745) q[3];
rz(2.839090492416748) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.930965153910959) q[0];
rz(-2.029077164900213) q[0];
ry(-0.054772728844645784) q[1];
rz(2.0914437568631747) q[1];
ry(-2.593261001434098) q[2];
rz(0.31222243027485685) q[2];
ry(-1.6130900326373308) q[3];
rz(2.972530914144789) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.038927298845393) q[0];
rz(-2.4273693962929768) q[0];
ry(0.7245405126796047) q[1];
rz(-0.4767393860314012) q[1];
ry(0.6128423416660852) q[2];
rz(1.6291574493948064) q[2];
ry(-2.545251979513274) q[3];
rz(-1.4335824663491252) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.9350415755552466) q[0];
rz(-1.9614753939967169) q[0];
ry(2.0452261853149984) q[1];
rz(0.6516141925678189) q[1];
ry(-2.458677937554624) q[2];
rz(-2.2545602346736073) q[2];
ry(-0.3505904182944018) q[3];
rz(1.800388500606462) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.6557498755348385) q[0];
rz(-0.19518073094794275) q[0];
ry(0.7828187250957478) q[1];
rz(2.7469080059278275) q[1];
ry(-3.1048079963594724) q[2];
rz(2.5503875614370672) q[2];
ry(0.14388399449789618) q[3];
rz(-2.440250480026696) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.1770348014460326) q[0];
rz(0.4638837479346405) q[0];
ry(2.278113211958818) q[1];
rz(0.36970766582738795) q[1];
ry(2.5474783976755213) q[2];
rz(-0.15765356336238234) q[2];
ry(-0.4469652001398172) q[3];
rz(1.70237458660365) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.7910132859299728) q[0];
rz(2.843145202609465) q[0];
ry(2.1922016922396237) q[1];
rz(1.2369944288428099) q[1];
ry(-0.4483076370168206) q[2];
rz(-0.10727924531673329) q[2];
ry(-1.1890728903183874) q[3];
rz(-2.9573138525895413) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.26785515067735094) q[0];
rz(-0.04786784431873404) q[0];
ry(-3.047783251414487) q[1];
rz(-2.756100220294756) q[1];
ry(-0.39044740714708315) q[2];
rz(1.0774515218242322) q[2];
ry(2.102697132312314) q[3];
rz(2.918973278937949) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.1292916625820002) q[0];
rz(1.4996059790960752) q[0];
ry(3.063874542504749) q[1];
rz(3.0648840794837304) q[1];
ry(1.7576261889637241) q[2];
rz(2.516629599966508) q[2];
ry(2.580858327694842) q[3];
rz(1.8542658281581947) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.7548484047513364) q[0];
rz(2.975006895020294) q[0];
ry(-1.6426046371129042) q[1];
rz(-0.22885078202976258) q[1];
ry(1.7806433039625789) q[2];
rz(1.4808829387856184) q[2];
ry(-3.128987235221874) q[3];
rz(-0.10706765309786627) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.6380846417474688) q[0];
rz(-0.9438125446749703) q[0];
ry(-1.9319202672079276) q[1];
rz(0.8354037652594908) q[1];
ry(-0.6193150393626085) q[2];
rz(2.5881277179059707) q[2];
ry(2.9113038133322675) q[3];
rz(-0.27447648811534325) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.5592336189317342) q[0];
rz(2.6469746308294106) q[0];
ry(-0.306837629841563) q[1];
rz(-0.746496701028069) q[1];
ry(1.9663367557975977) q[2];
rz(1.8554939853794192) q[2];
ry(1.816787412602788) q[3];
rz(1.2560171680890135) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.592061370801189) q[0];
rz(1.316151579622033) q[0];
ry(2.8773407034715124) q[1];
rz(0.9882211272960753) q[1];
ry(2.7029131327435167) q[2];
rz(-2.173425641514001) q[2];
ry(-2.7550634131691716) q[3];
rz(-2.1954459113915483) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.2788593053776027) q[0];
rz(-2.117215698401006) q[0];
ry(-1.5416917580128162) q[1];
rz(-1.0466853406754573) q[1];
ry(1.0645384017463666) q[2];
rz(-0.40248435595847193) q[2];
ry(-0.7782028472814849) q[3];
rz(-1.0948380277282417) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.6327540364113924) q[0];
rz(0.7165939283291287) q[0];
ry(-2.6148988981330374) q[1];
rz(2.2807248047462614) q[1];
ry(2.0614820785016073) q[2];
rz(2.38538008947868) q[2];
ry(-1.120652458884448) q[3];
rz(-2.348185372933618) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.08499855747831297) q[0];
rz(1.991118134383154) q[0];
ry(-1.8364108185730552) q[1];
rz(-0.16267763602336607) q[1];
ry(-0.02014971823160168) q[2];
rz(-1.4306879804866124) q[2];
ry(-2.101512389300317) q[3];
rz(2.4956248900400166) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.7649097224437069) q[0];
rz(0.0036721147374708246) q[0];
ry(0.8747406532870929) q[1];
rz(2.9588221962541503) q[1];
ry(-1.7054462780228212) q[2];
rz(-1.0275783348832406) q[2];
ry(1.0455730538674937) q[3];
rz(-3.006626770728039) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.775024966884712) q[0];
rz(-3.1232797735361926) q[0];
ry(3.1206220868602648) q[1];
rz(0.7062795668116181) q[1];
ry(-2.684095899843227) q[2];
rz(-1.8742996201343762) q[2];
ry(1.2809933910794546) q[3];
rz(-1.5693759391012352) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(3.1096090199879955) q[0];
rz(2.6979888752971815) q[0];
ry(2.6986615597081545) q[1];
rz(-1.8288226222984587) q[1];
ry(0.6197332074899097) q[2];
rz(1.2014870564056124) q[2];
ry(-1.0272286597782982) q[3];
rz(2.9629380077348197) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.909007906067994) q[0];
rz(-1.5446301591685847) q[0];
ry(1.9036821653666678) q[1];
rz(-2.7496569060161318) q[1];
ry(1.9564856973520577) q[2];
rz(-2.1458446888020237) q[2];
ry(-0.974801782976912) q[3];
rz(1.8507323772046076) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.293178678749906) q[0];
rz(2.0661717242053035) q[0];
ry(-0.6429794458353273) q[1];
rz(-0.6874666070727073) q[1];
ry(-2.196344473867117) q[2];
rz(2.521128932717486) q[2];
ry(-2.9675810694660627) q[3];
rz(-2.3985965138525462) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.0099385152415312) q[0];
rz(2.309958919391494) q[0];
ry(2.1759120393482907) q[1];
rz(0.21693076834553307) q[1];
ry(-0.0028521796194533877) q[2];
rz(1.2676041761534327) q[2];
ry(-1.5644212273664362) q[3];
rz(-2.8881859691769916) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.3410534677792563) q[0];
rz(-0.27325426637320416) q[0];
ry(1.6320161686875494) q[1];
rz(2.233299589125326) q[1];
ry(1.5499317695040526) q[2];
rz(-1.3827642122177564) q[2];
ry(3.0932730561269923) q[3];
rz(0.6937102424624478) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.138921640403563) q[0];
rz(-0.5107208512559733) q[0];
ry(-0.3106766740449264) q[1];
rz(0.18676964417674252) q[1];
ry(-2.931059487248118) q[2];
rz(0.7154605535704216) q[2];
ry(-2.3510732391799882) q[3];
rz(0.2829891047258888) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.001684617183882) q[0];
rz(-1.0979205665479999) q[0];
ry(2.0420564923251505) q[1];
rz(0.07457209881098813) q[1];
ry(-0.3028701540194323) q[2];
rz(-0.09593841462816875) q[2];
ry(-2.8384718791257484) q[3];
rz(0.7022708160718714) q[3];